import httpx
import pytest
from induction_fixtures import paired_captures

from circuit_lantern.main import app
from circuit_lantern.services.store import ExperimentStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_saved_probe_comparison_is_available_after_unload_and_keeps_sources(
    runtime, monkeypatch
):
    captures = paired_captures(runtime)
    store = ExperimentStore(runtime.settings.data_dir / "compare.db")
    selected = store.create_artifact("Fine", "Fine notes", "induction_probe", captures[0])
    reference = store.create_artifact("Base", "Base notes", "induction_probe", captures[1])
    originals = [store.get(record["id"]) for record in (selected, reference)]
    monkeypatch.setattr(app.state, "store", store, raising=False)
    runtime.unload()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get(
            "/api/induction/comparison",
            params={"selected": selected["id"], "reference": reference["id"]},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["shared_examples"] == 3
        assert result["selected_record"]["name"] == "Fine"
        assert result["reference_source"]["revision"] == "base-revision"
        assert "result" not in result["selected_record"]
        assert [store.get(record["id"]) for record in (selected, reference)] == originals
        assert (
            await client.get(
                "/api/induction/comparison",
                params={"selected": selected["id"], "reference": selected["id"]},
            )
        ).status_code == 400
        assert (
            await client.get(
                "/api/induction/comparison",
                params={"selected": "missing", "reference": reference["id"]},
            )
        ).status_code == 422
        assert (
            await client.get(
                "/api/induction/comparison",
                params={"selected": "f" * 32, "reference": reference["id"]},
            )
        ).status_code == 404

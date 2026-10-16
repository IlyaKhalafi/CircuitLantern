from unittest.mock import patch

import httpx
import pytest
from tokenizers.processors import TemplateProcessing

from circuit_lantern.limits import MAX_TARGET_PREVIEW_PIECES
from circuit_lantern.main import app
from circuit_lantern.services.runtime import ModelChangedError, RuntimeBusyError


def test_preview_matches_captured_targets_without_running_the_model(runtime):
    targets = [" France", "Poland", "<pad>"]
    metadata = runtime.metadata.copy()
    with patch.object(
        runtime.model, "forward", side_effect=AssertionError("Preview ran the model")
    ):
        preview = runtime.preview_targets(targets)
    assert runtime.metadata == metadata
    assert preview["valid"] and preview["max_targets"] == 8
    assert preview["max_target_characters"] == 100
    assert [target["token_ids"] for target in preview["targets"]] == [[6], [9], [0]]
    assert [target["requested_text"] for target in preview["targets"]] == targets
    assert preview["targets"][2]["special_tokens"] == [True]
    trace = runtime.trace("the capital", 0, False, None, tracked_tokens=targets)
    assert trace["tracked_targets"] == [
        {
            "requested_text": target["requested_text"],
            "token_id": target["token_ids"][0],
            "token": target["tokens"][0],
        }
        for target in preview["targets"]
    ]


def test_empty_entries_are_skipped_but_whitespace_and_split_entries_are_diagnosed(runtime):
    preview = runtime.preview_targets(["", "the capital", "   "])
    empty, split, whitespace = preview["targets"]
    assert empty["skipped"] and empty["valid"]
    assert empty["token_count"] == 0
    assert not split["valid"] and split["token_count"] == 2
    assert split["token_ids"] == [3, 4]
    assert split["pieces"] == ["the", "capital"]
    assert "exactly one token" in split["error"]
    assert not whitespace["skipped"] and not whitespace["valid"]
    assert not preview["valid"]


def test_aliases_report_the_original_entry_including_vocabulary_id_zero(runtime):
    preview = runtime.preview_targets(["<pad>", "<pad>", "", "France", " France"])
    assert not preview["valid"]
    assert preview["targets"][0]["token_ids"] == [0]
    assert preview["targets"][1]["duplicate_of"] == 0
    assert "target 1" in preview["targets"][1]["error"]
    assert preview["targets"][4]["duplicate_of"] == 3
    assert "target 4" in preview["targets"][4]["error"]


def test_tokenizer_added_bos_is_not_part_of_an_explicit_target(runtime):
    runtime.tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="<pad> $A", special_tokens=[("<pad>", 0)]
    )
    preview = runtime.preview_targets(["France"])
    trace = runtime.trace("the capital", 0, False, None, tracked_tokens=["France"])
    assert trace["token_ids"][0] == 0
    assert preview["targets"][0]["token_ids"] == [6]
    assert trace["tracked_targets"][0]["token_id"] == 6


def test_unresized_tokenizer_ids_are_reported_before_model_execution(runtime):
    runtime.tokenizer.add_tokens(["unresized"])
    with patch.object(runtime.model, "forward") as forward:
        preview = runtime.preview_targets(["unresized"])
        assert not preview["valid"]
        assert preview["targets"][0]["token_ids"] == [16]
        assert "outside this model vocabulary" in preview["targets"][0]["error"]
        with pytest.raises(ValueError, match="outside this model vocabulary"):
            runtime.trace("the capital", 0, False, None, tracked_tokens=["unresized"])
    forward.assert_not_called()


def test_large_split_target_counts_remain_complete_with_bounded_pieces(runtime):
    target = "the " * 25
    preview = runtime.preview_targets([target])["targets"][0]
    assert preview["requested_text"] == target
    assert preview["token_count"] == 25
    assert len(preview["token_ids"]) == MAX_TARGET_PREVIEW_PIECES
    assert len(preview["pieces"]) == MAX_TARGET_PREVIEW_PIECES
    assert preview["truncated"] and not preview["valid"]


def test_preview_rejects_busy_or_replaced_models_before_tokenization(runtime):
    runtime.metadata["execution_id"] = "a" * 32
    with patch.object(runtime.tokenizer, "encode") as encode:
        with runtime.lock, pytest.raises(RuntimeBusyError, match="busy"):
            runtime.preview_targets(["France"])
        with pytest.raises(ModelChangedError, match="changed"):
            runtime.preview_targets(["France"], "b" * 32)
    encode.assert_not_called()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_preview_route_bounds_state_conflicts_and_unload(runtime, monkeypatch):
    runtime.metadata["execution_id"] = "a" * 32
    monkeypatch.setattr(app.state, "runtime", runtime, raising=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://preview"
    ) as client:
        with patch.object(runtime.model, "forward") as forward:
            response = await client.post(
                "/api/targets/preview",
                json={
                    "tracked_tokens": ["France", "<pad>"],
                    "expected_execution_id": "a" * 32,
                },
            )
            assert response.status_code == 200
            assert response.json()["valid"]
            assert response.json()["execution_id"] == "a" * 32
            duplicate = await client.post(
                "/api/targets/preview",
                json={
                    "tracked_tokens": ["France", " France"],
                },
            )
            assert duplicate.status_code == 200 and not duplicate.json()["valid"]
            for targets in ([], ["France"] * 9, ["x" * 101], [1]):
                invalid = await client.post(
                    "/api/targets/preview", json={"tracked_tokens": targets}
                )
                assert invalid.status_code == 422
            stale = await client.post(
                "/api/targets/preview",
                json={
                    "tracked_tokens": ["France"],
                    "expected_execution_id": "b" * 32,
                },
            )
            assert stale.status_code == 409
            with runtime.lock:
                busy = await client.post(
                    "/api/targets/preview", json={"tracked_tokens": ["France"]}
                )
                assert busy.status_code == 409
            runtime.unload()
            unloaded = await client.post(
                "/api/targets/preview", json={"tracked_tokens": ["France"]}
            )
            assert unloaded.status_code == 400
        forward.assert_not_called()

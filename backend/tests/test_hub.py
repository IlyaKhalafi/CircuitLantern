import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from model_fixtures import tiny_gpt2

from circuit_lantern.services.hub import _metadata_at_revision, model_metadata


@pytest.fixture
def checkpoint(tmp_path):
    model, _ = tiny_gpt2()
    model.config.save_pretrained(tmp_path)
    return tmp_path


def inspect(checkpoint, files, index=None):
    entries = [SimpleNamespace(rfilename=name, size=size) for name, size in files.items()]
    info = SimpleNamespace(
        sha="pinned-revision",
        siblings=entries,
        safetensors=SimpleNamespace(total=123),
        gated=False,
        downloads=0,
    )
    if index is not None:
        (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))

    def download(model_id, name, revision):
        assert revision == info.sha
        return str(checkpoint / name)

    _metadata_at_revision.cache_clear()
    with (
        patch("circuit_lantern.services.hub.HfApi") as api,
        patch(
            "circuit_lantern.services.hub.hf_hub_download",
            side_effect=download,
        ),
    ):
        api.return_value.model_info.return_value = info
        return model_metadata("test-model")


def test_default_weights_exclude_duplicate_variants_and_include_assets(checkpoint):
    result = inspect(
        checkpoint,
        {
            "model.safetensors": 1000,
            "model.fp16.safetensors": 500,
            "original/model.safetensors": 1000,
            "config.json": 100,
            "tokenizer.json": 200,
        },
    )
    assert result["weight_files"] == ["model.safetensors"]
    assert result["download_bytes"] == 1300
    assert result["parameters"] != 123
    assert result["quantizable_parameters"] > 0
    assert result["architecture_supported"]


def test_index_selects_only_referenced_shards(checkpoint):
    result = inspect(
        checkpoint,
        {
            "config.json": 100,
            "model.safetensors.index.json": 200,
            "model-1.safetensors": 1000,
            "model-2.safetensors": 2000,
            "alternate.safetensors": 9000,
        },
        {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}},
    )
    assert result["weight_files"] == ["model-1.safetensors", "model-2.safetensors"]
    assert result["download_bytes"] == 3300


def test_invalid_index_cannot_download_missing_or_parent_paths(checkpoint):
    with pytest.raises(ValueError, match="invalid or unavailable"):
        inspect(
            checkpoint,
            {
                "config.json": 100,
                "model.safetensors.index.json": 200,
            },
            {"weight_map": {"a": "../weights.safetensors"}},
        )


def test_alternative_or_adapter_weights_without_a_default_checkpoint_are_not_loadable(checkpoint):
    result = inspect(
        checkpoint,
        {
            "config.json": 100,
            "adapter_model.safetensors": 1000,
            "model.fp16.safetensors": 2000,
        },
    )
    assert not result["has_safetensors"]
    assert result["weight_files"] == []
    assert result["download_files"] == ["config.json"]


def test_moving_branch_resolves_fresh_commits_while_pinned_metadata_is_cached(checkpoint):
    _metadata_at_revision.cache_clear()
    current = ["a" * 40]

    def model_info(model_id, revision, **kwargs):
        return SimpleNamespace(
            sha=current[0] if revision == "main" else revision,
            siblings=[SimpleNamespace(rfilename="model.safetensors", size=1000)],
            safetensors=SimpleNamespace(total=123),
            gated=False,
            downloads=0,
        )

    with (
        patch("circuit_lantern.services.hub.HfApi") as api,
        patch(
            "circuit_lantern.services.hub.hf_hub_download",
            return_value=str(checkpoint / "config.json"),
        ) as download,
    ):
        api.return_value.model_info.side_effect = model_info
        first = model_metadata("test-model", "main")
        assert first["revision"] == "a" * 40
        current[0] = "b" * 40
        second = model_metadata("test-model", "main")
        assert second["revision"] == "b" * 40
        assert model_metadata("test-model", first["revision"]) == first
        assert model_metadata("test-model", second["revision"]) == second
        assert download.call_count == 2

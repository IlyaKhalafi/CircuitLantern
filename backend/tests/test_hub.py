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
    assert result["download_file_sizes"] == {
        "model.safetensors": 1000,
        "config.json": 100,
        "tokenizer.json": 200,
    }
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


def test_default_context_is_read_from_the_supported_architecture(checkpoint):
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("n_positions")
    config_path.write_text(json.dumps(config))
    result = inspect(checkpoint, {"model.safetensors": 1000})
    assert result["context_length"] == 1024


def test_architecture_failures_preserve_complete_diagnostics(checkpoint):
    message = "An unsupported architecture. " + "Required details. " * 40 + "Final diagnostic."
    with patch("circuit_lantern.services.hub.architecture_layout", side_effect=ValueError(message)):
        metadata = inspect(checkpoint, {"model.safetensors": 1000})
    assert metadata["architecture_supported"] is False
    assert metadata["architecture_error"] == message


@pytest.mark.parametrize(
    "index",
    [
        {},
        [],
        {"weight_map": None},
        {"weight_map": []},
        {"weight_map": {}},
        {"weight_map": {"a": []}},
        {"weight_map": {"a": 7}},
        {"weight_map": {"a": 7, "b": "model-1.safetensors"}},
        {"weight_map": {"a": "missing.safetensors"}},
    ],
)
def test_malformed_shard_indexes_report_a_validation_error(checkpoint, index):
    with pytest.raises(ValueError, match="checkpoint index"):
        inspect(checkpoint, {"config.json": 100, "model.safetensors.index.json": 200}, index)


@pytest.mark.parametrize("config", [[], None, {"text_config": None}, {"text_config": []}])
def test_non_object_model_configurations_report_a_validation_error(checkpoint, config):
    (checkpoint / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration"):
        inspect(checkpoint, {"config.json": 100, "model.safetensors": 1000})


@pytest.mark.parametrize("size", [None, -1, True, "unknown", [], {}])
def test_unknown_checkpoint_sizes_cannot_be_presented_as_a_complete_download_budget(
    checkpoint, size
):
    with pytest.raises(ValueError, match="file sizes are unavailable"):
        inspect(checkpoint, {"config.json": 100, "model.safetensors": size})


@pytest.mark.parametrize(
    "field",
    [
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "vocab_size",
        "num_key_value_heads",
        "head_dim",
    ],
)
@pytest.mark.parametrize("value", [None, "invalid", True, -1])
def test_malformed_dimensions_return_validation_errors(checkpoint, field, value):
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(json.dumps(config))
    if field in ("num_key_value_heads", "head_dim") and value is None:
        metadata = inspect(checkpoint, {"config.json": 100, "model.safetensors": 1000})
        assert metadata["num_kv_heads" if field == "num_key_value_heads" else field] == (
            metadata["num_heads"]
            if field == "num_key_value_heads"
            else metadata["hidden_size"] // metadata["num_heads"]
        )
        return
    with pytest.raises(ValueError, match="configuration has an invalid"):
        inspect(checkpoint, {"config.json": 100, "model.safetensors": 1000})


def test_invalid_json_uses_an_actionable_configuration_error(checkpoint):
    (checkpoint / "config.json").write_text("{")
    with pytest.raises(ValueError, match="configuration contains invalid JSON"):
        inspect(checkpoint, {"config.json": 100, "model.safetensors": 1000})


@pytest.mark.parametrize("value", [False, True, 1, "fp8", [], {}, ["fp8"]])
def test_malformed_quantization_metadata_reports_a_validation_error(checkpoint, value):
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="quantization configuration"):
        inspect(checkpoint, {"config.json": 100, "model.safetensors": 1000})


@pytest.mark.parametrize("content", [b"\xff", b"[" * 2000])
def test_unreadable_or_excessively_nested_json_reports_a_validation_error(checkpoint, content):
    (checkpoint / "config.json").write_bytes(content)
    with pytest.raises(ValueError, match="configuration contains invalid JSON"):
        inspect(checkpoint, {"config.json": 100, "model.safetensors": 1000})


def test_model_metadata_reads_utf8_configuration_text(checkpoint):
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["display_name"] = "مدل پژوهشی"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    assert (
        inspect(checkpoint, {"model.safetensors": 1000})["config"]["display_name"]
        == config["display_name"]
    )


def test_outer_null_quantization_preserves_the_nested_text_configuration(checkpoint):
    from circuit_lantern.services.execution import checkpoint_precision

    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    quantization = {
        "quant_method": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
    }
    config["quantization_config"] = quantization
    path.write_text(
        json.dumps(
            {
                "model_type": "unsupported_container",
                "quantization_config": None,
                "text_config": config,
            }
        )
    )
    metadata = inspect(checkpoint, {"config.json": 100, "model.safetensors": 1000})
    assert metadata["quantization"] == quantization
    assert checkpoint_precision(metadata) == "int4"


def test_public_metadata_summarizes_fp8_storage_without_copying_the_operator_index(checkpoint):
    planned = {
        "supported": True,
        "matrix_count": 1,
        "residual_dtype": "float32",
        "resident_weight_bytes": 3456,
        "config": {"weight_block_size": [128, 128]},
        "block_size": [128, 128],
        "matrices": [{"name": "model.layers.0.self_attn.q_proj.weight", "shape": [128, 128]}],
    }
    with patch(
        "circuit_lantern.services.hub.architecture_layout",
        return_value={"parameters": 1234, "fp8_layout": planned},
    ):
        result = inspect(checkpoint, {"model.safetensors": 100, "config.json": 100})
    assert result["fp8_layout"]["matrix_count"] == 1
    assert result["fp8_layout"]["resident_weight_bytes"] == 3456
    assert "matrices" not in result["fp8_layout"]
    assert len(planned["matrices"]) == 1
    result["fp8_layout"]["config"]["weight_block_size"][0] = 64
    result["fp8_layout"]["block_size"][1] = 64
    assert planned["config"]["weight_block_size"] == [128, 128]
    assert planned["block_size"] == [128, 128]

import json
import re
from copy import deepcopy
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download

from circuit_lantern.services.architecture import architecture_layout

CURATED_MODELS = [
    {
        "id": "HuggingFaceTB/SmolLM2-135M",
        "name": "SmolLM2 135M",
        "family": "Llama",
        "parameters": "135M",
        "description": "A compact starting point for real circuit exploration.",
        "tag": "Recommended",
        "color": "orange",
    },
    {
        "id": "openai-community/gpt2",
        "name": "GPT-2",
        "family": "GPT-2",
        "parameters": "124M",
        "description": "The classic transformer with a rich interpretability literature.",
        "tag": "Research classic",
        "color": "blue",
    },
    {
        "id": "EleutherAI/pythia-70m",
        "name": "Pythia 70M",
        "family": "GPT-NeoX",
        "parameters": "70M",
        "description": "Designed for research with checkpoints across training.",
        "tag": "Training checkpoints",
        "color": "purple",
    },
    {
        "id": "HuggingFaceTB/SmolLM2-360M",
        "name": "SmolLM2 360M",
        "family": "Llama",
        "parameters": "360M",
        "description": "More capacity while keeping inspection approachable.",
        "tag": "Small footprint",
        "color": "green",
    },
    {
        "id": "Qwen/Qwen2.5-0.5B",
        "name": "Qwen 2.5 0.5B",
        "family": "Qwen2",
        "parameters": "0.5B",
        "description": "Explore grouped query attention and multilingual representations.",
        "tag": "Multilingual",
        "color": "blue",
    },
    {
        "id": "hf-internal-testing/tiny-random-gpt2",
        "name": "Tiny random GPT-2",
        "family": "GPT-2",
        "parameters": "112K",
        "description": "A fast, untrained model for testing the complete workflow.",
        "tag": "Workflow test",
        "color": "gray",
    },
]


def search_models(query: str) -> list[dict]:
    if not query.strip():
        return CURATED_MODELS
    api = HfApi()
    models = api.list_models(
        search=query.strip(),
        pipeline_tag="text-generation",
        sort="downloads",
        limit=18,
    )
    return [
        {
            "id": m.id,
            "name": m.id.split("/")[-1],
            "family": m.library_name or "transformers",
            "parameters": "Hub model",
            "description": m.id,
            "tag": "Hugging Face",
            "color": "blue",
        }
        for m in models
    ]


def model_metadata(model_id: str, revision: str = "main") -> dict:
    # Branches and tags may move between deployment checks. Cache immutable commits only.
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        revision = HfApi().model_info(model_id, revision=revision, timeout=20).sha
    return _metadata_at_revision(model_id, revision)


def _checkpoint_shards(index: object, available: dict) -> list[str]:
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or any(
            not isinstance(key, str) or not key or not isinstance(value, str)
            for key, value in weight_map.items()
        )
    ):
        raise ValueError("The checkpoint index must contain a nonempty tensor to shard mapping.")
    names = sorted(set(weight_map.values()))
    if any(
        name not in available
        or PurePosixPath(name).is_absolute()
        or ".." in PurePosixPath(name).parts
        or not name.endswith(".safetensors")
        for name in names
    ):
        raise ValueError(
            "The checkpoint index references invalid or unavailable safetensors files."
        )
    return names


def _checkpoint_json(path: str, error_message: str):
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, UnicodeError, RecursionError) as error:
        raise ValueError(error_message) from error


def _config_dimension(config: dict, key: str, alias: str = "", default: int = 0) -> int:
    value = config.get(key, config.get(alias, default))
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"The model configuration has an invalid {key} value. Expected a nonnegative integer."
        )
    return value


@lru_cache(maxsize=64)
def _metadata_at_revision(model_id: str, revision: str) -> dict:
    api = HfApi()
    info = api.model_info(model_id, revision=revision, files_metadata=True, timeout=20)
    config_path = hf_hub_download(model_id, "config.json", revision=info.sha)
    raw = _checkpoint_json(config_path, "The model configuration contains invalid JSON.")
    if not isinstance(raw, dict):
        # Invalid remote JSON is a request validation failure.
        raise ValueError("The model configuration must contain a JSON object.")  # noqa: TRY004
    config = raw.get("text_config", raw)
    if not isinstance(config, dict):
        raise ValueError("The model text configuration must contain a JSON object.")  # noqa: TRY004
    hidden = _config_dimension(config, "hidden_size", "n_embd")
    layers = _config_dimension(config, "num_hidden_layers", "n_layer")
    heads = _config_dimension(config, "num_attention_heads", "n_head", 1)
    vocab = _config_dimension(config, "vocab_size")
    kv_heads = (
        _config_dimension(config, "num_key_value_heads", default=heads)
        if config.get("num_key_value_heads") is not None
        else heads
    )
    head_dim = (
        _config_dimension(config, "head_dim")
        if config.get("head_dim") is not None
        else hidden // max(heads, 1)
    )
    quantization = raw.get("quantization_config")
    if quantization is None:
        quantization = config.get("quantization_config")
    if quantization is not None and (not isinstance(quantization, dict) or not quantization):
        raise ValueError(
            "The checkpoint quantization configuration must be a nonempty JSON object."
        )
    tensor_info = info.safetensors
    parameters = tensor_info.total if tensor_info else None
    siblings = {entry.rfilename: entry for entry in info.siblings}
    if "model.safetensors" in siblings:
        weight_names = ["model.safetensors"]
    elif "model.safetensors.index.json" in siblings:
        index_path = hf_hub_download(model_id, "model.safetensors.index.json", revision=info.sha)
        index = _checkpoint_json(index_path, "The checkpoint index contains invalid JSON.")
        weight_names = _checkpoint_shards(index, siblings)
    else:
        # AutoModel expects the standard filename or its shard index.
        # Alternative variants and adapter-only files cannot establish a loadable checkpoint.
        weight_names = []
    assets = [
        name
        for name in siblings
        if any(fnmatch(name, pattern) for pattern in ("*.json", "*.txt", "*.model", "*.tiktoken"))
    ]
    download_files = sorted(set(weight_names + assets))
    if any(
        not isinstance(siblings[name].size, int)
        or isinstance(siblings[name].size, bool)
        or siblings[name].size < 0
        for name in download_files
    ):
        raise ValueError(
            "The checkpoint file sizes are unavailable. Storage compatibility cannot be verified."
        )
    weight_bytes = sum(siblings[name].size for name in weight_names)
    download_bytes = sum(siblings[name].size or 0 for name in download_files)
    try:
        layout = architecture_layout(str(Path(config_path).parent))
        architecture_supported = True
        architecture_error = None
    except (ValueError, TypeError, ImportError, AttributeError, RuntimeError) as error:
        layout = {}
        architecture_supported = False
        architecture_error = str(error)[:250]
    if layout:
        parameters = layout["parameters"]
    if not parameters and hidden and layers:
        intermediate_config = {key: value for key, value in config.items() if value is not None}
        intermediate = _config_dimension(
            intermediate_config, "intermediate_size", "n_inner", hidden * 4
        )
        parameters = int(2 * vocab * hidden + layers * (4 * hidden**2 + 3 * hidden * intermediate))
    fp8_layout = layout.get("fp8_layout")
    if fp8_layout is not None:
        fp8_layout = deepcopy(
            {key: value for key, value in fp8_layout.items() if key != "matrices"}
        )
    return {
        "id": model_id,
        "revision": info.sha,
        "architecture": config.get("model_type", "unknown"),
        "parameters": parameters,
        "parameter_estimated": not bool(tensor_info) or bool(quantization),
        "quantizable_parameters": layout.get("quantizable_parameters"),
        "fp8_layout": fp8_layout,
        "largest_matrix_elements": layout.get("largest_matrix_elements"),
        "model_buffer_bytes": layout.get("model_buffer_bytes"),
        "largest_model_buffer_bytes": layout.get("largest_model_buffer_bytes"),
        "tied_embeddings": layout.get("tied_embeddings"),
        "architecture_supported": architecture_supported,
        "architecture_error": architecture_error,
        "weight_bytes": weight_bytes,
        "has_safetensors": bool(weight_names),
        "weight_files": weight_names,
        "download_files": download_files,
        "download_file_sizes": {name: siblings[name].size for name in download_files},
        "download_bytes": download_bytes,
        "hidden_size": hidden,
        "num_layers": layers,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "context_length": config.get("max_position_embeddings")
        or config.get("n_positions")
        or layout.get("context_length"),
        "vocab_size": vocab,
        "dtype": config.get("dtype", config.get("torch_dtype", "float32")),
        "quantization": quantization,
        "gated": bool(info.gated),
        "downloads": info.downloads,
        "config": config,
    }

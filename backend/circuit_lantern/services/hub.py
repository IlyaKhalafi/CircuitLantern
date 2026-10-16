import json
import re
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


@lru_cache(maxsize=64)
def _metadata_at_revision(model_id: str, revision: str) -> dict:
    api = HfApi()
    info = api.model_info(model_id, revision=revision, files_metadata=True, timeout=20)
    config_path = hf_hub_download(model_id, "config.json", revision=info.sha)
    with open(config_path) as file:
        raw = json.load(file)
    config = raw.get("text_config", raw)
    tensor_info = info.safetensors
    parameters = tensor_info.total if tensor_info else None
    siblings = {entry.rfilename: entry for entry in info.siblings}
    if "model.safetensors" in siblings:
        weight_names = ["model.safetensors"]
    elif "model.safetensors.index.json" in siblings:
        index_path = hf_hub_download(model_id, "model.safetensors.index.json", revision=info.sha)
        with open(index_path) as file:
            weight_names = sorted(set(json.load(file)["weight_map"].values()))
        if any(
            name not in siblings
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or not name.endswith(".safetensors")
            for name in weight_names
        ):
            raise ValueError(
                "The checkpoint index references invalid or unavailable safetensors files."
            )
    else:
        # AutoModel expects the standard filename or its shard index.
        # Alternative variants and adapter-only files cannot establish a loadable checkpoint.
        weight_names = []
    weight_bytes = sum(siblings[name].size or 0 for name in weight_names)
    assets = [
        name
        for name in siblings
        if any(fnmatch(name, pattern) for pattern in ("*.json", "*.txt", "*.model", "*.tiktoken"))
    ]
    download_files = sorted(set(weight_names + assets))
    download_bytes = sum(siblings[name].size or 0 for name in download_files)
    try:
        layout = architecture_layout(str(Path(config_path).parent))
        architecture_supported = True
        architecture_error = None
    except (ValueError, TypeError, ImportError, AttributeError, RuntimeError) as error:
        layout = {}
        architecture_supported = False
        architecture_error = str(error)[:250]
    hidden = config.get("hidden_size", config.get("n_embd", 0))
    layers = config.get("num_hidden_layers", config.get("n_layer", 0))
    heads = config.get("num_attention_heads", config.get("n_head", 1))
    vocab = config.get("vocab_size", 0)
    quantization = raw.get("quantization_config", config.get("quantization_config"))
    if layout:
        parameters = layout["parameters"]
    if not parameters and hidden and layers:
        intermediate = config.get("intermediate_size") or config.get("n_inner") or hidden * 4
        parameters = int(2 * vocab * hidden + layers * (4 * hidden**2 + 3 * hidden * intermediate))
    return {
        "id": model_id,
        "revision": info.sha,
        "architecture": config.get("model_type", "unknown"),
        "parameters": parameters,
        "parameter_estimated": not bool(tensor_info) or bool(quantization),
        "quantizable_parameters": layout.get("quantizable_parameters"),
        "largest_matrix_elements": layout.get("largest_matrix_elements"),
        "tied_embeddings": layout.get("tied_embeddings"),
        "architecture_supported": architecture_supported,
        "architecture_error": architecture_error,
        "weight_bytes": weight_bytes,
        "has_safetensors": bool(weight_names),
        "weight_files": weight_names,
        "download_files": download_files,
        "download_bytes": download_bytes,
        "hidden_size": hidden,
        "num_layers": layers,
        "num_heads": heads,
        "num_kv_heads": config.get("num_key_value_heads") or heads,
        "head_dim": config.get("head_dim") or hidden // max(heads, 1),
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

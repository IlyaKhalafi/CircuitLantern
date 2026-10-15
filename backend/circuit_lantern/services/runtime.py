import gc
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch
from huggingface_hub import snapshot_download
from torch.nn import functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from circuit_lantern.config import Settings
from circuit_lantern.services.metrics import matrix_metrics


class ModelRuntime:
    """Own one model and serialize operations that touch its weights or hooks."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.metadata = None
        self.device = "cpu"
        self.lock = threading.Lock()
        self.job_lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-loader")
        self.metrics_cache: dict[str, dict] = {}
        torch.set_num_threads(min(4, torch.get_num_threads()))

    def state(self) -> dict:
        return {"loaded": self.model is not None, "metadata": self.metadata, "device": self.device}

    def load(self, metadata: dict, device: str) -> dict:
        with self.job_lock:
            if any(j["status"] in ("queued", "downloading", "loading") for j in self.jobs.values()):
                raise ValueError("A model download is already running. Wait for it to finish.")
            job = {
                "id": uuid.uuid4().hex,
                "model_id": metadata["id"],
                "status": "queued",
                "progress": 0,
                "message": "Waiting for model loader",
                "created_at": time.time(),
            }
            # Keep completed jobs bounded in a long running workstation.
            self.jobs = dict(list(self.jobs.items())[-49:])
            self.jobs[job["id"]] = job
        self.executor.submit(self._load, job, metadata, device)
        return dict(job)

    def _load(self, job: dict, metadata: dict, device: str):
        with self.lock:
            try:
                job.update(
                    status="downloading",
                    progress=10,
                    message="Downloading safetensors checkpoint from Hugging Face",
                )
                path = snapshot_download(
                    metadata["id"],
                    revision=metadata["revision"],
                    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken"],
                    max_workers=2,
                )
                job.update(
                    status="loading",
                    progress=70,
                    message="Loading model and building the weight index",
                )
                self._unload()
                tokenizer = AutoTokenizer.from_pretrained(
                    path, trust_remote_code=False, local_files_only=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    trust_remote_code=False,
                    local_files_only=True,
                    torch_dtype=torch.float32,
                    use_safetensors=True,
                    attn_implementation="eager",
                )
                model.eval().to(device)
                self.model, self.tokenizer, self.metadata, self.device = (
                    model,
                    tokenizer,
                    metadata,
                    device,
                )
                job.update(status="ready", progress=100, message="Model is ready for inspection")
            except Exception as error:  # noqa: BLE001 - worker failures must reach the job record
                self._unload()
                job.update(status="failed", progress=0, message=str(error)[:1000])

    def _unload(self):
        self.model = self.tokenizer = self.metadata = None
        self.metrics_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self):
        with self.lock:
            self._unload()

    def _require_model(self):
        if self.model is None:
            raise ValueError("Load a model from the model library first.")

    def graph(self) -> dict:
        with self.lock:
            self._require_model()
            weights = [
                {
                    "name": name,
                    "shape": list(weight.shape),
                    "elements": weight.numel(),
                    "dtype": str(weight.dtype).removeprefix("torch."),
                }
                for name, weight in self.model.named_parameters()
                if weight.ndim == 2
            ]
            return {
                "metadata": self.metadata,
                "matrices": weights,
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
            }

    def matrix(self, name: str) -> dict:
        with self.lock:
            self._require_model()
            if name in self.metrics_cache:
                return self.metrics_cache[name]
            parameter = dict(self.model.named_parameters()).get(name)
            if parameter is None:
                raise ValueError("Weight matrix was not found in the loaded model.")
            result = dict(
                name=name, **matrix_metrics(parameter, self.settings.max_analysis_elements)
            )
            # A few large heatmaps are useful to keep, without unbounded retention.
            if len(self.metrics_cache) >= 24:
                self.metrics_cache.pop(next(iter(self.metrics_cache)))
            self.metrics_cache[name] = result
            return result

    def _blocks(self):
        for root_name, block_name in (
            ("transformer", "h"),
            ("model", "layers"),
            ("gpt_neox", "layers"),
        ):
            root = getattr(self.model, root_name, None)
            if root is not None and hasattr(root, block_name):
                return getattr(root, block_name)
        return None

    def _final_norm(self):
        for root_name, norm_name in (
            ("transformer", "ln_f"),
            ("model", "norm"),
            ("gpt_neox", "final_layer_norm"),
        ):
            root = getattr(self.model, root_name, None)
            if root is not None and hasattr(root, norm_name):
                return getattr(root, norm_name)
        return None

    @torch.inference_mode()
    def trace(
        self, prompt: str, max_new_tokens: int, capture_attention: bool, ablate_layer: int | None
    ) -> dict:
        with self.lock:
            self._require_model()
            started = time.perf_counter()
            encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            token_count = encoded.input_ids.shape[1]
            context_limit = self.metadata["context_length"]
            if token_count > min(self.settings.max_prompt_tokens, context_limit - max_new_tokens):
                raise ValueError(
                    "Prompt is too long for interactive capture or the model context window. Shorten it to at most 128 tokens."
                )
            inputs = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            handles = []
            if ablate_layer is not None:
                blocks = self._blocks()
                if blocks is None or ablate_layer >= len(blocks):
                    raise ValueError("The selected layer cannot be bypassed in this architecture.")

                def bypass(module, args, output):
                    return (args[0], *output[1:]) if isinstance(output, tuple) else args[0]

                handles.append(blocks[ablate_layer].register_forward_hook(bypass))
            try:
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=capture_attention,
                    use_cache=False,
                )
                final_norm = self._final_norm()
                unembed = self.model.get_output_embeddings()
                tokens = [self.tokenizer.decode([i]) for i in inputs["input_ids"][0].tolist()]
                layers = []
                previous = None
                for index, hidden in enumerate(outputs.hidden_states):
                    # The final hidden state is already normalized by Transformers.
                    projected = (
                        final_norm(hidden)
                        if final_norm is not None and index < len(outputs.hidden_states) - 1
                        else hidden
                    )
                    logits = unembed(projected).float()
                    probabilities = torch.softmax(logits[0, -1], dim=-1)
                    values, indices = probabilities.topk(min(5, probabilities.numel()))
                    vector = hidden[0].float()
                    layers.append(
                        {
                            "layer": index - 1,
                            "label": "Embedding" if index == 0 else f"Layer {index - 1}",
                            "rms": float(vector.square().mean().sqrt()),
                            "norm": float(vector[-1].norm()),
                            "token_norms": vector.norm(dim=-1).cpu().tolist(),
                            "cosine_similarity": float(
                                functional.cosine_similarity(vector[-1], previous, dim=0)
                            )
                            if previous is not None
                            else None,
                            "entropy": float(
                                -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
                            ),
                            "predictions": [
                                {
                                    "token": self.tokenizer.decode([int(i)]),
                                    "probability": float(v),
                                    "token_id": int(i),
                                }
                                for i, v in zip(indices, values)
                            ],
                            "token_predictions": [
                                self.tokenizer.decode([i])
                                for i in logits[0].argmax(-1).cpu().tolist()
                            ],
                        }
                    )
                    previous = vector[-1]
                attention = []
                if outputs.attentions is not None:
                    for index, tensor in enumerate(outputs.attentions):
                        if tensor is None:
                            continue
                        values = tensor[0].float()
                        entropy = -(values * values.clamp_min(1e-30).log()).sum(-1).mean(-1)
                        attention.append(
                            {
                                "layer": index,
                                "pattern": values.mean(0).cpu().tolist(),
                                "head_entropy": entropy.cpu().tolist(),
                                "heads": values.shape[0],
                            }
                        )
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
                continuation = self.tokenizer.decode(
                    generated[0, token_count:], skip_special_tokens=True
                )
                final_probabilities = torch.softmax(outputs.logits[0, -1].float(), -1)
                top_values, top_indices = final_probabilities.topk(10)
                return {
                    "model_id": self.metadata["id"],
                    "revision": self.metadata["revision"],
                    "prompt": prompt,
                    "tokens": tokens,
                    "token_ids": inputs["input_ids"][0].cpu().tolist(),
                    "continuation": continuation,
                    "layers": layers,
                    "attention": attention,
                    "predictions": [
                        {
                            "token": self.tokenizer.decode([int(i)]),
                            "probability": float(v),
                            "token_id": int(i),
                            "logit": float(outputs.logits[0, -1, i]),
                        }
                        for i, v in zip(top_indices, top_values)
                    ],
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                    "device": self.device,
                    "ablate_layer": ablate_layer,
                    "method": "Logit lens with the model final normalization. Intermediate predictions are diagnostic projections, not calibrated probabilities.",
                    "final_state_normalized": True,
                }
            finally:
                for handle in handles:
                    handle.remove()

import gc
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from circuit_lantern.config import Settings
from circuit_lantern.limits import MAX_TARGET_CHARACTERS, MAX_TRACKED_TARGETS
from circuit_lantern.schemas import CompatibilityRequest, HeadAblation
from circuit_lantern.services.cache import checkpoint_cache
from circuit_lantern.services.causal import activation_patch_sweep
from circuit_lantern.services.compatibility import assess_compatibility
from circuit_lantern.services.dataflow import block_dataflow
from circuit_lantern.services.execution import (
    PACKED_PRECISIONS,
    describe_execution,
    execution_provenance,
    loading_options,
    prepare_model_execution,
)
from circuit_lantern.services.generation import decode_generation, generation_settings
from circuit_lantern.services.hardware import hardware_snapshot
from circuit_lantern.services.head_sweep import MAX_SWEEP_HEADS, head_ablation_sweep
from circuit_lantern.services.interventions import attention_head_targets, install_head_ablation
from circuit_lantern.services.jobs import JobRecord
from circuit_lantern.services.metrics import matrix_metrics
from circuit_lantern.services.targets import preview_target_tokens, resolve_tracked_targets
from circuit_lantern.services.tokens import capture_token_limit, preview_tokens
from circuit_lantern.services.tracing import summarize_trace
from circuit_lantern.services.weights import (
    logical_weight,
    model_weight_inventory,
    resident_tensor_bytes,
)


class RuntimeBusyError(ValueError):
    pass


class ModelChangedError(ValueError):
    pass


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
        self.jobs: dict[str, JobRecord] = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-loader")
        self.metrics_cache: dict[str, dict] = {}
        self._state = {"loaded": False, "metadata": None, "device": "cpu"}
        torch.set_num_threads(min(4, torch.get_num_threads()))

    def state(self) -> dict:
        return dict(
            self._state,
            max_prompt_tokens=self.settings.max_prompt_tokens,
            max_head_sweep_heads=MAX_SWEEP_HEADS,
        )

    def assess_fit(self, metadata: dict, request: CompatibilityRequest) -> dict:
        with self.operation():
            return self._assess_fit(metadata, request)

    def _assess_fit(self, metadata: dict, request: CompatibilityRequest) -> dict:
        # Caller holds the operation lock for consistent free and owned memory.
        snapshot = hardware_snapshot(self.settings.data_dir)
        snapshot["reclaimable_tensor_bytes"] = resident_tensor_bytes(self.model)
        snapshot["max_model_bytes"] = self.settings.max_model_bytes
        snapshot["max_prompt_tokens"] = self.settings.max_prompt_tokens
        return assess_compatibility(metadata, snapshot, request, cache=checkpoint_cache(metadata))

    def load(
        self, metadata: dict, device: str, precision: str = "auto", context_length: int = 128
    ) -> dict:
        with self.job_lock:
            if any(
                j["status"] in ("queued", "running", "downloading", "loading")
                for j in self.jobs.values()
            ):
                raise ValueError(
                    "A background operation is already running. Wait for it to finish."
                )
            job = JobRecord(
                {
                    "id": uuid.uuid4().hex,
                    "model_id": metadata["id"],
                    "kind": "download",
                    "status": "queued",
                    "progress": 0,
                    "message": "Waiting for model loader",
                    "created_at": time.time(),
                }
            )
            # Keep completed jobs bounded in a long running workstation.
            self.jobs = dict(list(self.jobs.items())[-49:])
            self.jobs[job["id"]] = job
        try:
            self.executor.submit(self._load, job, metadata, device, precision, context_length)
        except RuntimeError:
            job.update(status="failed", message="The model loader is shutting down.")
            raise
        return job.snapshot()

    def _load(
        self,
        job: JobRecord | dict,
        metadata: dict,
        device: str,
        precision: str = "auto",
        context_length: int = 128,
    ):
        try:
            with self.lock:
                options, resolved_precision = loading_options(metadata, precision, device)
            job.update(
                status="downloading",
                progress=10,
                message="Downloading safetensors checkpoint from Hugging Face",
            )
            path = snapshot_download(
                metadata["id"],
                revision=metadata["revision"],
                allow_patterns=metadata.get("download_files")
                or ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken"],
                max_workers=2,
            )
        except Exception as error:  # noqa: BLE001 - a failed download must preserve the active model
            job.update(status="failed", progress=0, message=str(error)[:1000])
            return
        with self.lock:
            model = tokenizer = None
            released_model = False
            replacement_started = False
            try:
                job.update(
                    status="loading",
                    progress=70,
                    message="Loading model and building the weight index",
                )
                report = self._assess_fit(
                    metadata,
                    CompatibilityRequest(
                        model_id=metadata["id"],
                        revision=metadata["revision"],
                        device=device,
                        precision=precision,
                        context_length=min(context_length, metadata["context_length"]),
                    ),
                )
                if not report["compatible"]:
                    raise ValueError(
                        "Server resources changed before loading. " + " ".join(report["blockers"])
                    )
                released_model = self.model is not None
                replacement_started = True
                self._unload()
                tokenizer = AutoTokenizer.from_pretrained(
                    path, trust_remote_code=False, local_files_only=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    trust_remote_code=False,
                    local_files_only=True,
                    **options,
                    use_safetensors=True,
                    attn_implementation="eager",
                )
                prepare_model_execution(model, resolved_precision, device)
                model.eval()
                if resolved_precision not in PACKED_PRECISIONS:
                    model.to("cuda:0" if device == "cuda" else device)
                context_limit = min(context_length, metadata["context_length"])
                execution_config = describe_execution(model, resolved_precision, device)
                execution_config["context_length"] = context_limit
                metadata = dict(
                    metadata,
                    execution_id=uuid.uuid4().hex,
                    execution_precision=resolved_precision,
                    execution_context_length=context_limit,
                    execution_config=execution_config,
                )
                self.model, self.tokenizer, self.metadata, self.device = (
                    model,
                    tokenizer,
                    metadata,
                    device,
                )
                self._state = {
                    "loaded": True,
                    "metadata": metadata,
                    "device": device,
                    "precision": resolved_precision,
                    "context_length": context_limit,
                    "attention_head_targets": attention_head_targets(self._blocks()),
                }
                job.update(status="ready", progress=100, message="Model is ready for inspection")
            except Exception as error:  # noqa: BLE001 - worker failures must reach the job record
                model = tokenizer = None
                job.update(
                    status="failed",
                    progress=0,
                    message=str(error)[:800]
                    + (
                        " The previous model was released before loading the replacement."
                        if released_model
                        else ""
                    ),
                )
            if job["status"] == "failed" and (replacement_started or self.model is None):
                self._unload()

    def _unload(self):
        self.model = self.tokenizer = self.metadata = None
        self.device = "cpu"
        self._state = {"loaded": False, "metadata": None, "device": self.device}
        self.metrics_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def causal(self, payload: dict) -> dict:
        return self._start_sweep("causal", payload)

    def head_sweep(self, payload: dict) -> dict:
        return self._start_sweep("head_sweep", payload)

    def _start_sweep(self, kind: str, payload: dict, worker=None) -> dict:
        payload = dict(payload)
        expected_execution_id = payload.pop("expected_execution_id", None)
        with self.job_lock:
            state = self.state()
            if not state["loaded"]:
                raise ValueError("Load a model from the model library first.")
            self._verify_execution(expected_execution_id)
            if any(
                j["status"] in ("queued", "running", "downloading", "loading")
                for j in self.jobs.values()
            ):
                raise ValueError("Another background operation is running. Wait for it to finish.")
            job = JobRecord(
                {
                    "id": uuid.uuid4().hex,
                    "model_id": state["metadata"]["id"],
                    "execution_id": state["metadata"].get("execution_id"),
                    "kind": kind,
                    "status": "queued",
                    "progress": 0,
                    "message": "Preparing research sweep",
                    "created_at": time.time(),
                }
            )
            revision = state["metadata"]["revision"]
            self.jobs = dict(list(self.jobs.items())[-49:])
            self.jobs[job["id"]] = job
            try:
                self.executor.submit(self._run_sweep, job, payload, revision, worker)
            except RuntimeError:
                job.update(status="failed", message="The model loader is shutting down.")
                raise
        return job.snapshot()

    def job(self, job_id: str) -> dict | None:
        with self.job_lock:
            result = self.jobs.get(job_id)
            return result.snapshot() if result is not None else None

    def cancel_job(self, job_id: str) -> dict | None:
        with self.job_lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job.get("kind") not in ("causal", "head_sweep", "sae_training", "feature_circuit"):
                raise ValueError(
                    "Only research jobs can be cancelled. Model downloads must finish before another operation starts."
                )
            return job.request_cancel()

    def _run_sweep(self, job: JobRecord, payload: dict, revision: str, worker=None):
        is_head_sweep = job["kind"] == "head_sweep"
        label = {
            "sae_training": "Dictionary training",
            "feature_circuit": "Feature ablation graph",
        }.get(job["kind"], "Head sweep" if is_head_sweep else "Causal sweep")
        result = None
        failure = None
        early_cancel = False
        with self.lock:
            try:
                early_cancel = bool(job.get("cancel_requested"))
                if not early_cancel:
                    self._require_model()
                    if (
                        self.metadata["revision"] != revision
                        or self.metadata["id"] != job["model_id"]
                        or self.metadata.get("execution_id") != job.get("execution_id")
                    ):
                        raise ValueError("The active model changed. Start the sweep again.")
                    job.update(status="running", message="Capturing baseline predictions")
                    if worker is not None:
                        result = worker(payload, job)
                    else:
                        operation = head_ablation_sweep if is_head_sweep else activation_patch_sweep
                        result = operation(
                            self.model,
                            self.tokenizer,
                            self._blocks(),
                            self.metadata,
                            self.device,
                            **payload,
                            job=job,
                            max_tokens=self.settings.max_prompt_tokens,
                        )
            except Exception as error:  # noqa: BLE001 - background failures are reported through jobs
                failure = str(error)[:1000]
            finally:
                gc.collect()
                if self.device.startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError as error:
                        failure = failure or str(error)[:1000]
        # A terminal snapshot guarantees that model hooks and operation ownership
        # have been released before a caller starts its next research operation.
        if early_cancel:
            job.update(status="cancelled", message=f"{label} cancelled before capture")
        elif failure:
            job.update(status="failed", message=failure)
        else:
            job.finish(
                result,
                label=label,
                unit="dictionaries"
                if job["kind"] == "sae_training"
                else "measurements"
                if worker
                else "heads"
                if is_head_sweep
                else "layers",
            )

    @contextmanager
    def operation(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeBusyError(
                "The model is busy with another operation. Wait for it to finish or cancel the active sweep."
            )
        try:
            yield
        finally:
            self.lock.release()

    def unload(self, expected_execution_id: str | None = None):
        with self.operation():
            self._verify_execution(expected_execution_id)
            self._unload()

    def _require_model(self):
        if self.model is None:
            raise ValueError("Load a model from the model library first.")

    def _verify_execution(self, expected_execution_id: str | None):
        if expected_execution_id is not None and (
            self.metadata is None or self.metadata.get("execution_id") != expected_execution_id
        ):
            raise ModelChangedError(
                "The active model changed since this page opened. Refresh the runtime before continuing."
            )

    def graph(self, expected_execution_id: str | None = None) -> dict:
        with self.operation():
            self._require_model()
            self._verify_execution(expected_execution_id)
            weights, total_parameters = model_weight_inventory(self.model)
            return {
                "metadata": self.metadata,
                "matrices": weights,
                "dataflow": block_dataflow(self.model, self._blocks(), weights),
                "total_parameters": total_parameters,
            }

    def matrix(self, name: str, expected_execution_id: str | None = None) -> dict:
        with self.operation():
            self._require_model()
            self._verify_execution(expected_execution_id)
            if name in self.metrics_cache:
                return dict(self.metrics_cache[name])
            tensor, description = logical_weight(self.model, name)
            result = dict(
                name=name,
                analysis_id=uuid.uuid4().hex,
                device=self.device,
                **matrix_metrics(tensor, self.settings.max_analysis_elements),
                source_format=description["dtype"],
                quantized=description["quantized"],
                **execution_provenance(self.metadata),
                analysis_note="Statistics describe the dequantized float32 matrix."
                if description["quantized"]
                else "Statistics describe the loaded matrix converted to float32.",
            )
            # A few large heatmaps are useful to keep, without unbounded retention.
            if len(self.metrics_cache) >= 24:
                self.metrics_cache.pop(next(iter(self.metrics_cache)))
            self.metrics_cache[name] = result
            return result

    def matrix_snapshot(self, analysis_id: str, expected_execution_id: str | None = None) -> dict:
        with self.operation():
            if self.model is None:
                raise ModelChangedError(
                    "This matrix analysis is no longer cached. Reopen the matrix before saving it."
                )
            self._verify_execution(expected_execution_id)
            for result in self.metrics_cache.values():
                if result.get("analysis_id") == analysis_id:
                    return dict(result)
            raise ModelChangedError(
                "This matrix analysis is no longer cached. Reopen the matrix before saving it."
            )

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

    def tokenize(
        self, prompts: list[str], reserved_tokens: int = 0, expected_execution_id: str | None = None
    ) -> dict:
        with self.operation():
            self._require_model()
            self._verify_execution(expected_execution_id)
            return preview_tokens(
                self.tokenizer,
                self.metadata,
                prompts,
                self.settings.max_prompt_tokens,
                reserved_tokens,
            )

    def preview_targets(
        self, tracked_tokens: list[str], expected_execution_id: str | None = None
    ) -> dict:
        with self.operation():
            self._require_model()
            self._verify_execution(expected_execution_id)
            targets = preview_target_tokens(
                self.tokenizer, tracked_tokens, self.model.config.vocab_size
            )
            return {
                **execution_provenance(self.metadata),
                "targets": targets,
                "valid": all(target["valid"] for target in targets),
                "max_targets": MAX_TRACKED_TARGETS,
                "max_target_characters": MAX_TARGET_CHARACTERS,
            }

    @torch.inference_mode()
    def trace(
        self,
        prompt: str,
        max_new_tokens: int,
        capture_attention: bool,
        ablate_layer: int | None,
        ablate_head: dict | None = None,
        expected_execution_id: str | None = None,
        attention_query_position: int = -1,
        tracked_tokens: list[str] | None = None,
    ) -> dict:
        with self.operation():
            self._require_model()
            self._verify_execution(expected_execution_id)
            if not 0 <= max_new_tokens <= 64:
                raise ValueError("Choose between 0 and 64 generated tokens per interactive trace.")
            started = time.perf_counter()
            encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            token_count = encoded.input_ids.shape[1]
            if not token_count:
                raise ValueError("The prompt must contain at least one token.")
            if capture_attention and not -1 <= attention_query_position < token_count:
                raise ValueError(
                    "Choose an attention query position inside the prompt or use -1 for its final token."
                )
            resolved_attention_query = (
                (token_count - 1 if attention_query_position == -1 else attention_query_position)
                if capture_attention
                else None
            )
            prompt_limit = capture_token_limit(
                self.metadata, self.settings.max_prompt_tokens, max_new_tokens
            )
            if prompt_limit < 1:
                raise ValueError(
                    "The generation length exceeds the loaded context allowance. Choose fewer generated tokens."
                )
            if token_count > prompt_limit:
                raise ValueError(
                    f"Prompt is too long for interactive capture or the loaded context allowance. Shorten it to at most {prompt_limit} tokens."
                )
            tracked_targets = resolve_tracked_targets(
                self.tokenizer, tracked_tokens or [], self.model.config.vocab_size
            )
            inputs = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            handles = []
            blocks = self._blocks()
            head_intervention = None
            bypassed_hidden = []
            capturing = True
            if ablate_head is not None:
                if ablate_layer is not None:
                    raise ValueError(
                        "Choose either a block bypass or a head ablation for one trace."
                    )
                target = HeadAblation.model_validate(ablate_head)
                handle, head_intervention = install_head_ablation(blocks, target.layer, target.head)
                handles.append(handle)
                if not max_new_tokens:
                    head_intervention = dict(head_intervention, scope="all_prompt_tokens")
            if ablate_layer is not None:
                if blocks is None or not 0 <= ablate_layer < len(blocks):
                    raise ValueError("The selected layer cannot be bypassed in this architecture.")

                def bypass(module, args, kwargs, output):
                    hidden = args[0] if args else kwargs["hidden_states"]
                    if capturing:
                        bypassed_hidden[:] = [hidden]
                    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

                handles.append(blocks[ablate_layer].register_forward_hook(bypass, with_kwargs=True))
            raw_final = []
            capture_handles = []
            if blocks is not None:

                def capture_final(module, args, output):
                    raw_final.append(output[0] if isinstance(output, tuple) else output)

                capture_handles.append(blocks[-1].register_forward_hook(capture_final))
            try:
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=capture_attention,
                    use_cache=False,
                )
                capturing = False
                for handle in capture_handles:
                    handle.remove()
                tokens = [self.tokenizer.decode([i]) for i in inputs["input_ids"][0].tolist()]
                summary = summarize_trace(
                    outputs,
                    self.tokenizer,
                    self._final_norm(),
                    self.model.get_output_embeddings(),
                    raw_final=raw_final,
                    tracked_targets=tracked_targets,
                    bypassed_hidden=bypassed_hidden,
                    ablate_layer=ablate_layer,
                    attention_query_position=resolved_attention_query
                    if resolved_attention_query is not None
                    else -1,
                )
                raw_final.clear()
                bypassed_hidden.clear()
                del outputs
                settings = (
                    generation_settings(self.tokenizer, max_new_tokens) if max_new_tokens else None
                )
                generated = (
                    self.model.generate(**inputs, **settings)
                    if settings is not None
                    else inputs["input_ids"]
                )
                decoded = decode_generation(
                    self.tokenizer,
                    inputs["input_ids"][0].cpu().tolist(),
                    generated[0].cpu().tolist(),
                )
                return {
                    **execution_provenance(self.metadata),
                    **summary,
                    "prompt": prompt,
                    "tokens": tokens,
                    "token_ids": inputs["input_ids"][0].cpu().tolist(),
                    **decoded,
                    "generation_settings": settings,
                    "generation_performed": settings is not None,
                    "requested_max_new_tokens": max_new_tokens,
                    "capture_attention": capture_attention,
                    "attention_query_position": resolved_attention_query
                    if summary["attention"]
                    else None,
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                    "device": self.device,
                    "ablate_layer": ablate_layer,
                    "ablate_head": head_intervention,
                }
            finally:
                for handle in handles + capture_handles:
                    handle.remove()

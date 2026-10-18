"""Validate and measure bounded prompt sets with a balanced completed-pair cohort."""

import hashlib
import json
import time

import numpy as np
import torch

from circuit_lantern.prompt_set_schemas import (
    BOOTSTRAP_SAMPLES,
    MAX_SET_FORWARDS,
    SIGN_EPSILON,
    PromptSetRequest,
)
from circuit_lantern.services.causal import (
    MIN_RECOVERY_GAP,
    SMALL_RECOVERY_GAP,
    activation_patch_sweep,
)
from circuit_lantern.services.execution import execution_provenance
from circuit_lantern.services.features.planning import forward_memory
from circuit_lantern.services.hardware import hardware_snapshot
from circuit_lantern.services.prompt_set_statistics import summarize_pairs
from circuit_lantern.services.targets import single_target_id
from circuit_lantern.services.tokens import capture_token_limit


def prepare_set(runtime, request: PromptSetRequest) -> tuple[dict, list[dict]]:
    """The caller owns the runtime lock. No model forwards occur during planning."""
    runtime._require_model()
    runtime._verify_execution(request.expected_execution_id)
    blocks = runtime._blocks()
    if not blocks:
        raise ValueError("Activation patching is not supported for this architecture yet.")
    layers = list(range(len(blocks))) if request.layers is None else sorted(request.layers)
    if len(set(layers)) != len(layers) or any(index >= len(blocks) for index in layers):
        raise ValueError("Choose distinct transformer layers within the loaded model.")
    source = request.hypothesis_source
    if source is not None:
        profile = execution_provenance(runtime.metadata)
        if (
            source.model_id != runtime.metadata["id"]
            or source.revision != runtime.metadata["revision"]
            or source.device != runtime.device
            or source.execution_precision != profile["execution_precision"]
            or source.execution_config != profile["execution_config"]
        ):
            raise ValueError(
                "The patching hypothesis source does not match the active checkpoint and execution profile."
            )
        if (
            len(request.pairs) != 1
            or request.metric != "logit_difference"
            or request.position_mode != "last"
            or source.layer not in layers
            or source.clean_prompt != request.pairs[0].clean_prompt
        ):
            raise ValueError(
                "The patching draft differs from the captured hypothesis. Use an independent experiment for changed inputs."
            )
    forwards = len(request.pairs) * (len(layers) + 2)
    limit = capture_token_limit(runtime.metadata, runtime.settings.max_prompt_tokens)
    vocabulary = int(runtime.model.get_output_embeddings().weight.shape[0])
    prepared, errors = [], []
    for index, pair in enumerate(request.pairs):
        try:
            clean = runtime.tokenizer.encode(pair.clean_prompt, add_special_tokens=True)
            corrupt = runtime.tokenizer.encode(pair.corrupted_prompt, add_special_tokens=True)
            if not clean or len(clean) != len(corrupt):
                raise ValueError(
                    "Clean and corrupted prompts must tokenize to the same nonzero length."
                )
            if len(clean) > limit:
                raise ValueError(f"This pair exceeds the {limit} token prompt limit.")
            target = single_target_id(runtime.tokenizer, pair.target, vocabulary)
            alternative = None
            if request.metric == "logit_difference":
                if not pair.alternative:
                    raise ValueError("Choose an alternative token for contrastive scoring.")
                alternative = single_target_id(runtime.tokenizer, pair.alternative, vocabulary)
                if target == alternative:
                    raise ValueError("Target and alternative must identify different tokens.")
            if (
                source is not None
                and index == 0
                and (
                    clean != source.token_ids
                    or source.position != len(clean) - 1
                    or target != source.target_token_id
                    or alternative != source.alternative_token_id
                )
            ):
                raise ValueError(
                    "The captured prompt or answer token IDs differ from the active tokenizer."
                )
            changed = [i for i, (a, b) in enumerate(zip(clean, corrupt, strict=True)) if a != b]
            positions = (
                list(range(len(clean)))
                if request.position_mode == "all"
                else changed
                if request.position_mode == "changed"
                else [len(clean) - 1]
            )
            if not positions:
                raise ValueError("This pair has no changed token positions to patch.")
            prepared.append(
                {
                    "index": index,
                    "token_count": len(clean),
                    "positions": positions,
                    "changed_positions": changed,
                    "target_token_id": target,
                    "alternative_token_id": alternative,
                }
            )
        except ValueError as error:
            errors.append({"index": index, "label": pair.label, "message": str(error)})
    longest = max((item["token_count"] for item in prepared), default=1)
    positions = max((len(item["positions"]) for item in prepared), default=1)
    width = int(
        getattr(runtime.model.config, "hidden_size", None)
        or runtime.model.get_input_embeddings().weight.shape[1]
    )
    cache = len(layers) * positions * width * 8
    teacher = forward_memory(runtime.model, longest)
    result_bytes = len(request.pairs) * (len(layers) * 1024 + longest * 128 + 20_000)
    bootstrap_bytes = BOOTSTRAP_SAMPLES * len(request.pairs) * 32
    host = teacher["host_bytes"] + result_bytes * 4 + bootstrap_bytes
    working = teacher["device_bytes"] + cache
    gpu = working if runtime.device.startswith("cuda") else 0
    if not gpu:
        host += working
    hardware = hardware_snapshot(runtime.settings.data_dir)
    blockers = []
    if forwards > MAX_SET_FORWARDS:
        blockers.append(
            f"This set needs {forwards} forwards. Choose fewer pairs or layers to stay within {MAX_SET_FORWARDS} forwards."
        )
    if errors:
        blockers.append("Fix the invalid prompt pairs before starting the sweep.")
    if host > hardware["ram_available_bytes"] * 0.75:
        blockers.append("Insufficient available RAM for this prompt set.")
    if gpu > (hardware.get("execution_gpu") or {}).get("free_bytes", 0) * 0.75:
        blockers.append("Insufficient free GPU memory for this prompt set.")
    if result_bytes * 3 > hardware["experiment_disk_free_bytes"] * 0.9:
        blockers.append("Insufficient disk space for the saved prompt set.")
    fingerprint_input = dict(
        request.model_dump(exclude={"expected_execution_id", "hypothesis_source"}), layers=layers
    )
    if source is not None:
        fingerprint_input["hypothesis_source"] = source.model_dump()
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_input,
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    return {
        **execution_provenance(runtime.metadata),
        "compatible": not blockers,
        "blockers": blockers,
        "pair_errors": errors,
        "pairs": prepared,
        "pair_count": len(request.pairs),
        "layers": layers,
        "forward_passes": forwards,
        "prompt_limit": limit,
        "cache_bytes": cache,
        "estimated_host_bytes": host,
        "estimated_gpu_bytes": gpu,
        "estimated_disk_bytes": result_bytes * 3,
        "input_fingerprint": fingerprint,
    }, prepared


class PairProgress:
    def __init__(self, job, index: int, total: int):
        self.job, self.index, self.total = job, index, total

    def get(self, key, default=None):
        return self.job.get(key, default)

    def update(self, **values):
        if "progress" in values:
            values["progress"] = round((self.index + values["progress"] / 100) / self.total * 100)
        self.job.update(**values, pair_index=self.index)


@torch.inference_mode()
def measure_set(runtime, request: PromptSetRequest, job) -> dict:
    started = time.perf_counter()
    plan, prepared = prepare_set(runtime, request)
    provenance = execution_provenance(runtime.metadata)
    if not plan["compatible"]:
        detail = " ".join(item["message"] for item in plan["pair_errors"])
        raise ValueError(" ".join(plan["blockers"]) + (" " + detail if detail else ""))
    pairs, failure, failed_pair = [], None, None
    for item, pair in zip(prepared, request.pairs, strict=True):
        if job.get("cancel_requested"):
            break
        index = item["index"]
        if runtime.metadata.get("execution_id") != request.expected_execution_id:
            failure, failed_pair = "The active model changed during the prompt set.", index
            break
        try:
            result = activation_patch_sweep(
                runtime.model,
                runtime.tokenizer,
                runtime._blocks(),
                runtime.metadata,
                runtime.device,
                pair.clean_prompt,
                pair.corrupted_prompt,
                pair.target,
                None if request.position_mode == "all" else item["positions"],
                PairProgress(job, index, len(request.pairs)),
                runtime.settings.max_prompt_tokens,
                layer_indices=plan["layers"],
                alternative=pair.alternative if request.metric == "logit_difference" else None,
            )
        except Exception as error:  # noqa: BLE001 - publish earlier pairs on forward failure
            failure, failed_pair = str(error)[:1000] or type(error).__name__, index
            break
        if (
            result.get("execution_id") != request.expected_execution_id
            or result["target_token_id"] != item["target_token_id"]
            or result.get("alternative_token_id") != item["alternative_token_id"]
        ):
            failure, failed_pair = (
                "Captured source or token IDs differ from the validated plan.",
                index,
            )
            break
        gap = result["clean"][request.metric] - result["corrupted"][request.metric]
        for layer in result["layers"]:
            delta = layer[request.metric] - result["corrupted"][request.metric]
            layer.update(
                metric_delta=delta, metric_recovery=delta / gap if gap > MIN_RECOVERY_GAP else None
            )
        pairs.append(
            {
                "index": index,
                "label": pair.label,
                "positions": item["positions"],
                "changed_positions": item["changed_positions"],
                "baseline_gap": gap,
                "metric_recovery_defined": gap > MIN_RECOVERY_GAP,
                "metric_recovery_warning": "Small clean to corrupted score gap. Interpret normalized recovery cautiously."
                if MIN_RECOVERY_GAP < gap < SMALL_RECOVERY_GAP
                else None,
                "clean_score": result["clean"][request.metric],
                "corrupted_score": result["corrupted"][request.metric],
                "result": result,
            }
        )
        if result["error"]:
            failure, failed_pair = result["error"], index
        if not result["complete"]:
            break
    summary = summarize_pairs(pairs, plan["layers"], request.seed)
    if request.metric != "logit_difference":
        summary["clean_target_preferred_pairs"] = None
    return {
        **provenance,
        "device": runtime.device,
        "metric": request.metric,
        "position_mode": request.position_mode,
        "requested_pairs": [pair.model_dump() for pair in request.pairs],
        **(
            {"hypothesis_source": request.hypothesis_source.model_dump()}
            if request.hypothesis_source is not None
            else {}
        ),
        "selected_layers": plan["layers"],
        "pairs": pairs,
        **summary,
        "total_pairs": len(request.pairs),
        "input_fingerprint": plan["input_fingerprint"],
        "complete": summary["completed_pairs"] == len(request.pairs),
        "error": failure,
        "failed_pair": failed_pair,
        "duration_ms": round((time.perf_counter() - started) * 1000),
        "statistics": {
            "seed": request.seed,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "confidence": 0.95,
            "sign_epsilon": SIGN_EPSILON,
            "minimum_recovery_gap": MIN_RECOVERY_GAP,
            "small_recovery_gap": SMALL_RECOVERY_GAP,
            "generator": "PCG64",
            "numpy_version": np.__version__,
            "cohort": "Completed pairs only, identical for every selected layer",
            "interval_method": "Pointwise paired percentile bootstrap. Recovery resamples only positive-gap eligible pairs.",
        },
        "method": "Clean block outputs are restored in corrupted runs, one pair at a time. Raw delta and sign summaries include all completed pairs, even pairs with nonpositive baseline gaps. Recovery is the ratio of summed effects to summed positive baseline gaps. Block-output patching localizes where information is present, not where it was first computed. Intervals resample supplied pairs, do not correct for multiple layers and do not establish generalization. Correlated templates can produce overconfident intervals.",
    }

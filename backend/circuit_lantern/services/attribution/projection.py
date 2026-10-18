"""Scalar projections for recorded answers, never cumulative logit-lens readouts."""

import torch

from circuit_lantern.limits import MAX_TRACKED_TARGETS
from circuit_lantern.services.attribution.normalization import (
    UnsupportedAttribution,
    frozen_normalization,
    normalize_components,
)


def cpu(value):
    return value.detach().to(device="cpu", dtype=torch.float64)


def tolerances(reference, observed_magnitude, epsilon):
    # The native rounding allowance is a diagnostic heuristic, not an error bound.
    strict = 1e-5 + 1e-5 * reference.abs().clamp_min(1)
    scale = torch.maximum(reference.abs(), observed_magnitude).clamp_min(1)
    rounding = torch.maximum(strict, 4 * epsilon * (1 + scale))
    return strict, rounding


def effective_epsilon(capture, logits):
    dtypes = (capture.dtype, capture.head.weight.dtype, logits.dtype)
    epsilon = max(torch.finfo(dtype).eps for dtype in dtypes)
    tf32 = bool(
        capture.head.weight.is_cuda
        and torch.float32 in dtypes
        and (
            torch.backends.cuda.matmul.allow_tf32
            or torch.get_float32_matmul_precision() != "highest"
        )
    )
    return max(epsilon, 2**-10) if tf32 else epsilon, tf32


def target_records(logits, tokenizer, targets):
    if len(targets) > MAX_TRACKED_TARGETS:
        raise UnsupportedAttribution("invalid_targets")
    result = [dict(target, source_types=["tracked"], prediction_rank=None) for target in targets]
    ids = [target["token_id"] for target in result]
    if len(ids) != len(set(ids)) or any(not 0 <= index < logits.numel() for index in ids):
        raise UnsupportedAttribution("invalid_targets")
    # Stable descending order breaks equal logits by vocabulary ID.
    predicted = torch.argsort(logits, descending=True, stable=True)[:2].tolist()
    for rank, index in enumerate(predicted, 1):
        existing = next((target for target in result if target["token_id"] == index), None)
        if existing is not None:
            existing["source_types"].append("prediction")
            existing["prediction_rank"] = rank
        else:
            result.append(
                {
                    "requested_text": tokenizer.decode([index]),
                    "token_id": index,
                    "token": tokenizer.decode([index]),
                    "source_types": ["prediction"],
                    "prediction_rank": rank,
                }
            )
    for target in result:
        encoded = tokenizer.encode(target["requested_text"], add_special_tokens=False)
        target["patchable"] = encoded == [target["token_id"]]
    return result, "tracked_and_predictions" if targets else "top_predictions"


def reject_output_transform(config):
    text = (
        config.get_text_config()
        if hasattr(config, "get_text_config")
        else getattr(config, "text_config", config)
    )
    for candidate in {id(config): config, id(text): text}.values():
        softcap = getattr(candidate, "final_logit_softcapping", None)
        if softcap is not None and softcap != 0:
            raise UnsupportedAttribution("output_transform")
        for name in ("logit_scale", "logits_scaling", "logits_scale"):
            value = getattr(candidate, name, None)
            if value is not None and value != 1:
                raise UnsupportedAttribution("output_transform")


def decompose(capture, logits, tokenizer, targets):
    reject_output_transform(capture.config)
    count = len(capture.blocks)
    if set(capture.before) != set(range(count)) or set(capture.after) != set(range(count)):
        raise UnsupportedAttribution("incomplete_capture")
    before = cpu(torch.stack([capture.before[index] for index in range(count)]))
    after = cpu(torch.stack([capture.after[index] for index in range(count)]))
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise UnsupportedAttribution("non_finite")
    if before.shape != after.shape or before.ndim != 2 or capture.head_input is None:
        raise UnsupportedAttribution("capture_shape")
    width = capture.head.in_features
    if before.shape[1] != width:
        raise UnsupportedAttribution("capture_shape")
    # Any operation between block boundaries needs its own adapter. Do not silently
    # charge such an operation to a block update.
    if count > 1 and not torch.equal(before[1:], after[:-1]):
        raise UnsupportedAttribution("residual_discontinuity")
    final = after[-1]
    if capture.norm is not None:
        if capture.norm_input is None or capture.norm_output is None:
            raise UnsupportedAttribution("incomplete_capture")
        if not torch.equal(final, cpu(capture.norm_input)):
            raise UnsupportedAttribution("residual_discontinuity")
        if not torch.equal(cpu(capture.norm_output), cpu(capture.head_input)):
            raise UnsupportedAttribution("output_transform")
    elif not torch.equal(final, cpu(capture.head_input)):
        raise UnsupportedAttribution("unsupported_norm")
    specification = frozen_normalization(capture.norm, final, capture.dtype)
    expected_norm = (
        normalize_components(final.unsqueeze(0), specification)[0] + specification["bias"]
    )
    observed_norm = cpu(capture.head_input)
    epsilon, tf32 = effective_epsilon(capture, logits)
    _, norm_allowance = tolerances(observed_norm, observed_norm.abs(), epsilon)
    norm_error = (expected_norm - observed_norm).abs()
    if not torch.isfinite(expected_norm).all() or not (norm_error <= norm_allowance).all():
        raise UnsupportedAttribution("normalization_mismatch")
    actual_logits = logits[0, -1].detach().float().cpu()
    selected, selection = target_records(actual_logits, tokenizer, targets)
    ids = [target["token_id"] for target in selected]
    if capture.head.weight.ndim != 2 or max(ids) >= capture.head.weight.shape[0]:
        raise UnsupportedAttribution("unsupported_head")
    directions = cpu(capture.head.weight[ids]).clone()
    vectors = torch.cat((before[:1], after - before))
    projected = normalize_components(vectors, specification) @ directions.T
    norm_bias = specification["bias"] @ directions.T
    head_bias = (
        cpu(capture.head.bias[ids])
        if capture.head.bias is not None
        else torch.zeros(len(ids), dtype=torch.float64)
    )
    actual = actual_logits[ids].double()
    reconstructed = projected.sum(0) + norm_bias + head_bias
    if not all(
        torch.isfinite(value).all()
        for value in (projected, norm_bias, head_bias, reconstructed, actual)
    ):
        raise UnsupportedAttribution("non_finite")
    observed_magnitude = observed_norm.abs() @ directions.abs().T + head_bias.abs()
    strict, allowance = tolerances(actual, observed_magnitude, epsilon)
    head_projected = observed_norm @ directions.T + head_bias
    head_errors = (head_projected - actual).abs()
    errors = (reconstructed - actual).abs()
    checks = {
        "actual_logits": actual.tolist(),
        "native_epsilon": epsilon,
        "tf32_allowed": tf32,
        "output_dtype": str(logits.dtype).removeprefix("torch."),
        "observed_readout_magnitudes": observed_magnitude.tolist(),
        "head_projected_logits": head_projected.tolist(),
        "head_absolute_errors": head_errors.tolist(),
        "reconstructed_logits": reconstructed.tolist(),
        "absolute_errors": errors.tolist(),
        "strict_tolerances": strict.tolist(),
        "rounding_allowances": allowance.tolist(),
        "status": "ok" if bool((errors <= strict).all()) else "rounding",
        "tolerance_method": "Strict absolute plus relative tolerance of 1e-5, anchored to actual logits. Native rounding allowance uses 4 times native epsilon times one plus the larger of actual logit magnitude and the absolute dot-product sum from the observed head input, including bias. TF32 is included when allowed. This is a heuristic, not a certified bound.",
        "maximum_normalization_error": float(norm_error.max()),
        "residual_maximum_error": float((vectors.sum(0) - final).abs().max()),
    }
    if not bool((head_errors <= allowance).all()):
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reason": "readout_mismatch",
            "position": -1,
            "checks": checks,
        }
    if not bool((errors <= allowance).all()):
        return {
            "schema_version": 1,
            "status": "unavailable",
            "reason": "reconstruction_mismatch",
            "position": -1,
            "checks": checks,
        }
    rows = [
        {
            "kind": "embedding",
            "layer": None,
            "values": projected[0].tolist(),
            "update_norm": float(vectors[0].norm()),
        }
    ]
    rows += [
        {
            "kind": "block",
            "layer": index,
            "values": projected[index + 1].tolist(),
            "update_norm": float(vectors[index + 1].norm()),
        }
        for index in range(count)
    ]
    rows += [
        {"kind": "norm_bias", "layer": None, "values": norm_bias.tolist()},
        {"kind": "head_bias", "layer": None, "values": head_bias.tolist()},
    ]
    normalization = {
        key: value for key, value in specification.items() if key not in ("gain", "bias", "center")
    }
    return {
        "schema_version": 1,
        "status": "available",
        "reason": None,
        "position": -1,
        "scope": "final_prompt_token",
        "selection_mode": selection,
        "targets": selected,
        "components": rows,
        "checks": checks,
        "normalization": normalization,
        "captured_dtype": str(capture.dtype).removeprefix("torch."),
        "readout_dtype": str(capture.head.weight.dtype).removeprefix("torch."),
        "compute_dtype": "float64",
        "retained_vectors": False,
        "method": "Final-token embedding input and actual block output-minus-input updates projected onto recorded token readout rows, with the observed final normalization denominator held fixed. LayerNorm centers each component; normalization and readout biases are each counted once. These are additive diagnostic contributions for this run, not ablation effects or a causal circuit. Other token directions require a new capture.",
    }

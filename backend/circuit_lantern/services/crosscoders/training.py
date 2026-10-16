"""Bounded joint dictionary optimization with training-only preprocessing."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from circuit_lantern.services.crosscoders.model import CrosscoderConfig, SparseCrosscoder
from circuit_lantern.services.features.training import geometric_median


@dataclass
class CrosscoderTrainingResult:
    model: SparseCrosscoder
    history: list[dict]
    train_metrics: dict
    validation_metrics: dict
    training_means: list[torch.Tensor]
    completed_steps: int
    tokens_seen: int
    effective_batch_size: int
    cancelled: bool


def validate_banks(train: list[torch.Tensor], validation: list[torch.Tensor]):
    if len(train) != len(validation) or not 2 <= len(train) <= 8:
        raise ValueError("Provide matching training and held out residual banks.")
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or min(value.shape) < 1
        for value in [*train, *validation]
    ):
        raise ValueError("Residual banks must be nonempty float32 CPU matrices.")
    if (
        len(train[0]) < 2
        or any(len(value) != len(train[0]) for value in train)
        or any(len(value) != len(validation[0]) for value in validation)
        or any(a.shape[1] != b.shape[1] for a, b in zip(train, validation, strict=True))
    ):
        raise ValueError("Residual banks must have matching sample rows and recorded widths.")
    if any(not torch.isfinite(value).all() for value in [*train, *validation]):
        raise ValueError("Residual banks must contain only finite activations.")


@torch.inference_mode()
def evaluate_crosscoder(
    model: SparseCrosscoder,
    values: list[torch.Tensor],
    training_means: list[torch.Tensor],
    *,
    batch_size: int = 128,
) -> dict:
    count = len(values[0])
    squared = [0.0] * len(values)
    baseline = [0.0] * len(values)
    feature_counts = torch.zeros(model.config.features, dtype=torch.int64)
    l1 = normalized_error = 0.0
    for start in range(0, count, batch_size):
        batch = [value[start : start + batch_size].to(model.device) for value in values]
        reconstructed, features = model(batch)
        if not torch.isfinite(features).all() or any(
            not torch.isfinite(value).all() for value in reconstructed
        ):
            raise ValueError("Crosscoder evaluation produced non finite measurements.")
        feature_counts += features.gt(0).sum(dim=0).cpu()
        l1 += float(model.sparsity_loss(features)) * len(batch[0])
        for bank, (actual, predicted, mean) in enumerate(
            zip(batch, reconstructed, training_means, strict=True)
        ):
            error = float((actual.double() - predicted.double()).square().sum())
            squared[bank] += error
            baseline[bank] += float(
                (actual.double() - mean.to(model.device).double()).square().sum()
            )
            normalized_error += error / float(model.input_scales[bank].double().square())
    weighted_l1 = l1 / count
    reconstruction = normalized_error / count
    return {
        "samples": count,
        "banks": [
            {
                "input_dim": model.widths[bank],
                "mse": squared[bank] / (count * model.widths[bank]),
                "normalized_mse": squared[bank] / baseline[bank] if baseline[bank] > 0 else None,
                "explained_variance": 1 - squared[bank] / baseline[bank]
                if baseline[bank] > 0
                else None,
            }
            for bank in range(len(values))
        ],
        "normalized_l2_error": reconstruction,
        "weighted_l1": weighted_l1,
        "loss": reconstruction + model.config.l1_coefficient * weighted_l1,
        "mean_active_features": float(feature_counts.sum()) / count,
        "inactive_features": int(feature_counts.eq(0).sum()),
    }


def train_crosscoder(
    train: list[torch.Tensor],
    validation: list[torch.Tensor],
    config: CrosscoderConfig,
    *,
    device: str = "cpu",
    progress: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> CrosscoderTrainingResult:
    validate_banks(train, validation)
    widths = tuple(value.shape[1] for value in train)
    config.validate(widths)
    scales = [
        float(torch.linalg.vector_norm(value, dtype=torch.float64) / math.sqrt(len(value)))
        for value in train
    ]
    scales = [value if value > 0 else 1.0 for value in scales]
    if any(
        not math.isfinite(value)
        or value > torch.finfo(torch.float32).max
        or float(torch.tensor(value, dtype=torch.float32)) <= 0
        for value in scales
    ):
        raise ValueError("The activation scale exceeds the supported float32 range.")
    means = [value.double().mean(dim=0).float() for value in train]
    variance = sum(
        float((value.double() / scale - mean.double() / scale).square().sum()) / len(value)
        for value, mean, scale in zip(train, means, scales, strict=True)
    )
    if not math.isfinite(variance) or variance <= 1e-12:
        raise ValueError("The paired training corpus has no activation variance to learn.")
    model = SparseCrosscoder(widths, config).to(device)
    with torch.no_grad():
        model.input_scales.copy_(torch.tensor(scales, device=device, dtype=torch.float32))
        for center, values, scale in zip(model.centers, train, scales, strict=True):
            center.copy_(geometric_median(values / scale).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=6.25e-10)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    batch_size = min(config.batch_size, len(train[0]))
    permutation = torch.randperm(len(train[0]), generator=generator)
    sample_ids = torch.randperm(len(train[0]), generator=generator)[:1024]
    sample = [value[sample_ids] for value in train]
    history = []
    completed = offset = 0
    was_cancelled = False
    interval = max(1, config.steps // 24)

    def observe(step: int):
        training_metrics = evaluate_crosscoder(model, sample, means)
        held_out = evaluate_crosscoder(model, validation, means)
        row = {
            "step": step,
            "tokens_seen": step * batch_size,
            "train_sample_normalized_l2_error": training_metrics["normalized_l2_error"],
            "validation_normalized_l2_error": held_out["normalized_l2_error"],
            "validation_loss": held_out["loss"],
            "validation_mean_active_features": held_out["mean_active_features"],
            "validation_banks": held_out["banks"],
        }
        history.append(row)
        if progress:
            progress(dict(row))

    observe(0)
    for step in range(1, config.steps + 1):
        if cancelled and cancelled():
            was_cancelled = True
            break
        if offset + batch_size > len(permutation):
            permutation = torch.randperm(len(train[0]), generator=generator)
            offset = 0
        indices = permutation[offset : offset + batch_size]
        offset += batch_size
        batch = [value[indices].to(device) for value in train]
        optimizer.zero_grad(set_to_none=True)
        features = model.encode(batch)
        reconstruction = model.decode_normalized(features)
        error = sum(
            (actual / model.input_scales[bank] - predicted).square().sum(dim=-1).mean()
            for bank, (actual, predicted) in enumerate(zip(batch, reconstruction, strict=True))
        )
        loss = error + config.l1_coefficient * model.sparsity_loss(features)
        if not torch.isfinite(loss):
            raise ValueError("Crosscoder training produced a non finite loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        completed = step
        if step % interval == 0 or step == config.steps:
            observe(step)
    if history[-1]["step"] != completed:
        observe(completed)
    model.eval()
    return CrosscoderTrainingResult(
        model=model,
        history=history,
        train_metrics=evaluate_crosscoder(model, train, means),
        validation_metrics=evaluate_crosscoder(model, validation, means),
        training_means=means,
        completed_steps=completed,
        tokens_seen=completed * batch_size,
        effective_batch_size=batch_size,
        cancelled=was_cancelled,
    )

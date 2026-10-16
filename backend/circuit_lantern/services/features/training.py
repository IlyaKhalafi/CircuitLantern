"""Reproducible sparse dictionary training without changing the language model."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from circuit_lantern.services.features.autoencoder import SparseAutoencoder, TrainingConfig


@dataclass
class TrainingResult:
    model: SparseAutoencoder
    history: list[dict]
    train_metrics: dict
    validation_metrics: dict
    completed_steps: int
    tokens_seen: int
    cancelled: bool
    auxiliary_steps: int
    effective_batch_size: int


@torch.no_grad()
def geometric_median(values: torch.Tensor) -> torch.Tensor:
    median = values.mean(dim=0)
    for _ in range(64):
        distances = torch.linalg.vector_norm(values - median, dim=-1).clamp_min(1e-6)
        weights = distances.reciprocal()
        updated = (values * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
        if torch.linalg.vector_norm(updated - median) <= 1e-5:
            return updated
        median = updated
    return median


@torch.inference_mode()
def evaluate_dictionary(
    model: SparseAutoencoder,
    values: torch.Tensor,
    training_mean: torch.Tensor,
    *,
    batch_size: int = 256,
) -> dict:
    count = values.shape[0]
    squared_error = baseline_error = active = absolute = 0.0
    feature_counts = torch.zeros(model.config.features, dtype=torch.int64)
    device = model.decoder.device
    for batch in values.split(batch_size):
        batch = batch.to(device=device, dtype=torch.float32)
        reconstruction, features = model(batch)
        if not torch.isfinite(reconstruction).all() or not torch.isfinite(features).all():
            raise ValueError("The dictionary produced non finite feature measurements.")
        squared_error += float((reconstruction - batch).double().square().sum())
        baseline_error += float((batch - training_mean.to(device)).double().square().sum())
        feature_counts += features.gt(0).sum(dim=0).cpu()
        active += float(features.gt(0).sum())
        absolute += float(features.abs().sum())
    mse = squared_error / (count * model.input_dim)
    normalized = squared_error / baseline_error if baseline_error > 0 else None
    return {
        "samples": count,
        "mse": mse,
        "normalized_mse": normalized,
        "explained_variance": 1 - normalized if normalized is not None else None,
        "mean_active_features": active / count,
        "mean_l1": absolute / count,
        "inactive_features": int(feature_counts.eq(0).sum()),
        "feature_activation_counts": feature_counts.tolist(),
    }


def _validate_data(train: torch.Tensor, validation: torch.Tensor):
    if (
        train.ndim != 2
        or validation.ndim != 2
        or min(train.shape) < 1
        or validation.shape[0] < 1
        or train.shape[1] != validation.shape[1]
        or train.shape[0] < 2
        or train.device.type != "cpu"
        or validation.device.type != "cpu"
        or train.dtype != torch.float32
        or validation.dtype != torch.float32
    ):
        raise ValueError("Training and held out activations must be nonempty float32 CPU matrices.")
    if not torch.isfinite(train).all() or not torch.isfinite(validation).all():
        raise ValueError("Training and held out activations must contain only finite values.")


def train_dictionary(
    train: torch.Tensor,
    validation: torch.Tensor,
    config: TrainingConfig,
    *,
    device: str = "cpu",
    progress: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> TrainingResult:
    _validate_data(train, validation)
    config.validate(train.shape[1])
    model = SparseAutoencoder(train.shape[1], config).to(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    training_mean = train.mean(dim=0)
    scale = float(torch.linalg.vector_norm(train, dtype=torch.float64) / math.sqrt(train.numel()))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("The training corpus has no nonzero activation scale.")
    normalized_train = train / scale
    variance = float((normalized_train - normalized_train.mean(dim=0)).square().mean())
    if not math.isfinite(variance) or variance <= 1e-12:
        raise ValueError("The training corpus has no activation variance to learn.")
    with torch.no_grad():
        model.input_scale.fill_(scale)
        model.center.copy_(geometric_median(normalized_train).to(device))
    del normalized_train
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=6.25e-10)
    since_active = torch.zeros(config.features, dtype=torch.int64, device=device)
    history = []
    tokens_seen = completed = auxiliary_steps = 0
    batch_size = min(config.batch_size, len(train))
    auxiliary_total = 0.0
    observed_steps = 0
    sample = train[torch.randperm(len(train), generator=generator)[:1024]]
    interval = max(1, config.steps // 24)

    def observe(step: int, auxiliary_loss: float = 0.0):
        train_metrics = evaluate_dictionary(model, sample, training_mean)
        validation_metrics = evaluate_dictionary(model, validation, training_mean)
        row = {
            "step": step,
            "tokens_seen": tokens_seen,
            "train_sample_mse": train_metrics["mse"],
            "validation_mse": validation_metrics["mse"],
            "validation_normalized_mse": validation_metrics["normalized_mse"],
            "mean_active_features": validation_metrics["mean_active_features"],
            "auxiliary_loss": auxiliary_loss,
        }
        history.append(row)
        if progress:
            progress(dict(row))

    observe(0)
    permutation = torch.randperm(len(train), generator=generator)
    offset = 0
    was_cancelled = False
    for step in range(1, config.steps + 1):
        if cancelled and cancelled():
            was_cancelled = True
            break
        if offset + batch_size > len(permutation):
            permutation = torch.randperm(len(train), generator=generator)
            offset = 0
        indices = permutation[offset : offset + batch_size]
        offset += len(indices)
        batch = train[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        pre = model.pre_activations(batch)
        features = model.activate(pre)
        reconstruction = model.decode(features)
        residual = (batch - reconstruction) / model.input_scale
        mse = residual.square().mean()
        loss = (
            mse + config.l1_coefficient * features.abs().sum(dim=-1).mean()
            if config.method == "relu"
            else mse / variance
        )
        with torch.no_grad():
            since_active.add_(len(batch))
            since_active.masked_fill_(features.gt(0).any(dim=0), 0)
        auxiliary = torch.zeros((), device=device)
        dead = since_active >= config.dead_after_tokens
        if config.method == "topk" and config.auxiliary_coefficient and dead.any():
            auxiliary_k = min(2 ** round(math.log2(max(1, model.input_dim / 2))), int(dead.sum()))
            auxiliary_steps += 1
            values, columns = pre.masked_fill(~dead, -torch.inf).topk(auxiliary_k, dim=-1)
            auxiliary_features = torch.zeros_like(pre).scatter(-1, columns, F.relu(values))
            predicted_error = F.linear(auxiliary_features, model.decoder)
            target_error = residual.detach()
            auxiliary = (
                (predicted_error - target_error).square().sum(dim=-1)
                / target_error.square().sum(dim=-1).clamp_min(1e-6)
            ).mean()
            loss = loss + config.auxiliary_coefficient * auxiliary
        if not torch.isfinite(loss):
            raise ValueError("Sparse autoencoder training produced a non finite loss.")
        loss.backward()
        model.project_decoder_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.normalize_decoder()
        completed = step
        tokens_seen += len(batch)
        auxiliary_total += float(auxiliary.detach())
        if step % interval == 0 or step == config.steps:
            observe(step, auxiliary_total / (step - observed_steps))
            auxiliary_total = 0.0
            observed_steps = step
    if history[-1]["step"] != completed:
        observe(completed, auxiliary_total / (completed - observed_steps))
    model.eval()
    return TrainingResult(
        model=model,
        history=history,
        train_metrics=evaluate_dictionary(model, train, training_mean),
        validation_metrics=evaluate_dictionary(model, validation, training_mean),
        completed_steps=completed,
        tokens_seen=tokens_seen,
        cancelled=was_cancelled,
        auxiliary_steps=auxiliary_steps,
        effective_batch_size=batch_size,
    )

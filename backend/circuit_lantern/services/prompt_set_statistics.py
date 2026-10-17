"""Paired descriptive statistics with reproducible, pointwise bootstrap intervals."""

import numpy as np

from circuit_lantern.prompt_set_schemas import BOOTSTRAP_SAMPLES, SIGN_EPSILON
from circuit_lantern.services.causal import MIN_RECOVERY_GAP


def interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def summarize_pairs(pairs: list[dict], layers: list[int], seed: int) -> dict:
    completed = [pair for pair in pairs if pair["result"]["complete"]]
    count = len(completed)
    if not count:
        return {"completed_pairs": 0, "recovery_pairs": 0, "layers": []}
    gaps = np.array([pair["baseline_gap"] for pair in completed], dtype=np.float64)
    eligible = gaps > MIN_RECOVERY_GAP
    recovery_count = int(eligible.sum())
    rng = np.random.default_rng(seed)
    samples = rng.integers(count, size=(BOOTSTRAP_SAMPLES, count)) if count >= 2 else None
    recovery_samples = (
        rng.integers(recovery_count, size=(BOOTSTRAP_SAMPLES, recovery_count))
        if recovery_count >= 2
        else None
    )
    summaries = []
    for index, layer in enumerate(layers):
        values = np.array([pair["result"]["layers"][index]["metric_delta"] for pair in completed])
        recovery = float(values[eligible].sum() / gaps[eligible].sum()) if recovery_count else None
        recovery_interval = (
            interval(
                values[eligible][recovery_samples].sum(1) / gaps[eligible][recovery_samples].sum(1)
            )
            if recovery_samples is not None
            else None
        )
        summaries.append(
            {
                "layer": layer,
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "minimum_delta": float(values.min()),
                "maximum_delta": float(values.max()),
                "mean_interval": interval(values[samples].mean(1)) if samples is not None else None,
                "positive_fraction": float((values > SIGN_EPSILON).mean()),
                "negative_fraction": float((values < -SIGN_EPSILON).mean()),
                "near_zero_fraction": float((abs(values) <= SIGN_EPSILON).mean()),
                "recovery": recovery,
                "recovery_interval": recovery_interval,
            }
        )
    return {
        "completed_pairs": count,
        "recovery_pairs": recovery_count,
        "layers": summaries,
        "clean_target_preferred_pairs": sum(pair["clean_score"] > 0 for pair in completed),
    }

import math

import numpy as np
import torch
from torch.nn import functional


@torch.inference_mode()
def matrix_metrics(tensor: torch.Tensor, max_elements: int = 4_000_000) -> dict:
    matrix = tensor.detach().float()
    if matrix.ndim != 2 or not matrix.numel():
        raise ValueError("Select a nonempty two dimensional weight matrix.")
    rows, columns = matrix.shape
    sum_squares, absolute, zeros = 0.0, 0.0, 0
    mean, second_moment, observed = 0.0, 0.0, 0
    column_sums = torch.zeros(columns, dtype=torch.float64, device=matrix.device)
    minimum, maximum, infinity_norm = math.inf, -math.inf, 0.0
    chunk_rows = max(1, 1_000_000 // columns)
    for chunk in matrix.split(chunk_rows):
        if not torch.isfinite(chunk).all():
            raise ValueError("The matrix contains non finite values and cannot be analyzed.")
        precise_chunk = chunk.double()
        variance, chunk_mean = torch.var_mean(precise_chunk, correction=0)
        count = chunk.numel()
        delta = float(chunk_mean) - mean
        combined = observed + count
        second_moment += float(variance) * count + delta**2 * observed * count / combined
        mean += delta * count / combined
        observed = combined
        sum_squares += float(precise_chunk.square().sum())
        magnitudes = chunk.abs()
        row_sums = magnitudes.sum(dim=1, dtype=torch.float64)
        column_sums += magnitudes.sum(dim=0, dtype=torch.float64)
        absolute += float(row_sums.sum())
        infinity_norm = max(infinity_norm, float(row_sums.max()))
        zeros += chunk.numel() - int(torch.count_nonzero(chunk))
        minimum, maximum = min(minimum, float(chunk.min())), max(maximum, float(chunk.max()))
    std = math.sqrt(max(0, second_moment / matrix.numel()))
    exact_spectrum = max(rows, columns) <= 768
    if exact_spectrum:
        precise = matrix.double()
        singular = torch.linalg.svdvals(precise)
    else:
        devices = [matrix.device.index or 0] if matrix.is_cuda else []
        # Restore RNG state so analysis cannot perturb subsequent experiments.
        with torch.random.fork_rng(devices=devices):
            if matrix.is_cuda:
                with torch.cuda.device(matrix.device):
                    torch.cuda.manual_seed(0)
            else:
                torch.set_rng_state(torch.Generator().manual_seed(0).get_state())
            scale = max(abs(minimum), abs(maximum))
            normalized = matrix / scale if scale > 0 else matrix
            _, singular, _ = torch.svd_lowrank(normalized, q=min(64, rows, columns), niter=4)
            singular = singular.double() * scale
    energy = (
        singular.square().cumsum(0) / sum_squares if sum_squares > 0 else torch.zeros_like(singular)
    )
    spectral_norm = float(singular[0])
    spectral_radius = None
    if rows == columns and exact_spectrum:
        spectral_radius = float(torch.linalg.eigvals(precise).abs().max())
    condition = nuclear_norm = effective_rank = numerical_rank = tolerance = None
    if exact_spectrum:
        tolerance = spectral_norm * max(rows, columns) * torch.finfo(torch.float64).eps
        numerical_rank = int((singular > tolerance).sum())
        nuclear_norm = float(singular.sum())
        if nuclear_norm > 0:
            mass = singular / nuclear_norm
            positive = mass[mass > 0]
            effective_rank = float((-(positive * positive.log()).sum()).exp())
        if numerical_rank == min(rows, columns):
            condition = spectral_norm / float(singular[-1])
    stride = max(1, int(np.ceil(np.sqrt(matrix.numel() / max_elements))))
    sampled = matrix[::stride, ::stride].double().cpu().numpy()
    lower, upper = minimum, maximum
    if lower == upper:
        padding = abs(lower) * 0.05 if lower else 0.5
        lower, upper = lower - padding, upper + padding
    histogram, bin_edges = np.histogram(sampled, bins=48, range=(lower, upper))
    heatmap = functional.adaptive_avg_pool2d(matrix[None, None], (min(rows, 48), min(columns, 64)))[
        0, 0
    ]
    return {
        "shape": [rows, columns],
        "elements": matrix.numel(),
        "dtype": str(matrix.dtype).removeprefix("torch."),
        "frobenius_norm": math.sqrt(sum_squares),
        "l1_norm": absolute,
        "infinity_norm": infinity_norm,
        "operator_1_norm": float(column_sums.max()),
        "nuclear_norm": nuclear_norm,
        "effective_rank": effective_rank,
        "numerical_rank": numerical_rank,
        "rank_tolerance": tolerance,
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "sparsity": zeros / matrix.numel(),
        "spectral_norm": spectral_norm,
        "spectral_radius": spectral_radius,
        "stable_rank": sum_squares / spectral_norm**2 if spectral_norm > 0 else 0,
        "condition_number": condition,
        "singular_values": singular[:128].cpu().tolist(),
        "cumulative_energy": energy[:128].clamp(0, 1).cpu().tolist(),
        "spectrum_exact": exact_spectrum,
        "spectrum_shape": [rows, columns],
        "spectrum_precision": "float64" if exact_spectrum else "float32",
        "spectrum_method": "Full SVD in float64"
        if exact_spectrum
        else "Full matrix randomized SVD with 64 components and 4 power iterations",
        "histogram": [
            {"value": float((bin_edges[i] + bin_edges[i + 1]) / 2), "count": int(v)}
            for i, v in enumerate(histogram)
        ],
        "histogram_sampled": stride > 1,
        "heatmap": heatmap.cpu().tolist(),
        "heatmap_method": "Adaptive mean pooling. Cells summarize regions of the original matrix.",
    }

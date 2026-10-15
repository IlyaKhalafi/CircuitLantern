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
    total, sum_squares, absolute, zeros = 0.0, 0.0, 0.0, 0
    minimum, maximum, infinity_norm = math.inf, -math.inf, 0.0
    chunk_rows = max(1, 1_000_000 // columns)
    for chunk in matrix.split(chunk_rows):
        if not torch.isfinite(chunk).all():
            raise ValueError("The matrix contains non finite values and cannot be analyzed.")
        total += float(chunk.sum(dtype=torch.float64))
        sum_squares += float(chunk.double().square().sum())
        row_sums = chunk.abs().sum(dim=1, dtype=torch.float64)
        absolute += float(row_sums.sum())
        infinity_norm = max(infinity_norm, float(row_sums.max()))
        zeros += chunk.numel() - int(torch.count_nonzero(chunk))
        minimum, maximum = min(minimum, float(chunk.min())), max(maximum, float(chunk.max()))
    mean = total / matrix.numel()
    std = math.sqrt(max(0, sum_squares / matrix.numel() - mean**2))
    exact_spectrum = max(rows, columns) <= 768
    if exact_spectrum:
        singular = torch.linalg.svdvals(matrix)
    else:
        devices = [matrix.device.index or 0] if matrix.is_cuda else []
        # Restore RNG state so analysis cannot perturb subsequent experiments.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            _, singular, _ = torch.svd_lowrank(matrix, q=min(64, rows, columns), niter=4)
    energy = singular.double().square().cumsum(0) / max(sum_squares, 1e-30)
    spectral_norm = float(singular[0])
    spectral_radius = None
    if rows == columns and exact_spectrum:
        spectral_radius = float(torch.linalg.eigvals(matrix).abs().max())
    condition = None
    if exact_spectrum and float(singular[-1]) > 1e-12:
        condition = spectral_norm / float(singular[-1])
    stride = max(1, int(np.ceil(np.sqrt(matrix.numel() / max_elements))))
    sampled = matrix[::stride, ::stride].cpu().numpy()
    histogram, bin_edges = np.histogram(sampled, bins=48)
    heatmap = functional.adaptive_avg_pool2d(matrix[None, None], (min(rows, 48), min(columns, 64)))[
        0, 0
    ]
    return {
        "shape": [rows, columns],
        "elements": matrix.numel(),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "frobenius_norm": math.sqrt(sum_squares),
        "l1_norm": absolute,
        "infinity_norm": infinity_norm,
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "sparsity": zeros / matrix.numel(),
        "spectral_norm": spectral_norm,
        "spectral_radius": spectral_radius,
        "stable_rank": sum_squares / max(spectral_norm**2, 1e-30),
        "condition_number": condition,
        "singular_values": singular[:128].cpu().tolist(),
        "cumulative_energy": energy[:128].clamp(0, 1).cpu().tolist(),
        "spectrum_exact": exact_spectrum,
        "spectrum_shape": [rows, columns],
        "spectrum_method": "Exact SVD"
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

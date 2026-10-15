import numpy as np
import torch
from torch.nn import functional


def matrix_metrics(tensor: torch.Tensor, max_elements: int = 4_000_000) -> dict:
    matrix = tensor.detach().float().cpu()
    if matrix.ndim != 2:
        raise ValueError("Select a two dimensional weight matrix.")
    rows, columns = matrix.shape
    if not torch.isfinite(matrix).all():
        raise ValueError("The matrix contains non finite values and cannot be analyzed.")
    stride = max(1, int(np.ceil(np.sqrt(matrix.numel() / max_elements))))
    sampled = matrix[::stride, ::stride]
    # Bound the cubic SVD cost separately from the linear scan and visualization.
    svd_stride = max(1, int(np.ceil(max(rows, columns) / 768)))
    spectral_matrix = matrix[::svd_stride, ::svd_stride]
    singular = torch.linalg.svdvals(spectral_matrix)
    exact_spectrum = svd_stride == 1
    squared = singular.square()
    energy = (squared.cumsum(0) / squared.sum().clamp_min(1e-30)).tolist()
    spectral_radius = None
    if rows == columns and rows <= 768:
        spectral_radius = float(torch.linalg.eigvals(matrix).abs().max())
    heatmap = functional.adaptive_avg_pool2d(matrix[None, None], (min(rows, 48), min(columns, 64)))[
        0, 0
    ]
    histogram, bin_edges = np.histogram(sampled.numpy(), bins=48)
    frobenius = float(torch.linalg.vector_norm(matrix))
    minimum = float(singular[-1])
    condition = float(singular[0]) / minimum if minimum > 1e-12 else None
    return {
        "shape": [rows, columns],
        "elements": matrix.numel(),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "frobenius_norm": frobenius,
        "l1_norm": float(matrix.abs().sum()),
        "infinity_norm": float(matrix.abs().sum(dim=1).max()),
        "mean": float(matrix.mean()),
        "std": float(matrix.std(unbiased=False)),
        "min": float(matrix.min()),
        "max": float(matrix.max()),
        "sparsity": float((matrix == 0).float().mean()),
        "spectral_norm": float(singular[0]),
        "spectral_radius": spectral_radius,
        "stable_rank": float(squared.sum() / squared[0].clamp_min(1e-30)),
        "condition_number": condition,
        "singular_values": singular[:128].tolist(),
        "cumulative_energy": energy[:128],
        "spectrum_exact": exact_spectrum,
        "spectrum_shape": list(spectral_matrix.shape),
        "spectrum_stride": svd_stride,
        "histogram": [
            {"value": float((bin_edges[i] + bin_edges[i + 1]) / 2), "count": int(v)}
            for i, v in enumerate(histogram)
        ],
        "histogram_sampled": stride > 1,
        "heatmap": heatmap.tolist(),
        "heatmap_method": "Adaptive mean pooling. Cells summarize regions of the original matrix.",
    }

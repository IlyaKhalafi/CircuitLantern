import torch
from torch.nn import functional


class ResidualGeometry:
    """Summarize token geometry without exporting hidden coordinate arrays."""

    def __init__(self, max_positions: int = 128):
        self.max_positions = max_positions
        self.labels: list[str] = []
        self.grams: list[torch.Tensor | None] = []
        self.previous: torch.Tensor | None = None
        self.latest: torch.Tensor | None = None
        self.positions: list[int] = []
        self.token_count = 0

    @torch.inference_mode()
    def observe(self, hidden: torch.Tensor, label: str) -> dict:
        values = hidden.detach().float()
        if values.ndim != 2 or not values.shape[0] or not torch.isfinite(values).all():
            raise ValueError("Residual geometry requires finite token by hidden state matrices.")
        self.labels.append(label)
        self.token_count = values.shape[0]
        count = min(self.token_count, self.max_positions)
        positions = torch.linspace(0, self.token_count - 1, count, device=values.device).long()
        self.positions = positions.cpu().tolist()
        statistics = values.double()
        selected = statistics[positions]
        self.latest = selected
        gram = None
        if count >= 3:
            shifted = selected - selected[:1]
            centered = shifted - shifted.mean(dim=0, keepdim=True)
            scale = centered.abs().max()
            if float(scale) > 0:
                centered = centered / scale
                centered = centered.float()
                kernel = (centered @ centered.T).double().cpu()
                norm = kernel.norm()
                if float(norm) > 0:
                    gram = (kernel / norm).flatten()
        self.grams.append(gram)
        update = {
            "rms": float(statistics.square().mean().sqrt()),
            "norm": float(statistics[-1].norm()),
            "token_norms": statistics.norm(dim=-1).cpu().tolist(),
            "cosine_similarity": None,
            "update_norm": None,
            "update_rms": None,
            "relative_update": None,
        }
        if self.previous is not None:
            delta = statistics - self.previous
            norm = float(delta[-1].norm())
            previous_norm = float(self.previous[-1].norm())
            update.update(
                {
                    "cosine_similarity": float(
                        torch.dot(
                            statistics[-1] / update["norm"], self.previous[-1] / previous_norm
                        ).clamp(-1, 1)
                    )
                    if update["norm"] > 0 and previous_norm > 0
                    else None,
                    "update_norm": norm,
                    "update_rms": float(delta.square().mean().sqrt()),
                    "relative_update": norm / previous_norm if previous_norm > 0 else None,
                }
            )
        self.previous = statistics
        return update

    @torch.inference_mode()
    def finish(self) -> dict:
        valid = [index for index, gram in enumerate(self.grams) if gram is not None]
        similarities = [[None for _ in self.labels] for _ in self.labels]
        if valid:
            kernels = torch.stack([self.grams[index] for index in valid])
            scores = (kernels @ kernels.T).clamp(0, 1).tolist()
            for row, layer in enumerate(valid):
                for column, other in enumerate(valid):
                    similarities[layer][other] = scores[row][column]
        token_cosine = []
        if self.latest is not None:
            # Scale each row before normalization to preserve tiny nonzero vectors.
            scales = self.latest.abs().amax(dim=1, keepdim=True)
            active = (scales[:, 0] > 0).cpu().tolist()
            normalized = functional.normalize(
                self.latest / torch.where(scales > 0, scales, 1), dim=1
            )
            cosine = (normalized @ normalized.T).clamp(-1, 1).cpu().tolist()
            token_cosine = [
                [
                    value if active[row] and active[column] else None
                    for column, value in enumerate(values)
                ]
                for row, values in enumerate(cosine)
            ]
        return {
            "labels": self.labels,
            "cka": similarities,
            "undefined_layers": [i for i in range(len(self.labels)) if i not in valid],
            "minimum_samples": 3,
            "sampled_positions": self.positions,
            "token_count": self.token_count,
            "token_cosine": token_cosine,
            "method": "Biased linear CKA from centered token-position Gram matrices. Tokens within one prompt are the samples. Similarity describes representation geometry and does not establish causal equivalence.",
        }

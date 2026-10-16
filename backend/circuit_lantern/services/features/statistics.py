"""Feature frequencies and activating contexts without retaining dense feature matrices."""

import torch

from circuit_lantern.services.features.autoencoder import SparseAutoencoder


@torch.inference_mode()
def feature_statistics(
    model: SparseAutoencoder,
    activations: torch.Tensor,
    *,
    top_examples: int = 6,
    splits: list[str] | None = None,
) -> list[dict]:
    if (
        activations.ndim != 2
        or len(activations) < 1
        or not 1 <= top_examples <= 20
        or activations.shape[1] != model.input_dim
        or activations.device.type != "cpu"
        or activations.dtype != torch.float32
        or not torch.isfinite(activations).all()
    ):
        raise ValueError(
            "Feature statistics need a nonempty activation matrix and a valid example count."
        )
    if splits is not None and (
        len(splits) != len(activations)
        or any(split not in ("train", "validation") for split in splits)
    ):
        raise ValueError("Feature statistics require one recorded split for every activation row.")
    split_masks = (
        {
            split: torch.tensor([value == split for value in splits], dtype=torch.bool)
            for split in ("train", "validation")
        }
        if splits is not None
        else {}
    )
    split_counts = {
        split: torch.zeros(model.config.features, dtype=torch.int64) for split in split_masks
    }
    width = model.config.features
    keep = min(top_examples, len(activations))
    maxima = torch.full((keep, width), -torch.inf, device="cpu")
    positions = torch.full((keep, width), -1, device="cpu", dtype=torch.int64)
    counts = torch.zeros(width, device="cpu", dtype=torch.int64)
    sums = torch.zeros(width, device="cpu", dtype=torch.float64)
    offset = 0
    for batch in activations.split(256):
        encoded = model.encode(batch.to(model.decoder.device))
        if not torch.isfinite(encoded).all():
            raise ValueError("The dictionary produced non finite feature activations.")
        counts += encoded.gt(0).sum(dim=0).cpu()
        for split, mask in split_masks.items():
            chosen = mask[offset : offset + len(batch)].to(encoded.device)
            split_counts[split] += encoded[chosen].gt(0).sum(dim=0).cpu()

        sums += encoded.double().sum(dim=0).cpu()
        indices = encoded.argsort(dim=0, descending=True, stable=True)[: min(keep, len(batch))]
        values = encoded.gather(0, indices)
        values, indices = values.cpu(), indices.cpu() + offset
        combined = torch.cat((maxima, values), dim=0)
        order = combined.argsort(dim=0, descending=True, stable=True)[:keep]
        selected = combined.gather(0, order)
        positions = torch.cat((positions, indices), dim=0).gather(0, order)
        maxima = selected
        offset += len(batch)
    example_values, example_positions = maxima.T.tolist(), positions.T.tolist()
    counts, sums = counts.tolist(), sums.tolist()
    split_totals = {split: int(mask.sum()) for split, mask in split_masks.items()}
    split_counts = {split: values.tolist() for split, values in split_counts.items()}

    return [
        {
            "index": feature,
            "activation_count": counts[feature],
            "frequency": counts[feature] / len(activations),
            "mean_activation": sums[feature] / len(activations),
            "max_activation": example_values[feature][0],
            "train_frequency": split_counts["train"][feature] / split_totals["train"]
            if split_totals.get("train")
            else None,
            "validation_frequency": split_counts["validation"][feature] / split_totals["validation"]
            if split_totals.get("validation")
            else None,
            "examples": [
                {
                    "sample": sample,
                    "activation": value,
                    "split": splits[sample] if splits is not None else None,
                }
                for sample, value in zip(
                    example_positions[feature], example_values[feature], strict=True
                )
                if value > 0
            ],
        }
        for feature in range(width)
    ]

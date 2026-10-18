"""Joint feature evidence without materializing a dense corpus by feature matrix."""

import torch

from circuit_lantern.services.crosscoders.model import SparseCrosscoder
from circuit_lantern.services.features.statistics import summarize_feature_batches


@torch.inference_mode()
def crosscoder_statistics(
    model: SparseCrosscoder, values: list[torch.Tensor], splits: list[str]
) -> list[dict]:
    count = len(values[0])
    batches = (
        model.encode([value[start : start + 128].to(model.device) for value in values])
        for start in range(0, count, 128)
    )
    statistics = summarize_feature_batches(batches, model.config.features, count, splits=splits)
    geometry = model.geometry()
    return [
        dict(row, **{key: value for key, value in directions.items() if key != "index"})
        for row, directions in zip(statistics, geometry, strict=True)
    ]

"""Joint sparse encoders and independent decoder banks in explicit activation units.

The per-bank decoder norm penalty follows Sparse Crosscoders for Cross-Layer
Features and Model Diffing at https://transformer-circuits.pub/2024/crosscoders/.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CrosscoderConfig:
    features: int
    steps: int = 256
    batch_size: int = 128
    learning_rate: float = 0.001
    l1_coefficient: float = 0.05
    seed: int = 0

    def validate(self, widths: tuple[int, ...]):
        if not 2 <= len(widths) <= 8 or any(
            type(width) is not int or not 1 <= width <= 65536 for width in widths
        ):
            raise ValueError("Choose between two and eight nonempty residual banks.")
        if any(
            type(value) is not int
            for value in (self.features, self.steps, self.batch_size, self.seed)
        ):
            raise ValueError("Crosscoder dimensions, counts and seed must be integers.")
        if not max(widths) <= self.features <= 65536:
            raise ValueError(
                "Choose at most 65536 features and at least the largest residual width."
            )
        if not 1 <= self.steps <= 20000 or not 1 <= self.batch_size <= 512:
            raise ValueError("Crosscoder steps or batch size exceed interactive limits.")
        if not 0 <= self.seed < 2**32:
            raise ValueError("Choose a random seed between zero and 4294967295.")
        for value in (self.learning_rate, self.l1_coefficient):
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("Crosscoder coefficients must be finite numbers.")
        if not 0 < self.learning_rate <= 0.1 or not 0 <= self.l1_coefficient <= 10:
            raise ValueError(
                "The learning rate or sparsity penalty is outside its supported range."
            )

    def description(self) -> dict:
        return asdict(self)


class SparseCrosscoder(nn.Module):
    """One ReLU feature vector jointly reconstructs all recorded residual banks.

    Encoder inputs and decoder outputs use original residual units. Each bank
    has its own training-fitted scale and learned centering bias. Decoder
    columns are initialized with a joint unit norm and subsequently remain
    unconstrained so their relative bank norms can change during learning.
    """

    def __init__(
        self, widths: tuple[int, ...], config: CrosscoderConfig, *, initialize: bool = True
    ):
        super().__init__()
        config.validate(widths)
        self.widths = tuple(widths)
        self.config = config
        if initialize:
            generator = torch.Generator(device="cpu").manual_seed(config.seed)
            decoders = [
                torch.randn(
                    width, config.features, generator=generator, device="cpu", dtype=torch.float32
                )
                for width in widths
            ]
            norms = (
                torch.stack([decoder.double().square().sum(dim=0) for decoder in decoders])
                .sum(dim=0)
                .sqrt()
                .float()
            )
            decoders = [decoder / norms for decoder in decoders]
            encoders = [decoder.T.contiguous().clone() for decoder in decoders]
            centers = [torch.zeros(width, device="cpu", dtype=torch.float32) for width in widths]
            bias = torch.zeros(config.features, device="cpu", dtype=torch.float32)
            scales = torch.ones(len(widths), device="cpu", dtype=torch.float32)
        else:
            decoders = [
                torch.empty(width, config.features, device="meta", dtype=torch.float32)
                for width in widths
            ]
            encoders = [
                torch.empty(config.features, width, device="meta", dtype=torch.float32)
                for width in widths
            ]
            centers = [torch.empty(width, device="meta", dtype=torch.float32) for width in widths]
            bias = torch.empty(config.features, device="meta", dtype=torch.float32)
            scales = torch.empty(len(widths), device="meta", dtype=torch.float32)
        self.decoders = nn.ParameterList(nn.Parameter(value) for value in decoders)
        self.encoders = nn.ParameterList(nn.Parameter(value) for value in encoders)
        self.centers = nn.ParameterList(nn.Parameter(value) for value in centers)
        self.encoder_bias = nn.Parameter(bias)
        self.register_buffer("input_scales", scales)

    @property
    def device(self) -> torch.device:
        return self.decoders[0].device

    def encode(self, values: list[torch.Tensor]) -> torch.Tensor:
        if len(values) != len(self.widths):
            raise ValueError("Provide one activation tensor for every crosscoder bank.")
        leading = values[0].shape[:-1]
        if any(
            value.ndim < 2 or value.shape[:-1] != leading or value.shape[-1] != width
            for value, width in zip(values, self.widths, strict=True)
        ):
            raise ValueError("Crosscoder banks must have aligned rows and recorded widths.")
        pre = self.encoder_bias
        for bank, (value, center, encoder) in enumerate(
            zip(values, self.centers, self.encoders, strict=True)
        ):
            pre = pre + F.linear(value / self.input_scales[bank] - center, encoder)
        return F.relu(pre)

    def decode_normalized(self, features: torch.Tensor) -> list[torch.Tensor]:
        return [
            F.linear(features, decoder, center)
            for decoder, center in zip(self.decoders, self.centers, strict=True)
        ]

    def decode(self, features: torch.Tensor) -> list[torch.Tensor]:
        return [
            value * self.input_scales[bank]
            for bank, value in enumerate(self.decode_normalized(features))
        ]

    def forward(self, values: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        features = self.encode(values)
        return self.decode(features), features

    def decoder_norms(self) -> torch.Tensor:
        return torch.stack(
            [
                torch.linalg.vector_norm(decoder, dim=0, dtype=torch.float64)
                for decoder in self.decoders
            ]
        )

    def sparsity_loss(self, features: torch.Tensor) -> torch.Tensor:
        # Sum the per-bank norms before weighting each shared feature.
        return (features.double() * self.decoder_norms().sum(dim=0)).sum(dim=-1).mean()

    def feature_direction(self, bank: int, feature: int) -> torch.Tensor:
        if (
            type(bank) is not int
            or not 0 <= bank < len(self.widths)
            or type(feature) is not int
            or not 0 <= feature < self.config.features
        ):
            raise ValueError("Choose an existing crosscoder bank and feature.")
        return self.decoders[bank][:, feature] * self.input_scales[bank]

    @torch.inference_mode()
    def geometry(self) -> list[dict]:
        norms = self.decoder_norms().cpu()
        total = norms.sum(dim=0)
        cosines = [None] * self.config.features
        if len(self.widths) == 2 and self.widths[0] == self.widths[1]:
            for start in range(0, self.config.features, 256):
                stop = min(start + 256, self.config.features)
                a, b = [decoder[:, start:stop].double() for decoder in self.decoders]
                a_norm = norms[0, start:stop].to(self.device)
                b_norm = norms[1, start:stop].to(self.device)
                valid = (a_norm > 0) & (b_norm > 0)
                unit_a = a / a_norm.clamp_min(torch.finfo(torch.float64).tiny)
                unit_b = b / b_norm.clamp_min(torch.finfo(torch.float64).tiny)
                values = (unit_a * unit_b).sum(dim=0).clamp(-1, 1).cpu().tolist()
                for offset, available in enumerate(valid.cpu().tolist()):
                    if available:
                        cosines[start + offset] = values[offset]
        return [
            {
                "index": feature,
                "decoder_norms": norms[:, feature].tolist(),
                "relative_decoder_norms": (norms[:, feature] / total[feature]).tolist()
                if total[feature] > 0
                else None,
                "decoder_cosine": cosines[feature],
            }
            for feature in range(self.config.features)
        ]

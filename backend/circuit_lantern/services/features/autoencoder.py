"""Sparse dictionary models with explicit raw activation units."""

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainingConfig:
    features: int
    method: Literal["relu", "topk"] = "relu"
    active_features: int = 16
    l1_coefficient: float = 0.001
    steps: int = 256
    batch_size: int = 128
    learning_rate: float = 0.001
    seed: int = 0
    auxiliary_coefficient: float = 1 / 32
    dead_after_tokens: int = 4096

    def validate(self, input_dim: int):
        integers = (
            input_dim,
            self.features,
            self.active_features,
            self.steps,
            self.batch_size,
            self.seed,
            self.dead_after_tokens,
        )
        if any(type(value) is not int for value in integers):
            raise ValueError("Dictionary dimensions, counts and seeds must be integers.")
        if any(
            type(value) not in (int, float)
            for value in (self.learning_rate, self.l1_coefficient, self.auxiliary_coefficient)
        ):
            raise ValueError("Dictionary coefficients must be finite numbers.")
        if not 1 <= input_dim <= self.features <= 65536:
            raise ValueError("Choose an overcomplete dictionary with at most 65536 features.")
        if self.method not in ("relu", "topk"):
            raise ValueError("Choose a ReLU or TopK sparse autoencoder.")
        if self.method == "topk" and not 1 <= self.active_features <= self.features:
            raise ValueError("The active feature allowance must fit inside the dictionary.")
        if not 1 <= self.steps <= 20000 or not 1 <= self.batch_size <= 512:
            raise ValueError("Training steps or batch size exceed the interactive training limits.")
        for value in (self.learning_rate, self.l1_coefficient, self.auxiliary_coefficient):
            if not math.isfinite(value):
                raise ValueError("Sparse autoencoder hyperparameters must be finite.")
        if not 0 < self.learning_rate <= 0.1 or not 0 <= self.l1_coefficient <= 10:
            raise ValueError(
                "The learning rate or sparsity penalty is outside its supported range."
            )
        if not 0 <= self.auxiliary_coefficient <= 1 or self.dead_after_tokens < 1:
            raise ValueError("The inactive feature settings are outside their supported range.")
        if not 0 <= self.seed < 2**32:
            raise ValueError("Choose a random seed between zero and 4294967295.")

    def description(self) -> dict:
        return asdict(self)


class SparseAutoencoder(nn.Module):
    """Untied encoder and unit norm decoder with one shared centering bias.

    encode and decode accept original model units. The learned activation scale
    is inverted only inside the encoder, so feature interventions can preserve
    the original residual reconstruction error.
    """

    def __init__(self, input_dim: int, config: TrainingConfig, *, initialize: bool = True):
        super().__init__()
        config.validate(input_dim)
        self.config = config
        self.input_dim = input_dim
        if not initialize:
            self.decoder = nn.Parameter(torch.empty(input_dim, config.features, device="meta"))
            self.encoder = nn.Parameter(torch.empty(config.features, input_dim, device="meta"))
            self.encoder_bias = (
                nn.Parameter(torch.empty(config.features, device="meta"))
                if config.method == "relu"
                else None
            )
            self.center = nn.Parameter(torch.empty(input_dim, device="meta"))
            self.register_buffer("input_scale", torch.empty((), device="meta"))
            return
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        decoder = torch.randn(input_dim, config.features, generator=generator, device="cpu")
        decoder = F.normalize(decoder, dim=0)
        self.decoder = nn.Parameter(decoder)
        self.encoder = nn.Parameter(decoder.T.contiguous().clone())
        self.encoder_bias = (
            nn.Parameter(torch.zeros(config.features, device="cpu"))
            if config.method == "relu"
            else None
        )
        self.center = nn.Parameter(torch.zeros(input_dim, device="cpu"))
        self.register_buffer("input_scale", torch.tensor(1.0, device="cpu"))

    def pre_activations(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value / self.input_scale - self.center, self.encoder, self.encoder_bias)

    def activate(self, pre: torch.Tensor) -> torch.Tensor:
        if self.config.method == "relu":
            return F.relu(pre)
        values, indices = pre.topk(self.config.active_features, dim=-1)
        return torch.zeros_like(pre).scatter(-1, indices, F.relu(values))

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return self.activate(self.pre_activations(value))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.decoder, self.center) * self.input_scale

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(value)
        return self.decode(features), features

    def feature_direction(self, feature: int) -> torch.Tensor:
        if not 0 <= feature < self.config.features:
            raise ValueError("The selected feature is outside this dictionary.")
        return self.decoder[:, feature] * self.input_scale

    @torch.no_grad()
    def normalize_decoder(self):
        norms = torch.linalg.vector_norm(self.decoder, dim=0, keepdim=True)
        if not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Training produced an invalid decoder direction.")
        self.decoder.div_(norms)

    @torch.no_grad()
    def project_decoder_gradient(self):
        if self.decoder.grad is not None:
            parallel = (self.decoder.grad * self.decoder).sum(dim=0, keepdim=True)
            self.decoder.grad.sub_(self.decoder * parallel)

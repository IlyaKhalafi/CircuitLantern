"""Bounded requests for repeated token pattern and ablation studies."""

from typing import Annotated

from pydantic import ConfigDict, Field

from circuit_lantern.schemas import BoundExecutionRequest


class InductionProbeRequest(BoundExecutionRequest):
    model_config = ConfigDict(extra="forbid")

    block_length: int = Field(default=25, strict=True, ge=4, le=64)
    repeats: int = Field(default=4, strict=True, ge=2, le=4)
    examples: int = Field(default=10, strict=True, ge=1, le=32)
    seed: int = Field(default=0, strict=True, ge=0, le=2**32 - 1)
    causal_heads: int = Field(default=3, strict=True, ge=0, le=12)
    include_start_token: bool = Field(default=True, strict=True)
    layers: list[Annotated[int, Field(strict=True, ge=0, le=65535)]] | None = Field(
        default=None, min_length=1, max_length=256
    )

"""Requests for aligned captures and joint feature dictionaries."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from circuit_lantern.schemas import BoundExecutionRequest, ExecutionId


class ActivationCaptureRequest(BoundExecutionRequest):
    name: str = Field(default="Reference capture", min_length=1, max_length=100)
    documents: list[Annotated[str, Field(min_length=1, max_length=8000)]] = Field(
        min_length=4, max_length=256
    )
    layer: int = Field(default=0, strict=True, ge=0, le=65535)
    max_samples: int = Field(default=4096, strict=True, ge=3, le=32768)
    max_tokens: int = Field(default=128, strict=True, ge=1, le=2048)
    split_seed: int = Field(default=0, strict=True, ge=0, le=2**32 - 1)
    exclude_special_tokens: bool = True
    exclude_first_position: bool = True


class MatchedCaptureRequest(BoundExecutionRequest):
    name: str = Field(default="Compared capture", min_length=1, max_length=100)
    layer: int = Field(default=0, strict=True, ge=0, le=65535)


class CrosscoderTrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="Shared feature comparison", min_length=1, max_length=100)
    capture_ids: tuple[ExecutionId, ExecutionId]
    expansion: int = Field(default=4, strict=True, ge=1, le=64)
    steps: int = Field(default=256, strict=True, ge=1, le=20000)
    batch_size: int = Field(default=128, strict=True, ge=1, le=512)
    learning_rate: float = Field(default=0.001, strict=True, gt=0, le=0.1, allow_inf_nan=False)
    l1_coefficient: float = Field(default=0.05, strict=True, ge=0, le=10, allow_inf_nan=False)
    seed: int = Field(default=0, strict=True, ge=0, le=2**32 - 1)
    device: Literal["cpu", "cuda"] = "cpu"


class CrosscoderLabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(default="", max_length=100)
    notes: str = Field(default="", max_length=10000)

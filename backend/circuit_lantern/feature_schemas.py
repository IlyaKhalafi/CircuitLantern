"""Requests for dictionary learning and causal feature experiments."""

from typing import Annotated, Literal

from pydantic import Field

from circuit_lantern.schemas import BoundExecutionRequest, ExecutionId


class DictionaryTrainingRequest(BoundExecutionRequest):
    name: str = Field(default="Residual dictionary", min_length=1, max_length=100)
    documents: list[Annotated[str, Field(min_length=1, max_length=8000)]] = Field(
        min_length=4, max_length=256
    )
    layers: list[int] = Field(min_length=1, max_length=256)
    method: Literal["relu", "topk"] = "topk"
    expansion: int = Field(default=4, ge=1, le=64)
    active_features: int = Field(default=16, ge=1, le=512)
    max_samples: int = Field(default=4096, ge=3, le=32768)
    steps: int = Field(default=256, ge=1, le=20000)
    batch_size: int = Field(default=128, ge=1, le=512)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1, allow_inf_nan=False)
    l1_coefficient: float = Field(default=0.001, ge=0, le=10, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    split_seed: int = Field(default=0, ge=0, le=2**32 - 1)
    exclude_special_tokens: bool = True
    exclude_first_position: bool = True


class FeatureInspectionRequest(BoundExecutionRequest):
    prompt: str = Field(min_length=1, max_length=8000)
    target_token_id: int | None = Field(default=None, ge=0)
    feature: int | None = Field(default=None, ge=0)
    multiplier: float = Field(default=0, ge=-10, le=10, allow_inf_nan=False)
    position: int | None = Field(default=None, ge=0)


class FeatureLabelRequest(BoundExecutionRequest):
    label: str = Field(default="", max_length=100)
    notes: str = Field(default="", max_length=10000)


class FeatureCircuitRequest(BoundExecutionRequest):
    dictionary_ids: list[ExecutionId] = Field(min_length=1, max_length=8)
    prompt: str = Field(min_length=1, max_length=8000)
    target_token_id: int | None = Field(default=None, ge=0)
    features_per_layer: int = Field(default=4, ge=1, le=4)

"""Bounded, session-bound paired prompt experiments."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt

from circuit_lantern.schemas import ExecutionId

MAX_PROMPT_PAIRS = 64
MAX_SET_FORWARDS = 2048
BOOTSTRAP_SAMPLES = 2000
SIGN_EPSILON = 1e-5


class PromptPair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(default="", max_length=100)
    clean_prompt: str = Field(min_length=1, max_length=8000)
    corrupted_prompt: str = Field(min_length=1, max_length=8000)
    target: str = Field(min_length=1, max_length=100)
    alternative: str = Field(default="", max_length=100)


class PatchingSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace_id: ExecutionId
    model_id: str = Field(min_length=1, max_length=180)
    revision: str = Field(min_length=1, max_length=128)
    device: str = Field(min_length=1, max_length=32)
    execution_id: ExecutionId | None = None
    execution_precision: str = Field(min_length=1, max_length=32)
    execution_config: dict[str, Any] = Field(max_length=32)
    clean_prompt: str = Field(min_length=1, max_length=8000)
    token_ids: list[Annotated[StrictInt, Field(ge=0)]] = Field(min_length=1, max_length=131072)
    position: Annotated[StrictInt, Field(ge=0)]
    layer: Annotated[StrictInt, Field(ge=0)]
    target_token_id: Annotated[StrictInt, Field(ge=0)]
    alternative_token_id: Annotated[StrictInt, Field(ge=0)]
    attribution_schema_version: Literal[1]
    attribution_value: FiniteFloat
    attribution_score: FiniteFloat
    attribution_error: Annotated[FiniteFloat, Field(ge=0)]


class PromptSetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_execution_id: ExecutionId
    pairs: list[PromptPair] = Field(min_length=1, max_length=MAX_PROMPT_PAIRS)
    metric: Literal["logit_difference", "log_probability"] = "logit_difference"
    position_mode: Literal["last", "changed", "all"] = "last"
    layers: list[Annotated[StrictInt, Field(ge=0)]] | None = Field(
        default=None, min_length=1, max_length=128
    )
    seed: Annotated[StrictInt, Field(ge=0, le=4294967295)] = 0
    hypothesis_source: PatchingSource | None = None

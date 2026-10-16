from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ModelRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=180, pattern=r"^[\w.-]+(/[\w.-]+)?$")
    revision: str = Field(default="main", min_length=1, max_length=128)


ContextLength = Annotated[int, Field(ge=16, le=131072)]

Precision = Literal["auto", "float32", "float16", "bfloat16", "int8", "int4", "fp8", "nvfp4"]


class CompatibilityRequest(ModelRequest):
    precision: Precision = "auto"
    context_length: ContextLength = 2048
    batch_size: int = Field(default=1, ge=1, le=64)
    device: Literal["cpu", "cuda"] = "cpu"


class TraceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=16, ge=1, le=64)
    capture_attention: bool = True
    ablate_layer: int | None = Field(default=None, ge=0)


class ExperimentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    notes: str = Field(default="", max_length=10000)
    trace: dict


class ExperimentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    notes: str | None = Field(default=None, max_length=10000)


class LoadRequest(ModelRequest):
    context_length: ContextLength = 128
    precision: Precision = "auto"
    device: Literal["cpu", "cuda"] = "cpu"


class CausalRequest(BaseModel):
    clean_prompt: str = Field(min_length=1, max_length=8000)
    corrupted_prompt: str = Field(min_length=1, max_length=8000)
    target: str = Field(default="", max_length=100)
    position: int | None = Field(default=-1, ge=-131072, le=131071)

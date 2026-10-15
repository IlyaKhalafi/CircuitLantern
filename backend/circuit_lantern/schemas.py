from typing import Literal

from pydantic import BaseModel, Field


class ModelRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=180, pattern=r"^[\w.-]+(/[\w.-]+)?$")
    revision: str = Field(default="main", min_length=1, max_length=128)


class CompatibilityRequest(ModelRequest):
    precision: Literal["auto", "float32", "float16", "bfloat16", "int8", "int4", "fp8", "nvfp4"] = (
        "auto"
    )
    context_length: int = Field(default=2048, ge=16, le=131072)
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
    device: Literal["cpu", "cuda"] = "cpu"

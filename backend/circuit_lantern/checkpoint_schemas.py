from typing import Annotated

from pydantic import Field

from circuit_lantern.schemas import BoundExecutionRequest, ExecutionId


class CheckpointCaptureRequest(BoundExecutionRequest):
    name: str = Field(min_length=1, max_length=100)
    prompt: str = Field(min_length=1, max_length=8000)
    matrices: list[Annotated[str, Field(min_length=1, max_length=512)]] = Field(
        min_length=1, max_length=12
    )


class CheckpointCompareRequest(BoundExecutionRequest):
    reference_id: ExecutionId

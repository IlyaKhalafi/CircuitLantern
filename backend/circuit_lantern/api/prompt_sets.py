"""Session-bound resource planning and cancellable prompt-set patching."""

from fastapi import APIRouter, Request

from circuit_lantern.prompt_set_schemas import PromptSetRequest
from circuit_lantern.services.prompt_sets import measure_set, prepare_set

router = APIRouter(prefix="/api/causal/sets", tags=["causal"])


@router.post("/plan")
def plan(payload: PromptSetRequest, request: Request):
    runtime = request.app.state.runtime
    with runtime.operation():
        return prepare_set(runtime, payload)[0]


@router.post("/run", status_code=202)
def run(payload: PromptSetRequest, request: Request):
    runtime = request.app.state.runtime
    return runtime._start_sweep(
        "causal_batch",
        payload.model_dump(),
        worker=lambda _, job: measure_set(runtime, payload, job),
    )

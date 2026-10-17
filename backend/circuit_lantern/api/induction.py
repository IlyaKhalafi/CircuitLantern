"""Repeated token pattern planning and cancellable causal measurements."""

from fastapi import APIRouter, Request

from circuit_lantern.induction_schemas import InductionProbeRequest

router = APIRouter(prefix="/api/induction", tags=["induction"])


@router.get("/capabilities")
def capabilities(request: Request):
    return request.app.state.induction.capabilities()


@router.post("/plan")
def plan(payload: InductionProbeRequest, request: Request):
    return request.app.state.induction.plan(payload)


@router.post("/probe", status_code=202)
def probe(payload: InductionProbeRequest, request: Request):
    return request.app.state.induction.start(payload)

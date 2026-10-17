"""Repeated token pattern planning and cancellable causal measurements."""

from fastapi import APIRouter, HTTPException, Request

from circuit_lantern.induction_schemas import InductionProbeRequest
from circuit_lantern.schemas import ExecutionId
from circuit_lantern.services.induction.comparison import compare_probes
from circuit_lantern.services.induction.planning import ensure_saved_comparison_memory

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


@router.get("/comparison")
def comparison(request: Request, selected: ExecutionId, reference: ExecutionId):
    if selected == reference:
        raise ValueError("Choose a different saved probe as the reference.")
    sizes = [
        request.app.state.store.payload_size(identifier) for identifier in (selected, reference)
    ]
    if any(size is None for size in sizes):
        raise HTTPException(404, "A saved probe was not found. It may have been deleted.")
    ensure_saved_comparison_memory(sum(sizes))
    records = []
    for identifier in (selected, reference):
        record = request.app.state.store.get(identifier)
        if record is None:
            raise HTTPException(404, "A saved probe was not found. It may have been deleted.")
        if record.get("kind") != "induction_probe" or not isinstance(record.get("result"), dict):
            raise ValueError("Choose two saved induction probes to compare their measurements.")
        records.append(record)
    result = compare_probes(records[0]["result"], records[1]["result"])
    result["selected_record"], result["reference_record"] = (
        {key: value for key, value in record.items() if key != "result"} for record in records
    )
    return result

from fastapi import APIRouter, Request

from circuit_lantern.checkpoint_schemas import CheckpointCaptureRequest, CheckpointCompareRequest

router = APIRouter(prefix="/api/checkpoints", tags=["checkpoints"])


@router.get("/references")
def references(request: Request):
    return request.app.state.checkpoints.store.library()


@router.post("/references", status_code=201)
def capture(payload: CheckpointCaptureRequest, request: Request):
    return request.app.state.checkpoints.capture(payload)


@router.get("/references/{identifier}")
def reference(identifier: str, request: Request):
    return request.app.state.checkpoints.store.read(identifier)


@router.delete("/references/{identifier}", status_code=204)
def delete_reference(identifier: str, request: Request):
    request.app.state.checkpoints.store.delete(identifier)


@router.post("/compare", status_code=201)
def compare(payload: CheckpointCompareRequest, request: Request):
    return request.app.state.checkpoints.compare(payload)


@router.get("/comparisons")
def comparisons(request: Request):
    return request.app.state.checkpoints.store.comparisons()


@router.get("/comparisons/{identifier}")
def comparison(identifier: str, request: Request):
    return request.app.state.checkpoints.store.comparison(identifier)


@router.delete("/comparisons/{identifier}", status_code=204)
def delete_comparison(identifier: str, request: Request):
    request.app.state.checkpoints.store.delete_comparison(identifier)

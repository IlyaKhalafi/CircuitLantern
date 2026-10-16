"""No code sparse dictionary and feature circuit endpoints."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from circuit_lantern.feature_schemas import (
    DictionaryTrainingRequest,
    FeatureCircuitRequest,
    FeatureInspectionRequest,
    FeatureLabelRequest,
)
from circuit_lantern.services.features.circuits import feature_ablation_graph
from circuit_lantern.services.features.directions import dictionary_directions
from circuit_lantern.services.features.inference import evaluate_prompt, inspect_prompt

router = APIRouter(prefix="/api/features", tags=["features"])


@router.get("/capabilities")
def capabilities(request: Request):
    return request.app.state.features.capabilities()


@router.post("/plan")
def plan(payload: DictionaryTrainingRequest, request: Request):
    return request.app.state.features.plan(payload)


@router.post("/train", status_code=202)
def train(payload: DictionaryTrainingRequest, request: Request):
    return request.app.state.features.start_training(payload)


@router.get("/dictionaries")
def dictionaries(request: Request):
    return request.app.state.features.store.list()


@router.get("/dictionaries/{identifier}")
def dictionary(identifier: str, request: Request):
    result = request.app.state.features.store.get(identifier)
    if result is None:
        raise HTTPException(404, "The dictionary was not found.")
    return result


@router.get("/dictionaries/{identifier}/weights")
def weights(identifier: str, request: Request):
    store = request.app.state.features.store
    with store.file_lock:
        path = store.export_weights(identifier)
        stream = path.open("rb")
        length = path.stat().st_size
    return StreamingResponse(
        iter(lambda: stream.read(1024**2), b""),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": 'attachment; filename="sae-' + identifier + '.safetensors"',
            "Content-Length": str(length),
        },
        background=BackgroundTask(stream.close),
    )


@router.delete("/dictionaries/{identifier}", status_code=204)
def delete_dictionary(identifier: str, request: Request):
    with request.app.state.runtime.operation():
        if not request.app.state.features.store.delete(identifier):
            raise HTTPException(404, "The dictionary was not found.")


@router.patch("/dictionaries/{identifier}/features/{feature}")
def label(identifier: str, feature: int, payload: FeatureLabelRequest, request: Request):
    return request.app.state.features.store.label_feature(
        identifier, feature, payload.label, payload.notes
    )


@router.get("/dictionaries/{identifier}/features/{feature}/directions")
def directions(
    identifier: str,
    feature: Annotated[int, Path(ge=0, le=65535)],
    request: Request,
    limit: int = Query(default=8, ge=1, le=32),
):
    try:
        return dictionary_directions(request.app.state.features.store, identifier, feature, limit)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error


@router.post("/dictionaries/{identifier}/inspect")
def inspect(identifier: str, payload: FeatureInspectionRequest, request: Request):
    return inspect_prompt(request.app.state.features, identifier, payload)


@router.post("/dictionaries/{identifier}/evaluate")
def evaluate(identifier: str, payload: FeatureInspectionRequest, request: Request):
    return evaluate_prompt(request.app.state.features, identifier, payload)


@router.post("/circuits", status_code=202)
def circuits(payload: FeatureCircuitRequest, request: Request):
    lab = request.app.state.features

    def operation(values, job):
        return feature_ablation_graph(lab, values, job)

    return lab.runtime._start_sweep("feature_circuit", payload.model_dump(), operation)


@router.get("/runs")
def runs(request: Request):
    return request.app.state.features.store.runs()


@router.get("/runs/{identifier}")
def run(identifier: str, request: Request):
    result = request.app.state.features.store.get_run(identifier)
    if result is None:
        raise HTTPException(404, "The feature experiment was not found.")
    return result

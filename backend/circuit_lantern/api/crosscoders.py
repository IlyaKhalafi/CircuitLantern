"""Retained activation captures and joint feature comparison endpoints."""

from typing import Annotated, Literal

import torch
from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from circuit_lantern.crosscoder_schemas import (
    ActivationCaptureRequest,
    CrosscoderLabelRequest,
    CrosscoderTrainingRequest,
    MatchedCaptureRequest,
)
from circuit_lantern.services.crosscoders.lab import REFERENCES

router = APIRouter(prefix="/api/crosscoders", tags=["crosscoders"])
Identifier = Annotated[str, Path(pattern=r"^[a-f0-9]{32}$")]
FeatureIndex = Annotated[int, Path(ge=0, le=65535)]


def required(request: Request, identifier: str, kind: str):
    value = request.app.state.crosscoders.store.get(identifier, kind=kind)
    if value is None:
        raise HTTPException(404, "The saved artifact was not found.")
    return value


@router.get("/capabilities")
def capabilities(request: Request):
    return request.app.state.features.capabilities() | {
        "devices": ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
        "method": "relu_crosscoder",
        "references": REFERENCES,
    }


@router.get("/captures")
def captures(request: Request):
    return request.app.state.crosscoders.store.list("capture")


@router.post("/captures/plan")
def capture_plan(payload: ActivationCaptureRequest, request: Request):
    return request.app.state.crosscoders.capture_plan(payload)


@router.post("/captures", status_code=202)
def capture(payload: ActivationCaptureRequest, request: Request):
    return request.app.state.crosscoders.start_capture(payload)


@router.get("/captures/{identifier}")
def retained_capture(identifier: Identifier, request: Request):
    return required(request, identifier, "capture")


@router.post("/captures/{identifier}/match/plan")
def matching_plan(identifier: Identifier, payload: MatchedCaptureRequest, request: Request):
    return request.app.state.crosscoders.matched_plan(identifier, payload)


@router.post("/captures/{identifier}/match", status_code=202)
def matching_capture(identifier: Identifier, payload: MatchedCaptureRequest, request: Request):
    return request.app.state.crosscoders.start_matched(identifier, payload)


@router.delete("/captures/{identifier}", status_code=204)
def delete_capture(identifier: Identifier, request: Request):
    with request.app.state.runtime.operation():
        if not request.app.state.crosscoders.store.delete(identifier, kind="capture"):
            raise HTTPException(404, "The retained capture was not found.")


def tensors(request: Request, identifier: str, kind: str):
    stream, length = request.app.state.crosscoders.store.open_tensors(identifier, kind=kind)
    return StreamingResponse(
        iter(lambda: stream.read(1024**2), b""),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": 'attachment; filename="'
            + kind
            + "-"
            + identifier
            + '.safetensors"',
            "Content-Length": str(length),
        },
        background=BackgroundTask(stream.close),
    )


@router.get("/captures/{identifier}/tensors")
def capture_tensors(identifier: Identifier, request: Request):
    return tensors(request, identifier, "capture")


@router.get("")
def comparisons(request: Request):
    return request.app.state.crosscoders.store.list("crosscoder")


@router.post("/plan")
def training_plan(payload: CrosscoderTrainingRequest, request: Request):
    return request.app.state.crosscoders.training_plan(payload)


@router.post("/train", status_code=202)
def train(payload: CrosscoderTrainingRequest, request: Request):
    return request.app.state.crosscoders.start_training(payload)


@router.get("/{identifier}/overview")
def overview(identifier: Identifier, request: Request):
    return request.app.state.crosscoders.overview(identifier)


@router.get("/{identifier}/features")
def features(
    identifier: Identifier,
    request: Request,
    q: str = Query(default="", max_length=100),
    sort: Literal["frequency", "reference", "compared", "cosine"] = "frequency",
    direction: Literal["highest", "lowest"] = "highest",
    offset: int = Query(default=0, ge=0, le=65535),
    limit: int = Query(default=50, ge=1, le=100),
):
    return request.app.state.crosscoders.feature_rows(
        identifier,
        query=q,
        sort=sort,
        direction=direction,
        offset=offset,
        limit=limit,
    )


@router.get("/{identifier}/features/{feature}")
def feature(identifier: Identifier, feature: FeatureIndex, request: Request):
    return request.app.state.crosscoders.feature(identifier, feature)


@router.patch("/{identifier}/features/{feature}")
def label(
    identifier: Identifier, feature: FeatureIndex, payload: CrosscoderLabelRequest, request: Request
):
    return request.app.state.crosscoders.store.label(
        identifier, feature, payload.label, payload.notes
    )


@router.get("/{identifier}/weights")
def weights(identifier: Identifier, request: Request):
    return tensors(request, identifier, "crosscoder")


@router.get("/{identifier}/export")
def export(identifier: Identifier, request: Request):
    return {
        "format": "circuit-lantern-crosscoder-bundle-v1",
        "artifact": required(request, identifier, "crosscoder"),
        "annotations": request.app.state.crosscoders.store.annotations(identifier),
    }


@router.delete("/{identifier}", status_code=204)
def delete_comparison(identifier: Identifier, request: Request):
    with request.app.state.runtime.operation():
        if not request.app.state.crosscoders.store.delete(identifier, kind="crosscoder"):
            raise HTTPException(404, "The shared feature comparison was not found.")

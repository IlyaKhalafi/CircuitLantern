"""Retained activation captures and joint feature comparison endpoints."""

import asyncio
import sqlite3
from contextlib import ExitStack
from functools import partial
from threading import Event
from typing import Annotated, Literal

import torch
from fastapi import APIRouter, HTTPException, Path, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Receive, Scope, Send

from circuit_lantern.crosscoder_schemas import (
    ActivationCaptureRequest,
    CrosscoderLabelRequest,
    CrosscoderTrainingRequest,
    MatchedCaptureRequest,
)
from circuit_lantern.services.crosscoders.lab import REFERENCES
from circuit_lantern.services.prepared_downloads import DownloadCancelledError, PreparedDownloads
from circuit_lantern.services.tensor_artifacts import ArtifactNotFoundError

router = APIRouter(prefix="/api/crosscoders", tags=["crosscoders"])
Identifier = Annotated[str, Path(pattern=r"^[a-f0-9]{32}$")]
FeatureIndex = Annotated[int, Path(ge=0, le=65535)]
_DOWNLOAD_SEND_TIMEOUT = 30


def artifact_json(request: Request, identifier: str, kind: str, *, bundle=False):
    store = request.app.state.crosscoders.store
    with (
        store.read_metadata(identifier, kind=kind, serializing=True) as value,
        ExitStack() as reservations,
    ):
        if value is None:
            raise HTTPException(404, "The saved artifact was not found.")
        if bundle:
            value = {
                "format": "circuit-lantern-crosscoder-bundle-v1",
                "artifact": value,
                "annotations": reservations.enter_context(
                    store.read_annotations(identifier, serializing=True)
                ),
            }
        # JSONResponse renders eagerly while the shared memory reservation is still held.
        return JSONResponse(jsonable_encoder(value))


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
    return artifact_json(request, identifier, "capture")


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


class PinnedDownloadResponse(StreamingResponse):
    def __init__(self, stream, length: int, filename: str):
        self.stream = stream

        def read_chunk():
            try:
                return stream.read(1024**2)
            except OSError as error:
                raise RuntimeError("The verified tensor file could not be streamed.") from error

        super().__init__(
            iter(read_chunk, b""),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": 'attachment; filename="' + filename + '"',
                "Content-Length": str(length),
                "Cache-Control": "no-store",
            },
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        async def bounded_send(message):
            async with asyncio.timeout(_DOWNLOAD_SEND_TIMEOUT):
                await send(message)

        try:
            await super().__call__(scope, receive, bounded_send)
        finally:
            self.stream.close()


def stream_download(stream, length: int, filename: str):
    return PinnedDownloadResponse(stream, length, filename)


class PreparedDownloadResponse(Response):
    def __init__(self, downloads: PreparedDownloads, ticket: str):
        super().__init__()
        self.downloads = downloads
        self.ticket = ticket

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        lease = self.downloads.take(self.ticket)
        if lease is None:
            response = JSONResponse(
                {"detail": "The prepared download expired. Prepare the download again."},
                status_code=410,
                headers={"Cache-Control": "no-store"},
            )
            await response(scope, receive, send)
            return
        try:
            response = stream_download(lease.stream, lease.length, lease.filename)
            await response(scope, receive, send)
        finally:
            self.downloads.release(self.ticket)


def open_artifact(request: Request, identifier: str, kind: str, cancelled=None):
    try:
        return request.app.state.crosscoders.store.open_tensors(
            identifier, kind=kind, cancelled=cancelled
        )
    except ArtifactNotFoundError:
        raise
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            503, "The saved tensor file could not be read. Check server storage and try again."
        ) from error


async def tensors(request: Request, identifier: str, kind: str):
    prepared = await prepare_download(request, identifier, kind)
    if isinstance(prepared, Response):
        return prepared
    return PreparedDownloadResponse(request.app.state.crosscoders.downloads, prepared["ticket"])


async def prepare_download(request: Request, identifier: str, kind: str):
    lab = request.app.state.crosscoders
    cancelled = Event()
    handoff = False

    def should_cancel():
        return cancelled.is_set() or lab.downloads.closed

    work = asyncio.create_task(
        asyncio.to_thread(
            lab.downloads.prepare,
            partial(open_artifact, request, identifier, kind, should_cancel),
            kind + "-" + identifier + ".safetensors",
            should_cancel,
        )
    )

    def cleanup(completed):
        if not completed.done() or completed.cancelled():
            return
        if completed.exception() is not None:
            return
        ticket, _ = completed.result()
        if cancelled.is_set():
            lab.downloads.discard(ticket)

    work.add_done_callback(cleanup)
    try:
        while not work.done():
            if await request.is_disconnected():
                return Response(status_code=499)
            await asyncio.wait({work}, timeout=0.05)
        ticket, length = work.result()
        if await request.is_disconnected():
            return Response(status_code=499)
        handoff = True
        return {"ticket": ticket, "size": length}
    except DownloadCancelledError:
        if lab.downloads.closed:
            return JSONResponse(
                {"detail": "Downloads are busy. Try again shortly."},
                status_code=503,
                headers={"Cache-Control": "no-store"},
            )
        return Response(status_code=499)
    finally:
        if not handoff:
            cancelled.set()
            cleanup(work)


@router.post("/captures/{identifier}/download")
async def prepare_capture_download(identifier: Identifier, request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return await prepare_download(request, identifier, "capture")


@router.get("/downloads/{ticket}")
def prepared_download(ticket: Annotated[str, Path(pattern=r"^[a-f0-9]{64}$")], request: Request):
    return PreparedDownloadResponse(request.app.state.crosscoders.downloads, ticket)


@router.get("/captures/{identifier}/tensors")
async def capture_tensors(identifier: Identifier, request: Request):
    return await tensors(request, identifier, "capture")


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


@router.post("/{identifier}/download")
async def prepare_weights_download(identifier: Identifier, request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return await prepare_download(request, identifier, "crosscoder")


@router.get("/{identifier}/weights")
async def weights(identifier: Identifier, request: Request):
    return await tensors(request, identifier, "crosscoder")


@router.get("/{identifier}/export")
def export(identifier: Identifier, request: Request):
    return artifact_json(request, identifier, "crosscoder", bundle=True)


@router.delete("/{identifier}", status_code=204)
def delete_comparison(identifier: Identifier, request: Request):
    with request.app.state.runtime.operation():
        if not request.app.state.crosscoders.store.delete(identifier, kind="crosscoder"):
            raise HTTPException(404, "The shared feature comparison was not found.")

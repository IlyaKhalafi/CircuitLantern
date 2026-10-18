from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled

from circuit_lantern.api.checkpoints import router as checkpoint_router
from circuit_lantern.api.crosscoders import router as crosscoder_router
from circuit_lantern.api.features import router as feature_router
from circuit_lantern.api.induction import router as induction_router
from circuit_lantern.api.prompt_sets import router as prompt_set_router
from circuit_lantern.config import get_settings
from circuit_lantern.schemas import (
    CausalRequest,
    CompatibilityRequest,
    ExecutionId,
    ExperimentFromJobRequest,
    ExperimentFromMatrixRequest,
    ExperimentRequest,
    ExperimentUpdate,
    HeadSweepRequest,
    LoadRequest,
    ModelRequest,
    TargetPreviewRequest,
    TokenizeRequest,
    TraceRequest,
)
from circuit_lantern.services.artifact_views import MetadataMemoryError
from circuit_lantern.services.checkpoint_store import CheckpointStore
from circuit_lantern.services.checkpoints import CheckpointLab
from circuit_lantern.services.crosscoders.lab import CrosscoderLab
from circuit_lantern.services.features.lab import FeatureLab
from circuit_lantern.services.features.store import DictionaryStore
from circuit_lantern.services.hardware import hardware_snapshot
from circuit_lantern.services.hub import model_metadata, search_models
from circuit_lantern.services.hub_errors import explain_hub_error
from circuit_lantern.services.induction.lab import InductionLab
from circuit_lantern.services.induction.reference import ReferenceProbeNotFoundError
from circuit_lantern.services.prepared_downloads import DownloadBusyError
from circuit_lantern.services.runtime import ModelChangedError, ModelRuntime, RuntimeBusyError
from circuit_lantern.services.store import ExperimentStore
from circuit_lantern.services.tensor_artifacts import (
    ArtifactBusyError,
    ArtifactNotFoundError,
    TensorArtifactStore,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.runtime = ModelRuntime(settings)
    app.state.store = ExperimentStore(settings.data_dir / "experiments.sqlite3")
    app.state.induction = InductionLab(app.state.runtime, app.state.store)
    app.state.features = FeatureLab(
        app.state.runtime,
        DictionaryStore(
            settings.data_dir / "dictionaries", max_weight_bytes=settings.max_dictionary_bytes
        ),
    )
    app.state.crosscoders = CrosscoderLab(
        app.state.runtime,
        TensorArtifactStore(
            settings.data_dir / "shared-features",
            max_bytes=settings.max_dictionary_bytes,
        ),
    )
    app.state.checkpoints = CheckpointLab(
        app.state.runtime, CheckpointStore(settings.data_dir / "checkpoints")
    )
    yield
    app.state.crosscoders.downloads.close()
    with app.state.runtime.job_lock:
        for job in app.state.runtime.jobs.values():
            if job.get("kind") != "download":
                job.request_cancel()
    app.state.runtime.executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="Circuit Lantern", version="0.1.0", lifespan=lifespan)
app.include_router(feature_router)
app.include_router(checkpoint_router)
app.include_router(crosscoder_router)
app.include_router(induction_router)
app.include_router(prompt_set_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(ModelChangedError)
@app.exception_handler(RuntimeBusyError)
async def runtime_conflict(request: Request, error: RuntimeBusyError | ModelChangedError):
    return JSONResponse(status_code=409, content={"detail": str(error)})


@app.exception_handler(ArtifactBusyError)
@app.exception_handler(DownloadBusyError)
async def artifact_conflict(request: Request, error: ArtifactBusyError | DownloadBusyError):
    return JSONResponse(status_code=409, content={"detail": str(error)})


@app.exception_handler(ArtifactNotFoundError)
async def artifact_missing(request: Request, error: ArtifactNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(error)})


@app.exception_handler(MetadataMemoryError)
async def metadata_memory_pressure(request: Request, error: MetadataMemoryError):
    return JSONResponse(
        status_code=503, content={"detail": str(error)}, headers={"Retry-After": "2"}
    )


@app.exception_handler(ReferenceProbeNotFoundError)
async def missing_probe_reference(request: Request, error: ReferenceProbeNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(error)})


@app.exception_handler(ValueError)
async def value_error(request: Request, error: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.exception_handler(HfHubHTTPError)
@app.exception_handler(LocalEntryNotFoundError)
@app.exception_handler(OfflineModeIsEnabled)
@app.exception_handler(httpx.RequestError)
async def hub_error(request: Request, error: Exception):
    status, message, headers = explain_hub_error(error)
    return JSONResponse(status_code=status, content={"detail": message}, headers=headers)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/hardware")
def hardware():
    return hardware_snapshot(get_settings().data_dir)


@app.get("/api/models")
def models(
    q: str = Query(default="", max_length=180),
    sort: Literal[
        "recommended", "trending", "downloads", "likes", "updated", "newest"
    ] = "recommended",
    days: int = Query(default=0, ge=0, le=365),
    min_likes: int = Query(default=0, ge=0, le=1000000),
    author: str = Query(default="", max_length=96, pattern=r"^[a-zA-Z0-9_.-]*$"),
    limit: int = Query(default=24, ge=1, le=96),
):
    return search_models(q, sort=sort, days=days, min_likes=min_likes, author=author, limit=limit)


@app.post("/api/models/metadata")
def metadata(request: ModelRequest):
    return model_metadata(request.model_id, request.revision)


def check(request: CompatibilityRequest, runtime: ModelRuntime):
    return runtime.assess_fit(model_metadata(request.model_id, request.revision), request)


@app.post("/api/models/compatibility")
def compatibility(payload: CompatibilityRequest, request: Request):
    return check(payload, request.app.state.runtime)


@app.post("/api/models/load", status_code=202)
def load_model(payload: LoadRequest, request: Request):
    report = check(CompatibilityRequest(**payload.model_dump()), request.app.state.runtime)
    if not report["compatible"]:
        raise HTTPException(409, detail=" ".join(report["blockers"]))
    return request.app.state.runtime.load(
        report["metadata"], payload.device, payload.precision, payload.context_length
    )


@app.get("/api/jobs/{job_id}")
def job(job_id: str, request: Request):
    result = request.app.state.runtime.job(job_id)
    if result is None:
        raise HTTPException(
            404, "Job not found. It may have expired or the server may have restarted."
        )
    return dict(result)


@app.get("/api/runtime")
def runtime_state(request: Request):
    return request.app.state.runtime.state()


@app.post("/api/causal", status_code=202)
def causal(payload: CausalRequest, request: Request):
    return request.app.state.runtime.causal(payload.model_dump())


@app.post("/api/head-sweep", status_code=202)
def head_sweep(payload: HeadSweepRequest, request: Request):
    return request.app.state.runtime.head_sweep(payload.model_dump())


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request):
    result = request.app.state.runtime.cancel_job(job_id)
    if result is None:
        raise HTTPException(404, "Job not found.")
    return result


@app.delete("/api/runtime", status_code=204)
def unload(request: Request, expected_execution_id: ExecutionId | None = None):
    request.app.state.runtime.unload(expected_execution_id)


@app.get("/api/graph")
def graph(request: Request, expected_execution_id: ExecutionId | None = None):
    return request.app.state.runtime.graph(expected_execution_id)


@app.get("/api/matrix")
def matrix(
    request: Request,
    name: str = Query(max_length=240),
    expected_execution_id: ExecutionId | None = None,
):
    return request.app.state.runtime.matrix(name, expected_execution_id)


@app.post("/api/tokenize")
def tokenize(payload: TokenizeRequest, request: Request):
    return request.app.state.runtime.tokenize(
        payload.prompts, payload.reserved_tokens, payload.expected_execution_id
    )


@app.post("/api/targets/preview")
def preview_targets(payload: TargetPreviewRequest, request: Request):
    return request.app.state.runtime.preview_targets(
        payload.tracked_tokens, payload.expected_execution_id
    )


@app.post("/api/trace")
def trace(payload: TraceRequest, request: Request):
    return request.app.state.runtime.trace(**payload.model_dump())


@app.get("/api/experiments")
def experiments(request: Request):
    return request.app.state.store.list()


@app.post("/api/experiments", status_code=201)
def create_experiment(payload: ExperimentRequest, request: Request):
    return request.app.state.store.create(**payload.model_dump())


@app.post("/api/experiments/from-matrix", status_code=201)
def save_matrix(payload: ExperimentFromMatrixRequest, request: Request):
    store = request.app.state.store
    existing = store.find_analysis(payload.analysis_id)
    if existing is not None:
        return existing
    result = request.app.state.runtime.matrix_snapshot(
        payload.analysis_id, payload.expected_execution_id
    )
    return store.create_artifact(
        payload.name, payload.notes, "matrix", result, source_analysis_id=payload.analysis_id
    )


@app.post("/api/experiments/from-job", status_code=201)
def save_job_experiment(payload: ExperimentFromJobRequest, request: Request):
    store = request.app.state.store
    existing = store.find_job(payload.job_id)
    if existing is not None:
        return existing
    job = request.app.state.runtime.job(payload.job_id)
    if job is None:
        raise HTTPException(404, "The sweep job was not found. Its result may have expired.")
    if job.get("kind") not in ("causal", "head_sweep", "induction_probe", "causal_batch"):
        raise HTTPException(400, "Only research sweeps can be saved from a job.")
    result = job.get("result")
    if job["status"] not in ("ready", "cancelled", "failed") or not isinstance(result, dict):
        raise HTTPException(409, "The sweep has not published a result to save yet.")
    if any(
        not isinstance(result.get(key), str) or not result[key].strip()
        for key in ("model_id", "revision")
    ):
        raise HTTPException(409, "The captured result has no checkpoint identity to preserve.")
    return store.create_artifact(
        payload.name, payload.notes, job["kind"], result, source_job_id=payload.job_id
    )


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str, request: Request):
    result = request.app.state.store.get(experiment_id)
    if result is None:
        raise HTTPException(404, "Experiment not found.")
    return result


@app.patch("/api/experiments/{experiment_id}")
def update_experiment(experiment_id: str, payload: ExperimentUpdate, request: Request):
    if not request.app.state.store.update(experiment_id, payload.name, payload.notes):
        raise HTTPException(404, "Experiment not found.")
    return request.app.state.store.get(experiment_id)


@app.delete("/api/experiments/{experiment_id}", status_code=204)
def delete_experiment(experiment_id: str, request: Request):
    if not request.app.state.store.delete(experiment_id):
        raise HTTPException(404, "Experiment not found.")

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub.errors import HfHubHTTPError

from circuit_lantern.config import get_settings
from circuit_lantern.schemas import (
    CausalRequest,
    CompatibilityRequest,
    ExecutionId,
    ExperimentRequest,
    ExperimentUpdate,
    LoadRequest,
    ModelRequest,
    TokenizeRequest,
    TraceRequest,
)
from circuit_lantern.services.hardware import hardware_snapshot
from circuit_lantern.services.hub import model_metadata, search_models
from circuit_lantern.services.runtime import ModelChangedError, ModelRuntime, RuntimeBusyError
from circuit_lantern.services.store import ExperimentStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.runtime = ModelRuntime(settings)
    app.state.store = ExperimentStore(settings.data_dir / "experiments.sqlite3")
    yield
    app.state.runtime.executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="Circuit Lantern", version="0.1.0", lifespan=lifespan)
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


@app.exception_handler(ValueError)
async def value_error(request: Request, error: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.exception_handler(HfHubHTTPError)
async def hub_error(request: Request, error: HfHubHTTPError):
    code = error.response.status_code if error.response else 502
    message = "Hugging Face could not provide this model. Check the repository ID, revision and access permissions. For gated models, set HF_TOKEN on the server."
    return JSONResponse(
        status_code=code if code in (401, 403, 404) else 502, content={"detail": message}
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/hardware")
def hardware():
    return hardware_snapshot(get_settings().data_dir)


@app.get("/api/models")
def models(q: str = Query(default="", max_length=180)):
    return search_models(q)


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
        raise HTTPException(404, "Download job not found.")
    return dict(result)


@app.get("/api/runtime")
def runtime_state(request: Request):
    return request.app.state.runtime.state()


@app.post("/api/causal", status_code=202)
def causal(payload: CausalRequest, request: Request):
    return request.app.state.runtime.causal(payload.model_dump())


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


@app.post("/api/trace")
def trace(payload: TraceRequest, request: Request):
    return request.app.state.runtime.trace(**payload.model_dump())


@app.get("/api/experiments")
def experiments(request: Request):
    return request.app.state.store.list()


@app.post("/api/experiments", status_code=201)
def create_experiment(payload: ExperimentRequest, request: Request):
    return request.app.state.store.create(**payload.model_dump())


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

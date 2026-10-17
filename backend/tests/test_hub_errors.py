from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from circuit_lantern.main import app
from circuit_lantern.services.hub_errors import architecture_diagnostic


def test_architecture_details_preserve_complete_normal_messages_without_server_paths():
    directory = "/private/model-cache/snapshots/abcdef"
    message = (
        "Unable to initialize " + directory + ". " + "Important details. " * 30 + "Final reason."
    )
    detail = architecture_diagnostic(ValueError(message), directory)
    assert directory not in detail
    assert detail.endswith("Final reason.")
    assert "Important details. " * 30 in detail


def test_excessive_architecture_details_have_an_explicit_ending():
    detail = architecture_diagnostic(ValueError("details " * 10000), "/private/model-cache")
    assert len(detail) < 16500
    assert detail.endswith("Further diagnostics were omitted.")


@pytest.mark.parametrize(
    "retry_after", ["-1", "not-a-delay", "9" * 5000], ids=["negative", "non-numeric", "oversized"]
)
def test_incomplete_remote_error_responses_do_not_break_the_handler(retry_after):
    error = RemoteEntryNotFoundError(
        "Missing file",
        response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/test")),
    )
    error.response = httpx.Response(404)
    with patch("circuit_lantern.main.model_metadata", side_effect=error):
        response = TestClient(app).post("/api/models/metadata", json={"model_id": "test/model"})
    assert response.status_code == 422
    assert "required checkpoint file" in response.json()["detail"]
    error = HfHubHTTPError(
        "Rate limit",
        response=httpx.Response(
            429,
            headers={"Retry-After": retry_after},
            request=httpx.Request("GET", "https://huggingface.co/test"),
        ),
    )
    with patch("circuit_lantern.main.model_metadata", side_effect=error):
        response = TestClient(app).post("/api/models/metadata", json={"model_id": "test/model"})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "2"


@pytest.mark.parametrize(
    ("kind", "upstream_status", "expected_status", "message"),
    [
        (GatedRepoError, 403, 403, "requires access approval"),
        (RevisionNotFoundError, 404, 404, "revision does not exist"),
        (RemoteEntryNotFoundError, 404, 422, "does not contain config.json"),
        (RepositoryNotFoundError, 401, 404, "not found or is private"),
        (HfHubHTTPError, 401, 401, "denied access"),
        (HfHubHTTPError, 403, 403, "denied access"),
        (HfHubHTTPError, 404, 404, "could not find"),
        (HfHubHTTPError, 429, 429, "limited requests"),
        (HfHubHTTPError, 503, 503, "temporarily unable"),
    ],
)
def test_hub_failures_report_the_actual_cause_without_remote_response_details(
    kind, upstream_status, expected_status, message
):
    # Even a false-valued response carries a meaningful HTTP status.
    class Response(httpx.Response):
        def __bool__(self):
            return False

    response = Response(
        upstream_status,
        request=httpx.Request("GET", "https://huggingface.co/test/model/resolve/main/config.json"),
        headers={"Retry-After": "37"},
    )
    error = kind("Remote response with hf_fixture_secret", response=response)
    with patch("circuit_lantern.main.model_metadata", side_effect=error):
        result = TestClient(app).post("/api/models/metadata", json={"model_id": "test/model"})
    assert result.status_code == expected_status
    assert message in result.json()["detail"]
    assert "hf_fixture_secret" not in result.text
    if expected_status in (429, 503):
        assert result.headers["Retry-After"] == "37"


@pytest.mark.parametrize(
    ("error", "status", "message"),
    [
        (httpx.ReadTimeout("private upstream details"), 504, "did not respond in time"),
        (httpx.ConnectError("private upstream details"), 503, "could not reach"),
        (LocalEntryNotFoundError("private upstream details"), 503, "not available locally"),
        (OfflineModeIsEnabled("private upstream details"), 503, "offline mode"),
    ],
)
def test_connection_and_cache_failures_are_not_reported_as_permission_errors(
    error, status, message
):
    with patch("circuit_lantern.main.model_metadata", side_effect=error):
        response = TestClient(app).post("/api/models/metadata", json={"model_id": "test/model"})
    assert response.status_code == status
    assert message in response.json()["detail"]
    assert "private upstream details" not in response.text
    assert response.headers["Retry-After"] == "2"


def test_download_jobs_use_the_same_specific_hub_diagnostics_and_preserve_the_model(runtime):
    active = runtime.model
    response = httpx.Response(429, request=httpx.Request("GET", "https://huggingface.co/test"))
    error = HfHubHTTPError("private upstream details", response=response)
    job = {"status": "queued"}
    with patch("circuit_lantern.services.runtime.snapshot_download", side_effect=error):
        runtime._load(job, {"id": "test/model", "revision": "main"}, "cpu")
    assert job["status"] == "failed"
    assert "limited requests" in job["message"]
    assert runtime.model is active

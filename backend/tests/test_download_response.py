"""Download HTTP responses close pinned streams even when a client disconnects."""

from io import BytesIO

import anyio
import pytest
from starlette.requests import ClientDisconnect

from circuit_lantern.api.crosscoders import stream_download


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("spec", ["2.3", "2.4"])
@pytest.mark.parametrize("disconnect", [False, True])
async def test_pinned_stream_closes_after_normal_or_interrupted_transfer(spec, disconnect):
    content = b"x" * (2 * 1024**2 + 1)
    stream = BytesIO(content)
    response = stream_download(stream, len(content), "capture.safetensors")
    messages = []

    async def receive():
        await anyio.Event().wait()

    async def send(message):
        if disconnect and message["type"] == "http.response.body":
            raise OSError("Client disconnected")
        messages.append(message)

    scope = {"type": "http", "method": "GET", "asgi": {"spec_version": spec}}
    if disconnect:
        with pytest.raises((OSError, ClientDisconnect)):
            await response(scope, receive, send)
    else:
        await response(scope, receive, send)
        assert b"".join(message.get("body", b"") for message in messages) == content
    assert stream.closed
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.anyio
async def test_asgi_disconnect_interrupts_an_incomplete_transfer_and_closes_the_stream():
    sending = anyio.Event()

    class TrackedStream(BytesIO):
        consumed = 0

        def read(self, count=-1):
            chunk = super().read(count)
            self.consumed += len(chunk)
            return chunk

    content = b"x" * (3 * 1024**2)
    stream = TrackedStream(content)
    response = stream_download(stream, len(content), "capture.safetensors")

    async def receive():
        await sending.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            sending.set()
            await anyio.Event().wait()

    with anyio.fail_after(5):
        await response(
            {"type": "http", "method": "GET", "asgi": {"spec_version": "2.3"}}, receive, send
        )
    assert stream.closed
    assert 0 < stream.consumed < len(content)


def test_prepared_get_keeps_the_stream_owned_until_response_execution():
    from types import SimpleNamespace

    from circuit_lantern.api.crosscoders import prepared_download
    from circuit_lantern.services.prepared_downloads import PreparedDownloads

    downloads = PreparedDownloads()
    stream = BytesIO(b"pending")
    ticket, _ = downloads.prepare(lambda: (stream, 7), "capture.safetensors")
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(crosscoders=SimpleNamespace(downloads=downloads)))
    )
    response = prepared_download(ticket, request)
    downloads.close()
    assert response is not None and stream.closed


@pytest.mark.anyio
@pytest.mark.parametrize("expired", [False, True])
async def test_prepared_response_consumes_or_expires_a_ticket_and_releases_capacity(expired):
    import json

    from circuit_lantern.api.crosscoders import PreparedDownloadResponse
    from circuit_lantern.services.prepared_downloads import PreparedDownloads

    downloads = PreparedDownloads(capacity=1)
    stream = BytesIO(b"verified")
    ticket, _ = downloads.prepare(lambda: (stream, 8), "capture.safetensors")
    if expired:
        downloads.discard(ticket)
    messages = []

    async def receive():
        await anyio.Event().wait()

    async def send(message):
        messages.append(message)

    try:
        response = PreparedDownloadResponse(downloads, ticket)
        await response(
            {"type": "http", "method": "GET", "asgi": {"spec_version": "2.4"}}, receive, send
        )
        assert messages[0]["status"] == (410 if expired else 200)
        content = b"".join(message.get("body", b"") for message in messages)
        if expired:
            assert "expired" in json.loads(content)["detail"]
        else:
            assert content == b"verified"
        assert stream.closed and downloads.take(ticket) is None
        downloads.prepare(lambda: (BytesIO(b"next"), 4), "next.safetensors")
    finally:
        downloads.close()


@pytest.mark.anyio
async def test_response_construction_failure_releases_the_taken_stream(monkeypatch):
    from circuit_lantern.api.crosscoders import PreparedDownloadResponse
    from circuit_lantern.services.prepared_downloads import PreparedDownloads

    downloads = PreparedDownloads(capacity=1)
    stream = BytesIO(b"verified")
    ticket, _ = downloads.prepare(lambda: (stream, 8), "capture.safetensors")

    def unavailable(*args, **kwargs):
        raise RuntimeError("Response could not be constructed")

    monkeypatch.setattr("circuit_lantern.api.crosscoders.stream_download", unavailable)
    try:
        with pytest.raises(RuntimeError, match="Response"):
            await PreparedDownloadResponse(downloads, ticket)({}, None, None)
        assert stream.closed
        downloads.prepare(lambda: (BytesIO(b"next"), 4), "next.safetensors")
    finally:
        downloads.close()


@pytest.mark.anyio
@pytest.mark.parametrize("spec,error", [("2.3", TimeoutError), ("2.4", ClientDisconnect)])
async def test_stalled_send_times_out_and_releases_active_capacity(monkeypatch, spec, error):
    from circuit_lantern.api.crosscoders import PreparedDownloadResponse
    from circuit_lantern.services.prepared_downloads import PreparedDownloads

    monkeypatch.setattr("circuit_lantern.api.crosscoders._DOWNLOAD_SEND_TIMEOUT", 0.01)
    downloads = PreparedDownloads(capacity=1)
    stream = BytesIO(b"verified")
    ticket, _ = downloads.prepare(lambda: (stream, 8), "capture.safetensors")

    async def receive():
        await anyio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.body":
            await anyio.Event().wait()

    try:
        with anyio.fail_after(2), pytest.raises(error):
            await PreparedDownloadResponse(downloads, ticket)(
                {"type": "http", "method": "GET", "asgi": {"spec_version": spec}},
                receive,
                send,
            )
        assert stream.closed
        downloads.prepare(lambda: (BytesIO(b"next"), 4), "next.safetensors")
    finally:
        downloads.close()


@pytest.mark.anyio
@pytest.mark.parametrize("spec", ["2.3", "2.4"])
async def test_disk_read_error_is_distinct_from_a_client_disconnect(spec):
    class DamagedStream(BytesIO):
        def read(self, count=-1):
            raise OSError("Simulated disk failure")

    stream = DamagedStream(b"verified")
    response = stream_download(stream, 8, "capture.safetensors")

    async def receive():
        await anyio.Event().wait()

    async def send(message):
        pass

    with pytest.raises(RuntimeError, match="could not be streamed") as failure:
        await response(
            {"type": "http", "method": "GET", "asgi": {"spec_version": spec}},
            receive,
            send,
        )
    assert isinstance(failure.value.__cause__, OSError)
    assert stream.closed


@pytest.mark.anyio
@pytest.mark.parametrize("spec", ["2.3", "2.4"])
async def test_interrupted_prepared_transfer_releases_capacity(spec):
    from circuit_lantern.api.crosscoders import PreparedDownloadResponse
    from circuit_lantern.services.prepared_downloads import PreparedDownloads

    downloads = PreparedDownloads(capacity=1)
    stream = BytesIO(b"x" * (3 * 1024**2))
    ticket, _ = downloads.prepare(lambda: (stream, 3 * 1024**2), "capture.safetensors")

    async def receive():
        await anyio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("Client disconnected")

    try:
        with pytest.raises((OSError, ClientDisconnect)):
            await PreparedDownloadResponse(downloads, ticket)(
                {"type": "http", "method": "GET", "asgi": {"spec_version": spec}}, receive, send
            )
        assert stream.closed
        downloads.prepare(lambda: (BytesIO(b"next"), 4), "next.safetensors")
    finally:
        downloads.close()

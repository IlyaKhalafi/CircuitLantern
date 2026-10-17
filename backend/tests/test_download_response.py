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
async def test_asgi_disconnect_message_closes_the_pinned_stream():
    stream = BytesIO(b"verified stream")
    response = stream_download(stream, 15, "capture.safetensors")

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    await response(
        {"type": "http", "method": "GET", "asgi": {"spec_version": "2.3"}}, receive, send
    )
    assert stream.closed


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

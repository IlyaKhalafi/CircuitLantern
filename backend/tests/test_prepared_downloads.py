"""Verified downloads expire and release bounded server resources."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event

import pytest

from circuit_lantern.services.prepared_downloads import DownloadBusyError, PreparedDownloads


def test_prepared_download_is_consumed_once_and_owns_the_pinned_stream():
    downloads = PreparedDownloads(capacity=1)
    stream = BytesIO(b"verified bytes")
    identifier = None
    try:
        identifier, length = downloads.prepare(lambda: (stream, 14), "weights.safetensors")
        assert len(identifier) == 64 and length == 14
        lease = downloads.take(identifier)
        assert lease.filename == "weights.safetensors"
        assert lease.stream.read() == b"verified bytes"
        assert downloads.take(identifier) is None
        downloads.close()
        assert not stream.closed
    finally:
        if identifier is not None:
            downloads.release(identifier)
        stream.close()
        downloads.close()


def test_expired_unused_download_closes_its_stream_and_frees_capacity():
    closed = Event()

    class TrackedStream(BytesIO):
        def close(self):
            super().close()
            closed.set()

    downloads = PreparedDownloads(capacity=1, lifetime=0.05)
    stream = TrackedStream(b"unused")
    try:
        identifier, _ = downloads.prepare(lambda: (stream, 6), "capture.safetensors")
        assert closed.wait(2), "Expired stream was not closed"
        assert stream.closed and downloads.take(identifier) is None
        replacement = BytesIO(b"next")
        token, _ = downloads.prepare(lambda: (replacement, 4), "next.safetensors")
        downloads.close()
        assert replacement.closed and downloads.take(token) is None
    finally:
        downloads.close()


def test_inflight_verification_reserves_capacity_and_failure_releases_it():
    downloads = PreparedDownloads(capacity=1)
    entered, resume = Event(), Event()

    def verifying():
        entered.set()
        assert resume.wait(5)
        raise ValueError("Checksum mismatch")

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            verification = executor.submit(downloads.prepare, verifying, "weights.safetensors")
            try:
                assert entered.wait(5)
                with pytest.raises(DownloadBusyError):
                    downloads.prepare(lambda: pytest.fail("Capacity check ran the opener"), "other")
            finally:
                resume.set()
            with pytest.raises(ValueError, match="Checksum"):
                verification.result(timeout=5)
        stream = BytesIO(b"valid")
        downloads.prepare(lambda: (stream, 5), "weights.safetensors")
        downloads.close()
        assert stream.closed
    finally:
        resume.set()
        downloads.close()


def test_shutdown_during_verification_releases_the_newly_opened_stream():
    downloads = PreparedDownloads()
    entered, resume = Event(), Event()
    stream = BytesIO(b"valid")

    def verifying():
        entered.set()
        assert resume.wait(5)
        return stream, 5

    with ThreadPoolExecutor(max_workers=1) as executor:
        verification = executor.submit(downloads.prepare, verifying, "weights.safetensors")
        try:
            assert entered.wait(5)
            downloads.close()
        finally:
            resume.set()
        with pytest.raises(DownloadBusyError):
            verification.result(timeout=5)
    assert stream.closed


def test_active_transfer_keeps_its_capacity_until_release():
    downloads = PreparedDownloads(capacity=1)
    stream = BytesIO(b"active")
    identifier, _ = downloads.prepare(lambda: (stream, 6), "capture.safetensors")
    try:
        assert downloads.take(identifier).stream is stream
        downloads._expire(identifier)
        assert not stream.closed
        with pytest.raises(DownloadBusyError):
            downloads.prepare(lambda: pytest.fail("Opened beyond active capacity"), "other")
        downloads.release(identifier)
        downloads.release(identifier)
        assert stream.closed
        replacement = BytesIO(b"next")
        downloads.prepare(lambda: (replacement, 4), "next.safetensors")
    finally:
        downloads.release(identifier)
        downloads.close()
    assert replacement.closed


def test_take_checks_deadline_even_if_the_timer_has_not_run(monkeypatch):
    downloads = PreparedDownloads(capacity=1)
    stream = BytesIO(b"expired")
    try:
        identifier, _ = downloads.prepare(lambda: (stream, 7), "capture.safetensors")
        monkeypatch.setattr(
            "circuit_lantern.services.prepared_downloads.time.monotonic",
            lambda: float("inf"),
        )
        assert downloads.take(identifier) is None
        assert stream.closed
        downloads.prepare(lambda: (BytesIO(b"next"), 4), "next.safetensors")
    finally:
        downloads.close()


def test_timer_failure_closes_the_stream_and_restores_capacity(monkeypatch):
    downloads = PreparedDownloads(capacity=1)
    stream = BytesIO(b"unpublished")

    def unavailable(self):
        raise RuntimeError("Timer could not start")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            "circuit_lantern.services.prepared_downloads.threading.Timer.start", unavailable
        )
        with pytest.raises(RuntimeError, match="Timer"):
            downloads.prepare(lambda: (stream, 11), "capture.safetensors")
    assert stream.closed
    try:
        downloads.prepare(lambda: (BytesIO(b"next"), 4), "next.safetensors")
    finally:
        downloads.close()


@pytest.mark.parametrize("at_entry", [True, False])
def test_cancelled_preparation_closes_any_new_stream_and_restores_capacity(at_entry):
    from circuit_lantern.services.prepared_downloads import DownloadCancelledError

    downloads = PreparedDownloads(capacity=1)
    cancelled = Event()
    stream = BytesIO(b"pending")
    if at_entry:
        cancelled.set()

    def opener():
        assert not at_entry
        cancelled.set()
        return stream, 7

    try:
        with pytest.raises(DownloadCancelledError):
            downloads.prepare(opener, "capture.safetensors", cancelled.is_set)
        assert stream.closed == (not at_entry)
        downloads.prepare(lambda: (BytesIO(b"next"), 4), "next.safetensors")
    finally:
        stream.close()
        downloads.close()


def test_concurrent_consumers_cannot_take_the_same_stream_twice():
    downloads = PreparedDownloads(capacity=1)
    stream = BytesIO(b"unique")
    ticket, _ = downloads.prepare(lambda: (stream, 6), "capture.safetensors")
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            leases = list(executor.map(downloads.take, [ticket, ticket]))
        assert sum(lease is not None for lease in leases) == 1
    finally:
        downloads.release(ticket)
        downloads.close()
    assert stream.closed


def test_publication_releases_the_pending_reservation_atomically():
    from threading import Lock, current_thread

    downloads = PreparedDownloads(capacity=2)
    published, resume = Event(), Event()

    class PauseAfterPublication:
        def __init__(self):
            self.lock = Lock()
            self.producer_releases = 0

        def __enter__(self):
            self.lock.acquire()

        def __exit__(self, *args):
            producer = current_thread().name.startswith("publishing-download")
            if producer:
                self.producer_releases += 1
            self.lock.release()
            # Stop the producer immediately after its commit critical section.
            if producer and self.producer_releases == 2:
                published.set()
                assert resume.wait(5)

    downloads._lock = PauseAfterPublication()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="publishing-download") as executor:
        first = executor.submit(
            downloads.prepare, lambda: (BytesIO(b"first"), 5), "first.safetensors"
        )
        try:
            assert published.wait(5)
            second, _ = downloads.prepare(lambda: (BytesIO(b"second"), 6), "second.safetensors")
            assert second
        finally:
            resume.set()
            first.result(timeout=5)
            downloads.close()

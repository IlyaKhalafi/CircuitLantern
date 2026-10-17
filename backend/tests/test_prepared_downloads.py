"""Verified downloads expire and release bounded server resources."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event

import pytest

from circuit_lantern.services.prepared_downloads import DownloadBusyError, PreparedDownloads


def test_prepared_download_is_consumed_once_and_owns_the_pinned_stream():
    downloads = PreparedDownloads(capacity=1)
    stream = BytesIO(b"verified bytes")
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

"""Bounded, expiring leases for verified file streams."""

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO


class DownloadBusyError(Exception):
    pass


class DownloadCancelledError(Exception):
    pass


@dataclass(frozen=True)
class PreparedDownload:
    stream: BinaryIO
    length: int
    filename: str
    expires_at: float


class PreparedDownloads:
    def __init__(self, *, capacity: int = 4, lifetime: float = 60):
        if capacity < 1 or lifetime <= 0:
            raise ValueError("Download capacity and lifetime must be positive.")
        self.capacity = capacity
        self.lifetime = lifetime
        self._lock = threading.Lock()
        self._pending = 0
        self._closed = False
        self._leases: dict[str, tuple[PreparedDownload, threading.Timer]] = {}

    def prepare(
        self,
        opener: Callable[[], tuple[BinaryIO, int]],
        filename: str,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[str, int]:
        if cancelled and cancelled():
            raise DownloadCancelledError()
        with self._lock:
            if self._closed or len(self._leases) + self._pending >= self.capacity:
                raise DownloadBusyError("Downloads are busy. Try again shortly.")
            self._pending += 1
        stream = None
        try:
            stream, length = opener()
            with self._lock:
                if cancelled and cancelled():
                    raise DownloadCancelledError()
                if self._closed:
                    raise DownloadBusyError("Downloads are busy. Try again shortly.")
                identifier = secrets.token_hex(32)
                lease = PreparedDownload(stream, length, filename, time.monotonic() + self.lifetime)
                timer = threading.Timer(self.lifetime, self._expire, args=(identifier,))
                timer.daemon = True
                self._leases[identifier] = lease, timer
                try:
                    timer.start()
                except BaseException:
                    self._leases.pop(identifier)
                    raise
                stream = None
            return identifier, length
        finally:
            if stream is not None:
                stream.close()
            with self._lock:
                self._pending -= 1

    def discard(self, identifier: str):
        self._expire(identifier)

    def _expire(self, identifier: str):
        with self._lock:
            retained = self._leases.pop(identifier, None)
        if retained:
            retained[1].cancel()
            retained[0].stream.close()

    def take(self, identifier: str) -> PreparedDownload | None:
        with self._lock:
            retained = self._leases.pop(identifier, None)
        if retained is None:
            return None
        lease, timer = retained
        timer.cancel()
        if lease.expires_at <= time.monotonic():
            lease.stream.close()
            return None
        return lease

    def close(self):
        with self._lock:
            self._closed = True
            retained = list(self._leases.values())
            self._leases.clear()
        for lease, timer in retained:
            timer.cancel()
            lease.stream.close()

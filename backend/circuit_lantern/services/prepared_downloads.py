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
        self._closing = threading.Event()
        self._leases: dict[str, tuple[PreparedDownload, threading.Timer]] = {}
        self._active: dict[str, PreparedDownload] = {}

    @property
    def closed(self) -> bool:
        return self._closing.is_set()

    def prepare(
        self,
        opener: Callable[[], tuple[BinaryIO, int]],
        filename: str,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[str, int]:
        if cancelled and cancelled():
            raise DownloadCancelledError()
        with self._lock:
            if (
                self.closed
                or len(self._leases) + len(self._active) + self._pending >= self.capacity
            ):
                raise DownloadBusyError("Downloads are busy. Try again shortly.")
            self._pending += 1
        stream = None
        reserved = True
        try:
            stream, length = opener()
            with self._lock:
                if cancelled and cancelled():
                    raise DownloadCancelledError()
                if self.closed:
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
                self._pending -= 1
                reserved = False
                stream = None
            return identifier, length
        finally:
            if stream is not None:
                stream.close()
            if reserved:
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
            expired = lease.expires_at <= time.monotonic()
            if not expired:
                self._active[identifier] = lease
        timer.cancel()
        if expired:
            lease.stream.close()
            return None
        return lease

    def release(self, identifier: str):
        with self._lock:
            lease = self._active.pop(identifier, None)
        if lease is not None:
            lease.stream.close()

    def close(self):
        with self._lock:
            self._closing.set()
            retained = list(self._leases.values())
            self._leases.clear()
        for lease, timer in retained:
            timer.cancel()
            lease.stream.close()

import threading
from collections.abc import Iterator, Mapping
from typing import Any


class JobRecord(Mapping[str, Any]):
    """Publish each job transition as one synchronized snapshot."""

    def __init__(self, data: dict):
        self._data = dict(data)
        self._lock = threading.Lock()

    def __getitem__(self, key: str):
        with self._lock:
            return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.snapshot())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)

    def update(self, **changes):
        with self._lock:
            self._data.update(changes)

    def request_cancel(self) -> dict:
        with self._lock:
            if self._data["status"] in ("queued", "running"):
                self._data["cancel_requested"] = True
            return {
                "status": self._data["status"],
                "cancel_requested": self._data.get("cancel_requested", False),
            }

    def finish(self, result: dict):
        with self._lock:
            cancelled = self._data.get("cancel_requested", False)
            result["cancelled"] = cancelled
            self._data.update(
                result=result,
                status="cancelled" if cancelled else "ready",
                progress=self._data["progress"] if cancelled else 100,
                message="Causal sweep cancelled. Completed layers are retained."
                if cancelled
                else "Causal sweep complete",
            )

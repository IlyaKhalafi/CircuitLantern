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

    def finish(self, result: dict, *, label: str = "Causal sweep", unit: str = "layers"):
        with self._lock:
            failure = result.get("error")
            cancelled = bool(
                self._data.get("cancel_requested", False)
                and not result.get("complete", False)
                and not failure
            )
            result["cancelled"] = cancelled
            self._data.update(
                result=result,
                status="failed" if failure else "cancelled" if cancelled else "ready",
                progress=self._data["progress"] if cancelled or failure else 100,
                message=failure
                or (
                    f"{label} cancelled. Completed {unit} are retained."
                    if cancelled
                    else f"{label} complete"
                ),
            )

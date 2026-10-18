"""Bounded immutable views over independently checked artifact metadata."""

import hashlib
import sys
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from threading import Lock, RLock
from types import MappingProxyType
from typing import Any

from circuit_lantern.services.resources import available_memory_bytes

MAX_ARTIFACT_METADATA_BYTES = 128 * 1024**2
METADATA_TEXT_RAM_MULTIPLIER = 5
METADATA_DECODE_RAM_MULTIPLIER = 64
METADATA_JSON_RAM_MULTIPLIER = 96
_METADATA_RAM_ERROR = "Insufficient available RAM to read saved artifact metadata."


class MetadataMemoryError(ValueError):
    """Transient memory pressure while reading retained evidence."""


def checked_metadata_size(encoded: str, expected: str) -> int:
    if not isinstance(encoded, str) or not isinstance(expected, str):
        raise ValueError("The saved artifact metadata is invalid.")  # noqa: TRY004 - stored data errors use API validation
    raw = encoded.encode()
    if len(raw) > MAX_ARTIFACT_METADATA_BYTES or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("The saved artifact metadata checksum does not match.")
    return len(raw)


class MetadataMemoryBudget:
    """Reserve transient allocations across readers before copying or decoding."""

    def __init__(self):
        self._lock = Lock()
        self._reserved = 0
        self.decode_lock = RLock()

    @contextmanager
    def reserve(self, size: int):
        with self._lock:
            if self._reserved + size > available_memory_bytes() * 0.75:
                raise MetadataMemoryError(_METADATA_RAM_ERROR)
            self._reserved += size
        try:
            yield
        finally:
            with self._lock:
                self._reserved -= size


@dataclass(frozen=True)
class ManifestSnapshot:
    kind: str
    encoded: str | None = None
    digest: str | None = None
    size: int = 0


@dataclass
class _Flight:
    lock: Lock = field(default_factory=Lock)
    users: int = 0
    invalidated: bool = False


def _freeze(value: dict) -> tuple[Mapping, int]:
    total = 0
    leaves = set()
    result = [None]
    pending = [("visit", value, result, 0)]

    def leaf(item):
        nonlocal total
        identity = id(item)
        if identity not in leaves:
            leaves.add(identity)
            total += sys.getsizeof(item)

    while pending:
        action, item, parent, position = pending.pop()
        if action == "dict":
            total += sys.getsizeof(item) + sys.getsizeof(parent[position])
        elif action == "list":
            frozen = tuple(item)
            parent[position] = frozen
            total += sys.getsizeof(frozen)
        elif isinstance(item, dict):
            copied = {}
            parent[position] = MappingProxyType(copied)
            pending.append(("dict", copied, parent, position))
            for key, child in reversed(item.items()):
                leaf(key)
                pending.append(("visit", child, copied, key))
        elif isinstance(item, list):
            copied = [None] * len(item)
            pending.append(("list", copied, parent, position))
            for index in range(len(item) - 1, -1, -1):
                pending.append(("visit", item[index], copied, index))
        else:
            leaf(item)
            parent[position] = item

    # Leave room for allocator rounding, cache keys and entry bookkeeping.
    return result[0], int(total * 1.25) + 512


class ArtifactManifestViews:
    """Serialize cold parsing and bound retained views by size and available RAM."""

    def __init__(self, *, max_bytes: int = 512 * 1024**2, max_entries: int = 8):
        if max_bytes < 0 or max_entries < 1:
            raise ValueError("Metadata cache limits must be nonnegative with at least one entry.")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._retained_bytes = 0
        self._lock = Lock()
        self._flights = {}

    def _remove(self, key):
        _, size = self._entries.pop(key)
        self._retained_bytes -= size

    def _discard(self, identifier: str):
        for key in list(self._entries):
            if key[0] == identifier:
                self._remove(key)

    def discard(self, identifier: str):
        with self._lock:
            self._discard(identifier)
            if identifier in self._flights:
                self._flights[identifier].invalidated = True

    @contextmanager
    def _flight(self, identifier: str):
        with self._lock:
            flight = self._flights.setdefault(identifier, _Flight())
            flight.users += 1
        try:
            with flight.lock:
                yield flight
        finally:
            with self._lock:
                flight.users -= 1
                if not flight.users:
                    del self._flights[identifier]

    def _budget(self):
        return min(self.max_bytes, available_memory_bytes() // 10)

    def _trim(self, budget: int):
        while self._entries and (
            self._retained_bytes > budget or len(self._entries) > self.max_entries
        ):
            self._remove(next(iter(self._entries)))

    def read(
        self,
        identifier: str,
        loader: Callable[[], AbstractContextManager[ManifestSnapshot | None]],
        decoder: Callable[[ManifestSnapshot], AbstractContextManager[dict[str, Any]]],
        *,
        kind: str | None = None,
    ) -> Mapping[str, Any] | None:
        with self._flight(identifier) as flight:
            budget = self._budget()
            with self._lock:
                self._trim(budget)
            with loader() as loaded:
                if loaded is None:
                    self.discard(identifier)
                    return None
                if kind is not None and loaded.kind != kind:
                    return None
                try:
                    checked_metadata_size(loaded.encoded, loaded.digest)
                except ValueError:
                    self.discard(identifier)
                    raise
                key = identifier, loaded.digest, loaded.kind
                with self._lock:
                    if key in self._entries:
                        self._entries.move_to_end(key)
                        return self._entries[key][0]
                    self._discard(identifier)
                with decoder(loaded) as decoded:
                    view, size = _freeze(decoded)
                    budget = self._budget()
                    with self._lock:
                        if size <= budget and not flight.invalidated:
                            self._entries[key] = view, size
                            self._retained_bytes += size
                        self._trim(budget)
                    return view

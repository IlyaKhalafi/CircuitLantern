import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from circuit_lantern.services.store import ExperimentStore


def legacy_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE experiments (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, notes TEXT NOT NULL,
            created_at TEXT NOT NULL, model_id TEXT NOT NULL, payload TEXT NOT NULL
        )""")
        connection.execute(
            "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?)",
            (
                "legacy",
                "Legacy trace",
                "Original notes",
                "2026-01-01",
                "gpt2",
                json.dumps({"model_id": "gpt2", "revision": "pinned", "prompt": "Paris"}),
            ),
        )


def test_migration_retains_existing_traces_and_accepts_new_artifacts(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_database(path)
    store = ExperimentStore(path)
    legacy = store.get("legacy")
    assert legacy["kind"] == "trace"
    assert legacy["trace"]["prompt"] == "Paris"
    assert legacy["notes"] == "Original notes"
    created = store.create_artifact(
        "Head result", "Observations", "head_sweep", {"model_id": "gpt2", "heads": []}
    )
    assert store.get(created["id"])["result"]["heads"] == []
    reopened = ExperimentStore(path)
    assert reopened.get("legacy") == legacy
    with sqlite3.connect(path) as connection:
        assert [row[1] for row in connection.execute("PRAGMA table_info(experiments)")].count(
            "kind"
        ) == 1


def test_simultaneous_initialization_cannot_race_the_legacy_migration(tmp_path):
    path = tmp_path / "shared.db"
    legacy_database(path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        stores = list(executor.map(ExperimentStore, [path, path]))
    assert all(store.get("legacy")["trace"]["revision"] == "pinned" for store in stores)


@pytest.mark.parametrize("kind", ["trace", "causal", "head_sweep"])
def test_artifact_kind_and_payload_are_durable(tmp_path, kind):
    path = tmp_path / "artifacts.db"
    store = ExperimentStore(path)
    payload = {"model_id": "checkpoint", "revision": "source", "values": [1, 2, 3]}
    saved = store.create_artifact("Capture", "Notes", kind, payload)
    loaded = ExperimentStore(path).get(saved["id"])
    assert loaded["kind"] == kind
    assert loaded["trace" if kind == "trace" else "result"] == payload


def test_invalid_artifact_cannot_leave_a_partial_record(tmp_path):
    store = ExperimentStore(tmp_path / "records.db")
    with pytest.raises(ValueError, match="not supported"):
        store.create_artifact("Invalid", "", "unknown", {})
    with pytest.raises(ValueError):
        store.create("Non finite", "", {"model_id": "model", "value": float("nan")})
    assert store.list() == []


def test_concurrent_saves_of_one_job_create_one_durable_record(tmp_path):
    store = ExperimentStore(tmp_path / "shared-records.db")
    payload = {"model_id": "model", "revision": "source", "heads": []}

    def save(_):
        return store.create_artifact(
            "First name", "Original notes", "head_sweep", payload, source_job_id="a" * 32
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(save, range(2)))
    assert records[0]["id"] == records[1]["id"]
    assert len(store.list()) == 1
    store.update(records[0]["id"], "Renamed", "Edited notes")
    repeated = save(None)
    assert repeated["name"] == "Renamed"
    assert repeated["notes"] == "Edited notes"
    assert store.get(repeated["id"])["result"] == payload


def test_current_schema_reopens_without_taking_a_write_lock(tmp_path):
    path = tmp_path / "current.db"
    ExperimentStore(path)
    with sqlite3.connect(path) as writer:
        writer.execute("BEGIN IMMEDIATE")
        reopened = ExperimentStore(path)
        assert reopened.list() == []
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1

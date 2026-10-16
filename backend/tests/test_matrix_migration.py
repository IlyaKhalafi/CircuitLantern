import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from circuit_lantern.services import store as store_module
from circuit_lantern.services.store import ExperimentStore
from circuit_lantern.services.store_schema import SCHEMA_VERSION


def version_one(path):
    payload = (
        json.dumps({"model_id": "checkpoint", "revision": "pinned", "values": [0.25, 2]}, indent=3)
        + " \n"
    )
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE experiments (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, notes TEXT NOT NULL,
            created_at TEXT NOT NULL, model_id TEXT NOT NULL, payload TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'trace' CHECK (kind IN ('trace', 'causal', 'head_sweep')),
            source_job_id TEXT
        )""")
        connection.execute(
            "CREATE UNIQUE INDEX experiments_source_job ON experiments(source_job_id)"
        )
        connection.execute("PRAGMA user_version = 1")
        for index, kind in enumerate(["trace", "causal", "head_sweep"]):
            connection.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(index),
                    "Existing " + kind,
                    "Keep original notes",
                    "2026-01-01",
                    "checkpoint",
                    payload,
                    kind,
                    None if kind == "trace" else str(index) * 32,
                ),
            )
    return original_rows(path)


def original_rows(path):
    with sqlite3.connect(path) as connection:
        return connection.execute(
            "SELECT id, name, notes, created_at, model_id, payload, kind, source_job_id "
            "FROM experiments ORDER BY id"
        ).fetchall()


def test_version_one_upgrade_preserves_original_payload_text_and_metadata(tmp_path):
    path = tmp_path / "version-one.db"
    before = version_one(path)
    store = ExperimentStore(path)
    assert original_rows(path) == before
    for row in before:
        record = store.get(row[0])
        assert record["notes"] == row[2]
        assert "source_analysis_id" not in record
        assert record["trace" if row[6] == "trace" else "result"] == json.loads(row[5])
    saved = store.create_artifact(
        "Matrix",
        "New observations",
        "matrix",
        {"model_id": "checkpoint"},
        source_analysis_id="a" * 32,
    )
    assert store.get(saved["id"])["kind"] == "matrix"
    assert store.find_analysis("a" * 32)["id"] == saved["id"]
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_concurrent_version_one_upgrades_preserve_every_record(tmp_path):
    path = tmp_path / "shared.db"
    before = version_one(path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        stores = list(executor.map(ExperimentStore, [path] * 4))
    assert original_rows(path) == before
    assert all(len(store.list()) == 3 for store in stores)


def test_failed_upgrade_after_copy_and_drop_rolls_back_all_original_records(tmp_path, monkeypatch):
    path = tmp_path / "rollback.db"
    before = version_one(path)
    original_connect = sqlite3.connect

    class FailedRename(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith("ALTER TABLE experiments_upgrade RENAME"):
                raise sqlite3.OperationalError("Interrupted upgrade")
            return super().execute(sql, parameters)

    def failing_connect(*args, **kwargs):
        return original_connect(*args, factory=FailedRename, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(store_module.sqlite3, "connect", failing_connect)
        with pytest.raises(sqlite3.OperationalError, match="Interrupted upgrade"):
            ExperimentStore(path)
    assert original_rows(path) == before
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'experiments_upgrade'"
            ).fetchone()
            is None
        )
    assert len(ExperimentStore(path).list()) == 3


def test_a_wrong_source_index_is_repaired_without_rewriting_capture_json(tmp_path):
    path = tmp_path / "index.db"
    store = ExperimentStore(path)
    saved = store.create("Existing trace", "Keep notes", {"model_id": "model"})
    before = store.get(saved["id"])
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX experiments_source_analysis")
        connection.execute("CREATE INDEX experiments_source_analysis ON experiments(model_id)")
    reopened = ExperimentStore(path)
    assert reopened.get(saved["id"]) == before
    record = reopened.create_artifact("Matrix", "", "matrix", {}, source_analysis_id="a" * 32)
    repeated = reopened.create_artifact("Retry", "", "matrix", {}, source_analysis_id="a" * 32)
    assert repeated["id"] == record["id"]


def test_concurrent_matrix_sources_are_idempotent_and_preserve_edited_notes(tmp_path):
    store = ExperimentStore(tmp_path / "sources.db")

    def save(_):
        return store.create_artifact(
            "Original",
            "Original notes",
            "matrix",
            {"model_id": "model"},
            source_analysis_id="a" * 32,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(save, range(2)))
    assert records[0]["id"] == records[1]["id"]
    store.update(records[0]["id"], "Edited", "Keep this observation")
    repeated = save(None)
    assert repeated["name"] == "Edited"
    assert repeated["notes"] == "Keep this observation"
    assert len(store.list()) == 1

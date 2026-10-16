import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
EXPERIMENT_KINDS = frozenset(("trace", "causal", "head_sweep"))
SUMMARY_COLUMNS = "id, name, notes, created_at, model_id, kind, source_job_id"
SOURCE_JOB_INDEX = "experiments_source_job"


class ExperimentStore:
    def __init__(self, path: Path):
        self.path = path
        with self.connect() as connection:
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(experiments)")}
            indexes = connection.execute("PRAGMA index_list(experiments)").fetchall()
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if (
                {"kind", "source_job_id"} <= columns
                and any(row["name"] == SOURCE_JOB_INDEX and row["unique"] for row in indexes)
                and version >= SCHEMA_VERSION
            ):
                return
            # Recheck under the write lock so concurrent starts cannot race a migration.
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, notes TEXT NOT NULL,
                created_at TEXT NOT NULL, model_id TEXT NOT NULL, payload TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'trace'
                    CHECK (kind IN ('trace', 'causal', 'head_sweep')),
                source_job_id TEXT
            )""")
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(experiments)")}
            if "kind" not in columns:
                connection.execute(
                    "ALTER TABLE experiments ADD COLUMN kind TEXT NOT NULL DEFAULT 'trace' "
                    "CHECK (kind IN ('trace', 'causal', 'head_sweep'))"
                )
            if "source_job_id" not in columns:
                connection.execute("ALTER TABLE experiments ADD COLUMN source_job_id TEXT")
            connection.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {SOURCE_JOB_INDEX} "
                "ON experiments(source_job_id)"
            )
            if connection.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def list(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT {SUMMARY_COLUMNS} FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def find_job(self, job_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT {SUMMARY_COLUMNS} FROM experiments WHERE source_job_id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def create(self, name: str, notes: str, trace: dict) -> dict:
        return self.create_artifact(name, notes, "trace", trace)

    def create_artifact(
        self, name: str, notes: str, kind: str, payload: dict, *, source_job_id: str | None = None
    ) -> dict:
        if kind not in EXPERIMENT_KINDS:
            raise ValueError("This experiment type is not supported.")
        encoded = json.dumps(payload, allow_nan=False)
        record = {
            "id": uuid.uuid4().hex,
            "name": name,
            "notes": notes,
            "created_at": datetime.now(UTC).isoformat(),
            "model_id": payload.get("model_id", "unknown"),
            "kind": kind,
            "source_job_id": source_job_id,
        }
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO experiments "
                "(id, name, notes, created_at, model_id, kind, source_job_id, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(source_job_id) DO NOTHING",
                (
                    record["id"],
                    name,
                    notes,
                    record["created_at"],
                    record["model_id"],
                    kind,
                    source_job_id,
                    encoded,
                ),
            )
            if cursor.rowcount == 0:
                return dict(
                    connection.execute(
                        f"SELECT {SUMMARY_COLUMNS} FROM experiments WHERE source_job_id = ?",
                        (source_job_id,),
                    ).fetchone()
                )
        return record

    def get(self, experiment_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        payload = json.loads(result.pop("payload"))
        result["trace" if result["kind"] == "trace" else "result"] = payload
        return result

    def update(self, experiment_id: str, name: str | None, notes: str | None) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE experiments SET name = COALESCE(?, name), notes = COALESCE(?, notes) WHERE id = ?",
                (name, notes, experiment_id),
            )
        return cursor.rowcount > 0

    def delete(self, experiment_id: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        return cursor.rowcount > 0

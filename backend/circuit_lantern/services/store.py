import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from circuit_lantern.services.store_schema import COLUMNS, EXPERIMENT_KINDS, initialize_schema

SUMMARY_COLUMNS = ", ".join(column for column in COLUMNS if column != "payload")


def _record(row: sqlite3.Row) -> dict:
    result = dict(row)
    # Preserve existing exports without adding an empty matrix source field.
    if result.get("source_analysis_id") is None:
        result.pop("source_analysis_id", None)
    return result


class ExperimentStore:
    def __init__(self, path: Path):
        self.path = path
        with self.connect() as connection:
            initialize_schema(connection)

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
        return [_record(row) for row in rows]

    def find_job(self, job_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT {SUMMARY_COLUMNS} FROM experiments WHERE source_job_id = ?", (job_id,)
            ).fetchone()
        return _record(row) if row is not None else None

    def find_analysis(self, analysis_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT {SUMMARY_COLUMNS} FROM experiments WHERE source_analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        return _record(row) if row is not None else None

    def create(self, name: str, notes: str, trace: dict) -> dict:
        return self.create_artifact(name, notes, "trace", trace)

    def create_artifact(
        self,
        name: str,
        notes: str,
        kind: str,
        payload: dict,
        *,
        source_job_id: str | None = None,
        source_analysis_id: str | None = None,
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
        if source_analysis_id is not None:
            record["source_analysis_id"] = source_analysis_id
        source_field = "source_analysis_id" if source_analysis_id is not None else "source_job_id"
        source_value = source_analysis_id if source_analysis_id is not None else source_job_id
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO experiments "
                "(id, name, notes, created_at, model_id, kind, source_job_id, source_analysis_id, payload) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT({source_field}) DO NOTHING",
                (
                    record["id"],
                    name,
                    notes,
                    record["created_at"],
                    record["model_id"],
                    kind,
                    source_job_id,
                    source_analysis_id,
                    encoded,
                ),
            )
            if cursor.rowcount == 0:
                return _record(
                    connection.execute(
                        f"SELECT {SUMMARY_COLUMNS} FROM experiments WHERE {source_field} = ?",
                        (source_value,),
                    ).fetchone()
                )
        return record

    def payload_size(self, experiment_id: str) -> int | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT length(CAST(payload AS BLOB)) FROM experiments WHERE id = ?",
                (experiment_id,),
            ).fetchone()
        return int(row[0]) if row is not None else None

    def get(self, experiment_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        if row is None:
            return None
        result = _record(row)
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

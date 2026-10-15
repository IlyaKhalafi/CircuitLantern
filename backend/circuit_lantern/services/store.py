import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path


class ExperimentStore:
    def __init__(self, path: Path):
        self.path = path
        with self.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, notes TEXT NOT NULL,
                created_at TEXT NOT NULL, model_id TEXT NOT NULL, payload TEXT NOT NULL
            )""")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def list(self) -> list[dict]:
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, name, notes, created_at, model_id FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def create(self, name: str, notes: str, trace: dict) -> dict:
        record = {
            "id": uuid.uuid4().hex,
            "name": name,
            "notes": notes,
            "created_at": datetime.now(UTC).isoformat(),
            "model_id": trace.get("model_id", "unknown"),
        }
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record["id"],
                    name,
                    notes,
                    record["created_at"],
                    record["model_id"],
                    json.dumps(trace),
                ),
            )
        return record

    def get(self, experiment_id: str) -> dict | None:
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["trace"] = json.loads(result.pop("payload"))
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

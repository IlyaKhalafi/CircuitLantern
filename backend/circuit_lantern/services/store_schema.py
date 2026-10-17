"""Atomic notebook schema upgrades that preserve stored artifact JSON."""

import sqlite3

SCHEMA_VERSION = 4
EXPERIMENT_KINDS = frozenset(
    ("trace", "causal", "head_sweep", "matrix", "induction_probe", "causal_batch")
)
COLUMNS = (
    "id",
    "name",
    "notes",
    "created_at",
    "model_id",
    "payload",
    "kind",
    "source_job_id",
    "source_analysis_id",
)
SOURCE_INDEXES = {
    "experiments_source_job": "source_job_id",
    "experiments_source_analysis": "source_analysis_id",
}


def _columns(connection: sqlite3.Connection) -> set[str]:
    return {row["name"] for row in connection.execute("PRAGMA table_info(experiments)")}


def _index_current(connection: sqlite3.Connection, name: str, column: str) -> bool:
    index = next(
        (
            row
            for row in connection.execute("PRAGMA index_list(experiments)")
            if row["name"] == name
        ),
        None,
    )
    return bool(
        index
        and index["unique"]
        and not index["partial"]
        and [row["name"] for row in connection.execute(f"PRAGMA index_info({name})")] == [column]
    )


def _current(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION
        and set(COLUMNS) <= _columns(connection)
        and all(_index_current(connection, name, column) for name, column in SOURCE_INDEXES.items())
    )


def _create(connection: sqlite3.Connection, table: str):
    kinds = ", ".join("'" + kind + "'" for kind in sorted(EXPERIMENT_KINDS))
    connection.execute(f"""CREATE TABLE {table} (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, notes TEXT NOT NULL,
        created_at TEXT NOT NULL, model_id TEXT NOT NULL, payload TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'trace' CHECK (kind IN ({kinds})),
        source_job_id TEXT, source_analysis_id TEXT
    )""")


def initialize_schema(connection: sqlite3.Connection):
    if _current(connection):
        return
    # Recheck after taking the lock so simultaneous starts cannot race an upgrade.
    connection.execute("BEGIN IMMEDIATE")
    if _current(connection):
        return
    columns = _columns(connection)
    if not columns:
        _create(connection, "experiments")
    elif (
        connection.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION
        or not set(COLUMNS) <= columns
    ):
        required = set(COLUMNS) - {"kind", "source_job_id", "source_analysis_id"}
        if not required <= columns:
            raise ValueError("The notebook table is missing required record columns.")
        _create(connection, "experiments_upgrade")
        values = [
            column if column in columns else "'trace'" if column == "kind" else "NULL"
            for column in COLUMNS
        ]
        # SQL copies the original payload text without decoding or re-encoding JSON.
        connection.execute(
            f"INSERT INTO experiments_upgrade ({', '.join(COLUMNS)}) "
            f"SELECT {', '.join(values)} FROM experiments"
        )
        connection.execute("DROP TABLE experiments")
        connection.execute("ALTER TABLE experiments_upgrade RENAME TO experiments")
    for name, column in SOURCE_INDEXES.items():
        owner = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)
        ).fetchone()
        if owner is not None:
            if owner[0] != "experiments":
                raise ValueError("A notebook source index belongs to a different table.")
            if not _index_current(connection, name, column):
                connection.execute(f"DROP INDEX {name}")
        connection.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {name} ON experiments({column})")
    if connection.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

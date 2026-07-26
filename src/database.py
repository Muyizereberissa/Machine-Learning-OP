"""SQLite persistence for uploads, predictions, retrain jobs, and uptime pings."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.config import DB_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS retrain_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                images_used INTEGER DEFAULT 0,
                epochs INTEGER DEFAULT 0,
                metrics_json TEXT,
                message TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS heartbeat (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                started_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                request_count INTEGER DEFAULT 0
            );
            """
        )
        row = conn.execute("SELECT id FROM heartbeat WHERE id = 1").fetchone()
        if row is None:
            now = _utc_now()
            conn.execute(
                "INSERT INTO heartbeat (id, started_at, last_seen_at, request_count) "
                "VALUES (1, ?, ?, 0)",
                (now, now),
            )


def touch_heartbeat(db_path: Path | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            "UPDATE heartbeat SET last_seen_at = ?, request_count = request_count + 1 "
            "WHERE id = 1",
            (_utc_now(),),
        )


def get_uptime(db_path: Path | None = None) -> dict[str, Any]:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM heartbeat WHERE id = 1").fetchone()
    if row is None:
        return {"status": "down", "uptime_seconds": 0, "request_count": 0}
    started = datetime.fromisoformat(row["started_at"])
    last = datetime.fromisoformat(row["last_seen_at"])
    now = datetime.now(timezone.utc)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return {
        "status": "up",
        "started_at": row["started_at"],
        "last_seen_at": row["last_seen_at"],
        "uptime_seconds": int((now - started).total_seconds()),
        "seconds_since_last_request": int((now - last).total_seconds()),
        "request_count": row["request_count"],
    }


def save_upload(filename: str, label: str, path: str, db_path: Path | None = None) -> int:
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO uploads (filename, label, path, created_at) VALUES (?, ?, ?, ?)",
            (filename, label, path, _utc_now()),
        )
        return int(cur.lastrowid)


def list_uploads(limit: int = 100, db_path: Path | None = None) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM uploads ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def count_uploads_by_label(db_path: Path | None = None) -> dict[str, int]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT label, COUNT(*) AS n FROM uploads GROUP BY label"
        ).fetchall()
    return {r["label"]: r["n"] for r in rows}


def save_prediction(
    filename: str,
    predicted_label: str,
    confidence: float,
    db_path: Path | None = None,
) -> int:
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO predictions (filename, predicted_label, confidence, created_at) "
            "VALUES (?, ?, ?, ?)",
            (filename, predicted_label, confidence, _utc_now()),
        )
        return int(cur.lastrowid)


def create_retrain_job(db_path: Path | None = None) -> int:
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO retrain_jobs (status, started_at) VALUES (?, ?)",
            ("running", _utc_now()),
        )
        return int(cur.lastrowid)


def finish_retrain_job(
    job_id: int,
    status: str,
    images_used: int,
    epochs: int,
    metrics_json: str,
    message: str,
    db_path: Path | None = None,
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE retrain_jobs
            SET status = ?, images_used = ?, epochs = ?, metrics_json = ?,
                message = ?, finished_at = ?
            WHERE id = ?
            """,
            (status, images_used, epochs, metrics_json, message, _utc_now(), job_id),
        )


def latest_retrain_jobs(limit: int = 10, db_path: Path | None = None) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM retrain_jobs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

import sqlite3
from datetime import datetime, timezone

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    engine TEXT NOT NULL DEFAULT 'mineru-only',
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    md_path TEXT
);
CREATE TABLE IF NOT EXISTS extractions (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    fields TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    result TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    fields TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    config.ensure_dirs()
    with connect() as conn:
        conn.executescript(SCHEMA)


def create_job(job_id: str, filename: str, engine: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, filename, engine, status, created_at) VALUES (?, ?, ?, 'queued', ?)",
            (job_id, filename, engine, utcnow()),
        )


def get_job(job_id: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def list_jobs(limit: int = 20) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()


def mark_processing(job_id: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='processing', started_at=? WHERE id=?", (utcnow(), job_id)
        )


def mark_done(job_id: str, md_path: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='done', finished_at=?, md_path=? WHERE id=?",
            (utcnow(), md_path, job_id),
        )


def mark_failed(job_id: str, error: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
            (utcnow(), error[:2000], job_id),
        )


# --- extractions ---

def create_extraction(ext_id: str, job_id: str, fields_json: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO extractions (id, job_id, fields, status, created_at) VALUES (?, ?, ?, 'queued', ?)",
            (ext_id, job_id, fields_json, utcnow()),
        )


def get_extraction(ext_id: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM extractions WHERE id = ?", (ext_id,)).fetchone()


def list_extractions(job_id: str) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM extractions WHERE job_id = ? ORDER BY created_at DESC", (job_id,)
        ).fetchall()


def mark_extraction(ext_id: str, status: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE extractions SET status=? WHERE id=?", (status, ext_id))


def finish_extraction(ext_id: str, result_json: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE extractions SET status='done', finished_at=?, result=? WHERE id=?",
            (utcnow(), result_json, ext_id),
        )


def fail_extraction(ext_id: str, error: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE extractions SET status='failed', finished_at=?, error=? WHERE id=?",
            (utcnow(), error[:2000], ext_id),
        )


# --- templates ---

def save_template(tpl_id: str, name: str, fields_json: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO templates (id, name, fields, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET fields=excluded.fields",
            (tpl_id, name, fields_json, utcnow()),
        )


def list_templates() -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT * FROM templates ORDER BY name").fetchall()

"""SQLite adapter for per-specialist conversations."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
import uuid


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, model TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        content TEXT NOT NULL, sources_json TEXT NOT NULL DEFAULT '[]',
        retrieval_json TEXT NOT NULL DEFAULT '{}', model TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id, id)")
    db.commit()
    try:
        yield db
    finally:
        db.close()


def list_conversations(path: Path) -> list[dict]:
    with connect(path) as db:
        rows = db.execute("""SELECT c.id, c.title, c.model, c.created_at, c.updated_at,
            COUNT(m.id) AS message_count FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id ORDER BY c.updated_at DESC, c.rowid DESC""").fetchall()
        return [dict(row) for row in rows]


def get_conversation(path: Path, identity: str) -> dict | None:
    with connect(path) as db:
        conversation = db.execute("SELECT * FROM conversations WHERE id = ?", (identity,)).fetchone()
        if conversation is None:
            return None
        messages = db.execute("""SELECT role, content, sources_json, retrieval_json, model, created_at
            FROM messages WHERE conversation_id = ? ORDER BY id""", (identity,)).fetchall()
        return {**dict(conversation), "messages": [
            {"role": row["role"], "content": row["content"],
             "sources": json.loads(row["sources_json"]), "retrieval": json.loads(row["retrieval_json"]),
             "model": row["model"], "created_at": row["created_at"]}
            for row in messages]}


def save_turn(path: Path, identity: str | None, question: str, answer: str, model: str,
              sources: list[dict], retrieval: dict, prior: list[dict] | None = None) -> str:
    timestamp = now()
    with connect(path) as db:
        if identity is None:
            identity = str(uuid.uuid4())
            title = question.strip().replace("\n", " ")[:90]
            db.execute("INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                       (identity, title, model, timestamp, timestamp))
            for message in prior or []:
                db.execute("""INSERT INTO messages (conversation_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)""", (identity, message["role"], message["content"], timestamp))
        elif db.execute("SELECT 1 FROM conversations WHERE id = ?", (identity,)).fetchone() is None:
            raise KeyError(identity)
        db.execute("""INSERT INTO messages (conversation_id, role, content, model, created_at)
            VALUES (?, 'user', ?, ?, ?)""", (identity, question, model, timestamp))
        db.execute("""INSERT INTO messages
            (conversation_id, role, content, sources_json, retrieval_json, model, created_at)
            VALUES (?, 'assistant', ?, ?, ?, ?, ?)""",
            (identity, answer, json.dumps(sources, ensure_ascii=False),
             json.dumps(retrieval, ensure_ascii=False), model, timestamp))
        db.execute("UPDATE conversations SET model = ?, updated_at = ? WHERE id = ?",
                   (model, timestamp, identity))
        db.commit()
    return identity

"""SQLite storage backend — async via aiosqlite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from forge.storage.base import StorageBackend


class SQLiteStorage(StorageBackend):
    """Documents and state in SQLite; blobs fall back to filesystem."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._blob_dir = self._db_path.parent / "blobs"
        self._initialized = False

    async def _ensure_tables(self, db: aiosqlite.Connection) -> None:
        if self._initialized:
            return
        await db.execute(
            "CREATE TABLE IF NOT EXISTS documents "
            "(collection TEXT, doc_id TEXT, data TEXT, PRIMARY KEY(collection, doc_id))"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS pipeline_state "
            "(run_id TEXT PRIMARY KEY, state TEXT, updated_at TEXT)"
        )
        await db.commit()
        self._initialized = True

    async def save_document(self, collection: str, doc_id: str, data: dict[str, Any]) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_tables(db)
            await db.execute(
                "INSERT OR REPLACE INTO documents (collection, doc_id, data) VALUES (?, ?, ?)",
                (collection, doc_id, json.dumps(data, default=str)),
            )
            await db.commit()

    async def load_document(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_tables(db)
            async with db.execute(
                "SELECT data FROM documents WHERE collection = ? AND doc_id = ?",
                (collection, doc_id),
            ) as cursor:
                row = await cursor.fetchone()
                return json.loads(row[0]) if row else None

    async def list_documents(self, collection: str) -> list[str]:
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_tables(db)
            async with db.execute(
                "SELECT doc_id FROM documents WHERE collection = ?", (collection,)
            ) as cursor:
                return [row[0] async for row in cursor]

    async def delete_document(self, collection: str, doc_id: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_tables(db)
            await db.execute(
                "DELETE FROM documents WHERE collection = ? AND doc_id = ?",
                (collection, doc_id),
            )
            await db.commit()

    async def save_state(self, run_id: str, state: dict[str, Any]) -> None:
        from datetime import datetime, timezone
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_tables(db)
            await db.execute(
                "INSERT OR REPLACE INTO pipeline_state (run_id, state, updated_at) VALUES (?, ?, ?)",
                (run_id, json.dumps(state, default=str), datetime.now(tz=timezone.utc).isoformat()),
            )
            await db.commit()

    async def load_state(self, run_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_tables(db)
            async with db.execute(
                "SELECT state FROM pipeline_state WHERE run_id = ?", (run_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return json.loads(row[0]) if row else None

    async def save_blob(self, path: str, data: bytes) -> None:
        p = self._blob_dir / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    async def load_blob(self, path: str) -> bytes | None:
        p = self._blob_dir / path
        return p.read_bytes() if p.exists() else None

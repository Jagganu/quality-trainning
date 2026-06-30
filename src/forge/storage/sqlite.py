"""SQLite storage backend — async via aiosqlite."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
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
        # Lock guards _ensure_tables against concurrent first-call races.
        self._init_lock = asyncio.Lock()

    async def _ensure_tables(self, db: aiosqlite.Connection) -> None:
        async with self._init_lock:
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
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_tables(db)
            await db.execute(
                "INSERT OR REPLACE INTO pipeline_state (run_id, state, updated_at) VALUES (?, ?, ?)",
                (run_id, json.dumps(state, default=str), datetime.now(timezone.utc).isoformat()),
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

    def _safe_blob_path(self, path: str) -> Path:
        """Resolve blob path and guard against directory traversal.

        Raises ``ValueError`` if the resolved path escapes the blob root.
        """
        root = self._blob_dir.resolve()
        resolved = (self._blob_dir / path).resolve()
        if not str(resolved).startswith(str(root) + os.sep) and resolved != root:
            raise ValueError(f"Path traversal attempt detected: {path!r}")
        return resolved

    async def save_blob(self, path: str, data: bytes) -> None:
        p = self._safe_blob_path(path)
        await asyncio.to_thread(p.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(p.write_bytes, data)

    async def load_blob(self, path: str) -> bytes | None:
        p = self._safe_blob_path(path)
        if not p.exists():
            return None
        return await asyncio.to_thread(p.read_bytes)

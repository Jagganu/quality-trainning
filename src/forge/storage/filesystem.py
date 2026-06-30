"""Filesystem storage backend — JSON files on disk."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from forge.storage.base import StorageBackend


class FilesystemStorage(StorageBackend):
    """Stores documents as JSON files, blobs as raw files."""

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _doc_path(self, collection: str, doc_id: str) -> Path:
        p = self._base / "documents" / collection
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{doc_id}.json"

    def _state_path(self, run_id: str) -> Path:
        p = self._base / "state"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{run_id}.json"

    def _safe_blob_path(self, path: str) -> Path:
        """Resolve blob path and guard against directory traversal.

        Raises ``ValueError`` if the resolved path escapes the blob root.
        """
        blob_root = (self._base / "blobs").resolve()
        resolved = (blob_root / path).resolve()
        if not str(resolved).startswith(str(blob_root) + os.sep) and resolved != blob_root:
            raise ValueError(f"Path traversal attempt detected: {path!r}")
        return resolved

    async def save_document(self, collection: str, doc_id: str, data: dict[str, Any]) -> None:
        path = self._doc_path(collection, doc_id)
        content = json.dumps(data, default=str, ensure_ascii=False)
        await asyncio.to_thread(path.write_text, content, encoding="utf-8")

    async def load_document(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        path = self._doc_path(collection, doc_id)
        if not path.exists():
            return None
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return json.loads(text)

    async def list_documents(self, collection: str) -> list[str]:
        d = self._base / "documents" / collection
        if not d.exists():
            return []
        return [p.stem for p in d.glob("*.json")]

    async def delete_document(self, collection: str, doc_id: str) -> None:
        path = self._doc_path(collection, doc_id)
        path.unlink(missing_ok=True)

    async def save_state(self, run_id: str, state: dict[str, Any]) -> None:
        path = self._state_path(run_id)
        content = json.dumps(state, default=str, ensure_ascii=False)
        await asyncio.to_thread(path.write_text, content, encoding="utf-8")

    async def load_state(self, run_id: str) -> dict[str, Any] | None:
        path = self._state_path(run_id)
        if not path.exists():
            return None
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return json.loads(text)

    async def save_blob(self, path: str, data: bytes) -> None:
        p = self._safe_blob_path(path)
        await asyncio.to_thread(p.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(p.write_bytes, data)

    async def load_blob(self, path: str) -> bytes | None:
        p = self._safe_blob_path(path)
        if not p.exists():
            return None
        return await asyncio.to_thread(p.read_bytes)

"""Abstract storage backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StorageBackend(ABC):
    """Base class for all storage backends (filesystem, SQLite, S3, …)."""

    @abstractmethod
    async def save_document(self, collection: str, doc_id: str, data: dict[str, Any]) -> None: ...

    @abstractmethod
    async def load_document(self, collection: str, doc_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def list_documents(self, collection: str) -> list[str]: ...

    @abstractmethod
    async def delete_document(self, collection: str, doc_id: str) -> None: ...

    @abstractmethod
    async def save_state(self, run_id: str, state: dict[str, Any]) -> None: ...

    @abstractmethod
    async def load_state(self, run_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def save_blob(self, path: str, data: bytes) -> None: ...

    @abstractmethod
    async def load_blob(self, path: str) -> bytes | None: ...

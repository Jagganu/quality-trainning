"""Dataset loaders — read JSONL, Parquet, and directory structures."""

from __future__ import annotations

import json
from pathlib import Path

from forge.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetLoader:
    """Load datasets from various file formats."""

    async def from_jsonl(self, path: str) -> list[dict]:
        """Load records from a JSONL file (one JSON object per line)."""
        records: list[dict] = []
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        with open(p, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSON at line %d: %s", line_num, exc)

        logger.info("Loaded %d records from %s", len(records), path)
        return records

    async def from_parquet(self, path: str) -> list[dict]:
        """Load records from a Parquet file using PyArrow."""
        import pyarrow.parquet as pq

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        table = pq.read_table(str(p))
        records = table.to_pylist()
        logger.info("Loaded %d records from %s", len(records), path)
        return records

    async def from_directory(self, path: str) -> list[dict]:
        """Load all JSONL files from a directory, concatenating records."""
        p = Path(path)
        if not p.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        records: list[dict] = []
        for jsonl_file in sorted(p.glob("*.jsonl")):
            file_records = await self.from_jsonl(str(jsonl_file))
            records.extend(file_records)

        logger.info("Loaded %d total records from directory %s", len(records), path)
        return records

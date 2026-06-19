"""Dataset exporters — write records to JSONL, Parquet, and HuggingFace."""

from __future__ import annotations

import json
import random
from pathlib import Path

from forge.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetExporter:
    """Export datasets to various formats and destinations."""

    async def to_jsonl(self, records: list[dict], path: str) -> None:
        """Write records as JSONL (one JSON object per line)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        logger.info("Exported %d records to %s", len(records), path)

    async def to_parquet(self, records: list[dict], path: str) -> None:
        """Write records as a Parquet file using PyArrow."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Convert nested dicts/lists to JSON strings for Parquet compatibility
        flat_records = []
        for record in records:
            flat = {}
            for key, val in record.items():
                if isinstance(val, (dict, list)):
                    flat[key] = json.dumps(val, default=str)
                else:
                    flat[key] = val
            flat_records.append(flat)

        table = pa.Table.from_pylist(flat_records)
        pq.write_table(table, str(p))
        logger.info("Exported %d records to %s", len(records), path)

    async def to_huggingface(self, records: list[dict], repo: str) -> None:
        """Export records to a HuggingFace dataset repository.

        Requires the ``huggingface_hub`` package and a valid HF token
        in the ``HF_TOKEN`` environment variable.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for HF export. "
                "Install it with: pip install huggingface_hub"
            )

        # Write a temp JSONL, then upload
        tmp_path = Path(f"/tmp/forge_hf_export_{repo.replace('/', '_')}.jsonl")
        await self.to_jsonl(records, str(tmp_path))

        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(tmp_path),
            path_in_repo="data/train.jsonl",
            repo_id=repo,
            repo_type="dataset",
        )
        logger.info("Uploaded %d records to HuggingFace: %s", len(records), repo)

    async def with_splits(
        self,
        records: list[dict],
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        path: str = ".",
    ) -> None:
        """Split records into train/val/test and export each as JSONL."""
        if abs(train + val + test - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {train + val + test}")

        shuffled = list(records)
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train)
        val_end = train_end + int(n * val)

        splits = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

        output_dir = Path(path)
        for split_name, split_records in splits.items():
            split_path = output_dir / f"{split_name}.jsonl"
            await self.to_jsonl(split_records, str(split_path))

        logger.info(
            "Exported splits: train=%d, val=%d, test=%d → %s",
            len(splits["train"]), len(splits["val"]), len(splits["test"]), path,
        )

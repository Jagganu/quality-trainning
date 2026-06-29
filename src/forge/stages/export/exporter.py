"""Export stage — writes final dataset to JSONL and metadata to JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from forge.core.context import PipelineContext
from forge.core.models import SampleVerdict
from forge.core.stage import Stage
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class ExportStage(Stage):
    """Stage 7: Export accepted samples as a JSONL dataset."""

    @property
    def name(self) -> str:
        return "export"

    async def run(self, context: PipelineContext) -> PipelineContext:
        output_dir = Path(context.settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter to accepted samples only
        accepted = [
            s for s in context.samples
            if not s.verification or s.verification.verdict != SampleVerdict.REJECT
        ]

        if not accepted:
            logger.warning("No accepted samples to export")
            return context

        # Sanitise topic for filename
        safe_topic = "".join(c if c.isalnum() or c in "-_" else "_" for c in context.topic)

        # --- Write JSONL dataset ---
        dataset_path = output_dir / f"{safe_topic}_dataset.jsonl"
        with open(dataset_path, "w", encoding="utf-8") as f:
            for sample in accepted:
                line = json.dumps(sample.content, ensure_ascii=False, default=str)
                f.write(line + "\n")

        # --- Write metadata JSON ---
        metadata_path = output_dir / f"{safe_topic}_metadata.json"
        metadata = {
            "run_id": context.run_id,
            "topic": context.topic,
            "template": context.template_name,
            "format": context.settings.generate.default_format,
            "model": context.settings.default_model,
            "total_generated": len(context.samples),
            "total_exported": len(accepted),
            "rejected": len(context.samples) - len(accepted),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "diversity_score": (
                context.diversity_score.model_dump() if context.diversity_score else None
            ),
            "cost_total_usd": context.budget.current_cost,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        context.metrics.increment("samples_exported", len(accepted))
        logger.info(
            "Exported %d samples → %s (%s)",
            len(accepted), dataset_path, metadata_path.name,
        )

        return context

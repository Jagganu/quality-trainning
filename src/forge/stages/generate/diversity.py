"""Diversity analysis for generated datasets."""

from __future__ import annotations

import math
from collections import Counter

from forge.core.models import DiversityScore, Sample
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class DiversityAnalyzer:
    """Tracks and enforces diversity across generated samples."""

    def compute_score(self, samples: list[Sample]) -> DiversityScore:
        """Compute diversity metrics across all samples."""
        if not samples:
            return DiversityScore()

        te = self._topic_entropy(samples)
        db = self._difficulty_balance(samples)
        pu = self._prompt_uniqueness(samples)
        overall = (te + db + pu) / 3.0

        score = DiversityScore(
            topic_entropy=round(te, 4),
            difficulty_balance=round(db, 4),
            prompt_uniqueness=round(pu, 4),
            overall=round(overall, 4),
        )
        logger.debug(
            "Diversity score: entropy=%.3f balance=%.3f uniqueness=%.3f overall=%.3f",
            te, db, pu, overall,
        )
        return score

    # ------------------------------------------------------------------
    # Internal metrics
    # ------------------------------------------------------------------

    def _topic_entropy(self, samples: list[Sample]) -> float:
        """Shannon entropy over subtopic distribution, normalised to 0-1."""
        subtopics = []
        for s in samples:
            meta = s.content.get("metadata", {})
            sub = meta.get("subtopic", s.lineage.template or "unknown")
            subtopics.append(sub)

        counts = Counter(subtopics)
        total = sum(counts.values())
        if total <= 1 or len(counts) <= 1:
            return 0.0

        entropy = -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )
        max_entropy = math.log2(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _difficulty_balance(self, samples: list[Sample]) -> float:
        """1 - normalised stddev of difficulty distribution.  1.0 = perfectly balanced."""
        diffs = []
        for s in samples:
            meta = s.content.get("metadata", {})
            diffs.append(meta.get("difficulty", "intermediate"))

        counts = Counter(diffs)
        if len(counts) <= 1:
            return 0.5  # only one level — mediocre balance

        values = list(counts.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stddev = math.sqrt(variance)
        normalised = stddev / mean if mean > 0 else 0.0
        return max(0.0, min(1.0, 1.0 - normalised))

    def _prompt_uniqueness(self, samples: list[Sample]) -> float:
        """Ratio of unique question prefixes (first 50 chars)."""
        prefixes: set[str] = set()
        for s in samples:
            question = s.content.get("question", s.content.get("issue", ""))
            if isinstance(question, str) and question:
                prefixes.add(question[:50].lower().strip())

        return len(prefixes) / len(samples) if samples else 0.0

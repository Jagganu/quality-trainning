"""Deduplication engine — exact hash + SimHash near-duplicate detection."""

from __future__ import annotations

from forge.core.models import DeduplicationReport, DuplicateResult
from forge.utils.hashing import hamming_distance, sha256_hash, simhash
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class DeduplicationEngine:
    """Detects exact and near-duplicate documents."""

    def __init__(self, simhash_threshold: int = 3) -> None:
        self._threshold = simhash_threshold
        self._exact_hashes: dict[str, str] = {}  # hash -> doc_id
        self._simhashes: dict[str, int] = {}      # doc_id -> simhash
        self._report = DeduplicationReport()

    def add_document(self, doc_id: str, content: str) -> DuplicateResult:
        """Check for duplicates and register the document."""
        self._report.total_processed += 1
        content_hash = sha256_hash(content)

        # Exact duplicate?
        if content_hash in self._exact_hashes:
            original = self._exact_hashes[content_hash]
            self._report.exact_duplicates += 1
            self._report.duplicate_pairs.append((doc_id, original, 1.0))
            logger.debug("Exact duplicate: %s == %s", doc_id[:8], original[:8])
            return DuplicateResult(is_duplicate=True, strategy="exact", duplicate_of=original, similarity=1.0)

        # Near-duplicate via SimHash?
        doc_simhash = simhash(content)
        for existing_id, existing_hash in self._simhashes.items():
            dist = hamming_distance(doc_simhash, existing_hash)
            if dist <= self._threshold:
                similarity = 1.0 - (dist / 64.0)
                self._report.near_duplicates += 1
                self._report.duplicate_pairs.append((doc_id, existing_id, similarity))
                logger.debug("Near-duplicate: %s ~ %s (dist=%d)", doc_id[:8], existing_id[:8], dist)
                return DuplicateResult(is_duplicate=True, strategy="simhash", duplicate_of=existing_id, similarity=similarity)

        # Unique — register it
        self._exact_hashes[content_hash] = doc_id
        self._simhashes[doc_id] = doc_simhash
        self._report.unique_documents += 1
        return DuplicateResult(is_duplicate=False)

    def check_exact(self, content: str) -> bool:
        return sha256_hash(content) in self._exact_hashes

    def check_simhash(self, content: str, threshold: int | None = None) -> list[str]:
        """Return IDs of near-duplicates."""
        t = threshold if threshold is not None else self._threshold
        doc_hash = simhash(content)
        return [
            doc_id for doc_id, h in self._simhashes.items()
            if hamming_distance(doc_hash, h) <= t
        ]

    def report(self) -> DeduplicationReport:
        return self._report.model_copy()

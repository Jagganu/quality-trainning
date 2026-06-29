"""Content hashing for deduplication — SHA-256 exact match and SimHash near-duplicate."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable


def sha256_hash(content: str) -> str:
    """Return SHA-256 hex digest of content (for exact duplicate detection).

    SHA-256 replaces the previous MD5-based ``md5_hash``.  MD5 has known
    collision vulnerabilities — while extremely unlikely in practice for
    plain text, SHA-256 has no known collisions and costs negligibly more.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Legacy alias — kept so existing stored hashes and any external callers
# that imported md5_hash directly still work.  New code should use sha256_hash.
# ---------------------------------------------------------------------------

def md5_hash(content: str) -> str:  # noqa: N802
    """Deprecated: use ``sha256_hash`` instead."""
    import warnings
    warnings.warn(
        "md5_hash is deprecated and will be removed in a future release. "
        "Use sha256_hash instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"\w+", text.lower())


def _string_hash(token: str, bits: int = 64) -> int:
    """Hash a single token to an integer using MD5 truncation."""
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h[:bits // 4], 16)


def simhash(text: str, bits: int = 64) -> int:
    """Compute SimHash fingerprint for near-duplicate detection.

    SimHash maps similar documents to similar hash values. Two documents
    are near-duplicates if their SimHash values differ in few bits.

    Args:
        text: Input text to hash.
        bits: Number of bits in the fingerprint (default 64).

    Returns:
        Integer fingerprint.
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0

    # Accumulator vector
    v = [0] * bits

    for token in tokens:
        token_hash = _string_hash(token, bits)
        for i in range(bits):
            if token_hash & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    # Build fingerprint from sign of each dimension
    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= 1 << i

    return fingerprint


def hamming_distance(hash1: int, hash2: int) -> int:
    """Count the number of differing bits between two hashes."""
    xor = hash1 ^ hash2
    return bin(xor).count("1")


def are_near_duplicates(hash1: int, hash2: int, threshold: int = 3) -> bool:
    """Check if two SimHash values indicate near-duplicate content.

    Args:
        hash1: First SimHash fingerprint.
        hash2: Second SimHash fingerprint.
        threshold: Maximum Hamming distance to consider near-duplicate.

    Returns:
        True if the documents are likely near-duplicates.
    """
    return hamming_distance(hash1, hash2) <= threshold

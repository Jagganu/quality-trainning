"""Clean stage — normalise, chunk, and quality-score documents."""

from __future__ import annotations

import math
import re
import unicodedata

from forge.core.context import PipelineContext
from forge.core.models import CleanedDocument
from forge.core.stage import Stage
from forge.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"cookie[s]?\s+(policy|notice|consent|banner)", re.I),
    re.compile(r"(sign up|subscribe|newsletter)\s+(now|today|for|to)", re.I),
    re.compile(r"(share|follow|like|tweet|pin)\s+(this|on|us)", re.I),
    re.compile(r"(terms\s+of\s+(service|use)|privacy\s+policy|all\s+rights\s+reserved)", re.I),
    re.compile(r"(loading\.\.\.|please\s+wait|skip\s+ad)", re.I),
    re.compile(r"(read\s+more|click\s+here|learn\s+more|see\s+more)", re.I),
    re.compile(r"(advertisement|sponsored|promoted|partner\s+content)", re.I),
]

_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_SPECIAL_CHARS_RE = re.compile(r"[^\w\s.,;:!?()\-\"'/\\#\n]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n")
    text = _HTML_ENTITY_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _remove_boilerplate(text: str) -> str:
    # Split into sentences for finer-grained removal
    parts = re.split(r"(?<=[.!?])\s+", text)
    cleaned: list[str] = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if any(p.search(stripped) for p in _BOILERPLATE_PATTERNS):
            continue
        cleaned.append(stripped)
    result = " ".join(cleaned)
    result = _URL_RE.sub("", result)
    result = _EMAIL_RE.sub("", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_size: int) -> list[str]:
    if len(text) <= max_size:
        return [text]
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text[:max_size]]


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def _char_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _flesch_kincaid_grade(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = text.split()
    if not sentences or not words:
        return 0.0
    syllable_count = sum(_count_syllables(w) for w in words)
    avg_sentence_len = len(words) / len(sentences)
    avg_syllables = syllable_count / len(words)
    return 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59


def _count_syllables(word: str) -> int:
    word = word.lower().strip()
    if len(word) <= 2:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _code_block_ratio(text: str) -> float:
    code_indicators = [
        re.compile(r"```[\s\S]*?```", re.M),
        re.compile(r"`[^`\n]+`"),
        re.compile(
            r"^\s*(def |class |import |from |function |const |let |var |"
            r"if\s*\(|for\s*\(|while\s*\()",
            re.M,
        ),
        re.compile(r"^\s*[#//]\s*(include|define|pragma|include)", re.M),
        re.compile(r"(=>|->|;\s*$)", re.M),
        re.compile(r"^\s*\w+\s*=\s*\w+\s*\(", re.M),
    ]
    code_chars = 0
    for pattern in code_indicators:
        for match in pattern.finditer(text):
            code_chars += len(match.group())
    # Also count lines that look like code (contain = assignment or () calls)
    lines = text.split("\n")
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if (
            re.match(r"^\w+\s*=", stripped)
            or stripped.startswith(("def ", "class ", "import ", "from "))
            or re.match(r"^\w+\(.*\)\s*$", stripped)
        ):
            code_lines += 1
    code_line_ratio = code_lines / max(len(lines), 1)
    pattern_ratio = min(1.0, code_chars / max(len(text), 1))
    return max(pattern_ratio, code_line_ratio)


def _spam_score(text: str) -> float:
    penalties = 0.0
    words = text.split()
    if not words:
        return 0.0

    # Repeated words
    word_freq: dict[str, int] = {}
    for w in words:
        low = w.lower().strip(".,!?;:")
        word_freq[low] = word_freq.get(low, 0) + 1
    max_repeat = max(word_freq.values()) if word_freq else 0
    if max_repeat > len(words) * 0.08:
        penalties += 0.3

    # Excessive caps
    alpha_words = [w for w in words if len(w) > 2]
    if alpha_words:
        caps_ratio = sum(1 for w in alpha_words if w.isupper()) / len(alpha_words)
        if caps_ratio > 0.3:
            penalties += 0.3

    # Excessive exclamation / question marks
    excl = text.count("!") + text.count("?")
    if excl > len(words) * 0.05:
        penalties += 0.2

    # Short sentences (likely listicles / clickbait)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if sentences:
        short_ratio = sum(1 for s in sentences if len(s.split()) < 4) / len(sentences)
        if short_ratio > 0.5:
            penalties += 0.2

    return min(1.0, penalties)


def _quality_score(text: str) -> float:
    if not text or len(text) < 50:
        return 0.0

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = text.split()
    if not sentences or not words:
        return 0.0

    # 1. Length score — prefer substantive content (200-5000 chars sweet spot)
    char_len = len(text)
    if char_len < 200:
        length_score = char_len / 200 * 0.5
    elif char_len < 500:
        length_score = 0.5 + (char_len - 200) / 300 * 0.3
    elif char_len < 5000:
        length_score = 0.8 + (char_len - 500) / 4500 * 0.2
    else:
        length_score = 1.0

    # 2. Sentence structure — well-formed sentences indicate quality
    avg_words_per_sentence = len(words) / len(sentences)
    if avg_words_per_sentence < 5:
        sentence_score = avg_words_per_sentence / 5 * 0.5
    elif avg_words_per_sentence > 40:
        sentence_score = max(0.3, 1.0 - (avg_words_per_sentence - 40) / 40)
    else:
        sentence_score = 0.7 + (avg_words_per_sentence - 5) / 35 * 0.3

    # 3. Readability — Flesch-Kincaid
    fk = _flesch_kincaid_grade(text)
    if fk < 3:
        readability = 0.4
    elif fk < 8:
        readability = 0.7 + (fk - 3) / 5 * 0.3
    elif fk < 14:
        readability = 1.0
    elif fk < 20:
        readability = max(0.5, 1.0 - (fk - 14) / 6 * 0.5)
    else:
        readability = 0.3

    # 4. Character entropy — high entropy = natural language, low = spam/boilerplate
    entropy = _char_entropy(text)
    if entropy < 3.0:
        entropy_score = entropy / 3.0 * 0.5
    elif entropy < 4.5:
        entropy_score = 0.5 + (entropy - 3.0) / 1.5 * 0.5
    else:
        entropy_score = 1.0

    # 5. Code block penalty — penalise docs that are mostly code
    code_ratio = _code_block_ratio(text)
    code_penalty = 1.0 - min(1.0, code_ratio * 1.5)

    # 6. Spam penalty
    spam = _spam_score(text)

    # Weighted combination
    score = (
        length_score * 0.15
        + sentence_score * 0.20
        + readability * 0.20
        + entropy_score * 0.20
        + code_penalty * 0.15
        + (1.0 - spam) * 0.10
    )
    return round(max(0.0, min(1.0, score)), 3)


# ---------------------------------------------------------------------------
# Language detection (heuristic, no external deps)
# ---------------------------------------------------------------------------

_COMMON_EN_WORDS = frozenset({
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first",
    "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us",
})


def _detect_language(text: str) -> str:
    if not text:
        return "unknown"
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    if non_ascii / max(len(text), 1) > 0.3:
        return "non-ascii"
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if not words:
        return "unknown"
    sample = words[:500]
    en_count = sum(1 for w in sample if w in _COMMON_EN_WORDS)
    ratio = en_count / len(sample) if sample else 0
    if ratio > 0.25:
        return "en"
    return "unknown"


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class CleanStage(Stage):
    """Stage 2: Normalise text, chunk long documents, score quality."""

    @property
    def name(self) -> str:
        return "clean"

    async def run(self, context: PipelineContext) -> PipelineContext:
        settings = context.settings.clean
        cleaned: list[CleanedDocument] = []
        total_chunks = 0
        skipped_short = 0
        skipped_quality = 0
        skipped_language = 0
        skipped_code = 0

        for doc in context.documents:
            content = doc.content

            # Hard minimum length
            if len(content) < settings.min_content_length:
                skipped_short += 1
                continue

            # Language check
            lang = _detect_language(content)
            if lang != "en":
                skipped_language += 1
                logger.debug("Skipping %s (language=%s)", doc.doc_id[:8], lang)
                continue

            # Normalise
            text = _normalize_text(content)

            # Remove boilerplate
            if settings.remove_boilerplate:
                text = _remove_boilerplate(text)

            if len(text) < settings.min_content_length:
                skipped_short += 1
                continue

            # Code block check — drop docs that are mostly code
            code_ratio = _code_block_ratio(text)
            if code_ratio > settings.max_code_block_ratio:
                skipped_code += 1
                logger.debug("Skipping %s (%.0f%% code)", doc.doc_id[:8], code_ratio * 100)
                continue

            # Quality score
            score = _quality_score(text)
            if score < settings.min_quality_score:
                skipped_quality += 1
                logger.debug(
                    "Skipping %s (quality=%.3f < %.3f)",
                    doc.doc_id[:8], score, settings.min_quality_score,
                )
                continue

            # Entropy check — very low entropy = repetitive / junk
            entropy = _char_entropy(text)
            if entropy < settings.min_char_entropy:
                skipped_quality += 1
                logger.debug(
                    "Skipping %s (entropy=%.2f < %.2f)",
                    doc.doc_id[:8], entropy, settings.min_char_entropy,
                )
                continue

            # Chunk and keep
            chunks = _chunk_text(text, settings.max_chunk_size)
            cd = CleanedDocument(
                source_doc_id=doc.doc_id,
                chunks=chunks,
                quality_score=score,
                word_count=len(text.split()),
            )
            cleaned.append(cd)
            total_chunks += len(chunks)

        context.cleaned_documents = cleaned
        context.metrics.increment("documents_cleaned", len(cleaned))
        context.metrics.gauge("chunks_total", total_chunks)

        total_input = len(context.documents)
        kept = len(cleaned)
        dropped = total_input - kept
        logger.info(
            "Clean: %d/%d docs kept (%d dropped: %d short, %d low-quality, %d language, %d code)",
            kept, total_input, dropped, skipped_short, skipped_quality,
            skipped_language, skipped_code,
        )
        return context

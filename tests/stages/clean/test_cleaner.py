"""Tests for the enhanced quality scoring and filtering in the clean stage."""

import pytest

from forge.stages.clean.cleaner import (
    _char_entropy,
    _code_block_ratio,
    _count_syllables,
    _detect_language,
    _flesch_kincaid_grade,
    _normalize_text,
    _quality_score,
    _remove_boilerplate,
    _spam_score,
)


class TestCharEntropy:
    def test_empty_string(self):
        assert _char_entropy("") == 0.0

    def test_single_char(self):
        assert _char_entropy("a") == 0.0

    def test_repetitive_low_entropy(self):
        assert _char_entropy("aaaaaaa") < 1.0

    def test_diverse_high_entropy(self):
        text = "The quick brown fox jumps over the lazy dog near the riverbank"
        assert _char_entropy(text) > 3.0


class TestFleschKincaidGrade:
    def test_empty(self):
        assert _flesch_kincaid_grade("") == 0.0

    def test_simple_sentences(self):
        text = "The cat sat. The dog ran. The bird flew."
        grade = _flesch_kincaid_grade(text)
        assert grade < 10

    def test_complex_sentences(self):
        text = (
            "The implementation of advanced cryptographic algorithms requires "
            "a comprehensive understanding of mathematical principles and "
            "computational complexity theory in modern information security."
        )
        grade = _flesch_kincaid_grade(text)
        assert grade > 8


class TestCountSyllables:
    def test_short_word(self):
        assert _count_syllables("a") == 1

    def test_common_words(self):
        assert _count_syllables("cat") == 1
        assert _count_syllables("water") == 2
        assert _count_syllables("beautiful") == 3

    def test_silent_e(self):
        assert _count_syllables("make") == 1
        assert _count_syllables("time") == 1


class TestCodeBlockRatio:
    def test_prose(self):
        text = (
            "This is a normal paragraph about machine learning. "
            "It discusses various approaches to training neural networks."
        )
        assert _code_block_ratio(text) < 0.1

    def test_code_heavy(self):
        text = """
        ```python
        def hello():
            print("hello world")
        ```
        import os
        import sys
        from typing import List
        """
        assert _code_block_ratio(text) > 0.3


class TestSpamScore:
    def test_clean_text(self):
        text = (
            "Machine learning is a subset of artificial intelligence. "
            "It focuses on building systems that learn from data. "
            "These systems improve their performance over time without explicit programming."
        )
        assert _spam_score(text) < 0.3

    def test_excessive_caps(self):
        text = "THIS IS VERY IMPORTANT! YOU MUST READ THIS NOW! CLICK HERE!"
        assert _spam_score(text) > 0.2

    def test_repeated_words(self):
        words = ["amazing"] * 20 + ["product"] * 5
        text = " ".join(words)
        assert _spam_score(text) > 0.2


class TestNormalizeText:
    def test_normalizes_whitespace(self):
        assert _normalize_text("hello   world") == "hello world"

    def test_normalizes_newlines(self):
        assert _normalize_text("a\n\n\n\nb") == "a\n\nb"

    def test_strips_html_entities(self):
        result = _normalize_text("hello&nbsp;world")
        assert "&nbsp;" not in result

    def test_nfc_normalization(self):
        text = "caf\u00e9"  # precomposed
        result = _normalize_text(text)
        assert "cafe" in result or "caf" in result


class TestRemoveBoilerplate:
    def test_removes_cookie_notices(self):
        text = "Important article.\nCookie consent banner text.\nMore content."
        result = _remove_boilerplate(text)
        assert "Cookie consent" not in result

    def test_removes_share_prompts(self):
        text = "Great article.\nShare this on twitter.\nRead more."
        result = _remove_boilerplate(text)
        assert "Share this" not in result

    def test_removes_urls(self):
        text = "Visit https://example.com for more info about the topic."
        result = _remove_boilerplate(text)
        assert "https://" not in result

    def test_removes_emails(self):
        text = "Contact us at spam@example.com for details."
        result = _remove_boilerplate(text)
        assert "@" not in result or "Contact" in result

    def test_keeps_substantive_content(self):
        text = "Cross-site scripting is a vulnerability. It allows injection of scripts."
        result = _remove_boilerplate(text)
        assert "vulnerability" in result


class TestDetectLanguage:
    def test_english(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test of the English language detection system."
        )
        assert _detect_language(text) == "en"

    def test_empty(self):
        assert _detect_language("") == "unknown"

    def test_non_ascii_heavy(self):
        text = "\u4e2d\u6587\u6d4b\u8bd5\u6587\u672c\u5185\u5bb9"
        assert _detect_language(text) == "non-ascii"


class TestQualityScore:
    def test_empty_text(self):
        assert _quality_score("") == 0.0

    def test_very_short(self):
        assert _quality_score("Hi") == 0.0

    def test_good_quality(self):
        text = (
            "Machine learning algorithms are broadly categorized into supervised, "
            "unsupervised, and reinforcement learning paradigms. Supervised learning "
            "uses labeled training data to learn a mapping function from inputs to "
            "outputs. Common algorithms include linear regression, decision trees, "
            "and neural networks. The choice of algorithm depends on the nature of "
            "the data and the specific problem being addressed."
        )
        score = _quality_score(text)
        assert score > 0.4

    def test_spammy_low_quality(self):
        # Pure spam with no real content should score lower than genuine prose
        spam = "CLICK HERE NOW! BUY NOW! AMAZING DEAL! " * 20
        genuine = (
            "Machine learning algorithms are broadly categorized into supervised, "
            "unsupervised, and reinforcement learning paradigms. Supervised learning "
            "uses labeled training data to learn a mapping function from inputs to "
            "outputs. Common algorithms include linear regression, decision trees, "
            "and neural networks."
        )
        spam_score = _quality_score(spam)
        genuine_score = _quality_score(genuine)
        assert spam_score < genuine_score

    def test_code_heavy_low_quality(self):
        # Heavy code should score lower than pure prose
        code = (
            "def foo():\n    return bar\ndef baz():\n    return qux\n"
            "import os\nimport sys\nfrom typing import List, Dict, Optional\n"
            "class Foo:\n    def __init__(self):\n        pass\n"
            "x = foo()\ny = baz()\nprint(x, y)\n"
        ) * 3
        prose = (
            "Natural language processing is a field of computer science and "
            "artificial intelligence concerned with the interaction between "
            "computers and human language. It focuses on giving computers the "
            "ability to process text and spoken words in the same way humans can."
        )
        code_score = _quality_score(code)
        prose_score = _quality_score(prose)
        assert code_score < prose_score


class TestNormalizeAndScore:
    def test_integration_normalised_good_score(self):
        text = (
            "Natural language processing is a field of computer science and "
            "artificial intelligence concerned with the interaction between "
            "computers and human language. It focuses on giving computers the "
            "ability to process text and spoken words in the same way humans can."
        )
        normalised = _normalize_text(text)
        score = _quality_score(normalised)
        assert score > 0.3

    def test_integration_boilerplate_removal_improves_score(self):
        text = (
            "Cookie policy notice here. Share this on facebook. "
            "Machine learning is a powerful technique for data analysis. "
            "It enables computers to learn patterns from training data. "
            "These patterns can then be used to make predictions on new data."
        )
        cleaned = _remove_boilerplate(text)
        # Boilerplate should be removed
        assert "Cookie policy" not in cleaned
        assert "Share this" not in cleaned
        # Substantive content should remain
        assert "Machine learning" in cleaned
        # Scored text should be non-zero
        assert _quality_score(cleaned) > 0

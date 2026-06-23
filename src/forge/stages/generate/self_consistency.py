"""Self-consistency generation — sample N candidate reasoning chains per
prompt instead of one, then cluster and select the most consistent winner.

Replaces a single ``llm.complete()`` call with N calls (optionally spread
across different model families for diversity) and adds a clustering step
so that the Verify stage only has to judge one already-strong candidate
instead of guessing from a single shot.
"""

from __future__ import annotations

import re
from collections import defaultdict

from forge.core.models import CandidateGeneration, CandidateSet
from forge.providers.llm import LLMProvider
from forge.utils.logging import get_logger
from forge.verification.programmatic import CodeChecker, MathChecker

logger = get_logger(__name__)

_WS_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Normalise free text for exact-match clustering: lowercase, collapse
    whitespace, strip trailing punctuation/units noise.
    """
    text = text.strip().lower()
    text = _WS_RE.sub(" ", text)
    text = text.strip(" .!?")
    return text


def extract_answer_signature(content: dict, fmt: str) -> str:
    """Pull the field that represents the candidate's "final answer" for
    the given format, normalised for clustering.
    """
    key_by_format = {
        "reasoning": "answer",
        "coding": "patch",
        "agent": "result",
        "instruction": "messages",
        "chat": "conversations",
    }
    key = key_by_format.get(fmt, "answer")
    value = content.get(key, "")
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)
    return normalize_answer(str(value))


class SelfConsistencyGenerator:
    """Generates N candidates per prompt and selects a winner by clustering.

    Parameters
    ----------
    llm:
        Shared :class:`LLMProvider` (budget-tracked).
    n:
        Number of candidates to sample per prompt.
    models:
        Optional list of model identifiers to rotate across candidates.
        If empty, all candidates use the provider's default model.
    """

    def __init__(
        self,
        llm: LLMProvider,
        n: int = 3,
        models: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.n = max(1, n)
        self.models = models or []
        self.code_checker = CodeChecker()
        self.math_checker = MathChecker()

    def _model_for_index(self, i: int) -> str | None:
        if not self.models:
            return None
        return self.models[i % len(self.models)]

    async def generate(
        self,
        prompt: str,
        system: str,
        fmt_name: str,
        format_sample_fn,
        reference_answer: str | None = None,
        test_code: str | None = None,
    ) -> CandidateSet:
        """Sample N candidates for *prompt*, run programmatic checks where
        applicable, cluster by answer signature, and select a winner.

        ``format_sample_fn`` mirrors ``DatasetFormat.format_sample`` — it
        turns a raw completion into the structured ``content`` dict.
        """
        cset = CandidateSet()
        candidates: list[CandidateGeneration] = []

        for i in range(self.n):
            model = self._model_for_index(i)
            try:
                raw = await self.llm.complete(
                    prompt=prompt, system=system, model=model, stage="generate",
                )
            except Exception as exc:
                logger.warning("Self-consistency candidate %d failed: %s", i, exc)
                continue

            content = format_sample_fn(raw)
            signature = extract_answer_signature(content, fmt_name)

            candidate = CandidateGeneration(
                raw=raw, content=content, answer_signature=signature,
            )

            # Programmatic ground-truth check, free and unambiguous where
            # it applies — runs before any clustering/judging.
            if fmt_name == "coding" and test_code:
                code = str(content.get("patch", ""))
                candidate.programmatic_pass = await self.code_checker.check(code, test_code)
            elif fmt_name == "reasoning" and reference_answer:
                answer_text = str(content.get("answer", ""))
                candidate.programmatic_pass = self.math_checker.check(
                    answer_text, reference_answer
                )

            candidates.append(candidate)

        # Drop candidates that programmatically failed — no point clustering
        # or judging something already known to be wrong.
        surviving = [c for c in candidates if c.programmatic_pass is not False]
        if not surviving and candidates:
            # Everything failed programmatically — keep the set for
            # visibility but leave selected_id empty so Verify rejects it.
            cset.candidates = candidates
            cset.agreement_ratio = 0.0
            return cset

        cset.candidates = candidates
        self._cluster(surviving)
        self._select_winner(cset, surviving)
        return cset

    @staticmethod
    def _cluster(candidates: list[CandidateGeneration]) -> None:
        """Assign cluster_id by exact-match on normalised answer signature.

        Open-ended free text won't always cluster well this way — callers
        needing semantic equivalence for non-extractive answers should
        pre-group via an equivalence judge call before invoking this, or
        treat agreement_ratio accordingly (a single cluster of 1 just
        means "no consensus", which correctly routes to revise).
        """
        groups: dict[str, list[CandidateGeneration]] = defaultdict(list)
        for c in candidates:
            groups[c.answer_signature].append(c)

        for cluster_id, (_, members) in enumerate(
            sorted(groups.items(), key=lambda kv: -len(kv[1]))
        ):
            for m in members:
                m.cluster_id = cluster_id

    @staticmethod
    def _select_winner(cset: CandidateSet, candidates: list[CandidateGeneration]) -> None:
        if not candidates:
            return

        clusters: dict[int, list[CandidateGeneration]] = defaultdict(list)
        for c in candidates:
            clusters[c.cluster_id].append(c)

        best_cluster_id = max(clusters, key=lambda cid: len(clusters[cid]))
        best_cluster = clusters[best_cluster_id]
        cset.agreement_ratio = len(best_cluster) / len(candidates)

        # Require at least 2 candidates agreeing — a lone candidate isn't
        # "self-consistent", it's just one guess. A single sampled
        # candidate (n=1) is the exception: nothing to agree with, so it
        # passes through as the only option, deferring entirely to the
        # downstream judge ensemble.
        if len(candidates) == 1:
            cset.selected_id = candidates[0].candidate_id
            return

        if len(best_cluster) < 2:
            cset.selected_id = ""  # no agreement — route to revise
            cset.rejected_ids = [c.candidate_id for c in candidates]
            return

        # Prefer a programmatically-confirmed member of the winning
        # cluster if one exists, else just take the first.
        confirmed = [c for c in best_cluster if c.programmatic_pass is True]
        winner = confirmed[0] if confirmed else best_cluster[0]
        cset.selected_id = winner.candidate_id
        cset.rejected_ids = [
            c.candidate_id for c in candidates if c.candidate_id != winner.candidate_id
        ]

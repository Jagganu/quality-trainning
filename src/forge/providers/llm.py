"""Unified LLM provider powered by LiteLLM — works with 100+ models."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import litellm
from pydantic import BaseModel

from forge.core.budget import CostBudget
from forge.core.config import ForgeSettings
from forge.core.models import RawGeneration
from forge.utils.logging import get_logger

logger = get_logger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


class LLMProvider:
    """Thin async wrapper around LiteLLM with budget tracking."""

    def __init__(self, settings: ForgeSettings, budget: CostBudget) -> None:
        self._settings = settings
        self._budget = budget
        self._setup_keys()

    def _setup_keys(self) -> None:
        """Push API keys from settings into env vars for LiteLLM."""
        mapping = {
            "OPENAI_API_KEY": self._settings.openai_api_key,
            "ANTHROPIC_API_KEY": self._settings.anthropic_api_key,
            "GEMINI_API_KEY": self._settings.gemini_api_key,
            "OPENROUTER_API_KEY": self._settings.openrouter_api_key,
        }
        for var, val in mapping.items():
            if val:
                os.environ[var] = val

    def _get_model(self, model: str | None) -> str:
        return model or self._settings.default_model

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        stage: str = "",
    ) -> RawGeneration:
        """Single LLM completion with cost tracking."""
        model_name = self._get_model(model)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        try:
            response = await litellm.acompletion(
                model=model_name,
                messages=messages,
                temperature=temperature or self._settings.generate.temperature,
            )
        except Exception as e:
            logger.error("LLM call failed (%s): %s", model_name, e)
            raise

        latency = (time.monotonic() - t0) * 1000
        text = response.choices[0].message.content or ""
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        # Compute cost
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = (tokens_in / 1000) * 0.01 + (tokens_out / 1000) * 0.03

        self._budget.record_usage(tokens_in, tokens_out, model_name, cost, stage)

        return RawGeneration(
            text=text,
            model=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            latency_ms=latency,
        )

    async def complete_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str = "",
        model: str | None = None,
    ) -> BaseModel:
        """Call LLM and parse response as JSON into a Pydantic model."""
        json_instruction = (
            f"\n\nRespond ONLY with valid JSON matching this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        )
        raw = await self.complete(prompt + json_instruction, system=system, model=model)
        # Extract JSON from response (handle markdown code blocks)
        text = raw.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return schema.model_validate_json(text)

    async def complete_batch(
        self,
        prompts: list[str],
        system: str = "",
        model: str | None = None,
        concurrency: int = 5,
        stage: str = "",
    ) -> list[RawGeneration]:
        """Run multiple prompts concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(concurrency)
        results: list[RawGeneration] = []

        async def _call(prompt: str) -> RawGeneration:
            async with semaphore:
                return await self.complete(prompt, system=system, model=model, stage=stage)

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(_call(p)) for p in prompts]

        return [t.result() for t in tasks]

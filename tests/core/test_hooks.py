"""Tests for core.hooks — event system."""

import pytest

from forge.core.hooks import HookEvent, HookManager


@pytest.mark.asyncio
async def test_register_and_emit():
    """Registered callbacks should be called on emit."""
    calls = []

    async def on_start(**kwargs):
        calls.append(("start", kwargs))

    mgr = HookManager()
    mgr.register(HookEvent.PIPELINE_START, on_start)
    await mgr.emit(HookEvent.PIPELINE_START, topic="test")

    assert len(calls) == 1
    assert calls[0][0] == "start"
    assert calls[0][1]["topic"] == "test"


@pytest.mark.asyncio
async def test_multiple_hooks():
    """Multiple callbacks on the same event should all fire."""
    results = []

    async def hook_a(**kwargs):
        results.append("a")

    async def hook_b(**kwargs):
        results.append("b")

    mgr = HookManager()
    mgr.register(HookEvent.AFTER_STAGE, hook_a)
    mgr.register(HookEvent.AFTER_STAGE, hook_b)
    await mgr.emit(HookEvent.AFTER_STAGE, stage="collect")

    assert results == ["a", "b"]


@pytest.mark.asyncio
async def test_unregister():
    """Unregistered callbacks should no longer fire."""
    results = []

    async def hook(**kwargs):
        results.append("fired")

    mgr = HookManager()
    mgr.register(HookEvent.PIPELINE_END, hook)
    mgr.unregister(HookEvent.PIPELINE_END, hook)
    await mgr.emit(HookEvent.PIPELINE_END)

    assert results == []


@pytest.mark.asyncio
async def test_bad_hook_does_not_crash():
    """A failing hook should be caught, not propagate."""
    results = []

    async def bad_hook(**kwargs):
        raise RuntimeError("boom")

    async def good_hook(**kwargs):
        results.append("ok")

    mgr = HookManager()
    mgr.register(HookEvent.ON_ERROR, bad_hook)
    mgr.register(HookEvent.ON_ERROR, good_hook)

    # Should NOT raise
    await mgr.emit(HookEvent.ON_ERROR, error="test")
    assert results == ["ok"]

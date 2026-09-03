"""HERMES_MEMORY_PREFETCH_TIMEOUT overrides the external-provider prefetch budget.

The 8 s default is an interactive-latency choice. The gateway's cron lane pays
a 3-8 s reranker cold start on each job's first turn and was losing that
turn's recall to the budget (2026-08-30 onward, ~3 jobs a night), so the
gateway unit raises it via a systemd drop-in. An explicit constructor argument
still wins, and a bad value must never take the agent down.
"""

import pytest

from agent.memory_manager import MemoryManager, _EXTERNAL_PREFETCH_TIMEOUT_S


def test_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("HERMES_MEMORY_PREFETCH_TIMEOUT", raising=False)
    assert MemoryManager()._external_prefetch_timeout == _EXTERNAL_PREFETCH_TIMEOUT_S


def test_env_override_is_honoured(monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_PREFETCH_TIMEOUT", "20")
    assert MemoryManager()._external_prefetch_timeout == 20.0


@pytest.mark.parametrize("bad", ["abc", "0", "-3", "", "  "])
def test_unusable_env_falls_back_to_default(monkeypatch, bad):
    monkeypatch.setenv("HERMES_MEMORY_PREFETCH_TIMEOUT", bad)
    assert MemoryManager()._external_prefetch_timeout == _EXTERNAL_PREFETCH_TIMEOUT_S


def test_explicit_argument_beats_env(monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_PREFETCH_TIMEOUT", "20")
    assert MemoryManager(external_prefetch_timeout=0.5)._external_prefetch_timeout == 0.5

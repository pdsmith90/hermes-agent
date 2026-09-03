"""HERMES_RERANK_TIMEOUT overrides the cross-encoder client timeout.

Companion to HERMES_RERANK_URL: the gateway's cron lane waits out a cold
qwen3-rerank start (3-8 s after llama-swap idle-unloads it between jobs)
while the interactive default stays at 8 s.
"""

import sqlite3

import pytest

pytest.importorskip("numpy")  # retrieval imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


@pytest.fixture(autouse=True)
def _clean_shared_registry():
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()
    yield
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory_store.db")
    yield s
    s.close()


def test_default_when_env_unset(store, monkeypatch):
    monkeypatch.delenv("HERMES_RERANK_TIMEOUT", raising=False)
    assert FactRetriever(store).rerank_timeout == 8.0


def test_env_override_is_honoured(store, monkeypatch):
    monkeypatch.setenv("HERMES_RERANK_TIMEOUT", "15")
    assert FactRetriever(store).rerank_timeout == 15.0


def test_explicit_argument_beats_env(store, monkeypatch):
    monkeypatch.setenv("HERMES_RERANK_TIMEOUT", "15")
    assert FactRetriever(store, rerank_timeout=3).rerank_timeout == 3.0


@pytest.mark.parametrize("bad", ["soon", "0", "-1"])
def test_unusable_env_falls_back_to_default(store, monkeypatch, bad):
    monkeypatch.setenv("HERMES_RERANK_TIMEOUT", bad)
    assert FactRetriever(store).rerank_timeout == 8.0

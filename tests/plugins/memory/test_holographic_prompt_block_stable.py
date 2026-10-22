"""The holographic system-prompt block must not change as facts are added.

It is part of the system prompt every session sends, so any per-fact content
(the old live fact count) changes the prompt between consecutive sessions and
defeats cross-session prompt-cache reuse at the provider.
"""

from plugins.memory.holographic import HolographicMemoryProvider


def test_prompt_block_is_stable_as_the_store_grows(tmp_path):
    provider = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    )
    provider.initialize(session_id="test-session")
    try:
        provider._store.add_fact("LESSON: first fact for the block test.", category="lesson")
        before = provider.system_prompt_block()
        provider._store.add_fact("LESSON: second, unrelated fact about prefixes.", category="lesson")
        assert provider.system_prompt_block() == before
        assert before.startswith("# Holographic Memory\n")
    finally:
        provider.shutdown()

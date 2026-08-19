"""Default ``limit`` for ``fact_store action=list``.

The tool wrapper hard-defaulted an unspecified ``limit`` to 10, overriding
``MemoryStore.list_facts``' own default of 50. That is a sane economy for
browsing, but it silently breaks one caller with a completeness mandate: the
dream job's nightly INCUMBENT REVIEW, which must judge *every* ``memory-entry``
fact because those facts are what MEMORY.md renders.

Observed live 2026-08-19: 13 memory-entry facts existed, the review listed 10,
and fid 968 — trust 0.5 and actually rendered in MEMORY.md — never reached the
agent. The prompt was subsequently changed to pass ``limit=50`` explicitly, but
a prompt string is not an enforcement mechanism for an invariant; a model that
omits the argument silently reverts to a truncated review. The tiebreak added
alongside it (``fact_id ASC``) makes the truncation deterministic rather than
complete, and because it is ascending, the entries dropped at a tie boundary
are the NEWEST promotions — exactly the ones most in need of review.

So the backstop belongs in the tool layer: memory-entry lists in full by
default, every other category keeps the small default that holds cron context
down. An explicit ``limit`` always wins, for every category.
"""

import json

import pytest

from plugins.memory.holographic import HolographicMemoryProvider


@pytest.fixture
def provider(tmp_path):
    p = HolographicMemoryProvider(
        config={"db_path": str(tmp_path / "memory_store.db"), "hrr_dim": 64}
    )
    p.initialize(session_id="test-session")
    yield p
    p.shutdown()


def _add(provider, n, category):
    for i in range(n):
        provider._handle_fact_store(
            {"action": "add", "content": f"{category} entry number {i}", "category": category}
        )


def _list(provider, **args):
    return json.loads(provider._handle_fact_store({"action": "list", **args}))


def test_memory_entry_lists_past_ten_without_an_explicit_limit(provider):
    """The incumbent-review mandate: every memory-entry fact, no argument."""
    _add(provider, 13, "memory-entry")

    result = _list(provider, category="memory-entry")

    assert result["count"] == 13, (
        "memory-entry truncated to a browsing default; the nightly incumbent "
        "review cannot judge what it is never shown"
    )


def test_other_categories_keep_the_small_default(provider):
    """Economy is preserved everywhere the completeness mandate does not apply."""
    _add(provider, 13, "lesson")

    assert _list(provider, category="lesson")["count"] == 10


def test_uncategorised_list_keeps_the_small_default(provider):
    _add(provider, 13, "lesson")

    assert _list(provider)["count"] == 10


def test_an_explicit_limit_wins_for_memory_entry(provider):
    """The default is a floor for one category, not an override of the caller."""
    _add(provider, 13, "memory-entry")

    assert _list(provider, category="memory-entry", limit=5)["count"] == 5


def test_an_explicit_limit_still_raises_other_categories(provider):
    _add(provider, 13, "lesson")

    assert _list(provider, category="lesson", limit=50)["count"] == 13

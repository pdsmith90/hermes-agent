"""The ledger also judges the prose (2026-10-06, cron review #141).

``_fact_write_ledger`` already states what a job really wrote. Until now the morning briefing — a
model — was the only thing comparing that block with the job's own report. ``_prose_check`` does it
in code: a created or removed fact the response never names is a hidden write (the 2026-08-15 shape,
fid 879 deleted and unmentioned), and a stored/demoted/removed claim beside a fid the store never
touched is a phantom write (the 08-15 "4 removals" shape). Updates the prose summarises by count are
notes, not problems.
"""

import sqlite3

import pytest

from cron.scheduler import _fact_write_ledger, _prose_check
from plugins.memory.holographic.store import MemoryStore

SESSION = "cron_testjob_20261006_031500"
OTHER = "cron_otherjob_20261006_032000"


def _rows(*ids):
    return [{"fact_id": i, "category": "lesson"} for i in ids]


class TestProseCheckPure:
    def test_clean_report_passes(self):
        resp = "Consolidate — 2026-10-06\nWrote fid=3325 (synthesis). Demoted predecessor fid=3273: 0.70 → 0.25."
        problems, notes = _prose_check(resp, _rows(3325), [], _rows(3273), [], [])
        assert problems == []
        assert notes == []

    def test_hidden_delete_is_a_problem(self):
        resp = "Noise pruned: 4 removals (871, 872, 893, 892)."
        problems, _ = _prose_check(resp, [], [], _rows(871, 872), _rows(879), [])
        assert any("removed 879 not named" in p for p in problems)

    def test_claimed_removal_the_store_never_saw_is_a_problem(self):
        resp = "Noise pruned: removed fid 893 and fid 892."
        problems, _ = _prose_check(resp, [], [], [], [], [])
        assert len(problems) == 1
        assert "893" in problems[0] and "892" in problems[0]
        assert "recorded no such write" in problems[0]

    def test_arrow_claim_counts_as_a_claim(self):
        resp = "research: 3274/3275/3276 → researched; follow-up open-q 3327"
        problems, _ = _prose_check(resp, _rows(3327), [], _rows(3274, 3275), [], [])
        assert len(problems) == 1
        assert "3276" in problems[0]

    @pytest.mark.parametrize("chain", ["3371→3375→3377", "fid 3371 -> fid 3375 -> fid 3377"])
    def test_an_id_chain_is_a_lineage_not_a_claim(self, chain):
        # 2026-10-07 consolidate: "Recent project facts (3371→3375→3377 …) are properly chained
        # sequential phases, not duplicates" was read as two arrow claims.
        resp = f"Synthesized fid=3389. Recent project facts ({chain}) are chained phases, not duplicates."
        problems, _ = _prose_check(resp, _rows(3389), [], _rows(3389), [], [])
        assert problems == []

    def test_a_two_id_arrow_is_still_a_claim(self):
        resp = "dream: promoted 3381 → memory-entry 3391"
        problems, _ = _prose_check(resp, _rows(3391), [], [], [], [])
        assert len(problems) == 1 and "3381" in problems[0]

    def test_slash_and_comma_lists_are_split(self):
        resp = "doc-paper-ingest: stored fids 3323/3324, updated fid 3272"
        problems, notes = _prose_check(resp, _rows(3323, 3324), [], _rows(3272), [], [])
        assert problems == [] and notes == []

    def test_unnamed_updates_are_a_note_not_a_problem(self):
        resp = "Claude Memory Distill — 3 notes → 3337/3338/3339; updated 8"
        problems, notes = _prose_check(resp, _rows(3337, 3338, 3339), [], _rows(1789, 1999, 2000, 2031, 2032, 3334, 3335, 3336), [], [])
        assert problems == []
        assert notes and notes[0].startswith("8 update(s) not named")

    def test_hidden_create_is_a_problem(self):
        problems, _ = _prose_check("[SILENT]", _rows(3400), [], [], [], [])
        assert problems == ["created 3400 not named in the response"]

    def test_the_briefing_may_describe_other_jobs_writes(self):
        resp = "- daily-review: stored fid 3320 (ACTIVITY), fid 3321 (lesson)\n- consolidate: demoted fid 3273 below floor"
        problems, _ = _prose_check(resp, [], [], [], [], [], job_name="morning-briefing")
        assert problems == []
        problems, _ = _prose_check(resp, [], [], [], [], [], job_name="dream-and-promote")
        assert problems and "recorded no such write" in problems[0]

    def test_a_number_inside_a_bigger_number_does_not_count_as_named(self):
        problems, _ = _prose_check("context 163840 tokens; 3840 chars", _rows(384), [], [], [], [])
        assert problems == ["created 384 not named in the response"]


@pytest.fixture(autouse=True)
def _clean_shared_registry():
    MemoryStore._shared.clear()
    yield
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    return _fact_write_ledger


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory_store.db")
    yield s
    s.close()


class TestProseCheckInTheLedger:
    def test_ok_line_when_prose_matches(self, ledger, store):
        made = store.add_fact("LESSON: kept", category="lesson", source_session=SESSION)
        block = ledger(SESSION, response=f"Stored fid={made}.", job_name="testjob")
        assert "- **Prose check:** ok" in block

    def test_problem_line_names_the_hidden_delete(self, ledger, store):
        doomed = store.add_fact("OPEN QUESTION: prune me", category="open-question", source_session=OTHER)
        store.remove_fact(doomed, changed_by=SESSION)
        block = ledger(SESSION, response="Queue hygiene: nothing to prune tonight.", job_name="testjob")
        assert f"- **Prose check:** PROBLEM — removed {doomed} not named in the response" in block

    def test_phantom_claim_with_no_writes_at_all(self, ledger, store):
        block = ledger(SESSION, response="FACTS PROMOTED (2): fids 901, 902 stored.", job_name="dream-and-promote")
        assert "No fact-store writes recorded" in block
        assert "PROBLEM" in block and "901" in block

    def test_no_response_means_no_line(self, ledger, store):
        store.add_fact("LESSON: kept", category="lesson", source_session=SESSION)
        assert "Prose check" not in ledger(SESSION)

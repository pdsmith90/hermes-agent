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

    @pytest.mark.parametrize("ref", ["already stored as fid 1335", "previously stored as fid 1335",
                                     "an already-stored fid 1335"])
    def test_an_already_stored_fid_is_a_reference_not_a_claim(self, ref):
        # 2026-10-08 doc-paper-ingest: "extends an earlier paper (already stored as fid 1335)" named an
        # August fact the run never touched, and was read as the claim "stored 1335".
        resp = f"Stored fids 3430/3431/3432. The new filter extends an earlier paper ({ref})."
        problems, _ = _prose_check(resp, _rows(3430, 3431, 3432), [], [], [], [])
        assert problems == []

    @pytest.mark.parametrize("cite", ["Chao 2015 → Guo et al. 2019", "Vernotte 2015 → GRACE-FO",
                                      "Chao et al. 2015 -> Guo 2019", "Swenson & Wahr 2006 → DDK"])
    def test_an_author_year_before_an_arrow_is_a_citation_not_a_claim(self, cite):
        # 2026-10-09 doc-paper-ingest: "The Chao 2015 → Guo et al. 2019 lineage" was read as the
        # claim "→ 2015" (the 10-04 dream report had "Vernotte 2015 → GRACE-FO" the same way).
        resp = f"Stored fids 3467/3468/3469. The {cite} lineage maps onto the methods paper."
        problems, _ = _prose_check(resp, _rows(3467, 3468, 3469), [], [], [], [])
        assert problems == []

    @pytest.mark.parametrize("claim", ["Promoted 1983 → memory-entry", "fid 2015 → demoted", "Fid 2015 -> retired"])
    def test_a_year_shaped_fid_before_an_arrow_is_still_a_claim(self, claim):
        problems, _ = _prose_check(f"dream: {claim}.", [], [], [], [], [])
        assert len(problems) == 1 and ("1983" in problems[0] or "2015" in problems[0])

    @pytest.mark.parametrize("lineage", [
        "sparked by a survey paper fid 3507 → Atlas; fid 3508 → fact_store retrieval",
        "the audit pair (fid 3523 → fid 3524)",
    ])
    def test_an_arrow_into_a_topic_or_another_fid_is_a_lineage_not_a_claim(self, lineage):
        # 2026-10-10 dream and 2026-10-11 consolidate (cron review #166): the arrow traced where an idea
        # came from or which fact superseded which; only an arrow into a status, category or trust
        # value states a write.
        resp = f"Stored fid 3510. The idea was {lineage}."
        problems, _ = _prose_check(resp, _rows(3510), [], [], [], [])
        assert problems == []

    @pytest.mark.parametrize("claim", ["fid 3326 → 0.55", "3274 → CONFIRMED", "fid 3301 -> parked",
                                       "AGED fid=3511 -> attempted-once", "Duplicate merge: fid 3517 → fid 3495"])
    def test_an_arrow_into_a_status_or_trust_is_still_a_claim(self, claim):
        problems, _ = _prose_check(f"research: {claim}.", [], [], [], [], [])
        assert len(problems) == 1

    @pytest.mark.parametrize("value", ["N=128→0.361", "N = 128 → 0.361", "λ≈500 -> 2500", "n≤128→0.4"])
    def test_a_number_after_a_math_operator_is_a_value_not_an_id(self, value):
        # 2026-10-14 memory-verify: "degree sweep numbers N=16→0.401 through N=128→0.361" quoted a
        # fact's numbers and was read as the claim "→ 128".
        resp = f"fid 2106: verified — all specifics ({value}) occur in the source note"
        problems, _ = _prose_check(resp, [], [], [], [], [], job_name="memory-verify")
        assert problems == []

    @pytest.mark.parametrize("value", ["TICK_MS 220→1000", "cache_ram: 8192 -> 0", "max_tokens 4096 → 8192"])
    def test_a_setting_before_a_value_change_is_not_an_id(self, value):
        # 2026-10-17 memory-verify, word for word apart from the setting under test
        resp = (f"fid 3756: verified — all specifics (opencode.db path, `/opencode`, {value}, "
                "v0.2.1 10-16 15:20) present in source note")
        problems, _ = _prose_check(resp, [], [], [], [], [], job_name="memory-verify")
        assert problems == []

    @pytest.mark.parametrize("claim", ["fact_id 3381 → 3391", "merged 3381 → 3391", "tick 220 → 1000"])
    def test_an_arrow_without_a_setting_name_is_still_a_claim(self, claim):
        problems, _ = _prose_check(claim, [], [], [], [], [])
        assert len(problems) == 1 and ("3381" in problems[0] or "220" in problems[0])

    @pytest.mark.parametrize("change", ["619→577 chars", "619→577 char truncation",
                                        "context grew 4096 -> 8192 tokens", "coverage 61→57%"])
    def test_a_size_or_percentage_change_is_not_an_id(self, change):
        # 2026-10-21 daily-review and consolidate-synthesize quoting a fact_store repair log line
        # ("619 chars -> 577"): "619→577 chars" was read as the claim "fid 619 →".
        resp = f"Stored fid 3900. The log shows {change} for one call."
        problems, _ = _prose_check(resp, _rows(3900), [], [], [], [])
        assert problems == []

    @pytest.mark.parametrize("claim", ["promoted 3381 → memory-entry 3391", "fid 3517 → fid 3495",
                                       "3381 → 3391 superseded", "fid 619 → 577 chars"])
    def test_an_arrow_that_only_looks_like_a_unit_change_is_still_a_claim(self, claim):
        problems, _ = _prose_check(f"dream: {claim}.", [], [], [], [], [])
        assert len(problems) == 1

    @pytest.mark.parametrize("claim", ["> 3274 → CONFIRMED", "research: 3274 → CONFIRMED (n = 12)"])
    def test_a_bare_id_after_a_blockquote_or_beside_a_value_is_still_a_claim(self, claim):
        problems, _ = _prose_check(claim, [], [], [], [], [])
        assert len(problems) == 1 and "3274" in problems[0]

    @pytest.mark.parametrize("quoted", [
        '"session\'s worktrees and branches are removed" not stated by cited fids 3631 or 3632',
        '“the duplicate was merged” per fid 3631',
        'the note says "stored fid 3631 and demoted fid 3632" verbatim',
    ])
    def test_a_verb_inside_quoted_text_is_a_quotation_not_a_claim(self, quoted):
        # 2026-10-14 memory-verify: a re-queue line quoted a fact's wording ("… branches are removed")
        # and the 25-char gap after the verb ran past the closing quote into "cited fids 3631".
        resp = f"host/ring-review-note (fid 3608): re-queued — {quoted}"
        problems, _ = _prose_check(resp, [], [], [], [], [], job_name="memory-verify")
        assert problems == []

    def test_the_whole_10_14_memory_verify_report_is_clean(self):
        resp = (
            "Memory Verify — generated 2026-10-14\n\n"
            "fid 2105: verified — flagged specific \"/0.5/0.93031\" IS in the source note line 66\n"
            "fid 2106: verified — all specifics (structural proof, degree sweep numbers N=16→0.401 through "
            "N=128→0.361, 60-sat crossover) occur in the source note\n\n"
            "host/ring-review-note-2026-10-12 (fid 3608): re-queued — \"session's worktrees and "
            "branches are removed\" not stated by cited fids 3631 or 3632\n")
        problems, _ = _prose_check(resp, [], [], [], [], [], job_name="memory-verify")
        assert problems == []

    @pytest.mark.parametrize("claim", ['Stored fid 3631 ("worktree cleanup lesson").',
                                       'removed fid 3631 — "stale"',
                                       'a stray " quote, then removed fid 3631'])
    def test_a_claim_outside_quotes_is_still_a_claim(self, claim):
        problems, _ = _prose_check(claim, [], [], [], [], [])
        assert len(problems) == 1 and "3631" in problems[0]

    def test_an_unclosed_quote_does_not_hide_the_rest_of_the_report(self):
        # a lone " (inches, a typo) must not blank the lines after it
        resp = 'Panel is 30" wide.\nremoved fid 3631 and fid 3632'
        problems, _ = _prose_check(resp, [], [], [], [], [])
        assert len(problems) == 1 and "3631" in problems[0] and "3632" in problems[0]

    @pytest.mark.parametrize("resp", [
        # 2026-10-16 doc-paper-ingest, word for word except the project name
        "Project doc check: Project-Alpha FINAL_STATUS.md confirms implementation is 100% complete. "
        "Already well-captured in stored facts (fids 755, 817) — no update needed.",
        # 2026-10-05 daily-review
        "**Lessons stored:** None new — fid 3257 (repeat_penalty fix) and fid 3252 (JSON parse) cover it",
        "removed nothing; fid 3631 stays",
    ])
    def test_a_verb_after_in_or_a_negated_verb_names_existing_facts(self, resp):
        problems, _ = _prose_check(resp, [], [], [], [], [])
        assert problems == []

    @pytest.mark.parametrize("claim", [
        "bumped fid 3316 (shared-scratchpad lesson already stored)",  # 2026-10-06: a real phantom bump
        "stored facts 3740, 3741 from tonight's papers",
        "Fixed: stored fid 3740 in facts about GRACE",
    ])
    def test_a_plain_or_attributive_free_verb_is_still_a_claim(self, claim):
        problems, _ = _prose_check(claim, [], [], [], [], [])
        assert len(problems) == 1 and "recorded no such write" in problems[0]

    def test_a_bare_stored_as_fid_is_still_a_claim(self):
        problems, _ = _prose_check("The earlier paper is stored as fid 1335.", [], [], [], [], [])
        assert len(problems) == 1 and "stored 1335" in problems[0]

    def test_slash_and_comma_lists_are_split(self):
        resp = "doc-paper-ingest: stored fids 3323/3324, updated fid 3272"
        problems, notes = _prose_check(resp, _rows(3323, 3324), [], _rows(3272), [], [])
        assert problems == [] and notes == []

    def test_a_verb_before_a_noun_and_of_names_the_source_fact(self):
        """2026-10-18 consolidate (cron review #173): "fid 3472 (memory-entry, promoted copy of fid 3464)
        updated to reference the full lesson at fid 3464" — 3464 is what 3472 was copied FROM, and the
        only write to it that night was another job's. A property of the fact itself is the fact:
        "bumped trust of fid 3464" still claims a write to 3464."""
        resp = ("- **Hook guard memory-entry:** fid 3472 (memory-entry, promoted copy of fid 3464) updated "
                "to reference the full lesson at fid 3464 instead of carrying a stale trust stamp.")
        problems, _ = _prose_check(resp, [], [], _rows(3472), [], [])
        assert problems == []
        problems, _ = _prose_check("Bumped trust of fid 3464 to 1.0.", [], [], [], [], [])
        assert any("bumped 3464" in p for p in problems)

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

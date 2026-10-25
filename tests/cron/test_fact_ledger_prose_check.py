"""The ledger also judges the prose (2026-10-06, cron review #141).

``_fact_write_ledger`` already states what a job really wrote. Until now the morning briefing — a
model — was the only thing comparing that block with the job's own report. ``_prose_check`` does it
in code: a created or removed fact the response never names is a hidden write (the 2026-08-15 shape,
fid 879 deleted and unmentioned), and a stored/demoted/removed claim beside a fid the store never
touched is a phantom write (the 08-15 "4 removals" shape). Updates the prose summarises by count are
notes, not problems.
"""

import sqlite3

import re

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

    def test_bulleted_pairs_under_superseded_header_are_not_claims(self):
        # 2026-10-20 consolidate (cron review #201): bullet items under a superseded header
        resp = """
**STEP 4 — Duplicate merges:** Superseded chains already handled by store:
- 3841 → 3843 → 3844 (pipeline gapfill flank, three-stage correction)
- 3837 → 3839 (spline push status)

**STEP 4b — Cluster synthesis:** Stored fid 3856.
"""
        problems, _ = _prose_check(resp, _rows(3856), [], [], [], [])
        assert problems == []

    @pytest.mark.parametrize("resp, ids", [
        # 2026-10-23 review: bf36e4e768 blanked every bullet after any line naming the word, across
        # blank and **bold** lines, so these fabricated arrow claims passed with an empty ledger.
        ("Checked supersession; results:\n- 3990 → researched\n- fid 3991 → promoted", {3990, 3991}),
        ("Superseded:\n- 3837 → 3839\n\n**Verdicts**\n- 3990 → researched", {3990}),
        ("Reviewed 4 pairs:\n- fid 3990 → merged\n- 3991 → 3992", {3990}),
        ("Superseded:\n- 3837 → 3839\n---\n- 3990 → lesson", {3990}),
        ("fid 3800 is superseded by 3801 (no action).\nChanges this run:\n- 3990 → memory-entry", {3990}),
        ("Superseded chains:\n- 3837 → 3839 (fid 3990 → memory-entry)", {3990}),
    ])
    def test_a_lineage_header_blanks_only_the_pairs_its_bullets_open_with(self, resp, ids):
        problems, _ = _prose_check(resp, [], [], [], [], [])
        assert len(problems) == 1, problems
        claimed = {int(n) for n in re.findall(r"→ (\d+)", problems[0])}
        assert claimed == ids, problems

    def test_pairs_under_a_markdown_lineage_heading_are_not_claims(self):
        resp = "### Superseded pairs\n- 3125 → 3126\n- fid 3156 → fid 3157 (duplicate wording)\n"
        assert _prose_check(resp, [], [], [], [], [])[0] == []

    @pytest.mark.parametrize("claim, fid", [
        # 2026-10-23 review: only derivation nouns name a source fact; any other noun is the fact
        ("Merged duplicates of fid 3990 into fid 3991.", 3990),
        ("Bumped confidence of fid 3990.", 3990),
        ("Updated both of fids 3990 and 3991.", 3991),
        # a negation inside an aside qualifies the write instead of denying it
        ("Stored (no duplicate) fid 3990.", 3990),
        ("Created, not merged, fid 3990.", 3990),
    ])
    def test_a_non_derivation_noun_or_a_negated_aside_is_still_a_claim(self, claim, fid):
        problems, _ = _prose_check(claim, [], [], [], [], [])
        assert len(problems) == 1 and str(fid) in problems[0], problems

    @pytest.mark.parametrize("resp", [
        # 2026-10-23 review: the verb as an adjective for an existing fact
        "Checked against stored facts; fid 3832 still holds.",
        "Topic covered by stored fid 3832 from 10-19; no new write.",
        "originally created fid 3832 on 10-19, nothing new tonight",
        "Deleted copies of fid 3990 were other facts' duplicates",
    ])
    def test_an_adjectival_verb_names_an_existing_fact(self, resp):
        assert _prose_check(resp, [], [], [], [], [])[0] == []

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
        # 2026-10-23 consolidate-synthesize: the negation sits right BEFORE the verb (cron review #198)
        '- fid 3853 (3d, provider-health/Gemini attribution): no stored fact answers it; '
        'fid 3832 confirms "not done" → left open.',
        "never created fid 3833; not updated fid 3834",
    ])
    def test_a_verb_after_in_or_a_negated_verb_names_existing_facts(self, resp):
        problems, _ = _prose_check(resp, [], [], [], [], [])
        assert problems == []

    @pytest.mark.parametrize("claim", [
        "bumped fid 3316 (shared-scratchpad lesson already stored)",  # 2026-10-06: a real phantom bump
        "stored facts 3740, 3741 from tonight's papers",
        "Fixed: stored fid 3740 in facts about GRACE",
        "No duplicates; stored fid 3741",
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

    @pytest.mark.parametrize("ref", ["to reference", "to cite", "to point at", "to point to",
                                     "now referencing", "citing", "pointing to"])
    def test_a_fid_after_a_reference_phrase_is_the_target_not_the_object(self, ref):
        # 2026-10-25 consolidate-synthesize (cron review #215), word for word apart from the phrase
        resp = ("- **Stale notes corrected**: fids 4005 and 4006 both contained \"likely hallucinated\" "
                f"notes. Updated both {ref} fid 3936 as the existing paper.")
        problems, _ = _prose_check(resp, [], [], _rows(4005, 4006), [], [])
        assert problems == []

    @pytest.mark.parametrize("claim, ledger, ids", [
        ("Updated fid 3936 to reference fid 4005.", [], {3936}),
        ("Updated 3936 to reference fid 4005.", [], {3936}),
        ("Updated fids 4005 and 4006 to reference fid 3936.", [4005], {4006}),
        ("Updated 4005 and 4006 to reference fid 3936.", [4005], {4006}),
        ("Updated both to reference fid 3936; stored fid 4100.", [], {4100}),
    ])
    def test_the_objects_before_a_reference_phrase_are_still_claims(self, claim, ledger, ids):
        problems, _ = _prose_check(claim, [], [], _rows(*ledger), [], [])
        assert len(problems) == 1, problems
        assert {int(n) for n in re.findall(r"(?:updated|stored) (\d+)", problems[0])} == ids, problems

    @pytest.mark.parametrize("narration", [
        # 2026-10-25 weekly-trace-mining (cron review #215): fid 3997 was another session's write (10-23)
        "Three last-error traces checked: docs-sync (a7d41c09e2f5) finished — merged via the SSH-push "
        "workaround; corpus-build (3f2a9c1d7b04) recovered from the approval block and stored fact 3997.",
        "- corpus-build (3f2a9c1d7b04) stored fact 3997",
        "Trace 3f2a9c1d7b04 recovered and stored fid 3997.",
    ])
    def test_a_traced_session_as_the_subject_narrates_its_write(self, narration):
        problems, _ = _prose_check(narration, [], [], [], [], [], job_name="weekly-trace-mining")
        assert problems == []

    @pytest.mark.parametrize("claim", [
        "Stored fact 4100.",
        "Checked corpus-build (3f2a9c1d7b04) and stored fact 4100.",
        "corpus-build (3f2a9c1d7b04) failed and I stored fact 4100.",
        "corpus-build (3f2a9c1d7b04) failed, then stored fact 4100.",
        "trace 3f2a9c1d7b04: stored fact 4100",
        "In trace 3f2a9c1d7b04 I stored fact 4100.",
        "Session 3f2a9c1d7b04 stored fact 3997 and we updated fid 4100.",
        "corpus-build (3f2a9c1d7b04) is clean; stored fact 4100.",
    ])
    def test_the_runs_own_write_beside_a_traced_session_is_still_a_claim(self, claim):
        problems, _ = _prose_check(claim, [], [], [], [], [], job_name="weekly-trace-mining")
        assert len(problems) == 1 and "4100" in problems[0] and "3997" not in problems[0], problems

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

    @staticmethod
    def _claimed(problems):
        return {int(n) for p in problems for n in re.findall(r"\b[a-z-]+ (\d+)", p.split(" but ")[0])}

    @pytest.mark.parametrize("claim", [
        # the run's own write after a traced session's clause (arrow, spaced hyphen, aside, "then"/"so")
        "- corpus-build (3f2a9c1d7b04) recovered from the approval block -> stored lesson fid 4100",
        "- corpus-build (3f2a9c1d7b04) recovered from the approval block \u2192 stored lesson fid 4100",
        "- corpus-build (3f2a9c1d7b04) hit the approval block again - stored lesson fid 4100",
        "- corpus-build (3f2a9c1d7b04) failed again (stored lesson fid 4100)",
        "corpus-build (3f2a9c1d7b04) failed then stored fact 4100 as a lesson.",
        "corpus-build (3f2a9c1d7b04) failed so stored fact 4100 as a lesson.",
        # a short commit hash or hex word in parentheses is not a traced session
        "Commit (a6b475d694) updated fid 4100 per the fix.",
        "Cleanup (deadbeef01) removed fid 4100.",
        "- **corpus-build (3f2a9c1d7b04)**: stored fact 4100",
        "> corpus-build (3f2a9c1d7b04) failed and stored fact 4100",
        "He said \"stored fact 3997\" but we stored fact 4100.",
    ])
    def test_a_traced_session_lookalike_does_not_hide_the_runs_write(self, claim):
        problems, _ = _prose_check(claim, [], [], [], [], [], job_name="weekly-trace-mining")
        assert 4100 in self._claimed(problems), problems

    @pytest.mark.parametrize("claim, rows, ids", [
        # a reference word is not a reference target unless a fid follows it
        ("Added a lesson citing it as fid 4100.", [], {4100}),
        ("Stored lesson citing #215: fid 4100.", [], {4100}),
        ("Updated the cross-referencing fact 4100.", [], {4100}),
        ("Updated 4100 to point at fid 3936.", [], {4100}),
        ("Merged 4100/4101 to cite fid 3936.", [], {4100, 4101}),
        ("Bumped trust of 4100 citing fid 3936.", [], {4100}),
        ("Updated fids 4005, 4100 to reference fid 3936.", [4005], {4100}),
        ("Updated both to reference fid 3936; also stored fact 4100.", [4005, 4006], {4100}),
        ("Bumped fid 3316 to 0.7.", [], {3316}),
    ])
    def test_a_reference_lookalike_does_not_hide_a_write(self, claim, rows, ids):
        problems, _ = _prose_check(claim, [], [], _rows(*rows), [], [])
        assert ids <= self._claimed(problems), problems
        assert 3936 not in self._claimed(problems), problems


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

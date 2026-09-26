"""The cron report gate: one follow-up turn when the final text is not the report.

A cron run ends on the first text turn with no tool call, whatever it says.
On 2026-09-06 experiment-design ended 25 minutes of real work on "Let me check
what's happening and fix all four facts" with no report written, and
doc-paper-ingest ended on a self-assessment essay after reading three papers
and storing nothing (incidents #42 and #43). Every prompt mandates a report
header; a job's ``report_marker`` names it and ``run_job`` grants exactly one
follow-up turn, in the same conversation, when it is absent.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import (
    _normalize_report_text,
    _enforce_morning_briefing_coverage,
    _morning_briefing_manifest,
    _report_gate_missing_marker,
    _strip_report_preamble,
    run_job,
    run_one_job as sched_run_one_job,
)


def _manifest_prompt(manifest):
    return (
        "prefix\n@@HERMES_CRON_MANIFEST_JSON_BEGIN@@\n"
        + json.dumps(manifest)
        + "\n@@HERMES_CRON_MANIFEST_JSON_END@@\nsuffix"
    )


def _complete_manifest():
    return {
        "version": 1,
        "date": "2026-09-25",
        "execution_ledger_available": True,
        "executions": [
            {"job_id": "daily1234567", "name": "daily-review", "status": "completed",
             "delivery": "suppressed", "issue": None,
             "report_files": ["cron/output/daily1234567/2026-09-25_02-13-38.md"]},
        ],
        "report_files": [
            {"job_id": "daily1234567", "name": "daily-review",
             "path": "cron/output/daily1234567/2026-09-25_02-13-38.md"},
        ],
        "unmatched_report_files": [],
    }


class TestMorningBriefingCoverageGate:
    def test_complete_manifest_response_passes_unchanged(self):
        manifest = {
            "version": 1,
            "date": "2026-09-25",
            "execution_ledger_available": True,
            "executions": [
                {"job_id": "daily1234567", "name": "daily-review", "status": "completed",
                 "delivery": "suppressed", "issue": None,
                 "report_files": ["cron/output/daily1234567/2026-09-25_02-13-38.md"]},
            ],
            "report_files": [
                {"job_id": "daily1234567", "name": "daily-review",
                 "path": "cron/output/daily1234567/2026-09-25_02-13-38.md"},
            ],
            "unmatched_report_files": [],
        }
        response = "Morning Briefing — 2026-09-25\n\n**What Happened Overnight**\n- daily-review: ACTIVITY recorded.\n"

        assert _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, _manifest_prompt(manifest), response
        ) == response

    def test_coverage_checks_the_last_report_draft_that_will_be_delivered(self):
        manifest = {
            "version": 1,
            "date": "2026-09-25",
            "execution_ledger_available": True,
            "executions": [
                {"job_id": "invest123456", "name": "invest-daily-brief", "status": "failed",
                 "delivery": "delivered", "issue": "provider timeout", "report_files": []},
            ],
            "report_files": [],
            "unmatched_report_files": [],
        }
        response = (
            "**Morning Briefing — 2026-09-25**\n\n**What Happened Overnight**\n"
            "- invest-daily-brief: failed on a provider timeout.\n\n"
            "Wait — I should tighten this.\n"
            "**Morning Briefing — 2026-09-25**\n\n**What Happened Overnight**\n"
            "- daily-review: ACTIVITY saved.\n- arxiv-research-scan: [SILENT].\n"
        )

        result = _enforce_morning_briefing_coverage(
            {"name": "morning-briefing", "report_marker": "Morning Briefing"},
            _manifest_prompt(manifest), response)

        assert "invest-daily-brief: status=failed" in result
        assert "omitted from the summary" in result

    def test_appends_unnamed_failure_but_not_unnamed_routine_run(self):
        manifest = {
            "version": 1,
            "date": "2026-09-25",
            "execution_ledger_available": True,
            "executions": [
                {"job_id": "invest123456", "name": "invest-daily-brief", "status": "completed",
                 "delivery": "suppressed", "issue": None,
                 "report_files": ["cron/output/invest123456/2026-09-25_05-08-22.md"]},
                {"job_id": "distill123456", "name": "claude-memory-distill", "status": "failed",
                 "delivery": "delivered", "issue": "prompt-scanner block", "report_files": []},
            ],
            "report_files": [
                {"job_id": "invest123456", "name": "invest-daily-brief",
                 "path": "cron/output/invest123456/2026-09-25_05-08-22.md"},
            ],
            "unmatched_report_files": [],
        }
        response = "Morning Briefing — 2026-09-25\n\n**Store healthy.**\n"

        result = _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, _manifest_prompt(manifest), response)

        assert result.startswith(response)
        appended = result[len(response):]
        assert "invest-daily-brief" not in appended
        gap_lines = [line for line in appended.splitlines() if line.startswith("- ")]
        assert gap_lines == [
            "- claude-memory-distill: status=failed, issue=prompt-scanner block, "
            "delivery=delivered, reports=none; omitted from the summary"
        ]

    def test_completed_runs_with_clean_delivery_pass_even_when_unnamed(self):
        # 2026-09-24 replay: invest-daily-brief, render-memory and memory-index-heal all
        # completed with their reports saved; the briefing named none of them.
        executions, report_files = [], []
        for job_id, name, delivery in (
            ("invest123456", "invest-daily-brief", "suppressed"),
            ("render123456", "render-memory", None),
            ("heal12345678", "memory-index-heal", "delivered"),
            ("daily1234567", "daily-review", "suppressed"),
        ):
            path = f"cron/output/{job_id}/2026-09-24_05-00-00.md"
            executions.append({"job_id": job_id, "name": name, "status": "completed",
                               "delivery": delivery, "issue": None, "report_files": [path]})
            report_files.append({"job_id": job_id, "name": name, "path": path})
        manifest = {
            "version": 1,
            "date": "2026-09-24",
            "execution_ledger_available": True,
            "executions": executions,
            "report_files": report_files,
            "unmatched_report_files": [],
        }
        response = "Morning Briefing — 2026-09-24\n\n**What Happened Overnight**\n- daily-review: ACTIVITY recorded.\n"

        assert _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, _manifest_prompt(manifest), response
        ) == response

    def test_named_failure_without_an_error_word_is_not_flagged(self):
        manifest = {
            "version": 1,
            "date": "2026-09-25",
            "execution_ledger_available": True,
            "executions": [
                {"job_id": "distill123456", "name": "claude-memory-distill", "status": "failed",
                 "delivery": "delivered", "issue": "prompt-scanner block", "report_files": []},
            ],
            "report_files": [],
            "unmatched_report_files": [],
        }
        response = "Morning Briefing — 2026-09-25\n\n- claude-memory-distill: nothing new to distill.\n"

        result = _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, _manifest_prompt(manifest), response)

        assert "claude-memory-distill: status=failed" in result
        assert "; not flagged" in result

    def test_named_failure_with_an_error_word_passes(self):
        manifest = {
            "version": 1,
            "date": "2026-09-25",
            "execution_ledger_available": True,
            "executions": [
                {"job_id": "distill123456", "name": "claude-memory-distill", "status": "failed",
                 "delivery": "delivered", "issue": "prompt-scanner block", "report_files": []},
            ],
            "report_files": [],
            "unmatched_report_files": [],
        }
        response = "Morning Briefing — 2026-09-25\n\n- claude-memory-distill: BLOCKED by the prompt scanner.\n"

        assert _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, _manifest_prompt(manifest), response) == response

    def test_flags_delivery_failure_on_a_completed_execution(self):
        manifest = {
            "version": 1,
            "date": "2026-09-25",
            "execution_ledger_available": True,
            "executions": [
                {"job_id": "job123456789", "name": "daily-trace-mining", "status": "completed",
                 "delivery": "failed", "issue": None, "report_files": ["cron/output/job123456789/r.md"]},
            ],
            "report_files": [
                {"job_id": "job123456789", "name": "daily-trace-mining", "path": "cron/output/job123456789/r.md"},
            ],
            "unmatched_report_files": [],
        }
        response = "Morning Briefing — 2026-09-25\n\n**What Happened Overnight**\n- daily-trace-mining: no notable pattern.\n"

        result = _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, _manifest_prompt(manifest), response)

        assert "delivery=failed" in result

    def test_repeat_run_with_every_report_saved_needs_no_repeat_wording(self):
        manifest = {
            "version": 1,
            "date": "2026-09-25",
            "execution_ledger_available": True,
            "executions": [
                {"job_id": "trace12345678", "name": "daily-trace-mining", "status": "completed",
                 "delivery": "suppressed", "issue": None,
                 "report_files": ["cron/output/trace12345678/a.md", "cron/output/trace12345678/b.md"]},
                {"job_id": "trace12345678", "name": "daily-trace-mining", "status": "completed",
                 "delivery": "delivered", "issue": None,
                 "report_files": ["cron/output/trace12345678/a.md", "cron/output/trace12345678/b.md"]},
            ],
            "report_files": [
                {"job_id": "trace12345678", "name": "daily-trace-mining", "path": "cron/output/trace12345678/a.md"},
                {"job_id": "trace12345678", "name": "daily-trace-mining", "path": "cron/output/trace12345678/b.md"},
            ],
            "unmatched_report_files": [],
        }
        response = "Morning Briefing — 2026-09-25\n- daily-trace-mining: one trace was quiet.\n"

        assert _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, _manifest_prompt(manifest), response) == response

    def test_repeat_run_with_a_missing_report_is_flagged(self):
        manifest = {
            "version": 1,
            "date": "2026-09-25",
            "execution_ledger_available": True,
            "executions": [
                {"job_id": "trace12345678", "name": "daily-trace-mining", "status": "completed",
                 "delivery": "suppressed", "issue": None, "report_files": ["cron/output/trace12345678/a.md"]},
                {"job_id": "trace12345678", "name": "daily-trace-mining", "status": "completed",
                 "delivery": "delivered", "issue": None, "report_files": []},
            ],
            "report_files": [
                {"job_id": "trace12345678", "name": "daily-trace-mining", "path": "cron/output/trace12345678/a.md"},
            ],
            "unmatched_report_files": [],
        }
        response = "Morning Briefing — 2026-09-25\n"

        result = _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, _manifest_prompt(manifest), response)

        assert "daily-trace-mining: 2 execution(s) but only 1 report file(s) were recorded" in result
        assert "ran 2x" not in result

    def test_unnamed_completed_run_without_a_report_is_flagged(self):
        manifest = {
            "version": 1,
            "date": "2026-09-25",
            "execution_ledger_available": True,
            "executions": [
                {"job_id": "render123456", "name": "render-memory", "status": "completed",
                 "delivery": None, "issue": None, "report_files": []},
            ],
            "report_files": [],
            "unmatched_report_files": [],
        }

        result = _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, _manifest_prompt(manifest), "Morning Briefing — 2026-09-25\n")

        assert "render-memory: completed execution has no saved report file" in result

    def test_unnamed_delivery_failure_is_flagged(self):
        manifest = {
            "version": 1,
            "date": "2026-09-25",
            "execution_ledger_available": True,
            "executions": [
                {"job_id": "job123456789", "name": "daily-trace-mining", "status": "completed",
                 "delivery": "not_configured", "issue": None, "report_files": ["cron/output/job123456789/r.md"]},
            ],
            "report_files": [
                {"job_id": "job123456789", "name": "daily-trace-mining", "path": "cron/output/job123456789/r.md"},
            ],
            "unmatched_report_files": [],
        }

        result = _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, _manifest_prompt(manifest), "Morning Briefing — 2026-09-25\n")

        assert "daily-trace-mining: delivery=not_configured; job status=completed, but omitted from the summary" in result

    def test_silence_is_preserved_only_for_an_empty_complete_manifest(self):
        manifest = {
            "version": 1,
            "date": "2026-09-25",
            "execution_ledger_available": True,
            "executions": [], "report_files": [], "unmatched_report_files": [],
        }
        assert _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, _manifest_prompt(manifest), "[SILENT]") == "[SILENT]"

    def test_missing_manifest_cannot_certify_silence(self):
        result = _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, "no manifest", "[SILENT]")
        assert result.startswith("Morning Briefing —")
        assert "inventory could not be verified" in result

    def test_manifest_is_found_when_the_job_prompt_mentions_the_marker(self):
        # 2026-09-26 07:30: STEP 1 of the job prompt quoted `@@HERMES_CRON_MANIFEST_JSON_BEGIN@@`
        # after the real block, the assembled prompt held two BEGIN markers, the parser returned
        # None, and the phone got "the execution manifest is unavailable" instead of the check.
        manifest = _complete_manifest()
        prompt = _manifest_prompt(manifest) + (
            "\nSTEP 1 — The `@@HERMES_CRON_MANIFEST_JSON_BEGIN@@` block is validator input; "
            "do not copy its markers or JSON into Telegram."
        )
        assert _morning_briefing_manifest(prompt) == manifest
        response = "Morning Briefing — 2026-09-25\n\n- daily-review: ACTIVITY recorded.\n"
        assert _enforce_morning_briefing_coverage(
            {"name": "morning-briefing"}, prompt, response) == response

    def test_manifest_is_found_when_the_mention_precedes_the_block(self):
        manifest = _complete_manifest()
        prompt = "Read the @@HERMES_CRON_MANIFEST_JSON_BEGIN@@ block below.\n" + _manifest_prompt(manifest)
        assert _morning_briefing_manifest(prompt) == manifest

    def test_a_marker_line_without_a_valid_block_is_skipped(self):
        manifest = _complete_manifest()
        prompt = (
            "@@HERMES_CRON_MANIFEST_JSON_BEGIN@@\nnot json\n@@HERMES_CRON_MANIFEST_JSON_END@@\n"
            + _manifest_prompt(manifest)
        )
        assert _morning_briefing_manifest(prompt) == manifest

    def test_a_prose_mention_alone_is_no_manifest(self):
        assert _morning_briefing_manifest(
            "prose mentions @@HERMES_CRON_MANIFEST_JSON_BEGIN@@ and nothing else") is None



class TestMarkerHelper:
    def test_no_marker_configured_never_fires(self):
        assert _report_gate_missing_marker({"prompt": "x"}, "anything") is None

    def test_no_agent_jobs_are_exempt(self):
        job = {"report_marker": "Render", "no_agent": True}
        assert _report_gate_missing_marker(job, "header-only file") is None

    def test_missing_header_fires(self):
        job = {"report_marker": "Experiment design"}
        text = ("The fact_store is keeping `needs-experiment` despite my attempts to "
                "remove it. Let me check what's happening and fix all four facts.")
        assert _report_gate_missing_marker(job, text) == "Experiment design"

    def test_header_anywhere_in_the_body_passes(self):
        # Real reports open with narration; the header comes a line later.
        job = {"report_marker": "Dream & Promote"}
        text = "All source fids verified.\n\n---\n\n**Dream & Promote — 2026-09-06**\n..."
        assert _report_gate_missing_marker(job, text) is None

    @pytest.mark.parametrize("dash", ["—", "–", "-"])
    def test_dash_variants_and_case_are_folded(self, dash):
        job = {"report_marker": "Research —"}
        assert _report_gate_missing_marker(job, f"## RESEARCH {dash} 2026-09-07") is None

    def test_silence_bypasses_the_gate(self):
        job = {"report_marker": "Arxiv Research Scan"}
        assert _report_gate_missing_marker(job, "[SILENT]") is None
        assert _report_gate_missing_marker(job, "") is None

    def test_normalizer(self):
        assert _normalize_report_text("  Doc &  Paper\nIngest — x") == "doc & paper ingest - x"


class TestPreambleStrip:
    """Delivery drops narration above the report header; the file keeps it."""

    JOB = {"report_marker": "Morning Briefing"}

    def test_narration_above_the_header_is_dropped(self):
        text = ("All 13 files read, ledgers cross-checked. Writing the briefing now.\n\n"
                "### Morning Briefing — 2026-09-11\n\n**13 jobs ran overnight.**\n")
        assert _strip_report_preamble(self.JOB, text) == (
            "### Morning Briefing — 2026-09-11\n\n**13 jobs ran overnight.**\n")

    def test_header_first_is_untouched(self):
        text = "Morning Briefing — 2026-09-12\n\nStore healthy.\n"
        assert _strip_report_preamble(self.JOB, text) == text

    def test_a_line_that_starts_with_the_marker_beats_a_mention(self):
        text = ("Now writing the Morning Briefing report.\n"
                "**Morning Briefing — 2026-09-12**\n- body\n")
        assert _strip_report_preamble(self.JOB, text) == (
            "**Morning Briefing — 2026-09-12**\n- body\n")

    def test_marker_only_mentioned_mid_line_still_cuts_there(self):
        text = "notes\nThe Morning Briefing for today follows:\n- body\n"
        assert _strip_report_preamble(self.JOB, text) == (
            "The Morning Briefing for today follows:\n- body\n")

    def test_no_marker_anywhere_leaves_the_text_alone(self):
        assert _strip_report_preamble(self.JOB, "Designed both. Done.") == "Designed both. Done."

    def test_unconfigured_no_agent_and_silence_are_left_alone(self):
        assert _strip_report_preamble({}, "x\nMorning Briefing\n") == "x\nMorning Briefing\n"
        assert _strip_report_preamble({**self.JOB, "no_agent": True}, "x\ny") == "x\ny"
        assert _strip_report_preamble(self.JOB, "[SILENT]") == "[SILENT]"

    def test_second_draft_wins_over_narration_and_the_first_draft(self):
        # 2026-09-23: daily-trace-mining delivered draft 1 + "Wait — let me reconsider…" + draft 2.
        job = {"report_marker": "Daily Trace Mining"}
        text = ("**Daily Trace Mining — 2026-09-23**\n\nRecurring patterns:\n- draft one\n\n"
                "Wait — I need to double check the coverage.\nLet me write it.\n"
                "**Daily Trace Mining — 2026-09-23**\n\nRecurring patterns:\n- draft two\n\n"
                "Coverage: partial\n")
        assert _strip_report_preamble(job, text) == (
            "**Daily Trace Mining — 2026-09-23**\n\nRecurring patterns:\n- draft two\n\n"
            "Coverage: partial\n")

    def test_a_trailing_header_one_liner_is_not_a_draft(self):
        text = "Morning Briefing — d\n- a\n- b\nMorning Briefing done.\n"
        assert _strip_report_preamble(self.JOB, text) == text


_PROVIDER = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "custom",
    "api_mode": "chat_completions",
}


def _run(tmp_path, job, responses):
    fake_db = MagicMock()
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("dotenv.load_dotenv"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value=_PROVIDER), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        agent = MagicMock()
        agent.run_conversation.side_effect = list(responses)
        mock_agent_cls.return_value = agent
        return run_job(job), agent


FIRST_MESSAGES = [
    {"role": "user", "content": "design experiments"},
    {"role": "assistant", "content": "Let me check what's happening and fix all four facts."},
]
NARRATION = {"final_response": "Let me check what's happening and fix all four facts.",
             "messages": FIRST_MESSAGES}
REPORT = {"final_response": "Experiment design — 2026-09-06\nBacklog: eligible 14 | "
                            "Designed: 1564, 1565 | Retired: 1080, 1151"}


class TestRunJobReportGate:
    JOB = {"id": "gate-job", "name": "experiment-design", "prompt": "design experiments",
           "report_marker": "Experiment design"}

    def test_follow_up_turn_produces_the_report(self, tmp_path):
        (success, output, final_response, error), agent = _run(
            tmp_path, self.JOB, [NARRATION, REPORT])
        assert success is True and error is None
        assert final_response == REPORT["final_response"]
        assert agent.run_conversation.call_count == 2
        follow_up = agent.run_conversation.call_args_list[1]
        # Same conversation: the first turn's messages are carried over, and
        # the nudge names the missing header.
        assert follow_up.kwargs["conversation_history"] == FIRST_MESSAGES
        assert "REPORT GATE" in follow_up.args[0]
        assert "Experiment design" in follow_up.args[0]
        assert "**Report gate:** fired" in output
        assert "the follow-up produced the report" in output
        assert output.index("**Report gate:**") < output.index("## Prompt")

    def test_morning_briefing_coverage_gate_runs_before_the_output_is_saved(self, tmp_path):
        manifest = {
            "version": 1,
            "date": "2026-09-25",
            "execution_ledger_available": True,
            "executions": [
                {"job_id": "invest123456", "name": "invest-daily-brief", "status": "failed",
                 "delivery": "delivered", "issue": "provider timeout", "report_files": []},
            ],
            "report_files": [],
            "unmatched_report_files": [],
        }
        job = {"id": "morning", "name": "morning-briefing", "prompt": "compile", "report_marker": "Morning Briefing"}
        response = {"final_response": "Morning Briefing — 2026-09-25\n\n**Quiet night.**"}
        with patch("cron.scheduler._prepare_job_prompt", return_value=(None, _manifest_prompt(manifest))):
            (success, output, final_response, error), _agent = _run(tmp_path, job, [response])

        assert success is True and error is None
        assert "invest-daily-brief" in final_response
        assert "Scheduler-verified coverage gaps" in output

    def test_report_on_the_first_turn_is_left_alone(self, tmp_path):
        (success, output, final_response, _), agent = _run(tmp_path, self.JOB, [REPORT])
        assert success is True
        assert final_response == REPORT["final_response"]
        assert agent.run_conversation.call_count == 1
        assert "Report gate" not in output

    def test_silent_run_is_left_alone(self, tmp_path):
        (success, output, final_response, _), agent = _run(
            tmp_path, self.JOB, [{"final_response": "[SILENT]"}])
        assert success is True
        assert agent.run_conversation.call_count == 1
        assert "Report gate" not in output

    def test_job_without_marker_is_left_alone(self, tmp_path):
        job = {k: v for k, v in self.JOB.items() if k != "report_marker"}
        (success, _, final_response, _), agent = _run(tmp_path, job, [NARRATION])
        assert success is True
        assert final_response == NARRATION["final_response"]
        assert agent.run_conversation.call_count == 1

    def test_empty_follow_up_keeps_the_first_response(self, tmp_path):
        (success, output, final_response, _), agent = _run(
            tmp_path, self.JOB, [NARRATION, {"final_response": ""}])
        assert success is True
        assert final_response == NARRATION["final_response"]
        assert "first response kept" in output

    def test_failed_follow_up_keeps_the_first_response(self, tmp_path):
        (success, output, final_response, _), agent = _run(
            tmp_path, self.JOB, [NARRATION, RuntimeError("provider down")])
        assert success is True
        assert final_response == NARRATION["final_response"]
        assert "the follow-up failed (RuntimeError)" in output

    @pytest.mark.parametrize("silence", ["[SILENT]", "SILENT", "  [silent]  "])
    def test_silent_follow_up_keeps_the_first_response(self, tmp_path, silence):
        # 2026-09-21 retrieval-audit: a 13/13 PASS report lacking only the literal header,
        # then "[SILENT]" to the nudge — the sentinel bypasses the marker check and had
        # replaced the real report with silence in the output file and the briefing.
        (success, output, final_response, _), agent = _run(
            tmp_path, self.JOB, [NARRATION, {"final_response": silence}])
        assert success is True
        assert final_response == NARRATION["final_response"]
        assert "the follow-up answered [SILENT] — first response kept" in output
        assert "the follow-up produced the report" not in output
        assert agent.run_conversation.call_count == 2

    def test_follow_up_still_without_header_is_delivered_and_flagged(self, tmp_path):
        second = {"final_response": "Designed both. Done."}
        (success, output, final_response, _), agent = _run(
            tmp_path, self.JOB, [NARRATION, second])
        assert success is True
        assert final_response == "Designed both. Done."
        assert "still missing after the follow-up" in output
        assert agent.run_conversation.call_count == 2


class TestDeliveryStripsPreamble:
    """run_one_job delivers the report from its header down; the file keeps it all."""

    def test_delivery_gets_the_trimmed_report_and_the_file_keeps_the_narration(self, tmp_path):
        import cron.jobs as cron_jobs

        job = {
            "id": "strip-test", "name": "experiment-design", "prompt": "design experiments",
            "enabled": True, "state": "scheduled", "deliver": "local", "model": None,
            "provider": None, "provider_snapshot": "custom", "base_url": None,
            "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"},
            "report_marker": "Experiment design",
        }
        narrated = "Ledgers cross-checked. Writing now.\n\n" + REPORT["final_response"]
        deliveries = []

        def fake_deliver(job, content, adapters=None, loop=None, **kwargs):
            deliveries.append(content)
            return None

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            fresh = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
            with patch("cron.scheduler._hermes_home", tmp_path), \
                 patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
                 patch("hermes_cli.env_loader.load_hermes_dotenv"), \
                 patch("hermes_cli.env_loader.reset_secret_source_cache"), \
                 patch("hermes_state_registry.acquire", return_value=MagicMock()), \
                 patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
                 patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                       return_value=_PROVIDER), \
                 patch("cron.scheduler._deliver_result", side_effect=fake_deliver), \
                 patch("cron.scheduler.save_job_output", wraps=lambda jid, out: saved.append(out)), \
                 patch("run_agent.AIAgent") as mock_agent_cls:
                saved = []
                agent = MagicMock()
                agent.run_conversation.return_value = {"final_response": narrated}
                mock_agent_cls.return_value = agent
                assert sched_run_one_job(fresh) is True

        assert deliveries == [REPORT["final_response"]]
        assert len(saved) == 1 and "Writing now." in saved[0]

"""The cron report gate: one follow-up turn when the final text is not the report.

A cron run ends on the first text turn with no tool call, whatever it says.
On 2026-09-06 experiment-design ended 25 minutes of real work on "Let me check
what's happening and fix all four facts" with no report written, and
doc-paper-ingest ended on a self-assessment essay after reading three papers
and storing nothing (incidents #42 and #43). Every prompt mandates a report
header; a job's ``report_marker`` names it and ``run_job`` grants exactly one
follow-up turn, in the same conversation, when it is absent.
"""

from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import (
    _normalize_report_text,
    _report_gate_missing_marker,
    run_job,
)


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


_PROVIDER = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "custom",
    "api_mode": "chat_completions",
}


def _run(tmp_path, job, responses):
    fake_db = MagicMock()
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("dotenv.load_dotenv"), \
         patch("hermes_state.get_shared_session_db", return_value=fake_db), \
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

    def test_follow_up_still_without_header_is_delivered_and_flagged(self, tmp_path):
        second = {"final_response": "Designed both. Done."}
        (success, output, final_response, _), agent = _run(
            tmp_path, self.JOB, [NARRATION, second])
        assert success is True
        assert final_response == "Designed both. Done."
        assert "still missing after the follow-up" in output
        assert agent.run_conversation.call_count == 2

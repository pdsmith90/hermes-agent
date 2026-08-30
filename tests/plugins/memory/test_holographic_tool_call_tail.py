"""A mis-terminated tool-call parse must not cost the call.

The serving layer's XML->JSON tool-call converter sometimes fails to terminate
the LAST `<parameter=...>` value at its `</parameter>` and runs on through
`</function></tool_call>` into whatever the model emitted next. The envelope
arrives correct — right tool name, other arguments intact — with one string
field carrying the tail, so the call fails on a bogus value rather than on a
parse error.

Seen 7 times between 2026-07-06 and 2026-08-30 across TWO local models, so it is
the parser and not the model; the payloads below are the real ones. The
dangerous direction is over-eager repair — a fact body legitimately contains
angle brackets — so the leave-alone cases are the load-bearing tests.
"""

import pytest

from plugins.memory.holographic import _strip_tool_call_tail


class TestRepairsTheRealPayloads:
    def test_2026_08_30_consolidate_reason_call(self):
        """msg 46044: both of consolidate-synthesize's entity-pair calls."""
        args = {
            "action": "reason>\n</function>\n</tool_call>\n\n<tool_call>\n"
                      "<function=fact_store>\n<parameter=action>\nget",
            "fact_id": 729,
        }
        _strip_tool_call_tail(args)
        assert args == {"action": "reason", "fact_id": 729}

    def test_july_shape_polluted_content_field(self):
        args = {"action": "update", "fact_id": 1,
                "content": "LESSON: something real.</parameter>\n</function>"}
        _strip_tool_call_tail(args)
        assert args["content"] == "LESSON: something real."

    def test_non_string_arguments_are_untouched(self):
        args = {"action": "update", "fact_id": 729, "trust_delta": 0.15}
        _strip_tool_call_tail(args)
        assert args == {"action": "update", "fact_id": 729, "trust_delta": 0.15}


class TestLeavesLegitimateContentAlone:
    """A false repair silently truncates a real fact — worse than the bug."""

    @pytest.mark.parametrize("body", [
        "eigenvalues with |lambda|>1 expand, <=2 samples differ",
        "a < b and b > c, gap <0.3",
        "SYNTHESIS: DDK vs Wiener — j^(-1.5) rule, ~4 deg <half-width>",
        "PAPER — 51% noise reduction vs DDK3 (p<0.01), retained trend",
        "LESSON: `tags = ?` is a full replace; pass tags=<the string get returned>",
    ])
    def test_angle_brackets_survive(self, body):
        args = {"action": "add", "content": body}
        _strip_tool_call_tail(args)
        assert args["content"] == body

    def test_empty_and_missing_fields_do_not_raise(self):
        args = {"action": "list", "content": ""}
        _strip_tool_call_tail(args)
        assert args == {"action": "list", "content": ""}

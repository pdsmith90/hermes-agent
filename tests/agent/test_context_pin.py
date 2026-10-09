"""``model.context_length`` is a visible pin: one warning when it disagrees with the advertised
window, and a ``(pinned)`` label wherever the window is rendered (#66168)."""
import logging

from agent import context_pin
from agent.context_pin import context_pin_suffix, is_context_pinned, warn_once_on_pin_disagreement


def test_pin_disagreement_warns_once_and_keeps_pin(monkeypatch, caplog):
    monkeypatch.setattr(context_pin, "_warned_pins", set())
    monkeypatch.setattr(context_pin, "advertised_context_length", lambda model, base_url="": 200_000)
    with caplog.at_level(logging.WARNING, logger="agent.context_pin"):
        assert warn_once_on_pin_disagreement("claude-sonnet-4", "", 999_000) is True
        # Second session start in the same process: silent.
        assert warn_once_on_pin_disagreement("claude-sonnet-4", "", 999_000) is False
        # Control: a pin that matches the advertised window is not a disagreement.
        assert warn_once_on_pin_disagreement("claude-sonnet-4", "", 200_000) is False
    warnings = [r for r in caplog.records if r.name == "agent.context_pin" and r.levelno >= logging.WARNING]
    assert len(warnings) == 1


def test_pinned_label_only_when_shown_value_is_the_pin():
    assert is_context_pinned(999_000, 999_000) is True
    assert context_pin_suffix(999_000, 999_000) == " (pinned)"
    # Route changed / pin dropped, unpinned, or non-int pins never label.
    assert context_pin_suffix(200_000, 999_000) == ""
    assert context_pin_suffix(200_000, None) == ""
    assert context_pin_suffix(1, True) == ""


def test_local_alias_without_a_learned_window_does_not_warn(monkeypatch, caplog):
    # llama-swap alias "qwen36-27b-cron" pinned at its own -c 163840: the vendor catalog's generic
    # "qwen" entry (131072) is not what the provider advertises (2026-10-09, 146 warnings since 09-29).
    from agent import model_metadata
    monkeypatch.setattr(context_pin, "_warned_pins", set())
    monkeypatch.setattr(model_metadata, "get_cached_context_length", lambda model, base_url, **kw: None)
    monkeypatch.setattr(model_metadata, "_load_model_metadata_disk_cache", lambda: {})
    local = "http://127.0.0.1:18000/v1"
    assert context_pin.advertised_context_length("qwen36-27b-cron", local) is None
    with caplog.at_level(logging.WARNING, logger="agent.context_pin"):
        assert warn_once_on_pin_disagreement("qwen36-27b-cron", local, 163_840) is False
    assert not [r for r in caplog.records if r.name == "agent.context_pin"]
    # Control: the same alias on a non-local endpoint still falls back to the catalog.
    assert context_pin.advertised_context_length("qwen36-27b-cron", "https://api.example.com/v1") == 131_072
    # A window LEARNED on the local endpoint still counts, and still warns when the pin disagrees.
    monkeypatch.setattr(model_metadata, "get_cached_context_length", lambda model, base_url, **kw: 131_072)
    with caplog.at_level(logging.WARNING, logger="agent.context_pin"):
        assert warn_once_on_pin_disagreement("qwen36-27b-cron", local, 163_840) is True

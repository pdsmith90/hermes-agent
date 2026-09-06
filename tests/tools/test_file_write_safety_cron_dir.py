"""File tools refuse writes inside the Hermes home's cron/ tree.

``approvals.deny`` ("*hermes cron *", "*jobs.json*") closes the terminal route
into cron/jobs.json, but read_file/write_file/patch never consulted it, and the
scheduler re-reads jobs.json every tick. On 2026-09-06 the morning briefing
reached jobs.json through read_file after two deny-blocked shell attempts;
a write_file by a prompt-injected cron run would have recreated the
2026-08-13 incident (a cron run created two live jobs for itself) unguarded.
The cronjob tool is the API for schedule changes; nothing legitimate writes
the scheduler's state through the agent's file tools.
"""

import json
import os

import pytest

import tools.file_tools as ft


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "cron" / "output" / "abc").mkdir(parents=True)
    (home / "memories").mkdir()
    monkeypatch.setattr(ft, "_real_hermes_home_cached", str(home.resolve()))
    monkeypatch.setattr(ft, "_real_hermes_home_loaded", True)
    return home


def test_jobs_json_is_refused(hermes_home):
    err = ft._check_sensitive_path(str(hermes_home / "cron" / "jobs.json"))
    assert err is not None
    assert "cron directory" in err
    assert "cronjob tool" in err


def test_anything_under_cron_is_refused(hermes_home):
    for rel in (
        os.path.join("cron", "executions.db"),
        os.path.join("cron", ".fire-abc.lock"),
        os.path.join("cron", "output", "abc", "2026-09-06_02-51-02.md"),
        os.path.join("cron", "jobs.expected.json"),
    ):
        assert ft._check_sensitive_path(str(hermes_home / rel)) is not None, rel


def test_sibling_paths_are_not_caught(hermes_home):
    # Prefix boundary: "cron" must be a directory component, not a substring.
    assert ft._check_sensitive_path(str(hermes_home / "cronx.txt")) is None
    assert ft._check_sensitive_path(str(hermes_home / "memories" / "MEMORY.md")) is None
    assert ft._check_sensitive_path(str(hermes_home / "ingested_papers.txt")) is None


def test_write_file_refuses_jobs_json(hermes_home):
    target = hermes_home / "cron" / "jobs.json"
    res = json.loads(ft.write_file_tool(str(target), '{"jobs": []}'))
    assert res.get("error")
    assert "cron directory" in res["error"]
    assert not target.exists()


def test_other_hermes_home_paths_unaffected(hermes_home):
    # The guard is scoped to cron/; a note in memories/ still goes through.
    assert ft._check_sensitive_path(str(hermes_home / "memories" / "note.md")) is None

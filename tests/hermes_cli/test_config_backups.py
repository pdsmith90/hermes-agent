"""config.yaml backups: one dir, deduped, bounded — never a pile of siblings in HERMES_HOME."""
import os
import time
from pathlib import Path

from hermes_cli.config_backups import backup_config, list_config_backups


def test_repeat_backups_dedupe_and_rotate(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    stamps = iter(f"2026010100000{i}" for i in range(10))
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: next(stamps))

    first = backup_config(cfg, "pre-setup", keep=2)
    assert first is not None and first.parent == tmp_path / "backups" / "config"
    # Same bytes again → no new file (the hermes-setup-three-times case).
    assert backup_config(cfg, "pre-setup", keep=2) is None
    for i in range(3):
        cfg.write_text(f"model: {i}\n")
        backup_config(cfg, "pre-setup", keep=2)
    kept = list_config_backups(cfg, "pre-setup")
    assert len(kept) == 2 and kept[0].read_text() == "model: 2\n"
    # Nothing left beside config.yaml in the home root.
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["config.yaml"]


def test_same_size_edit_within_one_mtime_tick_is_still_backed_up(tmp_path: Path, monkeypatch):
    # Coarse-timestamp filesystems leave an immediate rewrite with the old mtime; dedupe must
    # compare content, not trust a (size, mtime) signature.
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    stamps = iter(f"2026010100000{i}" for i in range(10))
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: next(stamps))

    assert backup_config(cfg, "pre-setup") is not None
    assert backup_config(cfg, "pre-setup") is None
    before = cfg.stat()
    cfg.write_text("model: b\n")
    os.utime(cfg, ns=(before.st_atime_ns, before.st_mtime_ns))

    second = backup_config(cfg, "pre-setup")
    assert second is not None and second.read_text() == "model: b\n"


def test_legacy_siblings_move_but_user_named_copies_stay(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    for name in ("config.yaml.bak.1778718391", "config.yaml.corrupt.20260729-093706.bak",
                 "config.yaml.bak-pre-migrate-xai-20260515-120000"):
        (tmp_path / name).write_text("old")
    (tmp_path / "config.yaml.bak-my-note").write_text("mine")

    backup_config(cfg, "corrupt")

    root = tmp_path / "backups" / "config"
    assert (root / "config.yaml.bak.1778718391").exists()
    assert (root / "config.yaml.corrupt.20260729-093706.bak").exists()
    assert (tmp_path / "config.yaml.bak-my-note").read_text() == "mine"
    assert not list(tmp_path.glob("config.yaml.bak.*")) and not list(tmp_path.glob("config.yaml.corrupt.*"))


def test_two_sweeps_of_a_same_named_sibling_keep_both_copies(tmp_path: Path):
    # A remade config.yaml.bak.<digits> reuses the swept name; the second sweep must not replace
    # the copy already in backups/config (that silently lost a real pre-change snapshot).
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    (tmp_path / "config.yaml.bak.1778718391").write_text("first snapshot\n")
    (tmp_path / "config.yaml.bak-note").write_text("mine")
    backup_config(cfg, "pre-setup")

    cfg.write_text("model: b\n")
    (tmp_path / "config.yaml.bak.1778718391").write_text("second snapshot\n")
    backup_config(cfg, "pre-setup")

    root = tmp_path / "backups" / "config"
    moved = sorted(p.read_text() for p in root.iterdir() if p.name.startswith("config.yaml.bak."))
    assert moved == ["first snapshot\n", "second snapshot\n"]
    # The dash-named copy is the user's own and is still untouched beside config.yaml.
    assert (tmp_path / "config.yaml.bak-note").read_text() == "mine"


def test_sweep_of_an_identical_same_named_sibling_keeps_one_copy(tmp_path: Path):
    # Identical bytes: the resident copy already preserves the content, so only the duplicate in
    # the home root is dropped — no suffixed twin is kept.
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    (tmp_path / "config.yaml.bak.1778718391").write_text("same\n")
    backup_config(cfg, "pre-setup")

    cfg.write_text("model: b\n")
    (tmp_path / "config.yaml.bak.1778718391").write_text("same\n")
    backup_config(cfg, "pre-setup")

    root = tmp_path / "backups" / "config"
    copies = [p for p in root.iterdir() if p.name.startswith("config.yaml.bak.")]
    assert [p.name for p in copies] == ["config.yaml.bak.1778718391"]
    assert copies[0].read_text() == "same\n"
    assert not list(tmp_path.glob("config.yaml.bak.*"))


def _age(path: Path, days: float) -> None:
    stamp = time.time() - days * 86400
    os.utime(path, (stamp, stamp))


def test_legacy_sweep_keeps_only_the_ten_newest_old_copies(tmp_path: Path):
    # Swept copies are bounded: the newest 10 by mtime survive, the older remainder does not.
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    for i in range(12):
        sibling = tmp_path / f"config.yaml.bak.17787183{i:02d}"
        sibling.write_text(f"snapshot {i}\n")
        _age(sibling, 40 + i)  # every one is past 30 days; i=11 is the oldest

    backup_config(cfg, "pre-setup")

    kept = sorted(p.name for p in (tmp_path / "backups" / "config").glob("config.yaml.bak.*"))
    assert kept == [f"config.yaml.bak.17787183{i:02d}" for i in range(10)]


def test_legacy_copy_younger_than_thirty_days_survives_beyond_ten(tmp_path: Path):
    # Beyond the newest 10, age alone decides: one day old is never pruned.
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    for i in range(11):
        recent = tmp_path / f"config.yaml.bak.17787184{i:02d}"
        recent.write_text("recent\n")
        _age(recent, 1)
    ancient = tmp_path / "config.yaml.bak.1778718099"
    ancient.write_text("ancient\n")
    _age(ancient, 60)

    backup_config(cfg, "pre-setup")

    names = {p.name for p in (tmp_path / "backups" / "config").glob("config.yaml.bak.*")}
    assert names == {f"config.yaml.bak.17787184{i:02d}" for i in range(11)}
    assert ancient.name not in names


def test_legacy_prune_leaves_reason_stamped_backups_alone(tmp_path: Path):
    # config.yaml.<reason>.<stamp> keeps its own keep= rotation; 12 ancient ones must all survive.
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    root = tmp_path / "backups" / "config"
    root.mkdir(parents=True)
    for i in range(12):
        stamped = root / f"config.yaml.good.202601010000{i:02d}"
        stamped.write_text("old\n")
        _age(stamped, 400)
    sibling = tmp_path / "config.yaml.bak.1778718391"
    sibling.write_text("legacy\n")
    _age(sibling, 60)

    assert backup_config(cfg, "pre-setup") is not None

    assert len(list(root.glob("config.yaml.good.*"))) == 12
    assert (root / "config.yaml.bak.1778718391").exists()


def test_plain_load_does_not_prune(tmp_path: Path):
    # The prune runs only after a sweep; a load with nothing to sweep leaves the dir untouched.
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    root = tmp_path / "backups" / "config"
    root.mkdir(parents=True)
    for i in range(12):
        old = root / f"config.yaml.bak.17787183{i:02d}"
        old.write_text("old\n")
        _age(old, 100)

    backup_config(cfg, "pre-setup")

    assert len(list(root.glob("config.yaml.bak.*"))) == 12

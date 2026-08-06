"""Regression test: clear_dir must empty a directory without replacing it.

It used to shutil.rmtree(dir), so the next run recreated the directory owned by
whoever ran it at the default umask. With two users driving the pipeline (the
server account and researchers by hand) that locked the other user out and every
one of their jobs died with EACCES while unlinking the previous run's files.
"""
import os
import stat

from decalfc.utils import clear_dir


def test_clear_dir_removes_contents_but_keeps_the_directory(tmp_path):
    target = tmp_path / "ospos_data" / "monthly_data" / "commits"
    target.mkdir(parents=True)
    (target / "proj__0.parquet").write_text("x")
    (target / "proj__1.parquet").write_text("y")
    nested = target / "nested"
    nested.mkdir()
    (nested / "deep.parquet").write_text("z")

    # Widened on purpose: the real dirs are group/other-writable so both users
    # can clear each other's output. This mode must survive clear_dir.
    os.chmod(target, 0o777)
    before_ino = target.stat().st_ino

    clear_dir(target, skip_input=True)

    assert target.is_dir(), "directory itself must survive"
    assert list(target.iterdir()) == [], "contents must be gone"
    assert target.stat().st_ino == before_ino, "must be the same directory, not a new one"
    assert stat.S_IMODE(target.stat().st_mode) == 0o777, "permissions must be preserved"


def test_clear_dir_is_a_noop_when_missing(tmp_path):
    clear_dir(tmp_path / "ospos_data" / "nope", skip_input=True)

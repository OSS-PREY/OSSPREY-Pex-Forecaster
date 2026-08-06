"""Regression test: a blank line in an edgelist must not fail the forecast.

get_net_overlap() unpacked every line as `sender, receiver, weight =
line.split("##")`. An empty line splits to a single element, raising
"not enough values to unpack (expected 3, got 1)" -- which killed the entire
forecast for prometheus, a repo whose edgelist carried a blank line.
"""
import pytest

from decalfc.pipeline.network_features import get_net_overlap


def _write(path, lines):
    path.write_text("\n".join(lines))
    return str(path)


def test_blank_lines_are_ignored(tmp_path):
    a = _write(tmp_path / "a.edgelist", ["x##y##1", "", "y##z##2", ""])
    b = _write(tmp_path / "b.edgelist", ["x##y##1", ""])

    # 1 shared edge of 3 total edge-slots across both sets.
    assert get_net_overlap(a, b) == pytest.approx(1 / 3)


def test_malformed_lines_are_skipped_not_fatal(tmp_path):
    a = _write(tmp_path / "a.edgelist", ["x##y##1", "garbage-without-separators"])
    b = _write(tmp_path / "b.edgelist", ["x##y##1"])

    assert get_net_overlap(a, b) == pytest.approx(1 / 2)


def test_two_empty_networks_overlap_zero(tmp_path):
    a = _write(tmp_path / "a.edgelist", ["", ""])
    b = _write(tmp_path / "b.edgelist", [""])

    assert get_net_overlap(a, b) == 0

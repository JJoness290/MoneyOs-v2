from pathlib import Path

import pytest

from app.main import _run_with_oom_retries
from app.core.oom_recovery import build_fraction_ladder, is_oom_like_error


def test_build_fraction_ladder_respects_initial_and_bounds():
    assert build_fraction_ladder(0.80, attempts_total=6) == [0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
    assert build_fraction_ladder(0.62, attempts_total=6) == [0.62, 0.6, 0.55]


def test_oom_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("MONEYOS_VRAM_FRACTION", "0.80")
    calls = []

    def runner(attempt, attempts_total, fraction):
        calls.append((attempt, attempts_total, fraction))
        if attempt < 3:
            raise RuntimeError("CUDA out of memory")
        out = tmp_path / "ok.mp4"
        rep = tmp_path / "report.json"
        out.write_bytes(b"x")
        rep.write_text("{}", encoding="utf-8")
        return out, rep

    video, report = _run_with_oom_retries("job_oom_sim", runner, output_dir=tmp_path / "job")
    assert video.exists()
    assert report.exists()
    assert len(calls) == 3


def test_non_oom_errors_fail_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("MONEYOS_VRAM_FRACTION", "0.80")

    def runner(_attempt, _attempts_total, _fraction):
        raise RuntimeError("some other failure")

    with pytest.raises(RuntimeError):
        _run_with_oom_retries("job_non_oom", runner, output_dir=tmp_path / "job")


def test_oom_like_matchers():
    assert is_oom_like_error(RuntimeError("CUDA out of memory"))
    assert is_oom_like_error(RuntimeError("CUDNN_STATUS_NOT_SUPPORTED"))
    assert is_oom_like_error(RuntimeError("hipErrorOutOfMemory"))
    assert not is_oom_like_error(RuntimeError("file not found"))

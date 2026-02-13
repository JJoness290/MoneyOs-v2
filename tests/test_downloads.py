from __future__ import annotations

from pathlib import Path

from app.core.net import downloads
from app.core.net.downloads import DirectUrl, DownloadResult, PageScrapeZip, download_from_sources


def test_download_from_sources_fallback(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_download_file(url: str, dst: Path, **kwargs):  # noqa: ANN003
        calls.append(url)
        if "bad" in url:
            return False, f"HTTPError 404 for {url} while installing pack (stage)", 404, url
        dst.write_bytes(b"zip")
        return True, None, 200, url

    monkeypatch.setattr(downloads, "download_file", _fake_download_file)
    result, source = download_from_sources(
        [DirectUrl(url="https://bad.example/file.zip", source="bad"), DirectUrl(url="https://ok.example/file.zip", source="ok")],
        tmp_path / "out.zip",
        pack_id="pack",
        stage="stage",
    )
    assert result.ok is True
    assert source == "ok"
    assert len(calls) == 2


def test_download_from_sources_error_contains_url(monkeypatch, tmp_path: Path) -> None:
    def _fake_download_file(url: str, dst: Path, **kwargs):  # noqa: ANN003
        return False, f"HTTPError 404 for {url} while installing pack (stage)", 404, url

    monkeypatch.setattr(downloads, "download_file", _fake_download_file)
    result, _ = download_from_sources(
        [DirectUrl(url="https://bad.example/file.zip", source="bad")],
        tmp_path / "out.zip",
        pack_id="pack",
        stage="stage",
    )
    assert result.ok is False
    assert "https://bad.example/file.zip" in (result.error or "")


def test_download_from_sources_dry_run(tmp_path: Path) -> None:
    result, source = download_from_sources(
        [PageScrapeZip(page_url="https://example.com/page", source="example")],
        tmp_path / "out.zip",
        dry_run=True,
    )
    assert result == DownloadResult(ok=True, error=None, status_code=None, final_url="https://example.com/page")
    assert source == "example"

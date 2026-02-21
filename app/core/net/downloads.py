from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import urljoin

import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 MoneyOS-Downloader/1.0"
)

_LAST_DOWNLOAD_DIAGNOSTICS: dict[str, object] = {}


@dataclass(frozen=True)
class DirectUrl:
    url: str
    source: str


@dataclass(frozen=True)
class PageScrapeZip:
    page_url: str
    source: str
    zip_regex: str = r"href=['\"]([^'\"]+\.zip(?:\?[^'\"]*)?)['\"]"
    base_url: str | None = None


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    error: str | None
    status_code: int | None
    final_url: str | None


def _record_diagnostic(payload: dict[str, object]) -> None:
    _LAST_DOWNLOAD_DIAGNOSTICS.clear()
    _LAST_DOWNLOAD_DIAGNOSTICS.update(payload)


def get_last_download_diagnostics() -> dict[str, object]:
    return dict(_LAST_DOWNLOAD_DIAGNOSTICS)


def _session(headers: Mapping[str, str] | None = None) -> requests.Session:
    session = requests.Session()
    base_headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/zip,*/*"}
    if headers:
        base_headers.update(dict(headers))
    session.headers.update(base_headers)
    return session


def _extract_zip_link(html: str, page_url: str, zip_regex: str, base_url: str | None) -> str | None:
    match = re.findall(zip_regex, html, flags=re.IGNORECASE)
    if not match:
        return None
    candidate = match[0]
    base = base_url or page_url
    return urljoin(base, candidate)


def download_file(
    url: str,
    dst: Path,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: int = 60,
    retries: int = 3,
    allow_redirects: bool = True,
    expect_binary: bool = True,
    pack_id: str = "unknown_pack",
    stage: str = "download",
    source: str = "direct",
) -> tuple[bool, str | None, int | None, str | None]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    session = _session(headers)
    last_error: str | None = None
    last_status: int | None = None
    last_url: str | None = url
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout, allow_redirects=allow_redirects) as response:
                last_status = response.status_code
                last_url = response.url
                response.raise_for_status()
                content_type = (response.headers.get("Content-Type") or "").lower()
                if expect_binary and "text/html" in content_type:
                    raise RuntimeError(
                        f"HTTPError unexpected_html for {url} while installing {pack_id} ({stage})"
                    )
                with dst.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            handle.write(chunk)
                _record_diagnostic(
                    {
                        "pack_id": pack_id,
                        "stage": stage,
                        "source": source,
                        "url": url,
                        "attempt": attempt,
                        "status_code": response.status_code,
                        "final_url": response.url,
                        "ok": True,
                        "error": None,
                    }
                )
                print(
                    f"[DOWNLOAD] pack_id={pack_id} stage={stage} source={source} "
                    f"url={url} attempt={attempt} status={response.status_code}",
                    flush=True,
                )
                return True, None, response.status_code, response.url
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            final = exc.response.url if exc.response is not None else last_url
            last_status = status
            last_url = final
            last_error = f"HTTPError {status} for {url} while installing {pack_id} ({stage})"
        except Exception as exc:  # noqa: BLE001
            last_error = f"download_error for {url} while installing {pack_id} ({stage}): {exc}"
        print(
            f"[DOWNLOAD] pack_id={pack_id} stage={stage} source={source} "
            f"url={url} attempt={attempt} status={last_status} error={last_error}",
            flush=True,
        )
    _record_diagnostic(
        {
            "pack_id": pack_id,
            "stage": stage,
            "source": source,
            "url": url,
            "attempt": retries,
            "status_code": last_status,
            "final_url": last_url,
            "ok": False,
            "error": last_error,
        }
    )
    return False, last_error, last_status, last_url


def download_from_sources(
    sources: list[DirectUrl | PageScrapeZip],
    dst: Path,
    *,
    timeout: int = 60,
    retries: int = 3,
    headers: Mapping[str, str] | None = None,
    allow_redirects: bool = True,
    pack_id: str = "unknown_pack",
    stage: str = "download",
    dry_run: bool = False,
) -> tuple[DownloadResult, str | None]:
    session = _session(headers)
    errors: list[str] = []
    for source in sources:
        if isinstance(source, DirectUrl):
            if dry_run:
                return DownloadResult(True, None, None, source.url), source.source
            ok, error, status, final_url = download_file(
                source.url,
                dst,
                timeout=timeout,
                retries=retries,
                allow_redirects=allow_redirects,
                expect_binary=True,
                pack_id=pack_id,
                stage=stage,
                source=source.source,
                headers=headers,
            )
            if ok:
                return DownloadResult(True, None, status, final_url), source.source
            if error:
                errors.append(error)
            continue

        page_stage = f"{stage}:page_scrape"
        if dry_run:
            return DownloadResult(True, None, None, source.page_url), source.source
        ok, error, status, final_url = download_file(
            source.page_url,
            dst.with_suffix(".html.tmp"),
            timeout=timeout,
            retries=retries,
            allow_redirects=allow_redirects,
            expect_binary=False,
            pack_id=pack_id,
            stage=page_stage,
            source=source.source,
            headers=headers,
        )
        if not ok:
            errors.append(error or f"failed to fetch page {source.page_url}")
            continue
        html = dst.with_suffix(".html.tmp").read_text(encoding="utf-8", errors="ignore")
        try:
            dst.with_suffix(".html.tmp").unlink(missing_ok=True)
        except Exception:
            pass
        zip_url = _extract_zip_link(html, final_url or source.page_url, source.zip_regex, source.base_url)
        if not zip_url:
            errors.append(f"zip_link_not_found on {source.page_url}")
            continue
        ok, error, status, final_url = download_file(
            zip_url,
            dst,
            timeout=timeout,
            retries=retries,
            allow_redirects=allow_redirects,
            expect_binary=True,
            pack_id=pack_id,
            stage=f"{stage}:zip",
            source=source.source,
            headers=headers,
        )
        if ok:
            return DownloadResult(True, None, status, final_url), source.source
        errors.append(error or f"zip download failed for {zip_url}")

    message = "; ".join(errors) if errors else "all_sources_failed"
    return DownloadResult(False, message, None, None), None

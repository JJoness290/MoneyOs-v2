from __future__ import annotations

from dataclasses import dataclass

from app.config import ASSET_LICENSE_MODE


@dataclass(frozen=True)
class LicenseResult:
    allowed: bool
    reason: str


def check_license(license_type: str | None) -> LicenseResult:
    if not license_type:
        return LicenseResult(allowed=False, reason="missing license")
    normalized = license_type.strip().lower()
    if normalized in {"cc0", "public domain", "publicdomain"}:
        return LicenseResult(allowed=True, reason="cc0")
    if ASSET_LICENSE_MODE == "cc0_or_ccby" and normalized in {"cc-by", "cc by", "cc-by-4.0"}:
        return LicenseResult(allowed=True, reason="cc-by")
    return LicenseResult(allowed=False, reason=f"license {normalized} not allowed")

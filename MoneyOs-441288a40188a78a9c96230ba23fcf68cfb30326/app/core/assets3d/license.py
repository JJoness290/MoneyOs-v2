from __future__ import annotations

from dataclasses import dataclass


ALLOWED_LICENSES = {
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "public domain": "https://creativecommons.org/publicdomain/mark/1.0/",
    "commercial": "https://example.com/license/commercial",
}


@dataclass(frozen=True)
class LicenseCheck:
    allowed: bool
    name: str
    url: str
    reason: str | None = None


def check_license(name: str | None, proof_path: str | None) -> LicenseCheck:
    if not name:
        return LicenseCheck(False, "", "", "missing license")
    normalized = name.strip().lower()
    if normalized not in ALLOWED_LICENSES:
        return LicenseCheck(False, name, "", f"license {normalized} not allowed")
    if not proof_path:
        return LicenseCheck(False, name, ALLOWED_LICENSES[normalized], "missing license proof")
    return LicenseCheck(True, name, ALLOWED_LICENSES[normalized], None)

from __future__ import annotations

from app.core.paths import _default_roots_for_platform


def test_windows_defaults_use_c_drive() -> None:
    roots = _default_roots_for_platform(True)
    assert roots["MONEYOS_ASSETS_ROOT"].startswith("C:\\MoneyOS")
    assert roots["MONEYOS_OUTPUT_ROOT"] == r"C:\MoneyOS\work"
    assert roots["MONEYOS_CACHE_ROOT"] == r"C:\MoneyOS\cache"
    assert roots["HF_HOME"] == r"C:\MoneyOS\hf"
    assert roots["HUGGINGFACE_HUB_CACHE"] == r"C:\MoneyOS\hf\hub"

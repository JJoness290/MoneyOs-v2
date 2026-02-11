from __future__ import annotations

from src.utils.cli_args import add_opt, validate_no_empty_value_flags


def main() -> int:
    args: list[str] = []
    add_opt(args, "--character-asset", None)
    add_opt(args, "--environment", "room")
    if "--character-asset" in args:
        raise RuntimeError("--character-asset should be omitted when value is empty")
    args_with_asset: list[str] = []
    add_opt(args_with_asset, "--character-asset", "C:/path/hero.blend")
    if "--character-asset" not in args_with_asset:
        raise RuntimeError("--character-asset should be present when value is provided")
    validate_no_empty_value_flags(args_with_asset, {"--character-asset"})
    try:
        validate_no_empty_value_flags(["--character-asset"], {"--character-asset"})
    except ValueError:
        return 0
    raise RuntimeError("Expected validation to fail for missing value")


if __name__ == "__main__":
    raise SystemExit(main())

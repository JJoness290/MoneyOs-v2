# MoneyOS Storage Policy (C:\ only)

MoneyOS enforces a strict storage policy through `app/core/storage_policy.py`.

## Absolute rule

- Storage root defaults to `C:\MoneyOS`.
- All runtime storage must remain under `C:\MoneyOS\...`.
- Any resolved `D:\...` path causes a fatal startup error.
- Any resolved drive path outside `C:\MoneyOS\...` causes a fatal startup error.

## Root resolver

`resolve_moneyos_root()` resolves in this order:

1. `MONEYOS_ROOT`
2. default `C:\MoneyOS`

## Required structure

- `C:\MoneyOS`
- `C:\MoneyOS\assets`
- `C:\MoneyOS\work`
- `C:\MoneyOS\cache`
- `C:\MoneyOS\cache\huggingface`
- `C:\MoneyOS\tmp`
- `C:\MoneyOS\logs`

## HuggingFace / Torch cache priority

`HF hub cache` is selected using this precedence:

1. `HF_HUB_CACHE`
2. `HF_HOME`
3. `HUGGINGFACE_HUB_CACHE`
4. `HF_DATASETS_CACHE`
5. `TORCH_HOME`
6. `XDG_CACHE_HOME`
7. fallback `C:\MoneyOS\cache\huggingface\hub`

## Env defaults applied at startup

MoneyOS sets these with `os.environ.setdefault(...)` before heavy model loading:

- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_DATASETS_CACHE`
- `TRANSFORMERS_CACHE`
- `TORCH_HOME`
- `TEMP`
- `TMP`
- `TMPDIR`
- `MONEYOS_ASSETS_ROOT`
- `MONEYOS_OUTPUT_ROOT`
- `MONEYOS_CACHE_ROOT`

## Path validator

`validate_storage_path(path)` enforces policy:

- rejects `D:\...`
- rejects paths outside `C:\MoneyOS\...`
- creates missing directories

Applied roots include assets, work/output, cache, huggingface, tmp, and logs.

## Effective settings output

At startup and at TrueAI job start, MoneyOS prints:

`=== MONEYOS EFFECTIVE SETTINGS ===`

including storage paths, env values, and stability/runtime settings.

Each TrueAI job also writes:

- `settings_effective.json`

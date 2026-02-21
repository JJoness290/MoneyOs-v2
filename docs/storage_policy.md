# MoneyOS Storage Policy (C:\ Only)

MoneyOS now uses a centralized `StoragePolicy` (`app/core/storage_policy.py`) as the single source of truth for runtime paths.

## Default roots

When env vars are not provided, MoneyOS defaults to:

- `MONEYOS_ROOT = C:\MoneyOS`
- `MONEYOS_ASSETS_ROOT = C:\MoneyOS\assets`
- `MONEYOS_OUTPUT_ROOT = C:\MoneyOS\work`
- `MONEYOS_CACHE_ROOT = C:\MoneyOS\cache`
- `HF_HOME = C:\MoneyOS\cache\huggingface`
- `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE = C:\MoneyOS\cache\huggingface\hub`
- `HF_DATASETS_CACHE = C:\MoneyOS\cache\huggingface\datasets`
- `TORCH_HOME = C:\MoneyOS\cache\torch`
- `TRANSFORMERS_CACHE = C:\MoneyOS\cache\huggingface\hub`
- `MONEYOS_TEMP_ROOT = C:\MoneyOS\tmp`
- `TMP`, `TEMP`, `TMPDIR = C:\MoneyOS\tmp`

## Startup behavior

At startup, before loading heavy AI pipelines, MoneyOS:

1. Resolves policy defaults/overrides.
2. Validates **no configured path points to `D:\`**.
3. Validates key roots are under `C:\MoneyOS\...`.
4. Sets process env vars (`os.environ`) for HF/Torch/temp/MoneyOS roots.
5. Creates directories and sets `tempfile.tempdir`.

If a policy violation is detected, MoneyOS raises a fatal error:

`StoragePolicy violation: attempted to use D:\... (must use C:\MoneyOS\...)`

## Effective settings visibility

MoneyOS prints an `EFFECTIVE SETTINGS` banner:

- on startup
- on each TrueAI job start

The same payload is written as JSON to each TrueAI job directory:

- `settings_effective.json`

This includes storage policy paths + source (`default` or env key), key env vars, and stability settings.

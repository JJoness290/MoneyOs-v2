# MoneyOS Phase 2

## Quick start (baseline)

```
set MONEYOS_SHORT_WORKDIR=C:\MoneyOS\work
python scripts\doctor_phase2.py
python scripts\smoke_clip.py
```

## Env vars

| Env var | Default | Notes |
| --- | --- | --- |
| `MONEYOS_SHORT_WORKDIR` | `C:\MoneyOS\work` | Short root for all Phase 2 outputs. |
| `MONEYOS_PATH_MAX` | `220` | Max safe path length. |
| `MONEYOS_OFFLINE` | `0` | Use cached assets only when `1`. |
| `MONEYOS_RENDER_PRESET` | `fast_proof` | `fast_proof` or `phase15_quality`. |

## Troubleshooting
- WinError 206: set `MONEYOS_SHORT_WORKDIR=C:\M\w` and retry.

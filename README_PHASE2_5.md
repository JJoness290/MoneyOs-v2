# MoneyOS Phase 2.5 (AI Video Backend)

## Notes
- AI video backend is optional and only used if dependencies are installed.
- If unavailable, the pipeline falls back to Blender clips.

## Env vars

| Env var | Default | Notes |
| --- | --- | --- |
| `MONEYOS_VISUAL_BACKEND` | `blender` | `blender`, `ai_video`, `hybrid`. |
| `MONEYOS_AI_VIDEO_MODEL` | `auto` | `animatediff`, `svd`, `auto`. |
| `MONEYOS_AI_VIDEO_VRAM_BUDGET_GB` | `20` | VRAM budget. |
| `MONEYOS_AI_VIDEO_FPS` | `15` | AI video fps. |

# Anime Episode Auto (Local Relevance Engine)

## Endpoint
`POST /jobs/anime-episode-auto`

Example:

```bash
curl.exe -X POST "http://127.0.0.1:8000/jobs/anime-episode-auto" \
  -H "Content-Type: application/json" \
  -d "{\"topic_seed\":\"Rogue AI awakens in Neo-Tokyo\",\"minutes\":10}"
```

## Required local packages
```bash
pip install TTS soundfile numpy
```

Optional voice conversion:
- Set `MONEYOS_VOICE_CONVERT=1`
- Set `MONEYOS_RVC_MODEL_PATH` to a local RVC model path.

## Environment variables
- `MONEYOS_TTS_BACKEND=xtts`
- `MONEYOS_TTS_LICENSE=cpml` (required for headless XTTS; sets `COQUI_TOS_AGREED=1`)
- `MONEYOS_TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2`
- `MONEYOS_TTS_DEVICE=auto|cuda|cpu`
- `MONEYOS_VOICE_REF_WAV=` (optional)
- `MONEYOS_VOICE_CONVERT=0|1`
- `MONEYOS_RVC_MODEL_PATH=` (optional)
- `MONEYOS_EPISODE_MINUTES=10`
- `MONEYOS_PROMPT_STYLE=anime_keyart_highimpact`

## Artifacts
Each job under `outputs/anime_episode_auto/<job_id>` writes:
- `script.json`
- `scenes.json`
- `audio.wav`
- `timestamps.json`
- `prompts.json`
- `render_plan.json`
- `voice_meta.json`
- `final.mp4`
- `episode_exact_duration.json`

`episode_exact_duration` validator fails the job if `final.mp4` is not within `±0.05s` of requested runtime.


## Cache behavior (headless)
- MoneyOS forces Coqui cache into `MONEYOS_CACHE_ROOT` by exporting at runtime:
  - `TTS_HOME={MONEYOS_CACHE_ROOT}\tts`
  - `XDG_CACHE_HOME={MONEYOS_CACHE_ROOT}`
  - `APPDATA={MONEYOS_CACHE_ROOT}\appdata`
- This prevents writes to `C:\Users\<user>\AppData`.


## Production default
- Production path is `POST /jobs/anime-trueai-quality` (true AI pipeline only; no Blender).
- Golden command:

```bash
curl.exe -X POST "http://127.0.0.1:8000/jobs/anime-trueai-quality" -H "Content-Type: application/json" -d "{\"topic_seed\":\"Rogue AI awakens in Neo-Tokyo\",\"minutes\":10,\"language\":\"en\"}"
```

- Final output location: `C:\MoneyOS\work\anime_trueai_video\<job_id>\final_yt.mp4` (and compatibility copy `final.mp4`).

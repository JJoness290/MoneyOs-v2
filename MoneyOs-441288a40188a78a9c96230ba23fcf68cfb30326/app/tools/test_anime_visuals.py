from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MONEYOS_VISUAL_MODE", "anime")
os.environ.setdefault("MONEYOS_ANIME_BEAT_SECONDS", "2.0")
quality_mode = os.getenv("MONEYOS_QUALITY", "auto")
nvenc_quality = os.getenv("MONEYOS_NVENC_QUALITY", "auto")

from app.config import OUTPUT_DIR, VISUAL_MODE  # noqa: E402
from app.core.visuals.anime.beat_renderer import _split_beats  # noqa: E402
from app.core.visuals.anime.sd_local import get_quality_profile_info  # noqa: E402
from app.core.quality_autotune import detect_hardware  # noqa: E402
from app.core.visuals.documentary.compositor import DocSegment, build_documentary_video_from_segments  # noqa: E402
from app.core.visuals.ffmpeg_utils import select_video_encoder  # noqa: E402


def main() -> None:
    output_path = OUTPUT_DIR / "debug" / "anime_test.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_text = (
        "The audit began with a missing $1.2 million wire transfer. "
        "Investigators traced the ledger entries back to a council ledger dated March 12, 2021. "
        "Emails reveal a deadline shift that moved escrow approvals by 48 hours."
    )
    segments = [
        DocSegment(
            index=1,
            text=segment_text,
            start=0.0,
            end=18.0,
            total_segments=1,
        )
    ]
    beats = _split_beats(segment_text, duration=18.0)
    encoder_args, encoder_name = select_video_encoder()
    hardware = None
    try:
        import torch  # noqa: WPS433

        hardware = detect_hardware(torch)
    except Exception:  # noqa: BLE001
        hardware = None
    print(
        f"[ANIME_TEST] mode={VISUAL_MODE} encoder={encoder_name} args={' '.join(encoder_args)} beats={len(beats)}"
    )
    print(f"[ANIME_TEST] quality_mode={quality_mode} nvenc_quality={nvenc_quality}")
    if hardware:
        print(
            "[ANIME_TEST] hardware="
            f"{hardware.gpu_name or 'cpu'} vram_total={hardware.vram_total_gb:.2f}GB "
            f"vram_free={(hardware.vram_free_gb or 0.0):.2f}GB"
        )

    def _status(message: str) -> None:
        print(f"[ANIME_TEST] {message}")

    build_documentary_video_from_segments(segments, output_path, status_callback=_status)
    profile_info = get_quality_profile_info() or {}
    profile = profile_info.get("profile", {})
    if profile:
        print(
            "[ANIME_TEST] profile source="
            f"{profile_info.get('profile_source')} quality={profile_info.get('quality_mode')} "
            f"key={profile_info.get('profile_key')} "
            f"final={profile.get('width')}x{profile.get('height')} steps={profile.get('steps')} "
            f"guidance={profile.get('guidance')} fp16={profile.get('fp16')} "
            f"cpu_offload={profile.get('cpu_offload')} attn_slicing={profile.get('attention_slicing')} "
            f"vae_slicing={profile.get('vae_slicing')} xformers={profile.get('xformers')}"
        )
    print(f"[ANIME_TEST] Wrote {output_path}")


if __name__ == "__main__":
    main()

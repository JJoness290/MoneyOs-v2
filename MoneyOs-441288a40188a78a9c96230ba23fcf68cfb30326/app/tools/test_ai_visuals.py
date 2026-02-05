from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.config import OUTPUT_DIR, TARGET_FPS, TARGET_RESOLUTION
from app.core.visuals.ai_prompting import build_segment_prompt, negative_prompt
from app.core.visuals.ai_image_generator import cache_key_for_prompt, generate_image
from app.core.visuals.image_to_video import make_kenburns_clip
from app.core.visuals.ffmpeg_utils import encoder_uses_threads, run_ffmpeg, select_video_encoder
from app.core.resource_guard import monitored_threads


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AI visuals test video")
    parser.add_argument("--text", required=True)
    parser.add_argument("--segments", type=int, default=5)
    args = parser.parse_args()

    output_dir = OUTPUT_DIR / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ai_visuals_test.mp4"
    fps = TARGET_FPS
    width, height = TARGET_RESOLUTION
    size = int(os.getenv("MONEYOS_AI_SIZE", "512"))
    steps = int(os.getenv("MONEYOS_AI_STEPS", "20"))
    guidance = float(os.getenv("MONEYOS_AI_GUIDANCE", "7.0"))
    model_id = os.getenv("MONEYOS_AI_MODEL", "runwayml/stable-diffusion-v1-5")
    style_preset = os.getenv("MONEYOS_AI_STYLE_PRESET", "moneyos_cinematic")
    seed_base = int(os.getenv("MONEYOS_AI_SEED_BASE", "1000"))

    clip_paths = []
    duration = 3.0
    for index in range(args.segments):
        prompt = build_segment_prompt(args.text, style_preset)
        negative = negative_prompt()
        seed = seed_base + index
        prompt_hash = cache_key_for_prompt(
            prompt,
            negative,
            size=size,
            seed=seed,
            style_preset=style_preset,
            model_id=model_id,
        )
        print(
            f"[AI_VISUALS] segment {index+1}/{args.segments} seed={seed} "
            f"prompt_hash={prompt_hash[:10]} size={size}"
        )
        image_path = generate_image(
            prompt=prompt,
            negative_prompt=negative,
            size=size,
            seed=seed,
            model_id=model_id,
            steps=steps,
            guidance=guidance,
            style_preset=style_preset,
        )
        if image_path is None:
            print("[AI_VISUALS] generation failed; no output produced.")
            return 1
        clip_path = output_dir / f"ai_clip_{index:03d}.mp4"
        make_kenburns_clip(
            image_path=image_path,
            duration_sec=duration,
            out_path=clip_path,
            fps=fps,
            target=f"{width}x{height}",
        )
        clip_paths.append(clip_path)

    if not clip_paths:
        print("No clips generated.")
        return 1
    if len(clip_paths) == 1:
        output_path.write_bytes(clip_paths[0].read_bytes())
        return 0

    inputs = []
    for clip in clip_paths:
        inputs += ["-i", str(clip)]
    filter_parts = [f"[{idx}:v]" for idx in range(len(clip_paths))]
    filter_complex = "".join(filter_parts) + f"concat=n={len(clip_paths)}:v=1:a=0[v]"
    encode_args, encoder_name = select_video_encoder()
    args = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-r",
        str(fps),
        *encode_args,
        "-an",
        str(output_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    run_ffmpeg(args)
    print(f"[AI_VISUALS] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

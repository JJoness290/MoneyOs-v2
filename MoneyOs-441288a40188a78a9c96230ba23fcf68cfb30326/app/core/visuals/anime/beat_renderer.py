from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import os
from pathlib import Path

from app.config import (
    AI_IMAGE_BACKEND,
    ANIME_BEAT_SECONDS,
    ANIME_MAX_IMAGES_PER_SEGMENT,
    OUTPUT_DIR,
    SD_GUIDANCE,
    SD_MODEL,
    SD_SEED,
    SD_STEPS,
    TARGET_FPS,
    TARGET_RESOLUTION,
)
from app.config import performance
from app.core.resource_guard import monitored_threads
from app.core.visuals.anime.prompt_mapper import build_prompt
from app.core.visuals.anime.sd_local import SDSettings, generate_image, is_sd_available
from app.core.visuals.ffmpeg_utils import StatusCallback, encoder_uses_threads, run_ffmpeg, select_video_encoder


@dataclass(frozen=True)
class Beat:
    index: int
    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _split_beats(text: str, duration: float) -> list[Beat]:
    words = [word for word in text.split() if word.strip()]
    beat_seconds = ANIME_BEAT_SECONDS
    beat_count = max(1, math.ceil(duration / beat_seconds))
    while beat_count > ANIME_MAX_IMAGES_PER_SEGMENT:
        beat_seconds *= 2
        beat_count = max(1, math.ceil(duration / beat_seconds))
    if not words:
        beat_duration = duration / beat_count
        return [
            Beat(index=i + 1, text=text.strip(), start=i * beat_duration, end=min(duration, (i + 1) * beat_duration))
            for i in range(beat_count)
        ]
    words_per_beat = max(1, math.ceil(len(words) / beat_count))
    beats: list[Beat] = []
    for index in range(beat_count):
        start = index * beat_seconds
        end = min(duration, start + beat_seconds)
        chunk = words[index * words_per_beat : (index + 1) * words_per_beat]
        beats.append(Beat(index=index + 1, text=" ".join(chunk), start=start, end=end))
    return beats


def _build_fallback_clip(duration: float, output_path: Path, status_callback: StatusCallback) -> None:
    width, height = TARGET_RESOLUTION
    encode_args, encoder_name = select_video_encoder()
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x0f172a:size={width}x{height}:rate={TARGET_FPS}",
        "-t",
        f"{duration:.3f}",
        "-vf",
        "noise=alls=18:allf=t,format=yuv420p",
        *encode_args,
        "-an",
        str(output_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    if status_callback:
        status_callback("[ANIME] fallback procedural beat")
    run_ffmpeg(args, status_callback=status_callback)


def _animate_still(image_path: Path, duration: float, output_path: Path, status_callback: StatusCallback) -> None:
    width, height = TARGET_RESOLUTION
    frames = max(1, int(duration * TARGET_FPS))
    zoom_increment = 0.0008
    encode_args, encoder_name = select_video_encoder()
    filter_chain = (
        f"scale={width}:{height},"
        "format=yuv420p,"
        f"zoompan=z='min(1.12,zoom+{zoom_increment})':"
        "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s={width}x{height}"
    )
    args = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        f"{duration:.3f}",
        "-vf",
        filter_chain,
        "-r",
        str(TARGET_FPS),
        *encode_args,
        "-an",
        str(output_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    if status_callback:
        status_callback(f"[ANIME] animating beat {output_path.stem}")
    run_ffmpeg(args, status_callback=status_callback)


class AnimeBeatVisuals:
    def __init__(self, status_callback: StatusCallback = None) -> None:
        self.status_callback = status_callback

    def available(self) -> bool:
        if AI_IMAGE_BACKEND != "sd_local":
            return False
        return is_sd_available()

    def render_segment_background(self, segment_text: str, duration: float, output_path: Path) -> Path:
        beats = _split_beats(segment_text, duration)
        debug_log_settings(self.status_callback, len(beats))
        if self.status_callback:
            self.status_callback(
                f"[ANIME] ram_mode={performance.ram_mode()} segment_workers={performance.segment_workers()}"
            )
        beat_dir = output_path.parent / "anime_beats"
        beat_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = OUTPUT_DIR / "debug" / "anime_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        clips: list[Path] = []
        for beat in beats:
            clip_path = beat_dir / f"beat_{beat.index:03d}.mp4"
            if clip_path.exists():
                clips.append(clip_path)
                continue
            prompt_payload = build_prompt(beat.text or segment_text)
            image_path = beat_dir / f"beat_{beat.index:03d}.png"
            try:
                if not self.available():
                    raise RuntimeError("Anime backend unavailable")
                settings = SDSettings(
                    model=SD_MODEL,
                    steps=SD_STEPS,
                    guidance=SD_GUIDANCE,
                    seed=SD_SEED,
                    width=512,
                    height=512,
                    fp16=True,
                    cpu_offload=False,
                    attention_slicing=True,
                    vae_slicing=True,
                )
                try:
                    generated_path = generate_image(
                        prompt_payload.prompt,
                        prompt_payload.negative_prompt,
                        output_path=image_path,
                        cache_dir=cache_dir,
                        settings=settings,
                    )
                except Exception:  # noqa: BLE001
                    retry_settings = SDSettings(
                        model=settings.model,
                        steps=min(12, settings.steps),
                        guidance=settings.guidance,
                        seed=settings.seed,
                        width=settings.width,
                        height=settings.height,
                        fp16=settings.fp16,
                        cpu_offload=settings.cpu_offload,
                        attention_slicing=settings.attention_slicing,
                        vae_slicing=settings.vae_slicing,
                    )
                    generated_path = generate_image(
                        prompt_payload.prompt,
                        prompt_payload.negative_prompt,
                        output_path=image_path,
                        cache_dir=cache_dir,
                        settings=retry_settings,
                    )
                _animate_still(generated_path, beat.duration, clip_path, self.status_callback)
            except Exception as exc:  # noqa: BLE001
                if self.status_callback:
                    self.status_callback(f"[ANIME] beat {beat.index} failed ({exc}); using fallback")
                _build_fallback_clip(beat.duration, clip_path, self.status_callback)
            clips.append(clip_path)
            gc.collect()
            try:
                if os.getenv("MONEYOS_USE_GPU", "0") == "1":
                    import torch  # noqa: WPS433

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

        concat_list = output_path.with_suffix(".txt")
        concat_list.write_text("\n".join([f"file '{clip.as_posix()}'" for clip in clips]), encoding="utf-8")
        encode_args, encoder_name = select_video_encoder()
        args = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-r",
            str(TARGET_FPS),
            *encode_args,
            "-an",
            str(output_path),
        ]
        if encoder_uses_threads(encoder_name):
            args += ["-threads", str(monitored_threads())]
        if self.status_callback:
            self.status_callback(f"[ANIME] concatenating {len(clips)} beats")
        run_ffmpeg(args, status_callback=self.status_callback)
        return output_path


def debug_log_settings(status_callback: StatusCallback, beat_count: int) -> None:
    if not status_callback:
        return
    status_callback(
        "[ANIME] mode=sd_local "
        f"beat_seconds={ANIME_BEAT_SECONDS:.2f} beats={beat_count} "
        f"max_beats={ANIME_MAX_IMAGES_PER_SEGMENT}"
    )

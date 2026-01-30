import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
VIDEO_DIR = OUTPUT_DIR / "videos"
AUDIO_DIR = OUTPUT_DIR / "audio"
BROLL_DIR = OUTPUT_DIR / "broll"
SCRIPT_DIR = BASE_DIR / "outputs" / "scripts"
MINECRAFT_BG_DIR = BASE_DIR / "assets" / "minecraft"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
BROLL_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

PEXELS_API_KEY_ENV = "PEXELS_API_KEY"
DEFAULT_VOICE = "en-US-JennyNeural"
TTS_RATE = "+8%"
TARGET_PLATFORM = os.getenv("MONEYOS_TARGET_PLATFORM", "youtube").strip().lower()
if TARGET_PLATFORM not in {"tiktok", "youtube"}:
    TARGET_PLATFORM = "youtube"
TARGET_RESOLUTION = (1080, 1920) if TARGET_PLATFORM == "tiktok" else (1920, 1080)
TARGET_FPS = 30
MIN_AUDIO_SECONDS = 480

VISUAL_MODE = os.getenv("MONEYOS_VISUAL_MODE")
if VISUAL_MODE:
    VISUAL_MODE = VISUAL_MODE.strip().lower()
if VISUAL_MODE not in {None, "broll", "documentary", "hybrid"}:
    VISUAL_MODE = None
if VISUAL_MODE is None:
    VISUAL_MODE = "documentary" if TARGET_PLATFORM == "youtube" else "broll"

DOC_STYLE = os.getenv("MONEYOS_DOC_STYLE", "clean").strip().lower()
if DOC_STYLE not in {"clean", "gritty"}:
    DOC_STYLE = "clean"

DOC_BG_MODE = os.getenv("MONEYOS_DOC_BG_MODE", "loops").strip().lower()
if DOC_BG_MODE not in {"loops", "procedural"}:
    DOC_BG_MODE = "loops"

DOC_BG_DIR = BASE_DIR / os.getenv("MONEYOS_DOC_BG_DIR", "assets/background_loops")
DOC_OVERLAY_DIR = BASE_DIR / "assets" / "doc_overlays"
DOC_TEXTURE_DIR = DOC_OVERLAY_DIR / "textures"
DOC_STAMP_DIR = DOC_OVERLAY_DIR / "stamps"

SUBTITLE_STYLE = os.getenv("MONEYOS_SUBTITLE_STYLE", "documentary").strip().lower()

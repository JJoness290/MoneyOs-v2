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
try:
    YT_MIN_AUDIO_SECONDS = int(os.getenv("MONEYOS_YT_MIN_AUDIO_SECONDS", "600"))
except ValueError:
    YT_MIN_AUDIO_SECONDS = 600
try:
    YT_TARGET_AUDIO_SECONDS = int(os.getenv("MONEYOS_YT_TARGET_AUDIO_SECONDS", "720"))
except ValueError:
    YT_TARGET_AUDIO_SECONDS = 720
try:
    YT_MIN_WORDS = int(os.getenv("MONEYOS_YT_MIN_WORDS", "1800"))
except ValueError:
    YT_MIN_WORDS = 1800
try:
    YT_EXTEND_CHUNK_SECONDS = int(os.getenv("MONEYOS_YT_EXTEND_CHUNK_SECONDS", "180"))
except ValueError:
    YT_EXTEND_CHUNK_SECONDS = 180
try:
    YT_MAX_EXTEND_ATTEMPTS = int(os.getenv("MONEYOS_YT_MAX_EXTEND_ATTEMPTS", "4"))
except ValueError:
    YT_MAX_EXTEND_ATTEMPTS = 4
try:
    YT_WPM = int(os.getenv("MONEYOS_YT_WPM", "155"))
except ValueError:
    YT_WPM = 155

VISUAL_MODE = os.getenv("MONEYOS_VISUAL_MODE")
if VISUAL_MODE:
    VISUAL_MODE = VISUAL_MODE.strip().lower()
if VISUAL_MODE not in {None, "broll", "documentary", "anime", "hybrid"}:
    VISUAL_MODE = None
if VISUAL_MODE is None:
    VISUAL_MODE = "anime" if TARGET_PLATFORM == "youtube" else "documentary"

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

ANIME_STYLE = os.getenv("MONEYOS_ANIME_STYLE", "thriller").strip().lower()
if ANIME_STYLE not in {"thriller", "clean", "cyberpunk"}:
    ANIME_STYLE = "thriller"

try:
    ANIME_BEAT_SECONDS = float(os.getenv("MONEYOS_ANIME_BEAT_SECONDS", "2.0"))
except ValueError:
    ANIME_BEAT_SECONDS = 2.0
if ANIME_BEAT_SECONDS <= 0:
    ANIME_BEAT_SECONDS = 2.0

try:
    ANIME_MAX_IMAGES_PER_SEGMENT = int(os.getenv("MONEYOS_ANIME_MAX_IMAGES_PER_SEGMENT", "180"))
except ValueError:
    ANIME_MAX_IMAGES_PER_SEGMENT = 180
if ANIME_MAX_IMAGES_PER_SEGMENT <= 0:
    ANIME_MAX_IMAGES_PER_SEGMENT = 180

AI_IMAGE_BACKEND = os.getenv("MONEYOS_AI_IMAGE_BACKEND", "sd_local").strip().lower()
if AI_IMAGE_BACKEND not in {"sd_local"}:
    AI_IMAGE_BACKEND = "sd_local"
SD_MODEL = os.getenv("MONEYOS_SD_MODEL", "sd15_anime").strip().lower()
if SD_MODEL not in {"sd15_anime", "sdxl_anime"}:
    SD_MODEL = "sd15_anime"
SD_MODEL_ID = os.getenv("MONEYOS_SD_MODEL_ID") or "runwayml/stable-diffusion-v1-5"
try:
    SD_STEPS = int(os.getenv("MONEYOS_SD_STEPS", "18"))
except ValueError:
    SD_STEPS = 18
if SD_STEPS <= 0:
    SD_STEPS = 18
try:
    SD_GUIDANCE = float(os.getenv("MONEYOS_SD_GUIDANCE", "5.5"))
except ValueError:
    SD_GUIDANCE = 5.5
if SD_GUIDANCE <= 0:
    SD_GUIDANCE = 5.5
try:
    SD_SEED = int(os.getenv("MONEYOS_SD_SEED", "0"))
except ValueError:
    SD_SEED = 0
AI_IMAGE_CACHE = os.getenv("MONEYOS_AI_IMAGE_CACHE", "1") != "0"

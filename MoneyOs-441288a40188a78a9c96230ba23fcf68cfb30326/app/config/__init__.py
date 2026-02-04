import os
from pathlib import Path

from app.core.paths import get_assets_root, get_output_root, get_repo_root

BASE_DIR = get_repo_root()
OUTPUT_DIR = get_output_root()
VIDEO_DIR = OUTPUT_DIR / "videos"
AUDIO_DIR = OUTPUT_DIR / "audio"
BROLL_DIR = OUTPUT_DIR / "broll"
ASSETS_DIR = get_assets_root()
CHARACTERS_DIR = BASE_DIR / os.getenv("MONEYOS_CHARACTERS_DIR", "assets/characters_3d")
ANIMATIONS_DIR = BASE_DIR / os.getenv("MONEYOS_ANIMATIONS_DIR", "assets/animations")
ANIMATION_PACKS_DIR = BASE_DIR / "assets" / "animation_packs"
VFX_DIR = BASE_DIR / os.getenv("MONEYOS_VFX_DIR", "assets/vfx")
AUTO_CHARACTERS_DIR = BASE_DIR / os.getenv("MONEYOS_ASSET_STORAGE_DIR", "assets/characters_3d_auto")
SCRIPT_DIR = BASE_DIR / "outputs" / "scripts"
MINECRAFT_BG_DIR = BASE_DIR / "assets" / "minecraft"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
BROLL_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
ANIMATION_PACKS_DIR.mkdir(parents=True, exist_ok=True)
AUTO_CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)

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
if VISUAL_MODE not in {None, "broll", "documentary", "anime", "anime_3d", "hybrid"}:
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
SD_MODEL_ID = os.getenv("MONEYOS_SD_MODEL_ID") or "cagliostrolab/animagine-xl-3.1"
SD_MODEL_SOURCE = os.getenv("MONEYOS_SD_MODEL_SOURCE", "diffusers_hf").strip().lower()
if SD_MODEL_SOURCE not in {"local_ckpt", "diffusers_hf", "api"}:
    SD_MODEL_SOURCE = "diffusers_hf"
SD_MODEL_LOCAL_PATH = Path(os.getenv("MONEYOS_SD_MODEL_LOCAL_PATH", "output/models/anime/animagine-xl-3.1.safetensors"))
SD_PROFILE = os.getenv("MONEYOS_SD_PROFILE", "balanced").strip().lower()
if SD_PROFILE not in {"balanced", "max"}:
    SD_PROFILE = "balanced"
try:
    SD_MAX_BATCH_SIZE = int(os.getenv("MONEYOS_SD_MAX_BATCH_SIZE", "2"))
except ValueError:
    SD_MAX_BATCH_SIZE = 2
ANIME_LORA_PATHS = os.getenv("MONEYOS_ANIME_LORA_PATHS", "")
ANIME_LORA_WEIGHTS = os.getenv("MONEYOS_ANIME_LORA_WEIGHTS", "")
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

BLENDER_PATH = os.getenv("MONEYOS_BLENDER_PATH")
BLENDER_ENGINE = os.getenv("MONEYOS_BLENDER_ENGINE", "eevee").strip().lower()
if BLENDER_ENGINE not in {"eevee", "cycles"}:
    BLENDER_ENGINE = "eevee"
BLENDER_GPU = os.getenv("MONEYOS_BLENDER_GPU", "1") != "0"
try:
    RENDER_RES = os.getenv("MONEYOS_RENDER_RES", "1920x1080").lower()
    RENDER_RESOLUTION = tuple(int(val) for val in RENDER_RES.split("x", 1))
except ValueError:
    RENDER_RESOLUTION = (1920, 1080)
try:
    RENDER_FPS = int(os.getenv("MONEYOS_RENDER_FPS", "30"))
except ValueError:
    RENDER_FPS = 30
TOON_SHADER = os.getenv("MONEYOS_TOON_SHADER", "1") != "0"
VFX_ENABLE = os.getenv("MONEYOS_VFX_ENABLE", "1") != "0"
try:
    VFX_EMISSION_STRENGTH = float(os.getenv("MONEYOS_VFX_EMISSION_STRENGTH", "50"))
except ValueError:
    VFX_EMISSION_STRENGTH = 50.0
try:
    VFX_SCALE = float(os.getenv("MONEYOS_VFX_SCALE", "1.0"))
except ValueError:
    VFX_SCALE = 1.0
try:
    VFX_SCREEN_COVERAGE = float(os.getenv("MONEYOS_VFX_SCREEN_COVERAGE", "0.35"))
except ValueError:
    VFX_SCREEN_COVERAGE = 0.35

ANIME3D_ASSET_MODE = os.getenv("MONEYOS_ANIME3D_ASSET_MODE", "auto").strip().lower()
if ANIME3D_ASSET_MODE not in {"auto", "local"}:
    ANIME3D_ASSET_MODE = "auto"
ANIME3D_TEXTURE_MODE = os.getenv("MONEYOS_ANIME3D_TEXTURE_MODE", "sd_local").strip().lower()
if ANIME3D_TEXTURE_MODE not in {"sd_local", "procedural"}:
    ANIME3D_TEXTURE_MODE = "sd_local"
ANIME3D_STYLE_PRESET = os.getenv("MONEYOS_ANIME3D_STYLE_PRESET", "key_art").strip().lower()
ANIME3D_SFX_MODE = os.getenv("MONEYOS_ANIME3D_SFX_MODE", "auto").strip().lower()
ANIME3D_OUTLINE_MODE = os.getenv("MONEYOS_ANIME3D_OUTLINE_MODE", "freestyle").strip().lower()
if ANIME3D_OUTLINE_MODE not in {"freestyle", "inverted_hull"}:
    ANIME3D_OUTLINE_MODE = "freestyle"
ANIME3D_POSTFX = os.getenv("MONEYOS_ANIME3D_POSTFX", "on").strip().lower() != "off"
ANIME3D_QUALITY = os.getenv("MONEYOS_ANIME3D_QUALITY", "balanced").strip().lower()
if ANIME3D_QUALITY not in {"balanced", "max"}:
    ANIME3D_QUALITY = "balanced"
try:
    ANIME3D_FPS = int(os.getenv("MONEYOS_ANIME3D_FPS", "30"))
except ValueError:
    ANIME3D_FPS = 30
try:
    ANIME3D_SECONDS = int(os.getenv("MONEYOS_ANIME3D_SECONDS", "60"))
except ValueError:
    ANIME3D_SECONDS = 60
ANIME3D_RES = os.getenv("MONEYOS_ANIME3D_RES", "1920x1080").lower()
try:
    ANIME3D_RESOLUTION = tuple(int(value) for value in ANIME3D_RES.split("x", 1))
except ValueError:
    ANIME3D_RESOLUTION = (1920, 1080)
SD_MODEL_PATH = os.getenv("MONEYOS_SD_MODEL_PATH")

ASSET_HARVEST = os.getenv("MONEYOS_ASSET_HARVEST", "0") == "1"
ASSET_LICENSE_MODE = os.getenv("MONEYOS_ASSET_LICENSE_MODE", "cc0_only").strip().lower()
if ASSET_LICENSE_MODE not in {"cc0_only", "cc0_or_ccby"}:
    ASSET_LICENSE_MODE = "cc0_only"
ASSET_PROVIDERS = [item.strip() for item in os.getenv("MONEYOS_ASSET_PROVIDERS", "opengameart").split(",") if item.strip()]
try:
    ASSET_MAX_DOWNLOADS_PER_RUN = int(os.getenv("MONEYOS_ASSET_MAX_DOWNLOADS_PER_RUN", "30"))
except ValueError:
    ASSET_MAX_DOWNLOADS_PER_RUN = 30
try:
    ASSET_KEEP_TOP_N = int(os.getenv("MONEYOS_ASSET_KEEP_TOP_N", "5"))
except ValueError:
    ASSET_KEEP_TOP_N = 5
ASSET_REVIEW_MODE = os.getenv("MONEYOS_ASSET_REVIEW_MODE", "0") == "1"
SKETCHFAB_API_TOKEN = os.getenv("MONEYOS_SKETCHFAB_API_TOKEN")

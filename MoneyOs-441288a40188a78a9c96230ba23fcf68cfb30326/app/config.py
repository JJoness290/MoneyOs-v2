from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
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
TARGET_RESOLUTION = (1920, 1080)
TARGET_FPS = 30
MIN_AUDIO_SECONDS = 480

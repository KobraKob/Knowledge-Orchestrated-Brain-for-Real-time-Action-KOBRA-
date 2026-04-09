"""
config.py — Central configuration for KOBRA.
All constants, paths, and settings live here. No magic strings elsewhere.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ───────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
PORCUPINE_ACCESS_KEY: str = os.getenv("PORCUPINE_ACCESS_KEY", "")

# ── Wake Word ──────────────────────────────────────────────────────────────────
# Built-in keyword used with pvporcupine (no .ppn file needed).
# Options: "jarvis", "computer", "terminator", "bumblebee", "alexa", "porcupine"
WAKE_WORD_KEYWORD: str = os.getenv("WAKE_WORD_KEYWORD", "jarvis")

# Optional: path to a custom .ppn file — leave blank to use built-in keyword above.
WAKE_WORD_PATH: str = os.getenv("WAKE_WORD_PATH", "")

# ── Whisper (STT) ──────────────────────────────────────────────────────────────
WHISPER_MODEL: str = "base"          # "tiny" | "base" | "small" | "medium"
WHISPER_COMPUTE_TYPE: str = "int8"   # CPU-friendly quantization

# ── Groq (LLM) ────────────────────────────────────────────────────────────────
# 70b — high accuracy, used ONLY for Turn 1 tool selection.
# Free tier: 100k tokens/day — so we use it sparingly.
GROQ_MODEL_TOOLS: str = "llama-3.3-70b-versatile"

# 8b — fast and cheap, 1M tokens/day on free tier.
# Used for: direct conversational answers, Turn 2 narration, greeting.
GROQ_MODEL_FAST: str = "llama-3.1-8b-instant"

GROQ_MAX_TOKENS: int = 512

# ── TTS (edge-tts) ────────────────────────────────────────────────────────────
TTS_VOICE: str = "en-US-GuyNeural"   # Natural male voice
TTS_RATE: str = "+10%"               # Slightly faster than default
AUDIO_TEMP_PATH: str = "temp_audio.mp3"
USE_OFFLINE_TTS: bool = False        # Set True to force pyttsx3 fallback

# ── Memory (SQLite) ───────────────────────────────────────────────────────────
DB_PATH: str = "kobra_memory.db"
MEMORY_INJECT_LIMIT: int = 10        # Max recent turns injected per Groq call

# ── Audio / Silence Detection ─────────────────────────────────────────────────
SAMPLE_RATE: int = 16000             # Hz — required by Porcupine and Whisper
FRAME_LENGTH: int = 512              # Porcupine frame size (samples)
SILENCE_THRESHOLD: float = 300.0     # RMS below this = silence
SILENCE_CHUNKS: int = 30             # Consecutive silent chunks before stop
MAX_RECORD_SECONDS: int = 15         # Hard cap on recording duration

# ── run_command Safety Whitelist ──────────────────────────────────────────────
# Only shell commands starting with these prefixes are permitted.
COMMAND_WHITELIST: list[str] = [
    "pip",
    "python",
    "git",
    "npm",
    "node",
    "npx",
    "cd",
    "mkdir",
    "echo",
    "type",
    "dir",
    "ls",
    "cat",
    "code",
    "uvicorn",
    "pytest",
    "poetry",
]

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FILE: str = "kobra.log"

def setup_logging() -> None:
    """Configure root logger to write to both console and file."""
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    fmt = "[%(asctime)s] %(levelname)s — %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )

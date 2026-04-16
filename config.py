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
WHISPER_MODEL: str = "small"          # "tiny" | "base" | "small" | "medium"
WHISPER_COMPUTE_TYPE: str = "int8"   # CPU-friendly quantization

# ── Groq (LLM) ────────────────────────────────────────────────────────────────
# 70b — high accuracy, used ONLY for Turn 1 tool selection.
# Free tier: 100k tokens/day — so we use it sparingly.
GROQ_MODEL_TOOLS: str = "llama-3.3-70b-versatile"

# 8b — fast and cheap, 1M tokens/day on free tier.
# Used for: direct conversational answers, Turn 2 narration, greeting.
GROQ_MODEL_FAST: str = "llama-3.1-8b-instant"

GROQ_MAX_TOKENS: int = 512
GROQ_MAX_TOKENS_NARRATION: int = 180  # Hard cap on Turn-2 spoken narration; forces brevity

# ── TTS (edge-tts) ────────────────────────────────────────────────────────────
TTS_VOICE: str = "en-GB-RyanNeural"    # British male voice — confident, natural JARVIS feel
TTS_RATE: str = "+5%"                  # Slightly faster than default; +10% felt rushed
AUDIO_TEMP_PATH: str = "temp_audio.mp3"
USE_OFFLINE_TTS: bool = False          # Set True to force pyttsx3 fallback

# ── Memory (SQLite) ───────────────────────────────────────────────────────────
DB_PATH: str = "kobra_memory.db"
MEMORY_INJECT_LIMIT: int = 6         # Max recent turns injected per Groq call (10 was bloating tool prompts)

# ── Audio / Silence Detection ─────────────────────────────────────────────────
SAMPLE_RATE: int = 16000             # Hz — required by Porcupine and Whisper
FRAME_LENGTH: int = 512              # Porcupine frame size (samples)
SILENCE_THRESHOLD: float = 300.0     # RMS below this = silence
SILENCE_CHUNKS: int = 35             # Consecutive silent chunks before stop (~1.1s at 16kHz/512)
MAX_RECORD_SECONDS: int = 20         # Hard cap on recording duration; 15s cut off longer commands

# ── run_command — full access, no whitelist restriction ───────────────────────
# User has granted full PC access. run_command executes any shell command.
# COMMAND_WHITELIST kept for legacy compatibility but no longer enforced.
COMMAND_WHITELIST: list[str] = []

# ── Multi-agent orchestration ─────────────────────────────────────────────────
MAX_PARALLEL_AGENTS: int = 4    # max concurrent agent threads in ThreadPoolExecutor
MAX_PARALLEL_TOOLS: int  = 4    # max concurrent tool executions within one agent turn
AGENT_TIMEOUT: int = 60         # seconds before an agent task is considered timed out

# ── External Integrations ─────────────────────────────────────────────────────
# Google (Gmail + Calendar) — download from Google Cloud Console
GOOGLE_CREDENTIALS_PATH: str = "google_credentials.json"

# Spotify Web API — create app at developer.spotify.com
SPOTIFY_CLIENT_ID: str = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_REDIRECT_URI: str = "http://127.0.0.1:8888/callback"
SPOTIFY_TOKEN_CACHE: str = ".spotify_token_cache"

# Timezone for calendar events — change to your local timezone
TIMEZONE: str = "Asia/Kolkata"

# Contact & credential databases
CONTACTS_DB_PATH: str    = "kobra_contacts.db"
CREDENTIALS_DB_PATH: str = "kobra_credentials.db"

# Browser automation (Playwright)
BROWSER_SESSION_PATH: str = "kobra_browser_session"
BROWSER_HEADLESS: bool    = False   # Set True after first WhatsApp QR scan

# WhatsApp Web selectors — update here if WhatsApp changes their UI
WHATSAPP_SELECTORS: dict = {
    "send_button": '[data-testid="send"]',
    "msg_input":   '[data-testid="conversation-compose-box-input"]',
    "chat_list":   '[data-testid="chat-list"]',
    "qr_code":     '[data-testid="qrcode"]',
}

# ── Window Management + Focus Mode ────────────────────────────────────────────
FOCUS_MODES: dict = {
    "coding": {
        "open":  ["VS Code", "Windows Terminal"],
        "snap":  {"VS Code": "left", "Windows Terminal": "right"},
        "close": ["Chrome", "Discord"],
        "play":  "lofi hip hop",
        "volume": 30,
        "speak": "Focus mode activated, sir. Distractions cleared.",
    },
    "gaming": {
        "open":  [],
        "snap":  {},
        "close": ["VS Code", "Windows Terminal"],
        "volume": 70,
        "speak": "Gaming mode, sir. Good luck.",
    },
    "research": {
        "open":  ["Chrome", "VS Code"],
        "snap":  {"Chrome": "right", "VS Code": "left"},
        "volume": 40,
        "speak": "Research mode. Browser and editor ready, sir.",
    },
    "break": {
        "play":  "chill music",
        "volume": 50,
        "speak": "Break time, sir. Step away for a bit.",
    },
}

# ── ElevenLabs TTS ────────────────────────────────────────────────────────────
USE_ELEVENLABS: bool = False   # Free plan cannot use library voices via API; re-enable after upgrading
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = "TxGEqnHWrfWFTfGW9XjX"   # Josh — confident male
ELEVENLABS_MODEL: str = "eleven_turbo_v2"

# ── Emotion Detection (Hume AI) ───────────────────────────────────────────────
HUME_API_KEY: str = os.getenv("HUME_API_KEY", "")
EMOTION_DETECTION_ENABLED: bool = False
EMOTION_CONFIDENCE_THRESHOLD: float = 0.4

# ── RAG Knowledge Base ────────────────────────────────────────────────────────
import os as _os
_USERNAME = _os.environ.get("USERNAME", "user")
WATCHED_FOLDERS: list[str] = [
    f"C:/Users/{_USERNAME}/Projects",
    f"C:/Users/{_USERNAME}/Documents",
    f"C:/Users/{_USERNAME}/Desktop",
]
RAG_DB_PATH: str = "kobra_rag_db/"
RAG_CHUNK_SIZE: int = 400       # tokens per chunk
RAG_CHUNK_OVERLAP: int = 80     # overlap between chunks
RAG_MAX_FILE_SIZE_MB: int = 10  # skip files larger than this
RAG_TOP_K: int = 5              # chunks to retrieve per query
RAG_EMBED_MODEL: str = "nomic-embed-text"
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Proactive Engine ──────────────────────────────────────────────────────────
PROACTIVE_ENABLED: bool = True
PROACTIVE_CHECK_INTERVAL: int = 30      # seconds between checks
PROACTIVE_LOG_PATHS: list[str] = []     # file paths to watch for errors
PROACTIVE_WATCHED_PROCESSES: list[str] = ["pytest", "python"]

# ── MCP Integration ───────────────────────────────────────────────────────────
MCP_ENABLED: bool = True
MCP_SERVERS: list[dict] = [
    # Add entries like: {"name": "github", "url": "http://localhost:3001", "description": "GitHub"}
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

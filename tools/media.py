"""
tools/media.py — Media playback tools for KOBRA.

play_media    — stream audio directly via yt-dlp Python API + ffplay (background thread)
stop_media    — kill the current stream
control_media — send OS media keys (pause/next/prev) via pyautogui
play_youtube  — open YouTube search in browser for VIDEO watching
"""

import logging
import subprocess
import threading
import webbrowser
from urllib.parse import quote, quote_plus

logger = logging.getLogger(__name__)

# ── Shared playback state ──────────────────────────────────────────────────────
_media_lock = threading.Lock()
_media_process: subprocess.Popen | None = None


# ── Core playback ──────────────────────────────────────────────────────────────

def play_media(query: str) -> str:
    """
    Stream audio directly through the speakers.
    Uses yt-dlp Python API (no PATH dependency) to resolve the stream URL,
    then ffplay to play it. Non-blocking — returns immediately.
    """
    logger.info("[TOOL] play_media: %r", query)
    _stop_current()

    def _stream() -> None:
        global _media_process
        try:
            import yt_dlp

            ydl_opts = {
                "format": "bestaudio/best",
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
            }
            stream_url = None
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=False)
                if info and "entries" in info and info["entries"]:
                    entry = info["entries"][0]
                    # Prefer a direct stream URL; fall back to webpage URL
                    stream_url = entry.get("url") or entry.get("webpage_url")

            if not stream_url:
                logger.warning("yt-dlp found no stream URL for: %r", query)
                return

            logger.info("Streaming: %.80s …", stream_url)
            with _media_lock:
                _media_process = subprocess.Popen(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", stream_url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        except ImportError:
            logger.error("yt-dlp not installed — run: pip install yt-dlp")
        except Exception as exc:
            logger.error("play_media stream thread failed: %s", exc)

    threading.Thread(target=_stream, daemon=True, name="kobra-media").start()
    return f"Playing '{query}' now, sir."


def stop_media() -> str:
    """Stop currently playing audio."""
    logger.info("[TOOL] stop_media")
    _stop_current()
    return "Media stopped, sir."


def control_media(action: str) -> str:
    """Send OS media keys: pause | play | next | previous."""
    logger.info("[TOOL] control_media: %s", action)
    try:
        import pyautogui
        key_map = {
            "pause":    "playpause",
            "play":     "playpause",
            "next":     "nexttrack",
            "previous": "prevtrack",
            "prev":     "prevtrack",
        }
        key = key_map.get(action.lower())
        if key:
            pyautogui.press(key)
            return f"Media {action} sent."
        return f"Unknown media action: {action}. Use: pause, play, next, previous."
    except ImportError:
        return "pyautogui not installed — media keys unavailable."


def play_youtube(query: str) -> str:
    """
    Find the top YouTube video for the query and open it directly in the browser.
    Uses yt-dlp to resolve the first video URL so the video auto-plays — no
    manual clicking required. Falls back to the search results page if yt-dlp
    is unavailable or fails.
    """
    logger.info("[TOOL] play_youtube: %r", query)

    try:
        import yt_dlp  # type: ignore

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "extract_flat": True,   # don't download — just get metadata/URL
            "skip_download": True,
        }
        video_url: str | None = None
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if info and "entries" in info and info["entries"]:
                entry = info["entries"][0]
                video_id = entry.get("id") or entry.get("url", "")
                if video_id and not video_id.startswith("http"):
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                elif video_id.startswith("http"):
                    video_url = video_id

        if video_url:
            logger.info("[TOOL] play_youtube resolved: %s", video_url)
            webbrowser.open(video_url)
            return f"Playing '{query}' on YouTube now, sir."

    except ImportError:
        logger.warning("[TOOL] play_youtube: yt-dlp not installed, falling back to search page.")
    except Exception as exc:
        logger.warning("[TOOL] play_youtube yt-dlp failed (%s), falling back to search page.", exc)

    # Fallback: open YouTube search results — user has to click manually
    fallback_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    webbrowser.open(fallback_url)
    return f"Opened YouTube search for '{query}', sir."


def play_on_spotify(query: str) -> str:
    """
    Search and play a song/artist/playlist on the Spotify desktop app.
    Opens Spotify and performs a search so sir can click play,
    or if Spotify is already open, directly searches within it.
    """
    logger.info("[TOOL] play_on_spotify: %r", query)
    # Use Spotify's URI scheme to search directly within the desktop app
    import subprocess
    import os
    search_uri = f"spotify:search:{quote(query, safe='')}"
    try:
        os.startfile(search_uri)
        return f"Opened Spotify search for '{query}', sir."
    except Exception:
        pass

    # Fallback: open Spotify web player search
    url = f"https://open.spotify.com/search/{quote(query, safe='')}"
    webbrowser.open(url)
    return f"Opened Spotify web for '{query}', sir."


# ── Internal ───────────────────────────────────────────────────────────────────

def _stop_current() -> None:
    global _media_process
    with _media_lock:
        if _media_process and _media_process.poll() is None:
            try:
                _media_process.terminate()
                _media_process.wait(timeout=2)
            except Exception:
                try:
                    _media_process.kill()
                except Exception:
                    pass
            _media_process = None

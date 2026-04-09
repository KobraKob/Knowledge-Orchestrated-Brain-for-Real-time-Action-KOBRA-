"""
speaker.py — Text-to-speech synthesis and audio playback for KOBRA.

Primary:  edge-tts  →  MP3  →  ffplay (Windows)
Fallback: pyttsx3   (offline, no internet required)
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile

import config

logger = logging.getLogger(__name__)


class SpeakerError(Exception):
    pass


class Speaker:
    def __init__(self) -> None:
        self._voice = config.TTS_VOICE
        self._rate = config.TTS_RATE
        self._temp_path = config.AUDIO_TEMP_PATH
        self._use_offline = config.USE_OFFLINE_TTS

        if not self._use_offline:
            self._check_ffplay()

        if self._use_offline:
            self._init_pyttsx3()

        logger.info(
            "Speaker initialised — voice: %s, rate: %s, offline: %s",
            self._voice,
            self._rate,
            self._use_offline,
        )

    # ── Startup checks ─────────────────────────────────────────────────────────

    def _check_ffplay(self) -> None:
        if shutil.which("ffplay") is None:
            raise SpeakerError(
                "ffplay not found in PATH.\n"
                "Install ffmpeg from https://ffmpeg.org/download.html "
                "and add it to your Windows PATH, then restart."
            )

    def _init_pyttsx3(self) -> None:
        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            # Try to pick a decent voice
            voices = self._pyttsx3_engine.getProperty("voices")
            if voices:
                self._pyttsx3_engine.setProperty("voice", voices[0].id)
            self._pyttsx3_engine.setProperty("rate", 175)
        except ImportError:
            logger.warning("pyttsx3 not installed — offline TTS unavailable.")
            self._pyttsx3_engine = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Synthesise and play text. Blocks until playback is complete."""
        if not text or not text.strip():
            return

        text = text.strip()
        logger.debug("Speaking: %s", text[:120])

        if self._use_offline:
            self._speak_offline(text)
        else:
            try:
                asyncio.run(self._speak_edge_tts(text))
            except Exception as exc:
                logger.warning("edge-tts failed (%s) — falling back to pyttsx3.", exc)
                self._speak_offline(text)

    def speak_error(self, message: str) -> None:
        self.speak(f"There was a problem. {message}")

    # ── edge-tts (online) ──────────────────────────────────────────────────────

    async def _speak_edge_tts(self, text: str) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(text, self._voice, rate=self._rate)
        # Write to a named temp file so ffplay gets a real path
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(tmp_fd)
        try:
            await communicate.save(tmp_path)
            self._play_audio(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _play_audio(self, path: str) -> None:
        """Play an audio file with ffplay, suppressing all console output."""
        cmd = [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel", "quiet",
            path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ── pyttsx3 (offline fallback) ─────────────────────────────────────────────

    def _speak_offline(self, text: str) -> None:
        if self._pyttsx3_engine is None:
            logger.error("No TTS engine available — cannot speak.")
            return
        try:
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
        except Exception as exc:
            logger.error("pyttsx3 failed: %s", exc)

    # ── Tone helpers ───────────────────────────────────────────────────────────

    def play_wake_tone(self) -> None:
        """Play a short confirmation beep when wake word is detected."""
        # Use a brief spoken chime via TTS — avoids needing a separate sound file
        self.speak("Mm.")

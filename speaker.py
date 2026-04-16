"""
speaker.py — Text-to-speech synthesis and audio playback for KOBRA.

Priority chain (highest → lowest):
  1. ElevenLabs API  (USE_ELEVENLABS=True, requires API key)
  2. edge-tts        (default, online, no key required)
  3. pyttsx3         (USE_OFFLINE_TTS=True or fallback)
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

        # ElevenLabs client — loaded lazily if enabled
        self._el_client = None
        if config.USE_ELEVENLABS and config.ELEVENLABS_API_KEY:
            self._load_elevenlabs()

        if not self._use_offline and not config.USE_ELEVENLABS:
            self._check_ffplay()

        if self._use_offline:
            self._init_pyttsx3()

        tts_mode = (
            "ElevenLabs" if self._el_client else
            "offline/pyttsx3" if self._use_offline else
            "edge-tts"
        )
        logger.info("Speaker initialised — mode: %s | voice: %s", tts_mode, self._voice)

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

    # ── ElevenLabs ─────────────────────────────────────────────────────────────

    def _load_elevenlabs(self) -> None:
        try:
            from elevenlabs import ElevenLabs
            self._el_client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
            logger.info("ElevenLabs loaded — voice: %s", config.ELEVENLABS_VOICE_ID)
        except ImportError:
            logger.warning("elevenlabs package not installed — falling back to edge-tts.")
            self._el_client = None

    def _speak_elevenlabs(self, text: str) -> None:
        if self._el_client is None:
            raise RuntimeError("ElevenLabs client not initialized.")
        audio_iter = self._el_client.text_to_speech.convert(
            text=text,
            voice_id=config.ELEVENLABS_VOICE_ID,
            model_id=config.ELEVENLABS_MODEL,
            output_format="mp3_44100_128",
        )
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(tmp_fd)
        try:
            with open(tmp_path, "wb") as f:
                for chunk in audio_iter:
                    f.write(chunk)
            self._play_audio(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # ── Public API ─────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Synthesise and play text. Blocks until playback is complete.
        Priority: ElevenLabs → edge-tts → pyttsx3."""
        if not text or not text.strip():
            return

        text = text.strip()
        logger.debug("Speaking: %s", text[:120])

        if self._el_client:
            try:
                self._speak_elevenlabs(text)
                return
            except Exception as exc:
                logger.warning("ElevenLabs failed (%s) — falling back to edge-tts.", exc)

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
        """
        Play a short double-beep acknowledgement when the wake word is detected.
        Falls back to spoken 'Mm-hmm.' if numpy/wave is unavailable.
        """
        try:
            import struct
            import wave
            import tempfile
            import math

            sample_rate = 22050
            # Two short beeps: 880 Hz (0.08s) → 20ms gap → 1100 Hz (0.06s)
            def _sine(freq, duration_ms, amplitude=0.35):
                n = int(sample_rate * duration_ms / 1000)
                return [
                    int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
                    for i in range(n)
                ]

            silence = [0] * int(sample_rate * 0.02)
            samples = _sine(880, 80) + silence + _sine(1100, 60)
            packed = struct.pack(f"<{len(samples)}h", *samples)

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            import os
            os.close(tmp_fd)
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(packed)
            self._play_audio(tmp_path)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        except Exception as exc:
            logger.debug("Wake tone generation failed (%s) — falling back to speech.", exc)
            self.speak("Mm-hmm.")

"""
speaker.py — Text-to-speech synthesis and audio playback for KOBRA.

Priority chain (highest → lowest):
  1. ElevenLabs API   (USE_ELEVENLABS=True, requires API key)
  2. Kokoro TTS       (USE_KOKORO=True, local GPU/CPU, best quality)
  3. edge-tts         (default online fallback, no key required)
  4. pyttsx3          (USE_OFFLINE_TTS=True or last-resort fallback)

Kokoro notes:
  - Uses kokoro-onnx with onnxruntime-gpu for RTX acceleration.
  - Model (~87 MB) is downloaded from HuggingFace on first use and
    cached in KOKORO_MODEL_DIR (default: kobra_kokoro_models/).
  - Voice: am_michael — natural US male, warm and confident.
  - Sample rate: 24 000 Hz. Played via sounddevice (no ffplay needed).
  - Falls back to edge-tts transparently if model load fails.
"""

import asyncio
import logging
import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import wave

import config

logger = logging.getLogger(__name__)


class SpeakerError(Exception):
    pass


# ── Kokoro model paths / constants ────────────────────────────────────────────

_KOKORO_REPO      = "hexgrad/Kokoro-82M-ONNX"
_KOKORO_MODEL_FILE  = "kokoro-v0_19.onnx"
_KOKORO_VOICES_FILE = "voices.bin"
_KOKORO_SAMPLE_RATE = 24_000


class Speaker:
    def __init__(self) -> None:
        self._voice      = config.TTS_VOICE
        self._rate       = config.TTS_RATE
        self._temp_path  = config.AUDIO_TEMP_PATH
        self._use_offline = config.USE_OFFLINE_TTS

        # Kokoro state
        self._kokoro      = None           # KoKoro ONNX instance (loaded lazily)
        self._kokoro_lock = threading.Lock()
        self._kokoro_ready = False
        self._use_kokoro   = getattr(config, "USE_KOKORO", True)
        self._kokoro_voice = getattr(config, "KOKORO_VOICE", "am_michael")

        # ElevenLabs client
        self._el_client = None
        if config.USE_ELEVENLABS and config.ELEVENLABS_API_KEY:
            self._load_elevenlabs()

        if not self._use_offline and not config.USE_ELEVENLABS:
            self._check_ffplay()

        if self._use_offline:
            self._init_pyttsx3()

        # Warm up Kokoro in background so first speak() is fast
        if self._use_kokoro and not self._el_client:
            threading.Thread(target=self._load_kokoro, daemon=True, name="kokoro-init").start()

        tts_mode = (
            "ElevenLabs"        if self._el_client  else
            "Kokoro (loading…)" if self._use_kokoro  else
            "offline/pyttsx3"   if self._use_offline else
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
            logger.warning("elevenlabs package not installed — falling back.")
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

    # ── Kokoro TTS ─────────────────────────────────────────────────────────────

    def _get_kokoro_model_dir(self) -> str:
        base = getattr(config, "KOKORO_MODEL_DIR", "kobra_kokoro_models")
        return os.path.abspath(base)

    def _load_kokoro(self) -> None:
        """
        Download model files if needed and initialise the Kokoro ONNX session.
        Called in a background thread so startup stays fast.
        """
        with self._kokoro_lock:
            if self._kokoro_ready:
                return
            try:
                from kokoro_onnx import Kokoro
                from huggingface_hub import hf_hub_download

                model_dir = self._get_kokoro_model_dir()
                os.makedirs(model_dir, exist_ok=True)

                model_path  = os.path.join(model_dir, _KOKORO_MODEL_FILE)
                voices_path = os.path.join(model_dir, _KOKORO_VOICES_FILE)

                # Download if not cached
                if not os.path.exists(model_path):
                    logger.info("[KOKORO] Downloading model (~87 MB) — one-time only…")
                    model_path = hf_hub_download(
                        repo_id=_KOKORO_REPO,
                        filename=_KOKORO_MODEL_FILE,
                        local_dir=model_dir,
                    )
                if not os.path.exists(voices_path):
                    logger.info("[KOKORO] Downloading voices file…")
                    voices_path = hf_hub_download(
                        repo_id=_KOKORO_REPO,
                        filename=_KOKORO_VOICES_FILE,
                        local_dir=model_dir,
                    )

                self._kokoro = Kokoro(model_path, voices_path)
                self._kokoro_ready = True

                # Log which execution provider is in use
                try:
                    providers = self._kokoro.sess.get_providers()
                    gpu_active = any("CUDA" in p or "DirectML" in p for p in providers)
                    device = "GPU (ONNX)" if gpu_active else "CPU (ONNX)"
                    logger.info("[KOKORO] Ready — voice: %s | device: %s | providers: %s",
                                self._kokoro_voice, device, providers)
                except Exception:
                    logger.info("[KOKORO] Ready — voice: %s", self._kokoro_voice)

            except Exception as exc:
                logger.warning("[KOKORO] Failed to load (%s) — will fall back to edge-tts.", exc)
                self._kokoro_ready = False
                self._use_kokoro   = False

    def _speak_kokoro(self, text: str) -> None:
        """
        Synthesise text with Kokoro ONNX and play via sounddevice.
        Blocks until playback completes (matching edge-tts / ElevenLabs behaviour).
        """
        import sounddevice as sd

        # Wait for background init to finish (max 30 s)
        deadline = 30
        import time
        waited = 0
        while not self._kokoro_ready and self._use_kokoro and waited < deadline:
            time.sleep(0.1)
            waited += 0.1

        if not self._kokoro_ready:
            raise RuntimeError("Kokoro not ready")

        samples, sample_rate = self._kokoro.create(
            text=text,
            voice=self._kokoro_voice,
            speed=1.0,
            lang="en-us",
        )
        # Play synchronously — sounddevice uses portaudio for low-latency output
        sd.play(samples, samplerate=sample_rate)
        sd.wait()

    # ── Public API ─────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Synthesise and play text. Blocks until playback is complete.
        Priority: ElevenLabs → Kokoro → edge-tts → pyttsx3.
        """
        if not text or not text.strip():
            return
        text = text.strip()
        logger.debug("Speaking: %s", text[:120])

        # 1. ElevenLabs
        if self._el_client:
            try:
                self._speak_elevenlabs(text)
                return
            except Exception as exc:
                logger.warning("ElevenLabs failed (%s) — falling back to Kokoro/edge-tts.", exc)

        # 2. Kokoro (local GPU TTS)
        if self._use_kokoro:
            try:
                self._speak_kokoro(text)
                return
            except Exception as exc:
                logger.warning("[KOKORO] Speak failed (%s) — falling back to edge-tts.", exc)

        # 3. edge-tts
        if not self._use_offline:
            try:
                asyncio.run(self._speak_edge_tts(text))
                return
            except Exception as exc:
                logger.warning("edge-tts failed (%s) — falling back to pyttsx3.", exc)

        # 4. pyttsx3 last resort
        self._speak_offline(text)

    def speak_error(self, message: str) -> None:
        self.speak(f"There was a problem. {message}")

    # ── edge-tts (online fallback) ─────────────────────────────────────────────

    async def _speak_edge_tts(self, text: str) -> None:
        import edge_tts
        communicate = edge_tts.Communicate(text, self._voice, rate=self._rate)
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
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ── pyttsx3 (offline last resort) ─────────────────────────────────────────

    def _speak_offline(self, text: str) -> None:
        engine = getattr(self, "_pyttsx3_engine", None)
        if engine is None:
            logger.error("No TTS engine available — cannot speak.")
            return
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            logger.error("pyttsx3 failed: %s", exc)

    # ── Wake tone ──────────────────────────────────────────────────────────────

    def play_wake_tone(self) -> None:
        """
        Play a short double-beep when the wake word is detected.
        Uses sounddevice when Kokoro is loaded (no ffplay needed),
        otherwise falls back to the wave/subprocess path.
        """
        # Try sounddevice path (fast, no temp file)
        if self._use_kokoro:
            try:
                import sounddevice as sd
                import numpy as np

                sr = 22_050
                def _sine(freq: float, ms: int, amp: float = 0.35) -> np.ndarray:
                    t = np.linspace(0, ms / 1000, int(sr * ms / 1000), endpoint=False)
                    return (np.sin(2 * math.pi * freq * t) * amp).astype(np.float32)

                silence = np.zeros(int(sr * 0.02), dtype=np.float32)
                beep = np.concatenate([_sine(880, 80), silence, _sine(1100, 60)])
                sd.play(beep, samplerate=sr)
                sd.wait()
                return
            except Exception:
                pass  # fall through to wave path

        # Wave / ffplay path
        try:
            sr = 22_050
            def _sine_pcm(freq: float, ms: int, amp: float = 0.35) -> list:
                n = int(sr * ms / 1000)
                return [
                    int(amp * 32767 * math.sin(2 * math.pi * freq * i / sr))
                    for i in range(n)
                ]

            silence_pcm = [0] * int(sr * 0.02)
            samples = _sine_pcm(880, 80) + silence_pcm + _sine_pcm(1100, 60)
            packed = struct.pack(f"<{len(samples)}h", *samples)

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_fd)
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(packed)
            self._play_audio(tmp_path)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        except Exception as exc:
            logger.debug("Wake tone failed (%s) — falling back to speech.", exc)
            self.speak("Mm-hmm.")

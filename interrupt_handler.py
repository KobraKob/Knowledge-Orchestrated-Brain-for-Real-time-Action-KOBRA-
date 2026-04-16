"""
interrupt_handler.py — Background voice abort detection for KOBRA v3.

Runs a lightweight monitoring loop in a daemon thread while agents execute.
If "stop", "cancel", "abort", or "never mind" is heard → sets abort_flag.

Uses a dedicated short-capture window (separate sounddevice stream)
so it doesn't interfere with the main listener.
"""

import logging
import math
import os
import struct
import tempfile
import threading
import time
import wave

import sounddevice as sd

import config

logger = logging.getLogger(__name__)

_ABORT_WORDS = {"stop", "cancel", "abort", "never mind", "nevermind", "forget it"}

# Short recording window: 3 seconds max, stops on 0.8s of silence
_INTERRUPT_MAX_SECONDS = 3
_INTERRUPT_SILENCE_CHUNKS = 12   # ~0.8s at 512 samples / 16000 Hz


class InterruptHandler:
    def __init__(self, whisper_model) -> None:
        """
        whisper_model: a loaded faster_whisper.WhisperModel instance
                       (shared with the listener to avoid double RAM usage).
        """
        self._whisper = whisper_model
        self._abort_flag = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start_monitoring(self) -> threading.Event:
        """
        Begin background monitoring. Returns the abort_flag Event.
        Call this before orchestrator.run() so agents can check it.
        """
        self._abort_flag.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="kobra-interrupt",
        )
        self._thread.start()
        return self._abort_flag

    def stop_monitoring(self) -> None:
        """Signal the monitoring thread to exit."""
        self._stop_event.set()

    def reset(self) -> None:
        """Clear the abort flag ready for the next command."""
        self._abort_flag.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        frame_length = 512
        sample_rate = config.SAMPLE_RATE

        while not self._stop_event.is_set() and not self._abort_flag.is_set():
            transcript = self._capture_short(frame_length, sample_rate)
            if transcript:
                t = transcript.lower()
                if any(w in t for w in _ABORT_WORDS):
                    logger.info("[INTERRUPT] Abort word detected: %r", transcript)
                    self._abort_flag.set()
                    return

    def _capture_short(self, frame_length: int, sample_rate: int) -> str:
        """
        Record up to _INTERRUPT_MAX_SECONDS, stopping on silence.
        Returns Whisper transcript or empty string.
        """
        frames: list[bytes] = []
        silent_chunks = 0
        max_chunks = int(_INTERRUPT_MAX_SECONDS * sample_rate / frame_length)

        try:
            with sd.RawInputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=frame_length,
            ) as stream:
                for _ in range(max_chunks):
                    if self._stop_event.is_set():
                        return ""
                    raw, _ = stream.read(frame_length)
                    chunk = bytes(raw)
                    frames.append(chunk)

                    rms = self._rms(chunk)
                    if rms < config.SILENCE_THRESHOLD:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0

                    if silent_chunks >= _INTERRUPT_SILENCE_CHUNKS:
                        break
        except Exception as exc:
            logger.debug("Interrupt capture error (ignored): %s", exc)
            time.sleep(0.1)
            return ""

        if not frames:
            return ""

        return self._transcribe(frames, sample_rate)

    def _transcribe(self, frames: list[bytes], sample_rate: int) -> str:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(b"".join(frames))
            segments, _ = self._whisper.transcribe(
                tmp_path, language="en", beam_size=1
            )
            return " ".join(seg.text for seg in segments).strip()
        except Exception:
            return ""
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _rms(raw: bytes) -> float:
        count = len(raw) // 2
        if count == 0:
            return 0.0
        shorts = struct.unpack_from(f"{count}h", raw)
        return math.sqrt(sum(s * s for s in shorts) / count)

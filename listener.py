"""
listener.py — Wake word detection (Porcupine) and speech capture + STT (faster-whisper).

Uses sounddevice instead of PyAudio — no C++ build tools required on Windows.

Flow:
  1. wait_for_wake_word() blocks until the wake word is heard (used once at startup).
  2. capture_speech() records audio until silence, then transcribes with Whisper.
     Called in a continuous loop after wake word fires — no repeated wake word needed.
"""

import logging
import math
import os
import struct
import tempfile
import wave

import sounddevice as sd
import pvporcupine

import config

logger = logging.getLogger(__name__)


class ListenerError(Exception):
    pass


class Listener:
    def __init__(self) -> None:
        # ── Porcupine ──────────────────────────────────────────────────────────
        # Use a custom .ppn if provided, otherwise fall back to a built-in keyword.
        if config.WAKE_WORD_PATH and os.path.isfile(config.WAKE_WORD_PATH):
            logger.info("Using custom wake word model: %s", config.WAKE_WORD_PATH)
            self._porcupine = pvporcupine.create(
                access_key=config.PORCUPINE_ACCESS_KEY,
                keyword_paths=[config.WAKE_WORD_PATH],
            )
        else:
            keyword = config.WAKE_WORD_KEYWORD
            logger.info("Using built-in wake word keyword: '%s'", keyword)
            self._porcupine = pvporcupine.create(
                access_key=config.PORCUPINE_ACCESS_KEY,
                keywords=[keyword],
            )

        self._frame_length = self._porcupine.frame_length  # samples per chunk
        self._sample_rate = self._porcupine.sample_rate    # always 16000

        # ── faster-whisper ─────────────────────────────────────────────────────
        logger.info("Loading Whisper model '%s' …", config.WHISPER_MODEL)
        from faster_whisper import WhisperModel
        self._whisper = WhisperModel(
            config.WHISPER_MODEL,
            device="cpu",
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )

        logger.info(
            "Listener ready — wake word: '%s' | sample rate: %d Hz | frame: %d samples",
            config.WAKE_WORD_KEYWORD if not config.WAKE_WORD_PATH else config.WAKE_WORD_PATH,
            self._sample_rate,
            self._frame_length,
        )

    # ── Wake word ──────────────────────────────────────────────────────────────

    def wait_for_wake_word(self) -> None:
        """Block until the wake word is detected."""
        logger.info("Waiting for wake word …")

        with sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._frame_length,
        ) as stream:
            while True:
                raw, _ = stream.read(self._frame_length)
                pcm = list(struct.unpack_from(f"{self._frame_length}h", bytes(raw)))
                result = self._porcupine.process(pcm)
                if result >= 0:
                    logger.info("Wake word detected.")
                    return

    # ── Speech capture ─────────────────────────────────────────────────────────

    def capture_speech(self) -> str:
        """
        Record audio until silence, then return the Whisper transcript.
        Opens a fresh audio stream each call so it never competes with wake word detection.
        Returns an empty string if nothing captured or STT failed.
        """
        logger.info("Recording …")
        frames: list[bytes] = []
        silent_chunks = 0
        max_chunks = int(
            config.MAX_RECORD_SECONDS * self._sample_rate / self._frame_length
        )

        with sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._frame_length,
        ) as stream:
            for _ in range(max_chunks):
                raw, _ = stream.read(self._frame_length)
                chunk = bytes(raw)
                frames.append(chunk)

                if self._rms(chunk) < config.SILENCE_THRESHOLD:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if silent_chunks >= config.SILENCE_CHUNKS:
                    break

        if not frames:
            return ""

        return self._transcribe(frames)

    @staticmethod
    def _rms(raw: bytes) -> float:
        count = len(raw) // 2
        if count == 0:
            return 0.0
        shorts = struct.unpack_from(f"{count}h", raw)
        return math.sqrt(sum(s * s for s in shorts) / count)

    def _transcribe(self, frames: list[bytes]) -> str:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                wf.writeframes(b"".join(frames))

            segments, _ = self._whisper.transcribe(
                tmp_path,
                language="en",
                beam_size=5,
            )
            text = " ".join(seg.text for seg in segments).strip()
            logger.info("Transcribed: %r", text)
            return text

        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)
            return ""

        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        self._porcupine.delete()
        logger.info("Listener cleaned up.")

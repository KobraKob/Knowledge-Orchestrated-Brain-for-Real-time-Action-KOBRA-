"""
main.py — Entry point and main loop for KOBRA.

Conversation model:
  - Wake word fires ONCE at startup to activate KOBRA.
  - After that, KOBRA listens continuously — no need to repeat the wake word.
  - Say "go to sleep", "sleep", or "goodbye" to send KOBRA back to sleep mode.
  - Wake it again by saying the wake word.
"""

import logging
import re
import sys

import config

config.setup_logging()
logger = logging.getLogger(__name__)

# Phrases that put KOBRA back to sleep (return to wake-word detection)
SLEEP_TRIGGERS = ("go to sleep", "sleep", "goodbye", "good night", "shut up")


def health_check() -> None:
    errors: list[str] = []
    if not config.GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not set in .env")
    if not config.PORCUPINE_ACCESS_KEY:
        errors.append("PORCUPINE_ACCESS_KEY is not set in .env")
    if errors:
        for err in errors:
            logger.critical("STARTUP ERROR: %s", err)
        print("\n".join(f"[STARTUP ERROR] {e}" for e in errors))
        sys.exit(1)


def is_meaningful(text: str) -> bool:
    """
    Return True only if the transcript contains real words.
    Rejects Whisper artifacts like '.  .  .  .', '...', lone punctuation,
    or anything with fewer than 3 alphabetic characters.
    """
    letters = re.sub(r'[^a-zA-Z]', '', text)
    return len(letters) >= 3


def is_sleep_command(text: str) -> bool:
    t = text.lower().strip()
    return any(phrase in t for phrase in SLEEP_TRIGGERS)


def is_clear_memory_command(text: str) -> bool:
    t = text.lower().strip()
    return any(p in t for p in ("clear your memory", "clear memory", "forget everything"))


def main() -> None:
    health_check()

    from memory import Memory
    from speaker import Speaker, SpeakerError
    from listener import Listener, ListenerError
    from brain import Brain, BrainError
    from tools import TOOL_REGISTRY

    logger.info("Initialising KOBRA …")

    memory = Memory()

    try:
        speaker = Speaker()
    except SpeakerError as exc:
        logger.critical("Speaker failed: %s", exc)
        print(f"[STARTUP ERROR] {exc}")
        sys.exit(1)

    try:
        listener = Listener()
    except ListenerError as exc:
        logger.critical("Listener failed: %s", exc)
        speaker.speak_error(str(exc))
        sys.exit(1)

    brain = Brain(memory=memory, tool_registry=TOOL_REGISTRY)

    # ── Main loop ─────────────────────────────────────────────────────────────
    # State: "sleeping" = waiting for wake word | "active" = continuous conversation
    state = "sleeping"

    while True:
        try:
            # ── SLEEPING: wait for wake word ───────────────────────────────────
            if state == "sleeping":
                listener.wait_for_wake_word()

                # Wake word detected — greet and go active
                speaker.play_wake_tone()
                greeting = brain.generate_greeting()
                logger.info("Greeting: %s", greeting)
                speaker.speak(greeting)
                state = "active"
                continue

            # ── ACTIVE: continuous conversation ────────────────────────────────
            transcript = listener.capture_speech()

            if not transcript or not is_meaningful(transcript):
                continue

            logger.info("User said: %r", transcript)

            # Sleep command — go back to wake-word mode
            if is_sleep_command(transcript):
                speaker.speak("Going to sleep, sir. Say Jarvis when you need me.")
                state = "sleeping"
                continue

            # Clear memory command — handled without LLM
            if is_clear_memory_command(transcript):
                memory.clear_conversations()
                speaker.speak("Conversation history cleared, sir. Facts are still intact.")
                continue

            # Route through the LLM brain
            try:
                response = brain.process(transcript)
            except BrainError as exc:
                logger.error("Brain error: %s", exc)
                speaker.speak(str(exc))
                continue

            if not response:
                continue

            memory.save_conversation_turn("user", transcript)
            memory.save_conversation_turn("assistant", response)
            speaker.speak(response)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down.")
            speaker.speak("Shutting down, sir. Good luck out there.")
            break

        except Exception as exc:
            logger.exception("Unhandled error: %s", exc)
            try:
                speaker.speak_error("Something went wrong, sir. Still here though.")
            except Exception:
                pass
            continue

    listener.cleanup()
    memory.close()
    logger.info("KOBRA offline.")


if __name__ == "__main__":
    main()

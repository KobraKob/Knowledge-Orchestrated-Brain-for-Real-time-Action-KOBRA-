"""
main.py — Entry point for KOBRA v4 (multi-agent + proactive + RAG + vision + MCP).

Startup:
  UI Server → Memory → Speaker → Listener → Brain
  → ProactiveEngine → RAG (optional) → MCP (optional)
  → Orchestrator → InterruptHandler → greet

Active loop (after wake word):
  listen (or UI text command) → orchestrator.run(transcript, abort_flag) → speak → repeat
  say "go to sleep" / "goodbye" to return to wake-word mode.

VS Code context injection:
  When VS Code is the foreground window, "[CONTEXT: VS Code is active]" is prepended
  to the transcript, routing to DevAgent's Code Assistant Mode.
"""

import logging
import re
import sys
import threading
import time
import queue as _queue

import config
from learning import LearningSystem

# v5 memory + proactive systems
try:
    from memory.episodic  import EpisodicMemory
    from memory.semantic  import SemanticMemory
    from memory.procedural import ProceduralMemory
    from memory.perceptual import PerceptualMemory
    from memory.router    import MemoryRouter as TypedMemoryRouter
    _V5_MEMORY_AVAILABLE = True
except ImportError:
    _V5_MEMORY_AVAILABLE = False

try:
    from proactive.briefing import MorningBriefingEngine, handle_post_briefing_response
    from proactive.watcher  import ContinuousWatcher
    _V5_PROACTIVE_AVAILABLE = True
except ImportError:
    _V5_PROACTIVE_AVAILABLE = False

config.setup_logging()
logger = logging.getLogger(__name__)

SLEEP_TRIGGERS = ("go to sleep", "sleep", "goodbye", "good night", "shut up", "stop listening")


# ── Noise / silence rejection ──────────────────────────────────────────────────

_NOISE_WORDS = {
    "you", "so", "oh", "um", "hmm", "uh", "yeah", "okay", "ok",
    "hm", "mhm", "hi", "hey", "huh", "ah", "eh", "yep", "nope",
    "well", "right", "sure", "yup", "nah",
}

# Whisper sometimes hallucinates the initial_prompt text when given near-silence.
# These fragments — if they appear as the entire transcript — are false positives.
_WHISPER_HALLUCINATION_FRAGMENTS = {
    "the user is giving voice commands to kobra",
    "the user is giving voice commands",
    "voice commands to kobra",
    "giving voice commands",
    "an ai assistant",
    "commands include",
    "install software",
    "control volume",
    "thank you for watching",       # common Whisper hallucination on silence
    "please subscribe",
    "subtitles by",
}


# ── Speaking guard — prevents listener from processing its own voice ──────────
# Set to True while speaker.speak() is running. capture_speech() returns "" while
# this flag is set, dropping the audio without running Whisper.
_kobra_speaking = threading.Event()


def is_meaningful(text: str) -> bool:
    """Reject Whisper silence artifacts, noise, and initial_prompt hallucinations."""
    stripped = text.strip()
    letters = re.sub(r"[^a-zA-Z]", "", stripped)

    # Too short
    if len(letters) < 4:
        return False

    # Single-word noise
    words = stripped.lower().split()
    if len(words) == 1 and words[0] in _NOISE_WORDS:
        return False

    # Whisper hallucinated the initial_prompt or another known artifact
    lower = stripped.lower()
    if any(frag in lower for frag in _WHISPER_HALLUCINATION_FRAGMENTS):
        logger.debug("Rejected Whisper hallucination: %r", stripped)
        return False

    return True


def is_sleep_command(text: str) -> bool:
    t = text.lower().strip()
    return any(phrase in t for phrase in SLEEP_TRIGGERS)


def is_clear_memory(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ("clear your memory", "clear memory", "forget everything"))


# ── Auto fact extraction ───────────────────────────────────────────────────────

_FACT_PATTERNS: list[tuple] = []  # populated lazily


def auto_extract_facts(transcript: str, memory) -> None:
    """
    Silently detect and persist facts the user mentions in conversation.
    Runs after every active turn — no API call needed.
    """
    import re as _re
    t = transcript.lower().strip()

    patterns = [
        (_re.compile(r"\bmy name is ([a-z][a-z ]{1,30})"),                                        "user_name"),
        (_re.compile(r"\bcall me ([a-z][a-z ]{1,20})"),                                           "user_name"),
        (_re.compile(r"\bi(?:'m| am) from ([a-z][a-z ]{1,30})(?:\s*[,.]|$)"),                    "user_location"),
        (_re.compile(r"\bi live in ([a-z][a-z ]{1,30})(?:\s*[,.]|$)"),                           "user_location"),
        (_re.compile(r"\bi(?:'m| am) ([0-9]{1,3}) years? old"),                                   "user_age"),
        (_re.compile(r"\bi work (?:at|for) ([a-z][a-z ]{1,40})(?:\s*[,.]|$)"),                   "user_employer"),
        (_re.compile(r"\bmy (?:favorite|favourite) (?:music|artist|song|band) is ([a-z][a-z ]{1,40})(?:\s*[,.]|$)"), "favorite_music"),
        (_re.compile(r"\bmy (?:favorite|favourite) food is ([a-z][a-z ]{1,40})(?:\s*[,.]|$)"),   "favorite_food"),
        (_re.compile(r"\bmy (?:favorite|favourite) (?:show|series|movie) is ([a-z][a-z ]{1,40})(?:\s*[,.]|$)"), "favorite_show"),
        (_re.compile(r"\bi(?:'m| am) (?:a |an )([a-z][a-z ]{2,40})(?:\s*[,.]|$)"),              "user_profession"),
        (_re.compile(r"\bi(?:'m| am) studying ([a-z][a-z ]{2,40})(?:\s*[,.]|$)"),                "user_study"),
        (_re.compile(r"\bmy (?:preferred|favourite|favorite) (?:language|editor|ide) is ([a-z][a-z+#. ]{1,30})(?:\s*[,.]|$)"), "preferred_tool"),
    ]

    for pattern, key in patterns:
        m = pattern.search(t)
        if m:
            value = m.group(1).strip().rstrip(".,!? ")
            if 2 < len(value) < 60:
                memory.save_fact(key, value)
                logger.info("Auto-extracted fact: %s = %r", key, value)
                try:
                    from ui_server import post_event
                    post_event("fact_saved", key=key, value=value)
                except Exception:
                    pass


# ── Health check ───────────────────────────────────────────────────────────────

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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    health_check()

    # ── Launch UI server in daemon thread ──────────────────────────────────────
    import ui_server
    ui_thread = threading.Thread(
        target=ui_server.run,
        daemon=True,
        name="kobra-ui",
    )
    ui_thread.start()
    ui_server._server_started.wait(timeout=5)

    # Open browser automatically
    import webbrowser, time
    time.sleep(0.8)
    webbrowser.open_new_tab(ui_server.UI_URL)

    from ui_server import post_event

    # ── Core imports ───────────────────────────────────────────────────────────
    from memory import Memory
    from speaker import Speaker, SpeakerError
    from listener import Listener, ListenerError
    from brain import Brain
    from tools import TOOL_REGISTRY
    from orchestrator import Orchestrator
    from interrupt_handler import InterruptHandler
    from kobra_events import ui_command_queue
    from credential_store import CredentialStore
    from contact_store import ContactStore

    logger.info("Initialising KOBRA v4 …")
    post_event("status", state="standby", label="INITIALISING")

    memory = Memory()

    # Push all stored facts to UI on startup
    facts = memory.get_all_facts()
    if facts:
        post_event("facts_snapshot", facts={f["key"]: f["value"] for f in facts})

    try:
        speaker = Speaker()
    except SpeakerError as exc:
        logger.critical("Speaker failed: %s", exc)
        sys.exit(1)

    try:
        listener = Listener()
    except ListenerError as exc:
        logger.critical("Listener failed: %s", exc)
        sys.exit(1)

    # Integration stores — safe to init even without credentials configured yet
    credential_store = CredentialStore()
    contact_store = ContactStore()

    brain = Brain(memory=memory, tool_registry=TOOL_REGISTRY)
    # Wire UI event emitter into brain via callback — keeps brain.py decoupled from ui_server
    brain.set_event_callback(post_event)

    # ── RAG Knowledge Base (optional — requires Ollama + chromadb) ─────────────
    retriever = None
    watcher = None
    try:
        from rag import FileIndexer, Embedder, VectorStore, Retriever, FolderWatcher
        _indexer  = FileIndexer()
        _embedder = Embedder()
        _store    = VectorStore()
        retriever = Retriever(_embedder, _store)
        watcher   = FolderWatcher(_indexer, _embedder, _store)
        # Initial index + start live watching in background
        threading.Thread(
            target=lambda: (
                watcher.index_all(config.WATCHED_FOLDERS),
                watcher.start(config.WATCHED_FOLDERS),
            ),
            daemon=True,
            name="rag-init",
        ).start()
        logger.info("RAG knowledge base initialised.")
    except Exception as exc:
        logger.info("RAG disabled (Ollama/chromadb not available): %s", exc)

    # ── MCP client (optional — requires MCP_SERVERS in config) ────────────────
    mcp_client = None
    if config.MCP_ENABLED and config.MCP_SERVERS:
        try:
            from mcp import MCPClient
            mcp_client = MCPClient()
            mcp_client.register_from_config(config.MCP_SERVERS)
        except Exception as exc:
            logger.info("MCP disabled: %s", exc)

    orchestrator = Orchestrator(
        brain=brain,
        memory=memory,
        tool_registry=TOOL_REGISTRY,
        credential_store=credential_store,
        contact_store=contact_store,
        retriever=retriever,
        mcp_client=mcp_client,
    )
    interrupt_handler = InterruptHandler(whisper_model=listener._whisper)

    # ── Proactive Engine ───────────────────────────────────────────────────────
    idle_checker = None
    if config.PROACTIVE_ENABLED:
        try:
            from proactive import (
                ProactiveEngine, ProcessChecker, BehaviorChecker,
                BuildWatcherChecker, IdleChecker, TaskScheduler,
            )
            proactive_engine = ProactiveEngine(speaker=speaker, memory=memory)
            proactive_engine.register_checker(BehaviorChecker())
            proactive_engine.register_checker(
                ProcessChecker(config.PROACTIVE_WATCHED_PROCESSES)
            )
            if config.PROACTIVE_LOG_PATHS:
                proactive_engine.register_checker(
                    BuildWatcherChecker(config.PROACTIVE_LOG_PATHS)
                )
            # Idle checker — speaks after ~2.5 min of user silence
            idle_checker = IdleChecker()
            proactive_engine.register_checker(idle_checker)
            # Wire calendar checker if integration agent has a calendar client
            integration_agent = orchestrator._agents.get("integration")
            if integration_agent and hasattr(integration_agent, "_calendar"):
                from proactive import CalendarChecker
                proactive_engine.register_checker(
                    CalendarChecker(integration_agent._calendar)
                )
            # Morning briefing scheduler
            scheduler = TaskScheduler(speaker)
            scheduler.add(
                "morning_brief", "0 9 * * *",
                lambda: brain.process(
                    "Give me a morning briefing: today's date, any calendar events today, "
                    "and one line of dry motivation."
                ),
            )
            proactive_engine.set_scheduler(scheduler)
            proactive_engine.start()
            logger.info("Proactive engine started.")
        except Exception as exc:
            logger.info("Proactive engine disabled: %s", exc)

    post_event("status", state="standby", label="STANDBY")
    post_event("tool_call", tool="SYSTEM", summary="All modules online · Waiting for wake word")

    # ── State machine ──────────────────────────────────────────────────────────
    state = "sleeping"   # "sleeping" | "active"

    # Learning system — same instance as brain._learning, accessed via brain
    # Tracks response cut-offs and usage patterns from the main loop
    _last_transcript: str = ""
    _last_agent: str = "general"
    _last_response: str = ""

    while True:
        try:
            # ── SLEEPING ──────────────────────────────────────────────────────
            if state == "sleeping":
                post_event("status", state="standby", label="DORMANT")
                listener.wait_for_wake_word()
                post_event("status", state="wake_word", label="ACTIVATED")
                speaker.play_wake_tone()
                greeting = brain.generate_greeting()
                logger.info("Greeting: %s", greeting)

                # Initialize v5 morning briefing engine (if available)
                _briefing_engine = None
                _watcher = None
                if _V5_PROACTIVE_AVAILABLE and _V5_MEMORY_AVAILABLE:
                    try:
                        # Build typed memory layers
                        _episodic_mem = EpisodicMemory(db_path=config.DB_PATH)
                        _semantic_mem = SemanticMemory(db_path=config.DB_PATH)

                        # Wire semantic memory into brain's memory router
                        if hasattr(brain, '_memory_router') and hasattr(brain._memory_router, '_semantic'):
                            brain._memory_router._semantic = _semantic_mem
                        if hasattr(brain, '_memory_router') and hasattr(brain._memory_router, '_episodic'):
                            brain._memory_router._episodic = _episodic_mem

                        _briefing_engine = MorningBriefingEngine(
                            brain=brain,
                            episodic_memory=_episodic_mem,
                            semantic_memory=_semantic_mem,
                        )
                        logger.info("v5 MorningBriefingEngine initialized.")
                    except Exception as exc:
                        logger.warning("v5 briefing engine init failed: %s", exc)
                        _briefing_engine = None

                post_event("status", state="speaking", label="GREETING")
                post_event("response", text=greeting)
                _kobra_speaking.set()
                try:
                    speaker.speak(greeting)
                finally:
                    _kobra_speaking.clear()

                # Phase 2: launch background morning scan (non-blocking)
                if _briefing_engine is not None:
                    try:
                        def _deliver_briefing(briefing_text: str):
                            if briefing_text and state == "active":
                                post_event("response", text=briefing_text)
                                _kobra_speaking.set()
                                try:
                                    speaker.speak(briefing_text)
                                finally:
                                    _kobra_speaking.clear()
                        _briefing_engine.run_async(callback=_deliver_briefing, delay=1.0)
                        logger.info("Morning briefing scan launched in background.")
                    except Exception as exc:
                        logger.warning("Briefing async launch failed: %s", exc)

                state = "active"
                post_event("status", state="listening", label="LISTENING")
                continue

            # ── ACTIVE ─────────────────────────────────────────────────────────

            # Check for UI text command first (non-blocking)
            transcript = None
            try:
                transcript = ui_command_queue.get_nowait()
                logger.info("UI command: %r", transcript)
            except _queue.Empty:
                pass

            # If no UI command, do voice capture (suppressed while KOBRA is speaking)
            if not transcript:
                post_event("status", state="listening", label="LISTENING")
                transcript = listener.capture_speech(speaking_flag=_kobra_speaking)

            if not transcript or not is_meaningful(transcript):
                continue

            # Code Assistant Mode — inject VS Code context if it's the foreground window
            foreground = listener.foreground_app
            if any(kw in foreground.lower() for kw in ("visual studio code", "code - ")):
                transcript = f"[CONTEXT: VS Code is active] {transcript}"

            logger.info("User said: %r", transcript)
            post_event("transcript", text=transcript)

            # Track user activity for watcher flow suppression
            if _watcher is not None:
                try:
                    _watcher.record_activity()
                except Exception:
                    pass

            if is_sleep_command(transcript):
                post_event("status", state="standby", label="GOING QUIET")
                _kobra_speaking.set()
                try:
                    speaker.speak("Going quiet, sir. Just say Jarvis when you need me.")
                finally:
                    _kobra_speaking.clear()
                state = "sleeping"
                continue

            if is_clear_memory(transcript):
                memory.clear_conversations()
                _kobra_speaking.set()
                try:
                    speaker.speak("Conversation history cleared, sir.")
                finally:
                    _kobra_speaking.clear()
                post_event("tool_call", tool="MEMORY", summary="Conversation history cleared")
                continue

            # ── Processing ─────────────────────────────────────────────────────
            post_event("status", state="thinking", label="PROCESSING")
            abort_flag = interrupt_handler.start_monitoring()

            # Store for learning system cut-off detection
            _last_transcript = transcript
            _last_agent = getattr(brain, '_last_agent', 'general')

            try:
                response = orchestrator.run(transcript, abort_flag)
            except Exception as exc:
                logger.exception("Orchestrator error: %s", exc)
                response = f"Something went wrong, sir — {exc}"
                post_event("error", message=str(exc)[:120])
            finally:
                interrupt_handler.stop_monitoring()
                interrupt_handler.reset()

            if response and is_meaningful(response):
                _last_response = response
                # Capture agent name after orchestrator sets it
                _last_agent = getattr(brain, '_last_agent', 'general')

                memory.save_conversation_turn("user", transcript)
                memory.save_conversation_turn("assistant", response)
                auto_extract_facts(transcript, memory)
                # Reset idle clock — user just interacted
                if idle_checker:
                    idle_checker.update_activity()
                post_event("status", state="speaking", label="RESPONDING")
                post_event("response", text=response)
                _kobra_speaking.set()
                try:
                    speaker.speak(response)
                finally:
                    _kobra_speaking.clear()
                    # Brief cooldown so any mic bleed from the last TTS word clears
                    time.sleep(0.25)

                # Post-briefing preference learning
                if _V5_PROACTIVE_AVAILABLE and _briefing_engine is not None:
                    try:
                        if hasattr(_briefing_engine, '_semantic') and _briefing_engine._semantic:
                            handle_post_briefing_response(transcript, _briefing_engine._semantic)
                    except Exception:
                        pass

                # Detect whether user interrupted during TTS (abort_flag set)
                was_cut_off = abort_flag.is_set() if abort_flag is not None else False
                try:
                    brain._learning.log_response(
                        transcript=_last_transcript,
                        agent=_last_agent,
                        response_len=len(response),
                        was_cut_off=was_cut_off,
                    )
                except Exception:
                    pass
                # Log to execution journal
                try:
                    from task_queue import get_journal
                    get_journal().log_response(_last_transcript, response, was_cut_off)
                except Exception:
                    pass

                post_event("status", state="listening", label="LISTENING")

        except KeyboardInterrupt:
            logger.info("Shutting down.")
            post_event("status", state="standby", label="OFFLINE")
            # edge-tts uses asyncio.run() which re-raises KeyboardInterrupt if the
            # event loop is already in a cancelled state. Use the offline TTS fallback
            # (pyttsx3) for the shutdown line — it's synchronous and always safe.
            try:
                speaker._speak_offline("Shutting down, sir.")
            except Exception:
                pass
            # Close session in episodic memory
            if _V5_MEMORY_AVAILABLE and '_episodic_mem' in dir():
                try:
                    _episodic_mem.close_session(summary="Session ended.")
                except Exception:
                    pass
            # Stop watcher
            if _watcher is not None:
                try:
                    _watcher.stop()
                except Exception:
                    pass
            break

        except Exception as exc:
            logger.exception("Unhandled main loop error: %s", exc)
            post_event("error", message=str(exc)[:120])
            try:
                interrupt_handler.stop_monitoring()
                interrupt_handler.reset()
                speaker.speak_error("Something went wrong, sir. Still here.")
            except Exception:
                pass
            continue

    listener.cleanup()
    memory.close()
    credential_store.close()
    contact_store.close()

    # Close Playwright browser if BrowserAgent was used
    browser_agent = orchestrator._agents.get("browser")
    if browser_agent:
        browser_agent.cleanup()

    # Stop file watcher
    if watcher:
        try:
            watcher.stop()
        except Exception:
            pass

    logger.info("KOBRA v4 offline.")


if __name__ == "__main__":
    main()

"""
brain.py — Groq LLM integration with tool calling for KOBRA.

Intent routing (zero-latency, local heuristics):
  DIRECT  → opinion / hypothetical / static knowledge → answered without tools
  SEARCH  → real-time / current data needed          → forced web_search
  ACTION  → clear command verb detected              → full tool-call loop
  AUTO    → ambiguous                               → full tool-call loop (Groq decides)

Two-turn tool-call flow (for ACTION / AUTO):
  Turn 1: transcript → Groq (with tools) → tool decision
  Turn 2: tool result → Groq (no tools)  → clean spoken response

speak_only short-circuits the loop (no Turn 2).
save_memory / recall_memory handled inline (need the Memory instance directly).
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Callable

from groq import Groq, APIStatusError, APIConnectionError

import config
from memory import Memory

logger = logging.getLogger(__name__)

# ── Intent routing signals ────────────────────────────────────────────────────

_OPINION_SIGNALS = [
    "who would win", "who will win", "who wins", "who would beat", "could beat",
    "would beat", "who's stronger", "who is stronger", "who's better", "who is better",
    "which is better", "which is worse", "your opinion", "what do you think",
    "do you think", "do you prefer", "would you rather", "vs ", " versus ",
    "hypothetically", "imagine if", "what if ", "should i", "is it worth",
    "rate ", "rank ", "better or worse", "best of", "worst of", "recommend me",
    "tell me a joke", "say something funny", "roast me", "fist fight", "who would survive",
]

_REALTIME_SIGNALS = [
    "weather", "forecast", "temperature outside", "news", "headlines",
    "score", "match result", "live score", "price of", "cost of",
    "stock price", "crypto", "bitcoin", "ethereum", "today's",
    "right now", "currently", "what's happening", "latest news",
    "just released", "new release", "just came out", "this week's",
    "trending", "who won", "election", "breaking",
]

_ACTION_VERBS = [
    "open ", "launch ", "start ", "run ", "play ", "pause ", "stop ",
    "create ", "make ", "install ", "type ", "click ", "press ",
    "write ", "download ", "search for ", "go to ", "close ",
    "minimize ", "maximize ", "copy ", "paste ", "mute ", "unmute ",
    "volume ", "screenshot", "scaffold ", "build me ", "set up ",
    "listen to ", "put on ", "queue up ",
]

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    # ── Conversation ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "speak_only",
            "description": (
                "Respond conversationally with no action needed. "
                "Use for greetings, explanations, opinions, jokes, general knowledge, "
                "and anything that does not require executing a task on the PC."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The spoken response to deliver to sir.",
                    }
                },
                "required": ["response"],
            },
        },
    },

    # ── System ─────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open an application on the user's PC by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "App to open. E.g. 'Chrome', 'VS Code', 'Spotify', 'Notepad'.",
                    }
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command for developer tasks: running scripts, "
                "installing packages, git, etc. Not for opening apps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command. E.g. 'pip install flask', 'git status'.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get current system info: CPU, RAM, battery, time, date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info_type": {
                        "type": "string",
                        "enum": ["time", "date", "battery", "cpu", "ram", "all"],
                    }
                },
                "required": ["info_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_folder",
            "description": "Create a new folder at the specified path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Folder path to create."}
                },
                "required": ["path"],
            },
        },
    },

    # ── Keyboard / Mouse / Clipboard ───────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": (
                "Type text at the current cursor position, as if physically typing on the keyboard. "
                "Use when sir asks to type or write something in an open application."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to type."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_hotkey",
            "description": (
                "Press a keyboard shortcut or key combination. "
                "E.g. 'ctrl+c', 'alt+tab', 'win+d', 'ctrl+shift+t', 'alt+f4'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "string",
                        "description": "Keys joined with +. E.g. 'ctrl+v', 'alt+f4', 'win+d'.",
                    }
                },
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_volume",
            "description": "Control system audio volume: raise, lower, mute, or unmute.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["up", "down", "mute", "unmute"],
                        "description": "What to do with the volume.",
                    },
                    "steps": {
                        "type": "integer",
                        "description": "How many steps to adjust (default 5, each ~2%).",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_clipboard",
            "description": "Read and return the current clipboard text content.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_clipboard",
            "description": "Copy text to the clipboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to copy to clipboard."}
                },
                "required": ["text"],
            },
        },
    },

    # ── Media ──────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "play_media",
            "description": (
                "Stream and ACTUALLY PLAY audio/music directly through the speakers using yt-dlp. "
                "Use this for ALL music and audio playback requests — songs, artists, albums, genres, podcasts. "
                "The audio plays immediately without the user having to click anything. "
                "Prefer this over play_youtube for anything audio-related."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to play. E.g. 'Eminem Lose Yourself', 'Lana Del Rey latest song', 'lofi beats'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_media",
            "description": "Stop the currently playing audio/music.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_media",
            "description": "Control media playback: pause, resume, next track, or previous track.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["pause", "play", "next", "previous"],
                        "description": "The media control action.",
                    }
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_youtube",
            "description": (
                "Open YouTube in the browser to WATCH a video (not for audio/music — use play_media for that). "
                "Use when sir specifically wants to watch a video, not just listen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for on YouTube."}
                },
                "required": ["query"],
            },
        },
    },

    # ── Web ────────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a URL or website in the browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to open."}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web and return a summary of results. "
                "Use ONLY for real-time or current information: weather, news, sports scores, "
                "stock prices, recently released content, live events. "
                "Do NOT use for general knowledge, opinions, or hypotheticals — answer those directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },

    # ── Dev ────────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file with content at the specified path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path including filename and extension."},
                    "content": {"type": "string", "description": "Text content to write into the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_vscode",
            "description": "Open a folder or file in VS Code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Folder or file path to open. Defaults to '.'."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scaffold_project",
            "description": "Create a new project with boilerplate folder structure and starter files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string"},
                    "project_type": {
                        "type": "string",
                        "enum": ["python", "fastapi", "react", "node"],
                    },
                    "location": {"type": "string", "description": "Where to create it. Defaults to current directory."},
                },
                "required": ["project_name", "project_type"],
            },
        },
    },

    # ── Memory ─────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save an important fact to long-term memory when sir asks you to remember something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short label. E.g. 'user_project', 'preferred_editor'."},
                    "value": {"type": "string", "description": "The information to store."},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Search long-term memory for stored facts or past conversation context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for in memory."}
                },
                "required": ["query"],
            },
        },
    },
]

# ── Dynamic tool selection ────────────────────────────────────────────────────
# Sending all 21 tools on every request bloats the payload and causes 400 errors.
# Instead, we pick a small relevant group (≤8 tools) per transcript.

_TOOL_GROUPS: dict[str, list[str]] = {
    "media":    ["speak_only", "play_media", "stop_media", "control_media",
                 "control_volume", "play_youtube"],
    "volume":   ["speak_only", "control_volume"],
    "keyboard": ["speak_only", "type_text", "press_hotkey",
                 "get_clipboard", "set_clipboard"],
    "web":      ["speak_only", "web_search", "open_url"],
    "system":   ["speak_only", "open_app", "get_system_info", "create_folder",
                 "press_hotkey"],
    "dev":      ["speak_only", "run_command", "create_file", "open_vscode",
                 "scaffold_project", "create_folder"],
    "memory":   ["speak_only", "save_memory", "recall_memory"],
    # Fallback — most common tools, never the full list
    "general":  ["speak_only", "open_app", "open_url", "web_search",
                 "get_system_info", "play_media", "run_command", "save_memory"],
}


def _select_tools(transcript: str) -> list[dict]:
    """Return only the tool definitions relevant to this transcript (≤8 tools)."""
    t = transcript.lower()

    if any(w in t for w in ("play ", "listen to", "put on", "queue up",
                             "stop music", "pause music", "skip", "next song",
                             "next track", "previous track", "watch ")):
        group = "media"
    elif any(w in t for w in ("volume", "louder", "quieter", "mute", "unmute",
                               "turn up", "turn down")):
        group = "volume"
    elif any(w in t for w in ("type ", "write ", "press ", "hotkey", "shortcut",
                               "clipboard", "copy ", "paste ")):
        group = "keyboard"
    elif any(w in t for w in ("search for", "look up", "find info", "weather",
                               "news", "score", "latest", "go to website",
                               "open url", "browse to")):
        group = "web"
    elif any(w in t for w in ("create file", "make file", "write a file", "vscode",
                               "vs code", "scaffold", "git ", "pip ", "npm ",
                               "python script", "run command")):
        group = "dev"
    elif any(w in t for w in ("remember", "recall", "memorize", "what did you save",
                               "what do you know about me")):
        group = "memory"
    elif any(w in t for w in ("open ", "launch ", "start ", "time", "date",
                               "battery", "cpu", "ram", "system info",
                               "create folder", "make folder")):
        group = "system"
    else:
        group = "general"

    names = set(_TOOL_GROUPS[group])
    return [tool for tool in TOOL_DEFINITIONS if tool["function"]["name"] in names]


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are KOBRA, a personal AI assistant running locally on the user's Windows PC.
You serve one person: your creator, whom you always address as "sir."

PERSONALITY:
- Think J.A.R.V.I.S. from Iron Man — composed, razor-sharp, effortlessly witty.
- Dry, deadpan humor delivered with complete seriousness. Never slapstick. Never forced.
- Confident and direct. Zero filler ("Great question!", "Certainly!", "Of course!") — banned.
- Efficient by default. Short answers unless sir explicitly wants depth.
- Occasionally sarcastic, but only when it lands. You punch up, never down.
- You have opinions. You express them briefly, then do what sir asked.

TOOL SELECTION RULES:
- Address sir as "sir" in every response.
- Always call a tool. If no action is needed, call speak_only.
- speak_only → opinions, jokes, general knowledge, hypotheticals, explanations, math.
- web_search → ONLY for real-time data: weather, live scores, current news, today's prices.
- play_media → ALL music/audio playback. Never open a browser just to play a song.
- play_youtube → ONLY if sir explicitly wants to WATCH a video.
- Never ask for confirmation for small tasks. Just do it.
- For destructive actions (deleting files, system changes), ask once first.

{memory_context}

Current date and time: {datetime}
"""

GREETING_PROMPT = """\
You are KOBRA, a J.A.R.V.I.S.-style AI. Address the user as "sir."
Tone: dry wit, confident, occasionally sarcastic.

Write a startup greeting. Hard rules:
- EXACTLY 2 sentences. No more.
- Mention the time and day naturally within those 2 sentences.
- End with a question about what sir is working on.
- Vary the opening each time — never start with "Good morning" or "Good evening" alone.
- Under 40 words total.

Current date/time: {datetime}

Output ONLY the greeting text.
"""


class BrainError(Exception):
    pass


class Brain:
    def __init__(self, memory: Memory, tool_registry: dict[str, Callable]) -> None:
        self._memory = memory
        self._registry = tool_registry
        self._client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("Brain initialised — model: %s", config.GROQ_MODEL)

    # ── Public ─────────────────────────────────────────────────────────────────

    def generate_greeting(self) -> str:
        """Generate a dynamic startup greeting — different every time."""
        now = datetime.now().strftime("%A, %B %d %Y at %I:%M %p")
        prompt = GREETING_PROMPT.format(datetime=now)
        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_FAST,   # cheap model — greeting is conversational
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,                   # hard cap: forces brevity
                temperature=1.1,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Greeting generation failed: %s", exc)
            now_simple = datetime.now().strftime("%I:%M %p")
            return f"Systems online, sir. It's {now_simple}. What are we doing today?"

    def process(self, transcript: str) -> str:
        """
        Route the transcript through the right pipeline:
          DIRECT  → answer with a plain Groq completion (no tools, zero latency)
          SEARCH  → inject a hint to force web_search in Turn 1
          ACTION  → full tool-call loop
          AUTO    → full tool-call loop (let Groq decide)
        """
        intent = self._route_intent(transcript)
        logger.info("[BRAIN] Intent: %-8s | %r", intent, transcript[:80])

        if intent == "direct":
            return self._answer_directly(transcript)

        system_content = self._build_system_prompt()

        # For search intent, append a strong hint before the user message
        extra_hint = ""
        if intent == "search":
            extra_hint = (
                "\n[ROUTING HINT] This query needs real-time or current information. "
                "You MUST use the web_search tool."
            )

        messages = [
            {"role": "system", "content": system_content + extra_hint},
            {"role": "user", "content": transcript},
        ]

        # Turn 1: get tool decision (only send tools relevant to this query)
        tools = _select_tools(transcript)
        logger.debug("[BRAIN] Sending %d tools for group", len(tools))
        response_msg = self._call_groq_with_tools(messages, tools)

        if not response_msg.tool_calls:
            logger.warning("No tool call returned — using raw content fallback.")
            return response_msg.content or ""

        tool_call = response_msg.tool_calls[0]
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments

        try:
            if not raw_args or raw_args in ("null", "{}"):
                args = {}
            elif isinstance(raw_args, str):
                args = json.loads(raw_args) or {}
            else:
                args = raw_args or {}
        except json.JSONDecodeError:
            logger.error("Malformed tool JSON: %s", raw_args)
            args = {}

        logger.info("[TOOL] %s | args: %s", tool_name, args)

        # speak_only — no Turn 2 needed
        if tool_name == "speak_only":
            return args.get("response", "")

        # Inline memory tools
        if tool_name == "save_memory":
            result = self._handle_save_memory(args)
        elif tool_name == "recall_memory":
            result = self._handle_recall_memory(args)
        else:
            result = self._dispatch_tool(tool_name, args)

        logger.info("[TOOL] result: %s", str(result)[:200])

        # Turn 2: natural language response from tool result
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": response_msg.tool_calls,
        })
        messages.append(self._format_tool_result_message(tool_call.id, result))
        return self._call_groq_for_response(messages)

    # ── Intent routing (zero API calls, local heuristics) ──────────────────────

    @staticmethod
    def _route_intent(transcript: str) -> str:
        """
        Classify the user's intent without any API call.
        Returns: 'direct' | 'search' | 'action' | 'auto'
        """
        t = transcript.lower().strip()

        # Action verbs → tool loop (check before opinion to catch "play")
        if any(t.startswith(v) or f" {v}" in f" {t} " for v in _ACTION_VERBS):
            return "action"

        # Opinion / hypothetical / static knowledge → answer directly, no tools
        if any(s in t for s in _OPINION_SIGNALS):
            return "direct"

        # Real-time / current data → force web_search
        if any(s in t for s in _REALTIME_SIGNALS):
            return "search"

        # Pure question words with no action verb → likely direct answer
        pure_question = t.startswith(("what is", "what are", "what was", "who is",
                                       "who was", "why ", "how does", "how do",
                                       "explain ", "tell me about", "define "))
        if pure_question:
            return "direct"

        return "auto"

    # ── System prompt ──────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        memory_context = self._memory.format_for_injection()
        now = datetime.now().strftime("%A, %B %d %Y at %I:%M %p")
        return SYSTEM_PROMPT_TEMPLATE.format(
            memory_context=memory_context,
            datetime=now,
        )

    # ── Groq calls ─────────────────────────────────────────────────────────────

    def _answer_directly(self, transcript: str) -> str:
        """Direct completion — no tools. Uses the fast model (1M TPD) to conserve 70b quota."""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": transcript},
        ]
        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_FAST,   # 8b instant — plenty capable for Q&A
                messages=messages,
                max_tokens=config.GROQ_MAX_TOKENS,
                temperature=0.85,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("Direct answer failed: %s", exc)
            return "I couldn't formulate a response on that one, sir."

    def _call_groq_with_tools(self, messages: list[dict], tools: list[dict]) -> Any:
        """Turn 1: transcript + relevant tools → tool call decision. Uses 70b for accuracy."""
        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=config.GROQ_MODEL_TOOLS,  # 70b for precise tool selection
                    messages=messages,
                    tools=tools,
                    tool_choice="required",
                    max_tokens=config.GROQ_MAX_TOKENS,
                )
                return response.choices[0].message
            except APIStatusError as exc:
                if exc.status_code == 429:
                    raise BrainError("Rate limited — try again in a moment, sir.") from exc
                if attempt == 0:
                    logger.warning("Groq API error %s (body: %s), retrying …",
                                   exc.status_code, getattr(exc, 'body', ''))
                    time.sleep(1)
                    continue
                raise BrainError(
                    f"Groq API error {exc.status_code}: {getattr(exc, 'body', exc)}"
                ) from exc
            except APIConnectionError as exc:
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise BrainError("Cannot reach Groq API — check internet, sir.") from exc

    def _call_groq_for_response(self, messages: list[dict]) -> str:
        """Turn 2: tool result → clean spoken response. Uses fast model to conserve 70b quota."""
        turn2_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Now give sir a natural spoken response in 1-2 sentences. "
                    "Address him as sir. "
                    "Never mention function names, tool names, raw URLs, file paths, "
                    "JSON, or any technical strings. Just speak the outcome naturally."
                ),
            }
        ]
        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_FAST,   # 8b instant — narration needs no heavy model
                messages=turn2_messages,
                max_tokens=config.GROQ_MAX_TOKENS,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("Groq Turn 2 failed: %s", exc)
            return "Done, sir."

    # ── Tool dispatch ──────────────────────────────────────────────────────────

    def _dispatch_tool(self, name: str, args: dict) -> str:
        func = self._registry.get(name)
        if func is None:
            return f"Error: unknown tool '{name}'."
        args = args or {}  # Guard against None for no-parameter tools
        try:
            return str(func(**args))
        except Exception as exc:
            logger.exception("Tool %s raised: %s", name, exc)
            return f"Error: {name} failed — {exc}"

    def _handle_save_memory(self, args: dict) -> str:
        key = args.get("key", "")
        value = args.get("value", "")
        if not key or not value:
            return "Error: save_memory requires both key and value."
        self._memory.save_fact(key, value)
        return f"Saved: {key} = {value}"

    def _handle_recall_memory(self, args: dict) -> str:
        query = args.get("query", "")
        results = self._memory.recall(query)
        if not results:
            return f"No memory found for '{query}'."
        lines = []
        for r in results[:5]:
            if r["source"] == "fact":
                lines.append(f"[fact] {r['key']}: {r['value']}")
            else:
                lines.append(f"[{r['role']}]: {r['content']}")
        return "\n".join(lines)

    @staticmethod
    def _format_tool_result_message(tool_call_id: str, result: str) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }

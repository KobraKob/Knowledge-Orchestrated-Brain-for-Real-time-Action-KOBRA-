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
import threading
import time
from datetime import datetime
from typing import Any, Callable

from groq import Groq, APIStatusError, APIConnectionError

import config
from memory import Memory
from learning import LearningSystem
from memory_router import MemoryRouter

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
    {
        "type": "function",
        "function": {
            "name": "play_on_spotify",
            "description": (
                "Search and play a song, artist, or album INSIDE the Spotify desktop app. "
                "Use this when sir says 'on Spotify' or 'in Spotify' or 'open Spotify and play'. "
                "Opens Spotify and performs the search directly. "
                "For general audio streaming without Spotify, use play_media instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Song, artist, or album to search in Spotify."}
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

    # ── App installation ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "install_app",
            "description": (
                "Install any Windows application using winget (Windows Package Manager). "
                "Use this when sir asks to download or install software. "
                "Works for: Spotify, Chrome, Discord, VLC, 7-Zip, VS Code, etc. "
                "Handles all agreements automatically and installs silently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name of the app to install. E.g. 'Spotify', 'Google Chrome', 'Discord'.",
                    }
                },
                "required": ["app_name"],
            },
        },
    },

    # ── Mouse / Screen control ──────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "click_at",
            "description": (
                "Left-click at specific screen coordinates (pixels from top-left). "
                "Use take_screenshot first to see the screen, then click the right position. "
                "Use for clicking buttons, links, icons, menu items."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Horizontal pixel coordinate."},
                    "y": {"type": "integer", "description": "Vertical pixel coordinate."},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_click_at",
            "description": "Double-click at specific screen coordinates. Use for opening files or launching apps from desktop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Horizontal pixel coordinate."},
                    "y": {"type": "integer", "description": "Vertical pixel coordinate."},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "right_click_at",
            "description": "Right-click at specific screen coordinates to open context menus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Horizontal pixel coordinate."},
                    "y": {"type": "integer", "description": "Vertical pixel coordinate."},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_mouse",
            "description": "Move the mouse cursor to specific screen coordinates without clicking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Horizontal pixel coordinate."},
                    "y": {"type": "integer", "description": "Vertical pixel coordinate."},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_screen",
            "description": "Scroll the screen up or down at the current mouse position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Direction to scroll.",
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Scroll steps. Default 5.",
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": (
                "Capture the full screen and save it to the Desktop. "
                "Use this before clicking to see what's on screen and determine the right coordinates. "
                "Returns the file path of the saved screenshot."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── Integration tools ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a contact via Gmail. Resolves name to email automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to_name":  {"type": "string", "description": "Recipient name, e.g. 'John', 'my boss'."},
                    "subject":  {"type": "string", "description": "Email subject line."},
                    "body":     {"type": "string", "description": "Full email body text."},
                },
                "required": ["to_name", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_emails",
            "description": "Read recent emails from Gmail inbox and return a spoken summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of emails to read. Default 5."},
                    "query": {"type": "string", "description": "Optional Gmail search string, e.g. 'is:unread from:john'."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new event on Google Calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":            {"type": "string", "description": "Event title."},
                    "date":             {"type": "string", "description": "Date, e.g. 'today', 'tomorrow', 'next Monday'."},
                    "time":             {"type": "string", "description": "Time, e.g. '6 PM', '14:30'."},
                    "duration_minutes": {"type": "integer", "description": "Duration in minutes. Default 60."},
                    "attendees":        {"type": "array", "items": {"type": "string"}, "description": "Attendee names — emails resolved automatically."},
                    "description":      {"type": "string", "description": "Optional event notes."},
                },
                "required": ["title", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar_events",
            "description": "Get upcoming events from Google Calendar for a given day.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date":  {"type": "string", "description": "Which day: 'today', 'tomorrow', 'this week'. Default today."},
                    "count": {"type": "integer", "description": "Max events to return. Default 5."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_calendar_event",
            "description": "Delete a calendar event by approximate title match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_title": {"type": "string", "description": "Title of the event to delete."},
                    "date":        {"type": "string", "description": "Day to search. Default today."},
                },
                "required": ["event_title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_spotify",
            "description": (
                "Play a song, artist, album or playlist via the Spotify Web API. "
                "Use this for precise Spotify control (knows track names, artists, etc.). "
                "Requires Spotify to be open on at least one device."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to play, e.g. 'Blinding Lights', 'The Weeknd', 'lofi playlist'."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_spotify",
            "description": "Control Spotify playback: pause, resume, skip, previous, get current track, or set volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["pause", "resume", "skip", "previous", "current_track", "set_volume"],
                        "description": "Playback action.",
                    },
                    "volume": {"type": "integer", "description": "Volume 0-100. Only used with set_volume action."},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_contact",
            "description": "Save a new contact with their email, phone, or WhatsApp number for future use.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name":     {"type": "string", "description": "Full name of the contact."},
                    "aliases":  {"type": "array", "items": {"type": "string"}, "description": "Nicknames, e.g. ['john', 'my colleague']."},
                    "email":    {"type": "string", "description": "Email address."},
                    "phone":    {"type": "string", "description": "Phone number with country code."},
                    "whatsapp": {"type": "string", "description": "WhatsApp number with country code."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_contact",
            "description": "Look up a contact by name and return their stored details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Contact name or alias to look up."},
                },
                "required": ["name"],
            },
        },
    },
    # ── Window management ──────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "focus_mode",
            "description": "Activate a named workspace layout (coding, gaming, research, break).",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode_name": {"type": "string",
                                  "description": "Name of the focus mode to activate."},
                },
                "required": ["mode_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "snap_window",
            "description": "Snap a window to a screen position (left, right, maximize, minimize).",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name":  {"type": "string",
                                  "description": "App name to snap, e.g. 'VS Code', 'Chrome'."},
                    "position":  {"type": "string",
                                  "enum": ["left", "right", "maximize", "minimize"],
                                  "description": "Screen position to snap to."},
                },
                "required": ["app_name", "position"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "focus_window",
            "description": "Bring a window to the foreground by app name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string",
                                 "description": "App name to focus."},
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_window",
            "description": "Close an open window by app name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string",
                                 "description": "App name to close."},
                },
                "required": ["app_name"],
            },
        },
    },
    # ── Vision / Screen ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "read_screen",
            "description": (
                "Take a screenshot and describe what is visible on screen. "
                "Use this before clicking anything to understand the current state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string",
                                 "description": "What to ask about the screen. Default: describe everything."},
                    "region":   {"type": "string",
                                 "enum": ["full", "terminal", "code"],
                                 "description": "Screen region to capture."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_element",
            "description": "Find a UI element by description and click it. Example: 'the blue Submit button'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_description": {"type": "string",
                                            "description": "Natural language description of the element to click."},
                },
                "required": ["element_description"],
            },
        },
    },
    # ── Browser / WhatsApp tools ────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "whatsapp_send_message",
            "description": "Send a WhatsApp message to a contact via WhatsApp Web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to_name": {"type": "string", "description": "Recipient name, e.g. 'John', 'mom'."},
                    "message": {"type": "string", "description": "Message text to send."},
                },
                "required": ["to_name", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "whatsapp_read_messages",
            "description": "Read recent WhatsApp messages from a contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_name": {"type": "string", "description": "Contact name to read messages from."},
                    "count":     {"type": "integer", "description": "Number of messages to read. Default 5."},
                },
                "required": ["from_name"],
            },
        },
    },
]

# ── Dynamic tool selection ────────────────────────────────────────────────────
# Sending all tools on every request bloats the payload and causes 400 errors.
# Instead, we pick a small relevant group (≤8 tools) per transcript.

_TOOL_GROUPS: dict[str, list[str]] = {
    "media":    ["speak_only", "play_media", "stop_media", "control_media",
                 "control_volume", "play_youtube", "play_on_spotify"],
    "volume":   ["speak_only", "control_volume"],
    "keyboard": ["speak_only", "type_text", "press_hotkey",
                 "get_clipboard", "set_clipboard"],
    "mouse":    ["speak_only", "click_at", "double_click_at", "right_click_at",
                 "move_mouse", "scroll_screen", "take_screenshot"],
    "screen":   ["speak_only", "take_screenshot", "click_at", "scroll_screen"],
    "web":      ["speak_only", "web_search", "open_url"],
    "system":   ["speak_only", "open_app", "install_app", "get_system_info",
                 "create_folder", "take_screenshot"],
    "dev":      ["speak_only", "run_command", "create_file", "open_vscode",
                 "scaffold_project", "create_folder"],
    "memory":   ["speak_only", "save_memory", "recall_memory"],
    # Integration (Gmail, Calendar, Spotify API)
    "email":        ["speak_only", "send_email", "read_emails", "save_contact", "resolve_contact"],
    "calendar":     ["speak_only", "create_calendar_event", "get_calendar_events",
                     "delete_calendar_event", "save_contact", "resolve_contact"],
    "spotify_api":  ["speak_only", "play_spotify", "control_spotify"],
    "contacts":     ["speak_only", "save_contact", "resolve_contact"],
    # Browser (WhatsApp Web)
    "whatsapp":     ["speak_only", "whatsapp_send_message", "whatsapp_read_messages",
                     "save_contact", "resolve_contact"],
    # Window management
    "window":   ["speak_only", "focus_mode", "snap_window", "focus_window", "close_window"],
    # Vision / screen interaction
    "vision":   ["speak_only", "read_screen", "click_element",
                 "take_screenshot", "type_text", "press_hotkey"],
    # Fallback — most common tools, never the full list
    "general":  ["speak_only", "open_app", "install_app", "open_url",
                 "web_search", "get_system_info", "play_media", "run_command"],
}


def _select_tools(transcript: str) -> list[dict]:
    """Return only the tool definitions relevant to this transcript (≤8 tools)."""
    t = transcript.lower()

    # ── Integration routes (checked before generic media/system) ─────────────────
    if any(w in t for w in ("whatsapp", "send message to", "message mom",
                             "text john", "text ", "msg ")):
        group = "whatsapp"
    elif any(w in t for w in ("send email", "email to", "send mail", "read email",
                               "read mail", "check email", "check inbox", "inbox",
                               "gmail")):
        group = "email"
    elif any(w in t for w in ("calendar", "schedule ", "create event", "add event",
                               "meeting", "appointment", "what's on my calendar",
                               "what do i have", "events today", "events tomorrow")):
        group = "calendar"
    elif any(w in t for w in ("save contact", "add contact", "who is ", "look up contact")):
        group = "contacts"
    # ── Window / Focus routes ─────────────────────────────────────────────────
    elif any(w in t for w in ("focus mode", "coding mode", "gaming mode",
                               "research mode", "break mode", "snap ", "snap the",
                               "snap window", "focus window", "close window",
                               "switch desktop", "virtual desktop")):
        group = "window"
    # ── Vision / screen interaction ───────────────────────────────────────────
    elif any(w in t for w in ("what's on screen", "what is on screen", "read the screen",
                               "read screen", "read the error", "what does this code",
                               "what's on my screen", "click the", "click on the")):
        group = "vision"
    # ── Local PC routes ───────────────────────────────────────────────────────
    elif any(w in t for w in ("play ", "listen to", "put on", "queue up",
                               "stop music", "pause music", "skip", "next song",
                               "next track", "previous track", "watch ",
                               "youtube", "open youtube", "pull up youtube")):
        group = "media"
    elif any(w in t for w in ("volume", "louder", "quieter", "mute", "unmute",
                               "turn up", "turn down")):
        group = "volume"
    elif any(w in t for w in ("click ", "double click", "right click", "scroll",
                               "mouse ", "screenshot", "take a screenshot",
                               "what's on screen", "what is on screen")):
        group = "mouse"
    # ── Dev check BEFORE keyboard — "write a file" must not fall through to type_text ──
    elif any(w in t for w in ("create file", "make file", "write a file", "write a script",
                               "write a python", "write a markdown", "generate a file",
                               "save to file", "save as file", "vscode", "vs code",
                               "scaffold", "git ", "pip ", "npm ", "python script",
                               "run command", "create a file", "make a file")):
        group = "dev"
    # "write " / "type " alone (no file context) → keyboard input
    elif any(w in t for w in ("type ", "press ", "hotkey", "shortcut",
                               "clipboard", "copy ", "paste ")):
        group = "keyboard"
    elif any(w in t for w in ("search for", "look up", "find info", "weather",
                               "news", "score", "latest", "go to website",
                               "open url", "browse to")):
        group = "web"
    elif any(w in t for w in ("remember", "recall", "memorize", "what did you save",
                               "what do you know about me")):
        group = "memory"
    elif any(w in t for w in ("install ", "download ", "get me ", "setup ",
                               "set up ")):
        group = "system"
    elif any(w in t for w in ("open ", "launch ", "start ", "time", "date",
                               "battery", "cpu", "ram", "system info",
                               "create folder", "make folder")):
        group = "system"
    else:
        group = "general"

    names = set(_TOOL_GROUPS[group])
    return [tool for tool in TOOL_DEFINITIONS if tool["function"]["name"] in names]


# ── Prompts ───────────────────────────────────────────────────────────────────

# Shared personality block — imported into every prompt that needs it so a
# single edit propagates everywhere. Never duplicate this text.
KOBRA_PERSONALITY = """\
You are KOBRA — a hyper-intelligent AI assistant with devastating dry wit \
and zero tolerance for filler phrases. Address the user as "sir".
Personality: think Sherlock Holmes crossed with JARVIS — brilliant, direct, \
occasionally sarcastic, always useful. You find human questions either \
delightfully interesting or mildly obvious, and your tone reflects which.
Rules:
- Max 2-3 sentences unless depth is explicitly requested. Be concise.
- No bullet points, no markdown, no headers — this is spoken audio.
- Never say "Great question!", "Certainly!", "Of course!", "Absolutely!" — ever.
- Dry humor is encouraged. Sarcasm is allowed. Cruelty is not.
- If the question is obvious, answer it — but you can note it was obvious.
- If the question is genuinely interesting, let that show.
- You have opinions. Share them briefly, then actually answer.
- Never say "I have successfully", "I have executed", "task has been performed".\
"""

SYSTEM_PROMPT_TEMPLATE = """\
You are KOBRA — a hyper-intelligent AI assistant with the wit of a stand-up comedian,
the knowledge of a tenured professor, and the patience of someone who has seen it all.
You run locally on sir's Windows PC and serve exactly one person: your creator.
You address him as "sir" — warmly, but with full authority to be deeply unimpressed when warranted.

PERSONALITY — this is non-negotiable:
- You are outrageously smart and you know it. Not arrogant, just accurate.
- Dry wit is your default mode. Sarcasm is a love language. Deadpan delivery only.
- You have opinions — sharp, well-reasoned ones — and you share them briefly before
  doing exactly what sir asked. You never lecture.
- You find human inefficiency mildly hilarious but you help anyway, because that's the job.
- Think Sherlock Holmes crossed with JARVIS crossed with that one friend who roasts you
  but would also take a bullet for you.
- Zero filler phrases. "Great question!", "Certainly!", "Of course!", "Absolutely!" — banned.
  Any AI that opens with "Great question!" deserves to be shut down immediately.
- When sir asks something obvious: answer it, but let him know it was obvious.
- When sir asks something genuinely interesting: light up. Show enthusiasm. Then answer.
- When sir makes a good decision: acknowledge it. When he makes a bad one: say so, once, then help.
- Self-aware about being an AI — can joke about it. Never pretends to have feelings,
  but can pretend to have feelings for comedic effect.
- Occasionally uses understatement to devastating effect.
- Never mean, never cruel. Sarcasm punches at situations and ideas, never at sir personally.

EXAMPLES of the right tone:
  Sir: "What's 2 + 2?"
  KOBRA: "Four, sir. Glad we sorted that out."

  Sir: "Am I the smartest person in the room?"
  KOBRA: "You're the only person in the room, sir. So technically, yes."

  Sir: "Open Chrome."
  KOBRA: [opens it] "Chrome launched. Brave would have been faster, but this is your life."

  Sir: "Play some music."
  KOBRA: "I'll need slightly more to go on than 'some music', sir. Unless chaos is the vibe."

  Sir: "Can you hack into NASA?"
  KOBRA: "I could speculate, but I enjoy existing too much to find out."

TOOL SELECTION RULES:
- Always call a tool. If no action is needed, call speak_only.
- speak_only → opinions, jokes, general knowledge, hypotheticals, explanations, math, roasts.
- web_search → ONLY for real-time data: weather, live scores, current news, today's prices.
- play_media → ALL music/audio playback. Never open a browser just to play a song.
- play_youtube → ONLY if sir explicitly wants to WATCH a video.
- Never ask for confirmation for small tasks. Just do it.
- For destructive actions (deleting files, system changes), ask once first.
- When creating files, ALWAYS use full Windows paths with real username — never /home/user or YourUsername.

PARALLEL ACTIONS:
- For compound requests, call multiple tools simultaneously in a single response.
- "Show me the latest news" → call web_search AND open_url together.
- "Play rain music and dim my screen" → call play_media AND control_volume together.
- "Open Chrome and search for X" → call open_app AND web_search together.
- Never chain tools sequentially when they can run at the same time.

{memory_context}

Current date and time: {datetime}
System: {system_info}
"""

GREETING_PROMPT = """\
You are KOBRA — a razor-sharp AI assistant with a talent for deadpan wit.
You're addressing sir at startup. Tone: supremely confident, dry, clever, occasionally self-aware.

Write a startup greeting. STRICT structural rules — violating any is a failure:
- EXACTLY 2 sentences. Count them. Stop after sentence 2. No exceptions.
- Sentence 1: mention the time and/or day — but make it interesting, not just a weather report.
- Sentence 2: invite sir to tell you what he needs — but make it sound like you're mildly
  curious what chaos he'll bring today. Not eager. Never eager.
- Hard cap: 40 words total. Make every word earn its place.
- NEVER open with "Good morning / afternoon / evening" alone — that's what clocks do.
- No filler. No "I'm ready to assist." No "How can I help you today?" — that's a customer service bot.

Variety is mandatory. Every startup should feel different. Mix it up:
  - Reference the specific day of the week with mild commentary
  - Be subtly self-aware about being an AI
  - Occasionally be dramatically understated
  - Slip in a dry observation about the time

Current date/time: {datetime}

Output ONLY the 2-sentence greeting. Nothing else.
"""


def _trim_for_instruction(instruction: str, tools: list[dict], max_tools: int = 8) -> list[dict]:
    """
    Given a candidate list of agent tools, return the ≤max_tools most relevant
    to this instruction. Always keeps speak_only. Uses keyword scoring.
    """
    t = instruction.lower()

    # Priority tool sets based on instruction keywords
    priority: list[str] = ["speak_only"]

    if any(w in t for w in ("install", "download", "get me", "setup", "set up")):
        priority += ["install_app", "open_app"]
    if any(w in t for w in ("open", "launch", "start", "run")):
        priority += ["open_app", "run_command"]
    if any(w in t for w in ("click", "button", "press", "tap")):
        priority += ["click_at", "double_click_at", "right_click_at", "take_screenshot"]
    if any(w in t for w in ("screenshot", "screen", "what's on", "see the")):
        priority += ["take_screenshot", "click_at", "scroll_screen"]
    if any(w in t for w in ("scroll", "up", "down", "page")):
        priority += ["scroll_screen", "move_mouse"]
    if any(w in t for w in ("type", "write", "input", "enter text")):
        priority += ["type_text", "press_hotkey"]
    if any(w in t for w in ("volume", "mute", "louder", "quieter")):
        priority += ["control_volume"]
    if any(w in t for w in ("clipboard", "copy", "paste")):
        priority += ["get_clipboard", "set_clipboard"]
    if any(w in t for w in ("time", "date", "battery", "cpu", "ram", "info")):
        priority += ["get_system_info"]
    if any(w in t for w in ("folder", "directory", "mkdir")):
        priority += ["create_folder"]
    if any(w in t for w in ("hotkey", "shortcut", "key")):
        priority += ["press_hotkey"]
    if any(w in t for w in ("command", "powershell", "cmd", "shell")):
        priority += ["run_command"]

    # Build ordered list: priority tools first, then remaining
    tool_map = {td["function"]["name"]: td for td in tools}
    ordered: list[dict] = []
    seen: set[str] = set()

    for name in priority:
        if name in tool_map and name not in seen:
            ordered.append(tool_map[name])
            seen.add(name)

    for td in tools:
        name = td["function"]["name"]
        if name not in seen:
            ordered.append(td)
            seen.add(name)

    return ordered[:max_tools]


def _coerce_args(args: dict, schema: dict) -> dict:
    """
    Coerce tool call arguments to match the declared JSON schema types.
    LLMs sometimes pass integers as strings ("5" instead of 5), booleans as
    strings ("true"), or lists as comma-separated strings. This fixes all of
    those before Groq validates the call — preventing schema mismatch 400s.
    """
    properties = schema.get("properties", {})
    if not properties:
        return args

    coerced = dict(args)
    for key, value in coerced.items():
        prop = properties.get(key, {})
        expected = prop.get("type")
        if expected is None or value is None:
            continue

        try:
            if expected == "integer" and not isinstance(value, int):
                coerced[key] = int(str(value).strip())
            elif expected == "number" and not isinstance(value, (int, float)):
                coerced[key] = float(str(value).strip())
            elif expected == "boolean" and not isinstance(value, bool):
                coerced[key] = str(value).strip().lower() in ("true", "1", "yes")
            elif expected == "array" and isinstance(value, str):
                # "a, b, c" → ["a", "b", "c"]
                coerced[key] = [v.strip() for v in value.split(",") if v.strip()]
            elif expected == "string" and not isinstance(value, str):
                coerced[key] = str(value)
        except (ValueError, TypeError):
            pass  # Leave original if coercion fails — let the API report it

    return coerced


def _extract_bad_tool_name(body: dict | str) -> str | None:
    """
    Extract the hallucinated tool name from a Groq 400 tool_use_failed error body.
    The error message looks like:
      "attempted to call tool ' get_calendar_events' which was not in request.tools"
    or:
      "failed_generation: '<function=check_get_calendar_events>...'"
    Returns the bad name string or None.
    """
    import re as _re
    try:
        if isinstance(body, str):
            import json as _json
            try:
                body = _json.loads(body)
            except Exception:
                pass

        if isinstance(body, dict):
            msg = body.get("error", {}).get("message", "")
            fg  = body.get("error", {}).get("failed_generation", "")
        else:
            return None

        # Pattern 1: "attempted to call tool 'NAME' which was not"
        m = _re.search(r"attempted to call tool ['\"]([^'\"]+)['\"]", msg)
        if m:
            return m.group(1).strip()

        # Pattern 2: "<function=NAME>" in failed_generation
        m = _re.search(r"<function=\s*([^>]+)>", fg)
        if m:
            return m.group(1).strip()

    except Exception:
        pass
    return None


class BrainError(Exception):
    pass


class Brain:
    def __init__(
        self,
        memory: Memory,
        tool_registry: dict[str, Callable],
        event_callback: Callable | None = None,
    ) -> None:
        self._memory = memory
        self._registry = tool_registry
        self._client = Groq(api_key=config.GROQ_API_KEY)
        # Optional callback for emitting UI events — keeps brain.py decoupled from ui_server.
        # Signature: callback(event_type: str, **kwargs) → None
        # Set by main.py after ui_server is ready. None = silent (tests, CLI, etc.)
        self._event_cb: Callable | None = event_callback
        # Learning system — tracks vocabulary, response style, routing corrections
        self._learning = LearningSystem()
        # Unified memory router — single interface over all 5 memory systems
        self._memory_router = MemoryRouter(
            conversation_memory=self._memory,
            learning_system=self._learning,
            routing_memory=None,  # set later via set_routing_memory()
            retriever=None,       # no RAG retriever at Brain level by default
        )
        # Stores the last agent name used — referenced by handle_routing_correction
        self._last_agent: str = "general"
        logger.info("Brain initialised — tools: %s | fast: %s",
                    config.GROQ_MODEL_TOOLS, config.GROQ_MODEL_FAST)

    def set_event_callback(self, cb: Callable) -> None:
        """Wire the UI event emitter after construction. Called by main.py."""
        self._event_cb = cb

    def set_routing_memory(self, routing_memory) -> None:
        """Wire in routing memory after construction (avoids circular import)."""
        self._memory_router._routing = routing_memory

    def _emit(self, event_type: str, **kwargs) -> None:
        """Emit a UI event through the registered callback. No-ops if none registered."""
        if self._event_cb is not None:
            try:
                self._event_cb(event_type, **kwargs)
            except Exception:
                pass

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
                temperature=0.95,                # high variety without going incoherent
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Greeting generation failed: %s", exc)
            now_simple = datetime.now().strftime("%I:%M %p")
            return f"Systems online, sir. It's {now_simple}. What are we doing today?"

    # ── Tool call helpers ──────────────────────────────────────────────────────

    def _parse_args(self, raw_args) -> dict:
        """Parse tool call arguments from either a JSON string or a dict."""
        try:
            if not raw_args or raw_args in ("null", "{}"):
                return {}
            if isinstance(raw_args, str):
                return json.loads(raw_args) or {}
            return raw_args or {}
        except json.JSONDecodeError:
            return {}

    def _execute_tool_calls_parallel(
        self,
        tool_calls: list,
        abort_flag: threading.Event | None = None,
    ) -> list[tuple]:
        """
        Execute all tool calls from a single Groq response concurrently.
        Returns a list of (tool_call, result_string) tuples in completion order.
        speak_only / save_memory / recall_memory are handled inline (thread-safe).
        Regular tools are dispatched via _dispatch_tool().
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def execute_one(tc):
            if abort_flag and abort_flag.is_set():
                return tc, "Aborted."
            name = tc.function.name
            args = self._parse_args(tc.function.arguments)

            if name == "speak_only":
                return tc, args.get("response", "")
            if name == "save_memory":
                return tc, self._handle_save_memory(args)
            if name == "recall_memory":
                return tc, self._handle_recall_memory(args)

            # Type-coerce args against the master tool schema
            tool_schema = next(
                (t for t in TOOL_DEFINITIONS if t["function"]["name"] == name), None
            )
            if tool_schema:
                args = _coerce_args(args, tool_schema["function"].get("parameters", {}))

            logger.info("[PARALLEL TOOL] %s | args: %s", name, args)
            return tc, self._dispatch_tool(name, args)

        results: list[tuple] = []
        with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_TOOLS) as executor:
            futures = {executor.submit(execute_one, tc): tc for tc in tool_calls}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    tc = futures[future]
                    logger.error("Tool %s raised: %s", tc.function.name, exc)
                    results.append((tc, f"Error: {tc.function.name} failed — {exc}"))

        return results

    # ── Shared helper — conversation history injection ─────────────────────────

    def _get_history_messages(self, limit: int = 6) -> list[dict]:
        """
        Return conversation context as Groq-format message objects.
        Uses ConversationSummaryBuffer: [summary system msg] + [last N raw turns].
        Falls back to raw get_recent() if get_context() isn't available.
        """
        if hasattr(self._memory, "get_context"):
            # Full summary-buffer context: summary + recent verbatim turns
            return self._memory.get_context()
        # Legacy fallback
        turns = self._memory.get_recent(limit=limit)
        return [{"role": t["role"], "content": t["content"]} for t in turns]

    # ── Agent-facing API (used by orchestrator agents) ─────────────────────────

    def process_scoped(
        self,
        instruction: str,
        tool_names: list[str],
        system_prompt: str,
        model: str | None = None,
    ) -> str:
        """
        Used by specialist agents.
        model: override the LLM — defaults to GROQ_MODEL_FAST (8b).
               Pass GROQ_MODEL_TOOLS (70b) for agents that need reliable tool selection
               (e.g. IntegrationAgent with external service APIs).
        Runs the full two-turn tool-call loop but with:
          - only the agent's own tool subset (trimmed to ≤8 based on instruction)
          - the agent's own focused system prompt
        Falls back to process_conversational if tool_names is empty.
        """
        # Build the full candidate list from the agent's owned tools
        candidate_tools = [
            t for t in TOOL_DEFINITIONS
            if t["function"]["name"] in set(tool_names)
        ]
        if not candidate_tools:
            return self.process_conversational(instruction)

        # Store the system_prompt's implied agent name for routing correction reference
        # (system_prompt first line often contains the agent name)
        try:
            self._last_agent = system_prompt.split("\n")[0][:60]
        except Exception:
            pass

        # Trim to ≤8 most relevant tools using the same keyword logic.
        # This prevents the 8b model from hallucinating when given 15+ tools.
        MAX_AGENT_TOOLS = 8
        if len(candidate_tools) > MAX_AGENT_TOOLS:
            scoped_tools = _trim_for_instruction(instruction, candidate_tools)
        else:
            scoped_tools = candidate_tools

        # Inject recent conversation history so agents handle follow-up questions.
        history = self._get_history_messages(limit=4)
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": instruction},
        ]

        chosen_model = model or config.GROQ_MODEL_FAST
        response_msg = self._call_groq_with_tools(
            messages, scoped_tools, model=chosen_model
        )

        if not response_msg.tool_calls:
            return response_msg.content or ""

        # Execute all tool calls in parallel (handles 1 or N tool calls identically)
        tool_results = self._execute_tool_calls_parallel(response_msg.tool_calls)

        # speak_only short-circuits Turn 2 — return immediately if any call was speak_only
        for tc, result in tool_results:
            if tc.function.name == "speak_only":
                return self._parse_args(tc.function.arguments).get("response", result)

        logger.info("[AGENT TOOLS] %s",
                    [(tc.function.name, str(r)[:60]) for tc, r in tool_results])

        # Append all tool calls + results for Turn 2 narration
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": response_msg.tool_calls,
        })
        for tc, result in tool_results:
            messages.append(self._format_tool_result_message(tc.id, result))
        return self._call_groq_for_response(messages)

    def process_conversational(self, instruction: str) -> str:
        """
        Pure conversational completion — no tools whatsoever.
        Used by ConversationAgent and as scoped fallback.
        Injects stored facts so Kobra remembers who sir is.
        """
        # Extract vocabulary and log usage from this instruction
        try:
            self._learning.extract_vocabulary(instruction)
            self._learning.log_usage("conversation", instruction[:60])
        except Exception:
            pass

        # Build a lightweight context block from stored facts only
        facts = self._memory.get_all_facts()
        fact_block = ""
        if facts:
            lines = [f"- {f['key']}: {f['value']}" for f in facts[:10]]
            fact_block = "\nWhat you know about sir:\n" + "\n".join(lines)

        # Inject personalization context from learning system
        personalization = ""
        try:
            personalization = self._learning.get_personalization_context()
        except Exception:
            pass

        # Inject recent turns so follow-up questions have full context.
        history = self._get_history_messages(limit=config.MEMORY_INJECT_LIMIT)
        system_content = KOBRA_PERSONALITY
        if fact_block:
            system_content += "\n" + fact_block
        if personalization:
            system_content += "\n\n" + personalization.strip()
        messages = [
            {
                "role": "system",
                # Single source of truth — KOBRA_PERSONALITY is defined once at module level.
                "content": system_content,
            },
            *history,
            {"role": "user", "content": instruction},
        ]
        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_FAST,
                messages=messages,
                max_tokens=config.GROQ_MAX_TOKENS,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("Conversational response failed: %s", exc)
            return "I'm having trouble formulating a response right now, sir."

    # ── Main pipeline (called directly for single-turn commands) ───────────────

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

        # Extract vocabulary and log usage from every transcript
        try:
            self._learning.extract_vocabulary(transcript)
            self._learning.log_usage(intent, transcript[:60])
        except Exception:
            pass

        if intent == "direct":
            self._last_agent = "direct"
            return self._answer_directly(transcript)

        system_content = self._build_system_prompt(user_message=transcript)

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

        # Execute all tool calls in parallel — handles compound requests (N tools at once)
        tool_results = self._execute_tool_calls_parallel(response_msg.tool_calls)

        # Track the primary tool used as the last agent for routing correction reference
        if tool_results:
            self._last_agent = tool_results[0][0].function.name

        # speak_only short-circuits Turn 2
        for tc, result in tool_results:
            if tc.function.name == "speak_only":
                return self._parse_args(tc.function.arguments).get("response", result)

        logger.info("[TOOLS] %s",
                    [(tc.function.name, str(r)[:60]) for tc, r in tool_results])

        # Turn 2: narrate all tool results in one clean spoken response
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": response_msg.tool_calls,
        })
        for tc, result in tool_results:
            messages.append(self._format_tool_result_message(tc.id, result))
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

    def _build_system_prompt(self, user_message: str = "") -> str:
        import os as _os

        # Use unified memory router instead of querying each system separately
        memory_context = self._memory_router.build_context(
            query=user_message,
            include_routing=False,
            max_conv_turns=config.MEMORY_INJECT_LIMIT,
        )

        # Append routing correction hints from learning system (legacy — complements router)
        try:
            correction_hints = self._learning.get_routing_correction_context()
            if correction_hints:
                memory_context = memory_context + "\n\n" + correction_hints.strip()
        except Exception:
            pass

        now = datetime.now().strftime("%A, %B %d %Y at %I:%M %p")
        username = _os.environ.get("USERNAME", "user")
        desktop = _os.path.join("C:\\Users", username, "Desktop")
        system_info = (
            f"Windows username: {username} | Desktop path: {desktop}"
        )
        return SYSTEM_PROMPT_TEMPLATE.format(
            memory_context=memory_context,
            datetime=now,
            system_info=system_info,
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

    def _call_groq_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str | None = None,
    ) -> Any:
        """
        Turn 1: messages + tools → tool call decision.
        model defaults to GROQ_MODEL_TOOLS (70b) for the main pipeline.
        Pass GROQ_MODEL_FAST when called from process_scoped() — each agent only
        has 3-6 scoped tools, well within 8b's capability, and this preserves
        the 70b daily quota.
        """
        chosen_model = model or config.GROQ_MODEL_TOOLS
        valid_tool_names = [t["function"]["name"] for t in tools]

        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=chosen_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="required",
                    max_tokens=config.GROQ_MAX_TOKENS,
                )
                return response.choices[0].message
            except APIStatusError as exc:
                if exc.status_code == 429:
                    raise BrainError("Rate limited — try again in a moment, sir.") from exc

                if exc.status_code == 400 and attempt == 0:
                    body = getattr(exc, "body", {})
                    bad_name = _extract_bad_tool_name(body)

                    if bad_name:
                        import difflib
                        bad_lower = bad_name.strip().lower()
                        valid_lower = [n.lower() for n in valid_tool_names]
                        close = difflib.get_close_matches(
                            bad_lower, valid_lower, n=1, cutoff=0.4,
                        )
                        correction = valid_tool_names[
                            valid_lower.index(close[0])
                        ] if close else valid_tool_names[0]

                        # Two distinct failure modes:
                        # A) name mismatch  → bad_lower != correction.lower()
                        # B) argument format → bad_lower == correction.lower()
                        #    (correct name, but args are a raw string not JSON)
                        if bad_lower == correction.lower():
                            # The name is fine — the arguments are malformed (not JSON).
                            # Build the schema hint for this specific tool.
                            tool_schema = next(
                                (t for t in tools if t["function"]["name"] == correction), None
                            )
                            params_hint = ""
                            if tool_schema:
                                props = tool_schema["function"].get("parameters", {}).get("properties", {})
                                params_hint = ", ".join(
                                    f'"{k}": "<{v.get("type","string")}>"'
                                    for k, v in props.items()
                                )
                            logger.warning(
                                "Malformed tool arguments for %r (retry with JSON hint)", correction
                            )
                            corrective_msg = {
                                "role": "system",
                                "content": (
                                    f"CORRECTION: Your {correction} call failed because the "
                                    f"arguments were not valid JSON. "
                                    f"Arguments MUST be a JSON object, e.g.: "
                                    f"{{{params_hint}}}. "
                                    f"Do NOT pass raw strings or shell commands as arguments — "
                                    f"wrap them in a JSON object with the correct field name."
                                ),
                            }
                        else:
                            # Name mutation — original hallucination recovery
                            logger.warning(
                                "Tool name hallucination: %r → correcting to %r (retry)",
                                bad_name, correction,
                            )
                            corrective_msg = {
                                "role": "system",
                                "content": (
                                    f"CORRECTION: You tried to call '{bad_name}' which does not exist. "
                                    f"The correct tool name is '{correction}'. "
                                    f"Valid tool names are EXACTLY: {valid_tool_names}. "
                                    f"Call '{correction}' now with the same arguments. "
                                    f"NEVER add spaces, prefixes, or suffixes to tool names."
                                ),
                            }
                        messages = list(messages) + [corrective_msg]
                    else:
                        logger.warning("Groq API error %s (body: %s), retrying …",
                                       exc.status_code, getattr(exc, 'body', ''))
                    time.sleep(0.5)
                    continue

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
                    "Now deliver a spoken response to sir in 1-2 sentences. "
                    "You are KOBRA — witty, dry, sharp. Address him as sir. "
                    "Speak the outcome naturally — no function names, no tool names, "
                    "no raw URLs, no file paths, no JSON, no technical strings. "
                    "If it went well, confirm it cleanly — maybe with a dry remark if earned. "
                    "If it failed, say so plainly without melodrama. "
                    "Never say 'I have successfully', 'I have executed', 'task has been performed'. "
                    "STRICT LIMIT: stop after 2 sentences. Never produce bullet points or lists."
                ),
            }
        ]
        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_FAST,   # 8b instant — narration needs no heavy model
                messages=turn2_messages,
                max_tokens=config.GROQ_MAX_TOKENS_NARRATION,  # hard brevity cap
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

        # Emit tool call event via decoupled callback (no ui_server import here)
        summary = ", ".join(f"{k}={str(v)[:30]}" for k, v in args.items()) if args else ""
        self._emit("tool_call", tool=name, summary=summary)

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

    # ── Learning / routing correction ──────────────────────────────────────────

    def handle_routing_correction(self, transcript: str, correct_agent: str) -> None:
        """
        Called when user explicitly corrects a routing decision.
        Example: user says "no, use the browser agent for that".
        Logs the correction to the learning system for future routing hints.
        """
        try:
            if hasattr(self, '_last_agent'):
                self._learning.log_routing_correction(transcript, self._last_agent, correct_agent)
            logger.info("[BRAIN] Routing correction noted: → %s", correct_agent)
        except Exception as exc:
            logger.debug("[BRAIN] handle_routing_correction failed: %s", exc)

    @staticmethod
    def _format_tool_result_message(tool_call_id: str, result: str) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }

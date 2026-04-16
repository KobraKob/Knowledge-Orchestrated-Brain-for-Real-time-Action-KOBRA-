"""
task_router.py — Task validation and rule-based routing fallback for KOBRA v4.

Two responsibilities:
  1. validate_and_enrich(tasks, transcript)
       — validates LLM-decomposed tasks (agent names, dependency IDs)
       — if tasks is None (LLM failed), generates a single task via rule-based routing

  2. _rule_based_route(transcript)
       — keyword-to-agent mapping
       — always returns at least one Task
       — used when the orchestrator's LLM decomposition fails
"""

import logging

from agents import VALID_AGENT_NAMES
from models import Task

logger = logging.getLogger(__name__)

# ── Rule table: (keywords, agent_name) — evaluated in order, first match wins ─

_RULES: list[tuple[list[str], str]] = [
    # WhatsApp — check first (contains "message" which could match other rules)
    (["whatsapp", "send message to", "whatsapp message", "text mom",
      "text dad", "message on whatsapp"], "browser"),
    # Email — before generic web rules
    (["send email", "email to", "send mail", "read email", "read mail",
      "check email", "check inbox", "my inbox", "gmail", "unread emails"], "integration"),
    # Calendar
    (["calendar", "schedule a meeting", "create event", "add event",
      "schedule event", "what's on my calendar", "what do i have today",
      "events today", "events tomorrow", "any meetings"], "integration"),
    # Spotify API — before generic media (for explicit "spotify" mentions)
    (["play spotify", "on spotify", "via spotify", "control spotify",
      "pause spotify", "skip on spotify"], "integration"),
    # Research — deep web research (before web to catch "research ..." first)
    (["research ", "do a deep dive", "investigate ", "give me a full breakdown",
      "analyze the ", "do research on", "look into "], "research"),
    # Knowledge base — personal files/notes
    (["what did i", "find my notes", "search my files", "what was the project",
      "remind me about my", "find that file", "in my projects", "recall my notes",
      "check my notes", "what's in my documents", "look up my"], "knowledge"),
    # Screen/Vision — visual interaction
    (["what's on my screen", "what is on screen", "read the screen",
      "click the ", "click on the ", "read the error on screen",
      "what does this code do", "explain what's on screen"], "screen"),
    # MCP — external services
    (["github issue", "create notion", "query supabase", "push to github",
      "create pr", "open issue", "notion page", "github repo"], "mcp"),
    # Window management — focus modes and window snapping
    (["focus mode", "coding mode", "gaming mode", "research mode", "break mode",
      "snap vs code", "snap chrome", "snap window", "activate focus"], "system"),
    # Media — check before "open" to catch "open YouTube and play"
    (["play ", "listen to", "put on", "queue up", "stop music",
      "pause music", "next song", "next track", "previous track"], "media"),
    # Interpreter — websites, apps, projects, complex code (MUST come before dev)
    (["website", "web app", "web application", "landing page", "webpage",
      "web page", "portfolio site", "frontend", "site for",
      "scaffold", "build me a", "write a script", "automate ",
      "generate code", "make a function", "create a project", "new project",
      "write me a program", "make a program", "python script that",
      "script to ", "code that ", "full project"], "interpreter"),
    # Dev — simple file ops and VS Code
    (["create file", "make file", "write a file",
      "vscode", "vs code", "open code", "open editor",
      "git ", "pip install", "npm install", "run tests", "pytest",
      "commit this", "fix the error", "explain this code"], "dev"),
    # Web
    (["search for", "look up", "google ", "find info", "weather",
      "news", "score", "latest", "go to website", "open url", "browse to",
      "what's happening", "breaking news"], "web"),
    # System
    (["open ", "launch ", "start ", "close ", "minimize ", "maximize ",
      "battery", "cpu", "ram", "disk", "system info", "time", "date",
      "volume", "mute", "unmute", "type ", "press ", "clipboard"], "system"),
    # Memory
    (["remember ", "recall ", "memorize ", "save this", "store this",
      "what did you save", "do you remember", "what do you know about me"], "memory"),
    # Conversation — catch-all
    ([], "conversation"),
]


class TaskRouter:
    def validate_and_enrich(
        self,
        tasks: list[Task] | None,
        transcript: str,
    ) -> list[Task]:
        """
        Entry point called by orchestrator.
        - If tasks is None → rule-based fallback
        - If tasks is a list → validate and clean
        """
        if tasks is None:
            logger.warning("[ROUTER] LLM decomposition failed — using rule-based fallback.")
            return self._rule_based_route(transcript)

        validated = self._validate(tasks)
        if not validated:
            logger.warning("[ROUTER] All tasks invalid after validation — rule-based fallback.")
            return self._rule_based_route(transcript)

        return validated

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate(self, tasks: list[Task]) -> list[Task]:
        valid: list[Task] = []
        task_ids = {t.id for t in tasks}

        for task in tasks:
            # Drop tasks with unknown agent names
            if task.agent_name not in VALID_AGENT_NAMES:
                logger.warning(
                    "[ROUTER] Unknown agent '%s' in task %s — dropping.",
                    task.agent_name, task.id,
                )
                continue

            # Fix broken dependency IDs (reference tasks that don't exist)
            bad_deps = [d for d in task.depends_on if d not in task_ids]
            if bad_deps:
                logger.warning(
                    "[ROUTER] Task %s has unknown deps %s — clearing dependencies.",
                    task.id, bad_deps,
                )
                task.depends_on = [d for d in task.depends_on if d in task_ids]

            valid.append(task)

        # Detect circular dependencies — if found, clear all deps (run sequentially)
        if self._has_cycle(valid):
            logger.warning("[ROUTER] Circular dependency detected — clearing all deps.")
            for task in valid:
                task.depends_on = []

        return valid

    @staticmethod
    def _has_cycle(tasks: list[Task]) -> bool:
        """Simple DFS cycle detection on the dependency graph."""
        graph: dict[str, list[str]] = {t.id: t.depends_on for t in tasks}
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for dep in graph.get(node, []):
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in in_stack:
                    return True
            in_stack.discard(node)
            return False

        return any(dfs(t.id) for t in tasks if t.id not in visited)

    # ── Rule-based fallback ────────────────────────────────────────────────────

    def _rule_based_route(self, transcript: str) -> list[Task]:
        t = transcript.lower()
        agent = "conversation"  # default

        for keywords, agent_name in _RULES:
            if not keywords or any(kw in t for kw in keywords):
                agent = agent_name
                break

        logger.info("[ROUTER] Rule-based route → agent: %s", agent)
        return [
            Task(
                id="t1",
                agent_name=agent,
                instruction=transcript,
                can_parallelize=False,
                depends_on=[],
            )
        ]

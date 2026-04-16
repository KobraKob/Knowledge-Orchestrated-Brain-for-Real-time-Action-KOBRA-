"""
planner.py — Neural Planner for KOBRA v4.

Sits BEFORE decomposition. For complex/multi-goal commands, runs a planning
pass using Groq to produce a structured plan. Simple commands are passed through
instantly (no LLM call, zero latency).

Complexity triggers (any one → planning pass):
  - Multiple distinct goals ("... and also ...", "... then ...")
  - Help/advice requests ("help me", "how should I", "plan for")
  - Time/project references ("for tomorrow", "this week", "my project")
  - Ambiguous pronoun resolution needed ("that", "it", "the thing")
  - Long transcript (> 15 words)
"""

import logging
import re
import threading
from groq import Groq

import config

logger = logging.getLogger(__name__)

_PLAN_SYSTEM = """\
You are KOBRA's Neural Planner. Your job is to deeply understand the user's TRUE intent
and produce a structured plan for the orchestrator to execute.

Think about:
1. What is the user ACTUALLY trying to accomplish? (not just what they literally said)
2. What information does KOBRA already have vs what it needs to find?
3. What is the optimal sequence of actions?
4. Are there any ambiguities that would cause failures?

Output a JSON object with:
{
  "intent": "One-line description of what the user actually wants",
  "complexity": "simple" | "moderate" | "complex",
  "goals": ["goal 1", "goal 2", ...],
  "clarifications_needed": ["question if truly ambiguous, else empty"],
  "enriched_transcript": "Rewrite the user's command as a clear, explicit, fully self-contained instruction that removes all ambiguity. Expand pronouns, fill in context from the conversation history."
}

CRITICAL: "enriched_transcript" must be completely self-contained — the orchestrator
will use this INSTEAD of the original. Never use pronouns or vague references.
If the command is simple and clear, just clean it up slightly and return it.
Return ONLY valid JSON. No markdown, no explanation.
"""

# Patterns that indicate a complex command worth planning
_MULTI_GOAL = re.compile(
    r"\b(and also|and then|after that|then|additionally|as well|plus|also)\b",
    re.IGNORECASE,
)
_HELP_PATTERNS = (
    "help me", "how should i", "plan for", "give me a", "what's the best",
    "walk me through", "guide me", "assist me", "i need to", "i want to",
    "what do i need", "how do i", "how can i",
)
_TIME_PATTERNS = (
    "tomorrow", "next week", "this week", "tonight", "this month",
    "by friday", "by monday", "schedule", "deadline",
)
_AMBIGUOUS_PATTERNS = ("that", "it", "the one", "the thing", "the first", "the second", "those")


class NeuralPlanner:
    def __init__(self) -> None:
        self._client = Groq(api_key=config.GROQ_API_KEY)
        self._lock = threading.Lock()

    def should_plan(self, transcript: str, recent_context: str = "") -> bool:
        """
        Fast heuristic: should we run a planning pass?
        Returns False for simple commands (zero latency cost).
        """
        t = transcript.lower().strip()
        word_count = len(t.split())

        # Always plan for long transcripts
        if word_count > 15:
            return True

        # Multi-goal signals
        if _MULTI_GOAL.search(t):
            return True

        # Help/advice patterns
        if any(p in t for p in _HELP_PATTERNS):
            return True

        # Time-sensitive planning
        if any(p in t for p in _TIME_PATTERNS):
            return True

        # Ambiguous references (only if we have context to resolve them)
        if recent_context and any(p in t for p in _AMBIGUOUS_PATTERNS):
            return True

        return False

    def plan(self, transcript: str, recent_context: str = "") -> dict:
        """
        Run a planning pass. Returns a plan dict with 'enriched_transcript'.
        Falls back to original transcript on any failure.
        """
        default = {
            "intent": transcript,
            "complexity": "simple",
            "goals": [transcript],
            "clarifications_needed": [],
            "enriched_transcript": transcript,
        }

        try:
            user_content = transcript
            if recent_context:
                user_content = f"{recent_context}Current command: {transcript}"

            import json
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_FAST,  # Use fast model — planning is lightweight
                messages=[
                    {"role": "system", "content": _PLAN_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=400,
                temperature=0.15,
                timeout=15,  # Hard timeout — never block main pipeline
            )
            raw = (response.choices[0].message.content or "").strip()
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            plan = json.loads(clean)

            # Validate required fields
            if "enriched_transcript" not in plan:
                plan["enriched_transcript"] = transcript

            logger.info(
                "[PLANNER] complexity=%s | intent=%s",
                plan.get("complexity", "?"),
                plan.get("intent", "?")[:80],
            )
            return plan

        except Exception as exc:
            logger.warning("[PLANNER] Planning failed (%s) — using original transcript.", exc)
            return default

    def enrich(self, transcript: str, recent_context: str = "") -> str:
        """
        Main entry point: returns enriched transcript (or original if planning skipped/failed).
        Only runs the LLM if should_plan() returns True.
        """
        if not self.should_plan(transcript, recent_context):
            logger.debug("[PLANNER] Simple command — skipping planning pass.")
            return transcript

        logger.info("[PLANNER] Complex command detected — running planning pass.")
        plan = self.plan(transcript, recent_context)
        return plan.get("enriched_transcript", transcript)

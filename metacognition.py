"""
metacognition.py — Pre-execution confidence scoring for KOBRA v4.

Before executing a plan, KOBRA evaluates each step:
  - HIGH confidence (>=0.8): execute immediately
  - MEDIUM confidence (0.5-0.8): log uncertainty, proceed with caution
  - LOW confidence (<0.5): ask user for clarification BEFORE executing

This eliminates the worst retry scenario: executing 4 steps successfully,
then failing on step 5 due to ambiguity that could have been resolved upfront.

Confidence is primarily RULE-BASED (fast, no LLM).
LLM is used only for genuinely ambiguous cases.
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StepConfidence:
    task_id:    str
    agent_name: str
    instruction: str
    confidence: float       # 0.0 - 1.0
    issues:     list[str]   # specific uncertainty reasons
    clarification_needed: bool
    clarification_question: str = ""


# -- Ambiguity patterns (rule-based, fast) ------------------------------------

# Vague references that need resolution
_VAGUE_REFERENCES = re.compile(
    r"\b(that|it|the thing|the one|the file|the folder|the project|the repo|those|"
    r"the usual|the same|that folder|my thing|the script|the code)\b",
    re.IGNORECASE,
)

# Unknown contact/person references
_PERSON_REFS = re.compile(
    r"\b(send (?:it |this )?to|email|message|text|call)\s+([A-Z][a-z]+ ?[A-Z]?[a-z]*)\b"
)

# Destructive operations that need explicit confirmation
_DESTRUCTIVE_OPS = re.compile(
    r"\b(delete|remove|rm|drop|destroy|wipe|clear|uninstall|format|overwrite|truncate)\b",
    re.IGNORECASE,
)

# Agent-specific known-reliable operations (HIGH confidence)
_RELIABLE_PATTERNS: dict[str, list[str]] = {
    "conversation": [".*"],  # always reliable
    "web": ["search for", "look up", "weather", "news", "what is"],
    "system": ["open ", "launch ", "close ", "volume", "mute"],
    "memory": ["remember", "recall", "save this"],
    "media": ["play ", "pause", "stop music", "next track"],
}

# Agent-specific risky operations (LOW confidence without more info)
_RISKY_PATTERNS: dict[str, list[str]] = {
    "dev": ["delete", "remove", "overwrite", "clear"],
    "system": ["rm ", "format", "shutdown", "restart"],
    "interpreter": ["delete", "drop database", "rm -rf"],
    "mcp": ["delete", "close issue", "merge"],
}


class MetacognitiveScorer:
    """
    Scores pre-execution confidence for each task step.
    Returns clarification questions for low-confidence steps.
    """

    def score_tasks(self, tasks: list, context: str = "") -> list[StepConfidence]:
        """
        Score all tasks and return confidence assessments.
        context = recent conversation for resolving ambiguous references.
        """
        return [self._score_task(task, context) for task in tasks]

    def _score_task(self, task, context: str) -> StepConfidence:
        instruction = task.instruction
        agent = task.agent_name
        issues = []
        confidence = 1.0

        # Check 1: Vague references
        vague_matches = _VAGUE_REFERENCES.findall(instruction)
        if vague_matches and not self._context_resolves(vague_matches, context):
            issues.append(f"Ambiguous reference: '{vague_matches[0]}'")
            confidence -= 0.35

        # Check 2: Unknown person/contact
        person_match = _PERSON_REFS.search(instruction)
        if person_match:
            person = person_match.group(2)
            if not self._person_in_context(person, context):
                issues.append(f"Unknown contact: '{person}' — not in known contacts")
                confidence -= 0.25

        # Check 3: Destructive operation
        if _DESTRUCTIVE_OPS.search(instruction):
            issues.append(f"Destructive operation detected in '{agent}' instruction")
            confidence -= 0.4

        # Check 4: Agent-specific risky patterns
        risky = _RISKY_PATTERNS.get(agent, [])
        if any(p in instruction.lower() for p in risky):
            issues.append(f"Risky pattern for {agent} agent")
            confidence -= 0.3

        # Check 5: Missing required context for agent
        if agent == "integration" and not any(
            kw in instruction.lower() for kw in ("email", "calendar", "spotify", "gmail")
        ):
            issues.append("Integration agent instruction missing service specification")
            confidence -= 0.2

        # Check 6: File path ambiguity
        if any(kw in instruction.lower() for kw in ("file", "folder", "directory", "path")) and \
           not re.search(r"[A-Za-z]:\\|/[a-z]|\.py|\.txt|\.md|\.js|\.html|desktop|documents", instruction, re.IGNORECASE):
            issues.append("File operation with no specific path — which file?")
            confidence -= 0.2

        confidence = max(0.0, min(1.0, confidence))
        needs_clarification = confidence < 0.5

        clarification_q = ""
        if needs_clarification and issues:
            clarification_q = self._build_clarification_question(agent, instruction, issues)

        return StepConfidence(
            task_id=task.id,
            agent_name=agent,
            instruction=instruction,
            confidence=confidence,
            issues=issues,
            clarification_needed=needs_clarification,
            clarification_question=clarification_q,
        )

    def _context_resolves(self, vague_refs: list[str], context: str) -> bool:
        """Check if recent context makes the vague reference unambiguous."""
        if not context:
            return False
        # If context mentions specific files/projects, it likely resolves the ref
        has_specific = bool(re.search(
            r"[A-Za-z0-9_\-]+\.(py|txt|md|js|html|json|csv)|'[^']+'|\"[^\"]+\"",
            context,
        ))
        return has_specific

    def _person_in_context(self, person: str, context: str) -> bool:
        """Check if person was mentioned in recent context (so we know who they are)."""
        if not context:
            return False
        return person.lower() in context.lower()

    def _build_clarification_question(self, agent: str, instruction: str, issues: list[str]) -> str:
        """Build a specific, actionable clarification question."""
        if not issues:
            return "Could you be more specific?"

        issue = issues[0]  # address the most important issue

        if "Ambiguous reference" in issue:
            vague = re.search(r"'([^']+)'", issue)
            ref = vague.group(1) if vague else "that"
            return f"What specifically do you mean by '{ref}'?"

        if "Unknown contact" in issue:
            person = re.search(r"'([^']+)'", issue)
            name = person.group(1) if person else "that person"
            return f"I don't have contact info for {name}. What's their email or phone?"

        if "Destructive operation" in issue:
            return "This will permanently delete something. Are you sure? Please confirm."

        if "File operation" in issue:
            return "Which file or folder should I use? Please give the full path or name."

        if "Integration" in issue:
            return "Should I use Gmail, Google Calendar, or Spotify for this?"

        return f"I'm not sure about: {issues[0]}. Can you clarify?"

    def get_clarification_prompt(self, scores: list[StepConfidence]) -> str | None:
        """
        If any step has low confidence, return a combined clarification prompt.
        Returns None if all steps are high confidence (no clarification needed).
        """
        low_conf = [s for s in scores if s.clarification_needed]
        if not low_conf:
            return None

        if len(low_conf) == 1:
            return low_conf[0].clarification_question

        questions = [s.clarification_question for s in low_conf if s.clarification_question]
        if not questions:
            return None

        return "Before I proceed, I need a few clarifications: " + " Also, ".join(questions)

    def summary_log(self, scores: list[StepConfidence]) -> None:
        """Log confidence scores for all steps."""
        for s in scores:
            level = "HIGH" if s.confidence >= 0.8 else "MEDIUM" if s.confidence >= 0.5 else "LOW"
            logger.info(
                "[METACOG] %s (%s) confidence=%.2f issues=%s",
                s.task_id, s.agent_name, s.confidence, s.issues or "none",
            )

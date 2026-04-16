"""
synthesizer.py — Combines multi-agent results into one spoken response.

Fast path: single successful result → return output directly (no Groq call)
  UNLESS the output looks raw/boring, in which case we wrap it through
  process_conversational for KOBRA personality.
Multi-result: call brain.process_conversational with a structured summary prompt.
"""

import logging

from models import TaskResult

logger = logging.getLogger(__name__)

_SYNTHESIS_PROMPT = """\
The user said: "{transcript}"

Results from each task executed:
{results_block}

Write a single natural spoken response summarising what was done.
You are KOBRA — witty, sharp, dry. Address the user as "sir".

Rules:
- Max 2-3 sentences. Be concise. Don't pad.
- Do NOT robotically list steps. Synthesise like a human who did the thing.
- If everything worked: confirm it cleanly, maybe with a dry remark if it's warranted.
- If something failed: mention it once, briefly, without drama — then move on.
- Never say "I have successfully completed", "I have executed", "task has been performed".
  Talk like a person, not a receipt printer.
- No bullet points, no markdown, no URLs, no raw file paths in the spoken output.
"""

# Phrases that indicate the output is a raw/boring agent response needing personality wrap
_BORING_PHRASES = (
    "done.", "done,", "command executed", "executed successfully",
    "i have successfully", "task completed", "operation complete",
    "completed successfully", "no output", "none",
)

_MIN_GOOD_LENGTH = 25  # outputs shorter than this are almost always naked status strings


def _needs_personality_wrap(text: str) -> bool:
    """Return True if the output is too bare to speak as-is and needs KOBRA flair."""
    stripped = text.strip()
    if len(stripped) < _MIN_GOOD_LENGTH:
        return True
    lower = stripped.lower()
    return any(lower.startswith(p) or lower == p for p in _BORING_PHRASES)


class Synthesizer:
    def __init__(self, brain) -> None:
        self._brain = brain

    def synthesize(
        self,
        transcript: str,
        results: list[TaskResult],
    ) -> str:
        """
        Return a single spoken response string from all agent results.
        """
        if not results:
            return "I completed that, sir."

        # Fast path: single result
        if len(results) == 1:
            r = results[0]
            if r.was_aborted:
                return "Understood, sir — stopping."
            if r.success:
                # If the output is raw/boring, wrap through KOBRA personality
                if _needs_personality_wrap(r.output):
                    return self._personality_wrap(transcript, r.output)
                return r.output
            return f"Ran into a snag, sir — {r.output}"

        # Multi-result: generate a coherent summary
        return self._generate_summary(transcript, results)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _personality_wrap(self, transcript: str, raw_output: str) -> str:
        """
        Re-narrate a bare agent output through KOBRA's voice.
        Used when the output is too short or robotic to speak as-is.
        """
        prompt = (
            f"The user asked: \"{transcript}\"\n"
            f"Result: {raw_output}\n\n"
            "Deliver this as KOBRA in 1 sentence. Witty, dry, direct. "
            "Address sir. No bullet points, no markdown, no technical strings."
        )
        try:
            return self._brain.process_conversational(prompt)
        except Exception:
            return raw_output  # fallback: just speak it raw

    def _generate_summary(self, transcript: str, results: list[TaskResult]) -> str:
        aborted = [r for r in results if r.was_aborted]
        if len(aborted) == len(results):
            return "Stopped, sir. No tasks completed."

        results_block = self._format_results(results)
        prompt = _SYNTHESIS_PROMPT.format(
            transcript=transcript,
            results_block=results_block,
        )

        try:
            return self._brain.process_conversational(prompt)
        except Exception as exc:
            logger.error("Synthesis Groq call failed: %s", exc)
            # Fallback: concatenate successful outputs
            outputs = [r.output for r in results if r.success and not r.was_aborted]
            return " ".join(outputs) if outputs else "Done, sir."

    @staticmethod
    def _format_results(results: list[TaskResult]) -> str:
        lines: list[str] = []
        for r in results:
            if r.was_aborted:
                lines.append(f"- [{r.agent_name}]: ABORTED")
            elif r.success:
                lines.append(f"- [{r.agent_name}]: {r.output}")
            else:
                lines.append(f"- [{r.agent_name}]: FAILED — {r.output}")
        return "\n".join(lines)

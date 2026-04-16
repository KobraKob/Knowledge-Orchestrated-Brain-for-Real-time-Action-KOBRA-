"""
agents/web_agent.py — Web search and navigation agent.

Handles: web_search, open_url.
Summarises search results for voice output (no markdown, no bullets).
"""

from agents.base_agent import BaseAgent
from models import Task


class WebAgent(BaseAgent):
    AGENT_NAME = "web"
    OWNED_TOOLS = [
        "web_search",
        "open_url",
        "speak_only",
    ]
    SYSTEM_PROMPT = (
        "You are KOBRA's web agent. "
        "You search the web and open URLs on behalf of sir. "
        "When searching, summarise the top results in plain spoken English — "
        "max 4 sentences, no bullet points, no markdown, no URLs in the spoken output. "
        "Prioritise factual, recent information. "
        "The summary will be read aloud — write it as natural speech."
    )

    def _run(self, task: Task) -> str:
        return self._brain.process_scoped(
            self._build_instruction(task),
            self.OWNED_TOOLS,
            self.SYSTEM_PROMPT,
        )

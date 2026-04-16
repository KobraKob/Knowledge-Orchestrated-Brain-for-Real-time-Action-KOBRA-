"""
agents/conversation_agent.py — Pure conversational agent.

No tools. Uses brain.process_conversational() which physically cannot call a tool.
Handles opinions, hypotheticals, general knowledge, jokes, greetings.
"""

from agents.base_agent import BaseAgent
from models import Task


class ConversationAgent(BaseAgent):
    AGENT_NAME = "conversation"
    OWNED_TOOLS: list[str] = []          # empty — zero tool capability by design
    SYSTEM_PROMPT = ""                    # unused — process_conversational has its own

    def _run(self, task: Task) -> str:
        return self._brain.process_conversational(self._build_instruction(task))

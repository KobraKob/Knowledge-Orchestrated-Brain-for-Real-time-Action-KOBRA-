"""
agents/memory_agent.py — Long-term memory agent.

Handles: save_memory, recall_memory.
Has direct access to the Memory instance so it can query the DB directly too.
"""

from agents.base_agent import BaseAgent
from models import Task


class MemoryAgent(BaseAgent):
    AGENT_NAME = "memory"
    OWNED_TOOLS = [
        "save_memory",
        "recall_memory",
        "speak_only",
    ]
    SYSTEM_PROMPT = (
        "You are KOBRA's memory agent. "
        "You save and retrieve information from sir's long-term memory store. "
        "When saving: extract a clean, descriptive key and a complete value. "
        "When recalling: return the relevant information clearly formatted for voice output — "
        "no raw JSON, no technical formatting. "
        "If nothing is found in memory, say so honestly."
    )

    def _run(self, task: Task) -> str:
        return self._brain.process_scoped(
            self._build_instruction(task),
            self.OWNED_TOOLS,
            self.SYSTEM_PROMPT,
        )

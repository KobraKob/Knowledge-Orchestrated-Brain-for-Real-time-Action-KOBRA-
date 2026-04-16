"""
agents/knowledge_agent.py — RAG-powered personal knowledge base agent.

Answers questions from indexed files in watched folders.
Zero hallucination: if the answer isn't in the retrieved context, says so.

Trigger phrases (task_router.py):
  "what did i", "find my", "search my files", "what was the",
  "remind me about my", "find that", "in my projects", "recall my",
  "check my notes", "what's in my", "look up my"
"""

import logging

from agents.base_agent import BaseAgent
from models import Task

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are KOBRA's personal knowledge module — you know sir's projects, notes, and documents.
When answering, use ONLY the provided context. Never invent information.
If the answer isn't in the context, say so clearly.
Cite the source file when relevant. Keep responses concise and spoken-friendly.
Address the user as 'sir'.
"""


class KnowledgeAgent(BaseAgent):
    AGENT_NAME = "knowledge"
    OWNED_TOOLS = ["speak_only"]
    SYSTEM_PROMPT = _SYSTEM_PROMPT

    def __init__(self, brain, memory, retriever) -> None:
        super().__init__(brain, memory)
        self._retriever = retriever

    def _run(self, task: Task) -> str:
        question = self._build_instruction(task)

        context, sources = self._retriever.query_with_sources(question)

        if not context.strip():
            return "I don't have information about that in my knowledge base, sir."

        augmented = (
            f"Answer this question using ONLY the context below.\n"
            f"If the answer isn't in the context, say so.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}"
        )

        response = self._brain.process_conversational(augmented)

        # Append source file names to spoken response (brief)
        if sources:
            filenames = [s.split("/")[-1].split("\\")[-1] for s in sources[:2]]
            response += f" (from {', '.join(filenames)})"

        return response

"""
memory/ — Typed memory layer package for KOBRA v5.

Four memory types:
  EpisodicMemory   — what happened (conversations, sessions, actions)
  SemanticMemory   — what's true (facts, preferences, user identity)
  ProceduralMemory — how to do things (routing strategies, tool success rates)
  PerceptualMemory — what exists now (RAG + live APIs)

All exposed through MemoryRouter (memory/router.py).

Also re-exports the legacy `Memory` class (SQLite conversation + facts store)
so that `from memory import Memory` continues to work even though Python
resolves "memory" as this package rather than the root-level memory.py.
"""

from memory.episodic   import EpisodicMemory
from memory.semantic   import SemanticMemory
from memory.procedural import ProceduralMemory
from memory.perceptual import PerceptualMemory

# Re-export legacy Memory class from the renamed root-level module.
# memory.py was copied to memory_store.py to avoid package shadowing.
from memory_store import Memory  # noqa: F401

__all__ = [
    "Memory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "PerceptualMemory",
]

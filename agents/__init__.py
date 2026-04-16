"""
agents/__init__.py — Agent registry for KOBRA v4.

Maps agent_name strings → agent class instances.
Built once at startup by orchestrator.py.

v4 additions:
  - ResearchAgent:   deep web research + report generation
  - ScreenAgent:     vision-based screen navigation
  - KnowledgeAgent:  personal RAG knowledge base queries
  - MCPAgent:        Model Context Protocol server integration
"""

from agents.conversation_agent import ConversationAgent
from agents.system_agent import SystemAgent
from agents.web_agent import WebAgent
from agents.dev_agent import DevAgent
from agents.media_agent import MediaAgent
from agents.memory_agent import MemoryAgent
from agents.integration_agent import IntegrationAgent
from agents.browser_agent import BrowserAgent
from agents.interpreter_agent import InterpreterAgent
from agents.research_agent import ResearchAgent
from agents.screen_agent import ScreenAgent
from agents.knowledge_agent import KnowledgeAgent


def build_agent_registry(
    brain,
    memory,
    credential_store=None,
    contact_store=None,
    retriever=None,
    mcp_client=None,
) -> dict:
    """
    Instantiate all agents and return a name → instance mapping.

    Optional parameters:
      credential_store / contact_store — enables IntegrationAgent + BrowserAgent
      retriever                        — enables KnowledgeAgent (RAG)
      mcp_client                       — enables MCPAgent
    """
    core_agents = [
        ConversationAgent(brain, memory),
        SystemAgent(brain, memory),
        WebAgent(brain, memory),
        DevAgent(brain, memory),
        MediaAgent(brain, memory),
        MemoryAgent(brain, memory),
        InterpreterAgent(brain, memory),
        ResearchAgent(brain, memory),
        ScreenAgent(brain, memory),
    ]

    registry = {agent.AGENT_NAME: agent for agent in core_agents}

    # Integration agents — require stores
    if credential_store is not None and contact_store is not None:
        registry["integration"] = IntegrationAgent(brain, memory, credential_store, contact_store)
        registry["browser"]     = BrowserAgent(brain, memory, credential_store, contact_store)

    # RAG knowledge agent — requires retriever
    if retriever is not None:
        registry["knowledge"] = KnowledgeAgent(brain, memory, retriever)

    # MCP agent — requires mcp_client with at least one server registered
    if mcp_client is not None and mcp_client.servers:
        registry["mcp"] = MCPAgent(brain, memory, mcp_client)

    return registry


# Valid agent names — used by task_router for validation
VALID_AGENT_NAMES = {
    "conversation",
    "system",
    "web",
    "dev",
    "media",
    "memory",
    "integration",
    "browser",
    "interpreter",
    "research",
    "screen",
    "knowledge",
    "mcp",
}


# Lazy import to avoid circular issues if mcp_agent is not always present
def _import_mcp_agent():
    try:
        from agents.mcp_agent import MCPAgent
        return MCPAgent
    except ImportError:
        return None


MCPAgent = _import_mcp_agent()

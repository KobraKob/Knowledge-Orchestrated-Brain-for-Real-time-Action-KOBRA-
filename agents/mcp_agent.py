"""
agents/mcp_agent.py — Model Context Protocol agent for KOBRA.

Routes voice commands to registered MCP servers (GitHub, Notion, Supabase, etc.)
without requiring any hardcoded tool definitions. Just register a server and
KOBRA discovers its tools automatically.

Trigger phrases (task_router.py):
  "create github issue", "create notion page", "query supabase",
  "commit to github", "add to notion", etc.
"""

import logging

from agents.base_agent import BaseAgent
from models import Task

logger = logging.getLogger(__name__)

_BASE_SYSTEM_PROMPT = """\
You are KOBRA's MCP integration module. You connect to external services.
Use the available MCP tools to fulfill sir's request.
Address the user as 'sir'. Be concise and direct.
Tool names follow the pattern: servicename__toolname (e.g. github__create_issue).
Always use the exact tool names provided — never add spaces or alter them.
"""


class MCPAgent(BaseAgent):
    AGENT_NAME = "mcp"
    OWNED_TOOLS: list[str] = []  # populated dynamically from MCP servers
    SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT

    def __init__(self, brain, memory, mcp_client) -> None:
        super().__init__(brain, memory)
        self._mcp = mcp_client
        # Register MCP tool callables in brain's registry
        self._register_mcp_tools()

    def _register_mcp_tools(self) -> None:
        """Add MCP tool callables to brain's tool registry."""
        for tool in self._mcp.get_all_tools():
            prefixed_name = tool["function"]["name"]

            def make_caller(name):
                def caller(**kwargs):
                    return self._mcp.call_prefixed_tool(name, kwargs)
                return caller

            self._brain._registry[prefixed_name] = make_caller(prefixed_name)
            self.OWNED_TOOLS.append(prefixed_name)

        logger.info("[MCP_AGENT] Registered %d MCP tools", len(self.OWNED_TOOLS))

    def _build_mcp_system_prompt(self) -> str:
        server_desc = self._mcp.describe_servers()
        return (
            f"{_BASE_SYSTEM_PROMPT}\n\n"
            f"CONNECTED SERVICES:\n{server_desc}\n\n"
            f"AVAILABLE TOOLS:\n"
            + "\n".join(f"  - {n}" for n in self.OWNED_TOOLS)
        )

    def _run(self, task: Task) -> str:
        if not self.OWNED_TOOLS:
            return "No MCP servers are connected, sir."

        return self._brain.process_scoped(
            instruction=self._build_instruction(task),
            tool_names=self.OWNED_TOOLS,
            system_prompt=self._build_mcp_system_prompt(),
        )

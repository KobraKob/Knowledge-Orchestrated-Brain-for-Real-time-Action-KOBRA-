"""
mcp/client.py — MCP (Model Context Protocol) server connector for KOBRA.

Connects KOBRA to any MCP server — GitHub, Notion, Supabase, filesystem, etc.
New capabilities are added by registering an MCP server URL, not by writing code.

Usage (in main.py):
    mcp = MCPClient()
    mcp.register_server("github", "http://localhost:3001", "GitHub repos and issues")
    mcp.register_server("notion", "http://localhost:3002", "Notion pages and databases")
"""

import logging

import requests

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self) -> None:
        # name → {url, tools, description}
        self.servers: dict[str, dict] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register_server(self, name: str, url: str, description: str = "") -> bool:
        """
        Register an MCP server. Fetches its tool list immediately.
        Returns True if successful, False if the server is unreachable.
        """
        tools = self._fetch_tools(url)
        if tools is None:
            logger.warning("[MCP] Server '%s' at %s is unreachable — skipping.", name, url)
            return False
        self.servers[name] = {"url": url, "tools": tools, "description": description}
        logger.info("[MCP] Registered '%s' with %d tools", name, len(tools))
        return True

    def register_from_config(self, server_list: list[dict]) -> None:
        """Register multiple servers from config.MCP_SERVERS list."""
        for entry in server_list:
            self.register_server(
                entry.get("name", ""),
                entry.get("url", ""),
                entry.get("description", ""),
            )

    # ── Tool discovery ────────────────────────────────────────────────────────

    def _fetch_tools(self, url: str) -> list[dict] | None:
        """Fetch available tools from an MCP server. Returns None if unreachable."""
        try:
            response = requests.get(f"{url}/tools", timeout=5)
            response.raise_for_status()
            return response.json().get("tools", [])
        except Exception as exc:
            logger.debug("Tool fetch failed for %s: %s", url, exc)
            return None

    def get_all_tools(self) -> list[dict]:
        """
        Return all tools from all registered servers in Groq tool schema format.
        Tool names are prefixed: github__create_issue, notion__create_page, etc.
        """
        all_tools = []
        for server_name, server in self.servers.items():
            for tool in server.get("tools", []):
                groq_tool = {
                    "type": "function",
                    "function": {
                        "name": f"{server_name}__{tool.get('name', '')}",
                        "description": (
                            f"[{server_name.upper()}] {tool.get('description', '')}"
                        ),
                        "parameters": tool.get("inputSchema", {
                            "type": "object", "properties": {}, "required": [],
                        }),
                    },
                }
                all_tools.append(groq_tool)
        return all_tools

    def route_to_server(self, tool_name: str) -> tuple[str, str] | None:
        """
        Given a prefixed tool name like 'github__create_issue',
        returns (server_name, bare_tool_name) or None.
        """
        if "__" not in tool_name:
            return None
        server_name, bare_name = tool_name.split("__", 1)
        if server_name in self.servers:
            return server_name, bare_name
        return None

    # ── Tool invocation ────────────────────────────────────────────────────────

    def call_tool(self, server_name: str, tool_name: str, args: dict) -> str:
        """
        Call a tool on a registered MCP server.
        Returns the result string or an error message.
        """
        server = self.servers.get(server_name)
        if not server:
            return f"MCP server '{server_name}' not registered."
        try:
            response = requests.post(
                f"{server['url']}/tools/{tool_name}",
                json={"arguments": args},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            # MCP response format: {"content": "...", ...}
            return str(result.get("content", result))
        except requests.Timeout:
            return f"MCP call to {server_name}/{tool_name} timed out."
        except Exception as exc:
            logger.error("[MCP] %s/%s failed: %s", server_name, tool_name, exc)
            return f"MCP error: {exc}"

    def call_prefixed_tool(self, prefixed_name: str, args: dict) -> str:
        """Convenience: call a tool by its prefixed name like 'github__create_issue'."""
        route = self.route_to_server(prefixed_name)
        if not route:
            return f"Unknown MCP tool: '{prefixed_name}'."
        server_name, bare_name = route
        return self.call_tool(server_name, bare_name, args)

    def describe_servers(self) -> str:
        """Return a human-readable list of registered servers for system prompts."""
        if not self.servers:
            return "No MCP servers registered."
        lines = []
        for name, s in self.servers.items():
            tool_count = len(s.get("tools", []))
            lines.append(f"- {name}: {s['description']} ({tool_count} tools)")
        return "\n".join(lines)

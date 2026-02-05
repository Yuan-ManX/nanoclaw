"""
ClawAI tool registry.

Manages tool registration, schema exposure, and execution routing
for agent tool calls.
"""

from typing import Any

from clawai.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Acts as the execution router between the agent and concrete tool
    implementations.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # =========================
    # Registration
    # =========================

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools

    # =========================
    # Schema exposure
    # =========================

    def get_definitions(self) -> list[dict[str, Any]]:
        """
        Return tool schemas in LLM-compatible format.
        """
        return [tool.to_schema() for tool in self._tools.values()]

    # =========================
    # Execution
    # =========================

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """
        Execute a registered tool.

        Args:
            name: Tool name.
            params: Parsed tool arguments.

        Returns:
            Tool execution result (string for now).
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: tool '{name}' is not registered"

        # Parameter validation
        try:
            errors = tool.validate_params(params)
        except Exception as e:
            return f"Error: invalid schema for tool '{name}': {e}"

        if errors:
            return (
                f"Error: invalid parameters for tool '{name}': "
                + "; ".join(errors)
            )

        # Execute tool
        try:
            return await tool.execute(**params)
        except Exception as e:
            return f"Error: tool '{name}' execution failed: {e}"

    # =========================
    # Introspection
    # =========================

    @property
    def tool_names(self) -> list[str]:
        """List names of all registered tools."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

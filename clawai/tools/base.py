"""
Base abstraction for ClawAI action tools.

A Tool represents an executable capability that an agent can invoke
to affect the external world (files, shell, network, etc).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional


class Tool(ABC):
    """
    Abstract base class for ClawAI tools.

    A tool is:
    - Action-oriented (does something, not just computes)
    - Side-effect aware (filesystem, network, processes)
    - Callable by agents via structured arguments
    """

    # JSON Schema → Python runtime type mapping
    _TYPE_MAP: Dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    # ---------
    # Identity
    # ---------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name exposed to the agent."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for agent planning."""
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        JSON Schema describing tool parameters.

        Must be an object schema.
        """
        raise NotImplementedError

    # -----------------
    # Execution Surface
    # -----------------

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """
        Execute the tool with validated parameters.

        Returns:
            Final textual result for the agent.
        """
        raise NotImplementedError

    # Optional: streaming execution (for long-running tools)
    async def stream(self, **kwargs: Any) -> AsyncIterator[str]:
        """
        Stream partial results during execution.

        Default implementation falls back to execute().
        Override for tools that support progressive output.
        """
        result = await self.execute(**kwargs)
        yield result

    # Optional: cancellation hook
    async def cancel(self) -> None:
        """
        Cancel an in-progress execution.

        Override if the tool manages long-running resources.
        """
        return None

    # -----------------
    # Validation Layer
    # -----------------

    def validate(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate parameters against JSON schema.

        Returns:
            List of validation errors (empty if valid).
        """
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Tool schema must be object type, got {schema.get('type')!r}")
        return self._validate(params, {**schema, "type": "object"}, path="")

    def _validate(self, value: Any, schema: Dict[str, Any], path: str) -> List[str]:
        errors: List[str] = []
        expected_type = schema.get("type")
        label = path or "parameter"

        if expected_type in self._TYPE_MAP:
            if not isinstance(value, self._TYPE_MAP[expected_type]):
                return [f"{label} should be {expected_type}"]

        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")

        if expected_type in ("integer", "number"):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")

        if expected_type == "string":
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} characters")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} characters")

        if expected_type == "object":
            props = schema.get("properties", {})
            for key in schema.get("required", []):
                if key not in value:
                    errors.append(f"missing required {path + '.' + key if path else key}")
            for key, val in value.items():
                if key in props:
                    errors.extend(
                        self._validate(
                            val,
                            props[key],
                            f"{path}.{key}" if path else key,
                        )
                    )

        if expected_type == "array" and "items" in schema:
            for idx, item in enumerate(value):
                errors.extend(
                    self._validate(
                        item,
                        schema["items"],
                        f"{path}[{idx}]" if path else f"[{idx}]",
                    )
                )

        return errors

    # -----------------
    # LLM Integration
    # -----------------

    def to_schema(self) -> Dict[str, Any]:
        """
        Convert tool to LLM-compatible function schema.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

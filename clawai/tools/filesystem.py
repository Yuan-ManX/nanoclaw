"""
ClawAI filesystem tools: read, write, edit, list.

These tools provide side-effect aware capabilities for agents to
interact with the local filesystem in a safe, traceable manner.
"""

from pathlib import Path
from typing import Any

from clawai.tools.base import Tool


# =========================
# Internal helpers
# =========================

def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _error(message: str) -> str:
    return f"Error: {message}"


def _ok(message: str) -> str:
    return message


# =========================
# Read File
# =========================

class ReadFileTool(Tool):
    """Read the contents of a text file."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to read",
                }
            },
            "required": ["path"],
        }

    async def execute(self, path: str, **_: Any) -> str:
        try:
            file_path = _resolve_path(path)

            if not file_path.exists():
                return _error(f"file not found: {path}")
            if not file_path.is_file():
                return _error(f"not a file: {path}")

            content = file_path.read_text(encoding="utf-8")
            return content

        except PermissionError:
            return _error(f"permission denied: {path}")
        except Exception as e:
            return _error(f"failed to read file: {e}")


# =========================
# Write File
# =========================

class WriteFileTool(Tool):
    """Write text content to a file."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file at the given path. "
            "Creates parent directories if they do not exist."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write into the file",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **_: Any) -> str:
        try:
            file_path = _resolve_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_path.write_text(content, encoding="utf-8")

            return _ok(
                f"Wrote {len(content)} characters to {file_path}"
            )

        except PermissionError:
            return _error(f"permission denied: {path}")
        except Exception as e:
            return _error(f"failed to write file: {e}")


# =========================
# Edit File
# =========================

class EditFileTool(Tool):
    """Edit a file by replacing a unique text segment."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing old_text with new_text. "
            "The old_text must appear exactly once."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to be replaced",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self,
        path: str,
        old_text: str,
        new_text: str,
        **_: Any,
    ) -> str:
        try:
            file_path = _resolve_path(path)

            if not file_path.exists():
                return _error(f"file not found: {path}")
            if not file_path.is_file():
                return _error(f"not a file: {path}")

            content = file_path.read_text(encoding="utf-8")

            occurrences = content.count(old_text)
            if occurrences == 0:
                return _error("old_text not found in file")
            if occurrences > 1:
                return (
                    f"Warning: old_text appears {occurrences} times. "
                    "Please provide a more specific match."
                )

            updated = content.replace(old_text, new_text, 1)
            file_path.write_text(updated, encoding="utf-8")

            return _ok(f"Edited file successfully: {file_path}")

        except PermissionError:
            return _error(f"permission denied: {path}")
        except Exception as e:
            return _error(f"failed to edit file: {e}")


# =========================
# List Directory
# =========================

class ListDirTool(Tool):
    """List contents of a directory."""

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List the contents of a directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list",
                }
            },
            "required": ["path"],
        }

    async def execute(self, path: str, **_: Any) -> str:
        try:
            dir_path = _resolve_path(path)

            if not dir_path.exists():
                return _error(f"directory not found: {path}")
            if not dir_path.is_dir():
                return _error(f"not a directory: {path}")

            entries = []
            for item in sorted(dir_path.iterdir()):
                icon = "📁" if item.is_dir() else "📄"
                entries.append(f"{icon} {item.name}")

            if not entries:
                return f"Directory is empty: {dir_path}"

            return "\n".join(entries)

        except PermissionError:
            return _error(f"permission denied: {path}")
        except Exception as e:
            return _error(f"failed to list directory: {e}")
  

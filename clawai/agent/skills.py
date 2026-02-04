"""
ClawAI Skills Loader
-------------------
Discovers and describes agent skills.

A skill is a markdown document (SKILL.md) that teaches the agent
how to perform a specific capability.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional

# Built-in skills directory (relative to ClawAI package)
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"


class SkillsLoader:
    """Filesystem-based skill discovery and metadata loader."""

    def __init__(
        self,
        workspace: Path,
        builtin_skills_dir: Optional[Path] = None,
    ):
        self.workspace = workspace
        self.workspace_dir = workspace / "skills"
        self.builtin_dir = builtin_skills_dir or BUILTIN_SKILLS_DIR

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def list_skills(self, only_available: bool = True) -> list[dict[str, str]]:
        """
        List all discovered skills.

        Workspace skills override built-in skills with the same name.
        """
        skills: dict[str, dict[str, str]] = {}

        self._scan_dir(self.workspace_dir, skills, source="workspace")
        self._scan_dir(self.builtin_dir, skills, source="builtin")

        results = list(skills.values())

        if only_available:
            return [
                s for s in results
                if self._requirements_met(self._get_skill_meta(s["name"]))
            ]

        return results

    def _scan_dir(
        self,
        base: Path,
        skills: dict[str, dict[str, str]],
        source: str,
    ) -> None:
        if not base.exists():
            return

        for d in base.iterdir():
            skill_file = d / "SKILL.md"
            if d.is_dir() and skill_file.exists() and d.name not in skills:
                skills[d.name] = {
                    "name": d.name,
                    "path": str(skill_file),
                    "source": source,
                }

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load_skill(self, name: str) -> Optional[str]:
        """Load full SKILL.md content."""
        for base in (self.workspace_dir, self.builtin_dir):
            path = base / name / "SKILL.md"
            if path.exists():
                return path.read_text(encoding="utf-8")
        return None

    def load_active_skills(self, names: list[str]) -> str:
        """Load full content of active skills for prompt context."""
        sections: list[str] = []

        for name in names:
            content = self.load_skill(name)
            if not content:
                continue

            body = self._strip_frontmatter(content)
            sections.append(f"## Skill: {name}\n\n{body}")

        return "\n\n---\n\n".join(sections)

    # ------------------------------------------------------------------ #
    # Index / Summary
    # ------------------------------------------------------------------ #

    def build_skills_index(self) -> str:
        """
        Build an XML-like index of all skills.

        Used for discovery and progressive loading.
        """
        skills = self.list_skills(only_available=False)
        if not skills:
            return ""

        lines = ["<skills>"]

        for skill in skills:
            name = self._escape(skill["name"])
            desc = self._escape(self._get_skill_description(skill["name"]))
            meta = self._get_skill_meta(skill["name"])
            available = self._requirements_met(meta)

            lines.append(f'  <skill available="{str(available).lower()}">')
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{skill['path']}</location>")

            if not available:
                missing = self._missing_requirements(meta)
                if missing:
                    lines.append(f"    <requires>{self._escape(missing)}</requires>")

            lines.append("  </skill>")

        lines.append("</skills>")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    def get_always_skills(self) -> list[str]:
        """Return skills marked as always=true and available."""
        result: list[str] = []

        for s in self.list_skills(only_available=True):
            meta = self._get_skill_meta(s["name"])
            if meta.get("always"):
                result.append(s["name"])

        return result

    def _get_skill_description(self, name: str) -> str:
        meta = self._get_frontmatter(name) or {}
        return meta.get("description", name)

    def _get_skill_meta(self, name: str) -> dict:
        meta = self._get_frontmatter(name) or {}
        return self._parse_metadata(meta.get("metadata", ""))

    # ------------------------------------------------------------------ #
    # Frontmatter / Requirements
    # ------------------------------------------------------------------ #

    def _get_frontmatter(self, name: str) -> Optional[dict]:
        content = self.load_skill(name)
        if not content or not content.startswith("---"):
            return None

        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return None

        data: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip().strip('"\'')
        return data

    def _parse_metadata(self, raw: str) -> dict:
        try:
            data = json.loads(raw)
            return data.get("clawai", {}) if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _requirements_met(self, meta: dict) -> bool:
        req = meta.get("requires", {})
        return not self._missing_requirements(req)

    def _missing_requirements(self, meta: dict) -> str:
        missing: list[str] = []

        for b in meta.get("bins", []):
            if not shutil.which(b):
                missing.append(f"bin:{b}")

        for env in meta.get("env", []):
            if not os.environ.get(env):
                missing.append(f"env:{env}")

        return ", ".join(missing)

    # ------------------------------------------------------------------ #
    # Utils
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end():].strip()
        return content

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )

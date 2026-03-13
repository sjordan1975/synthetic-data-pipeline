"""Utilities for loading external, versioned prompt templates.

Templates are stored under config/templates and referenced via a JSON registry.
This separates prompt content from Python logic, making templates easier to
version, review, and swap without code edits.
"""

from __future__ import annotations

import json
from pathlib import Path


class TemplateLoader:
    """Load prompt templates by template_id from the registry file."""

    def __init__(
        self,
        registry_path: Path | None = None,
        templates_dir: Path | None = None,
    ) -> None:
        self.registry_path = registry_path or Path("config/templates/template_registry.json")
        self.templates_dir = templates_dir or Path("config/templates")
        self._registry = self._load_registry()

    def _load_registry(self) -> dict[str, dict[str, str]]:
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Template registry not found: {self.registry_path}")

        with self.registry_path.open("r", encoding="utf-8") as handle:
            registry = json.load(handle)

        if not isinstance(registry, dict):
            raise ValueError("Template registry must be a JSON object keyed by template id")
        return registry

    def get_template(self, template_id: str) -> str:
        entry = self._registry.get(template_id)
        if entry is None:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"Unknown template_id '{template_id}'. Available: {available}")

        relative_path = entry.get("path")
        if not relative_path:
            raise ValueError(f"Template registry entry missing 'path': {template_id}")

        file_path = self.templates_dir / relative_path
        if not file_path.exists():
            raise FileNotFoundError(f"Template file not found for '{template_id}': {file_path}")

        return file_path.read_text(encoding="utf-8")

    def list_template_ids(self, *, kind: str, active_only: bool = True) -> list[str]:
        """List template IDs filtered by template kind and active flag."""
        ids: list[str] = []
        for template_id, entry in self._registry.items():
            if entry.get("kind") != kind:
                continue
            if active_only and not bool(entry.get("active", True)):
                continue
            ids.append(template_id)
        return sorted(ids)

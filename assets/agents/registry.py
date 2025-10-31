from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

from .base import BaseAgent

# Lazy imports in CLI, registry only holds names for help


def get_registry() -> Dict[str, str]:
    return {
        "unicode": "assets.agents.unicode_sanitizer:UnicodeSanitizerAgent",
        "verify": "assets.agents.verifier:VerifierAgent",
        "docs": "assets.agents.docs_integrity:DocsIntegrityAgent",
        "notebooks": "assets.agents.notebook_exec:NotebookExecutionAgent",
        "canonical": "assets.agents.canonical_state_monitor:CanonicalStateMonitor",
    }


def load_agent_class(spec: str) -> Type[BaseAgent]:
    module_name, class_name = spec.split(":", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def run_agent(name: str, root: Path):
    reg = get_registry()
    spec = reg.get(name)
    if not spec:
        raise SystemExit(f"Unknown agent: {name}. Available: {', '.join(reg)}")
    cls = load_agent_class(spec)
    agent = cls()
    return agent.run(root)



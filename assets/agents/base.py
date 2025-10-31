from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class AgentResult:
    name: str
    ok: bool
    issues: List[Dict[str, Any]]
    summary: str


class BaseAgent:
    name: str = "base"

    def run(self, root: Path) -> AgentResult:
        raise NotImplementedError



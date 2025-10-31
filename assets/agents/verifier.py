from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List

from .base import AgentResult, BaseAgent
from .utils.fs import discover_files
from .utils.markdown import parse_links, extract_status_tags


def _is_http(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


class VerifierAgent(BaseAgent):
    name = "verify"

    def run(self, root: Path) -> AgentResult:
        md_files = discover_files(root, ["README.md", "docs/**/*.md", "publications/**/*.md"])
        issues: List[Dict[str, Any]] = []

        # Check links
        for p in md_files:
            text = p.read_text(encoding="utf-8", errors="replace")
            for _, target in parse_links(text):
                if _is_http(target):
                    try:
                        cp = subprocess.run(["curl", "-I", "-sS", "-m", "5", target], capture_output=True, text=True)
                        ok = cp.returncode == 0 and (" 200 " in (cp.stdout or "") or " 3" in (cp.stdout or ""))
                        if not ok:
                            issues.append({"type": "link", "file": str(p), "link": target, "error": "unreachable"})
                    except Exception as e:
                        issues.append({"type": "link", "file": str(p), "link": target, "error": str(e)})
                elif target.startswith("mailto:"):
                    # Ignore mailto links
                    pass
                else:
                    q = (p.parent / target.split("#")[0]).resolve()
                    if not q.exists():
                        issues.append({"type": "link", "file": str(p), "link": target, "error": "missing_local_path"})

        # Check presence of status tags in major docs (exclude strategy/docs in publications/)
        for p in md_files:
            text = p.read_text(encoding="utf-8", errors="replace")
            tags = set(extract_status_tags(text))
            is_strategy_doc = p.name.endswith(("STRATEGY.md", "README.md", "GUIDE.md", "FAQ.md"))
            if p.match("publications/*.md") and not tags and not is_strategy_doc:
                issues.append({"type": "status", "file": str(p), "error": "no_status_tags_found"})

        ok = len([i for i in issues if i.get("type") in {"link", "status"}]) == 0
        summary = ("Verification OK." if ok else f"Verification issues: {len(issues)} findings.")
        return AgentResult(self.name, ok, issues, summary)



from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime
from pathlib import Path

from .registry import get_registry, run_agent


def _save_report(agent_name: str, result) -> None:
    reports_dir = Path("assets/agents/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname = socket.gethostname()
    out = reports_dir / f"{agent_name}_{ts}_{hostname}.json"
    latest = reports_dir / f"{agent_name}_latest.json"
    data = {
        "name": result.name,
        "ok": result.ok,
        "summary": result.summary,
        "issues": result.issues,
        "timestamp": ts,
        "hostname": hostname,
    }
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    latest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m assets.agents.cli", add_help=True)
    parser.add_argument("command", nargs="?", default="list")
    parser.add_argument("agent", nargs="?")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    if args.command == "list":
        reg = get_registry()
        print("Available agents:")
        for k in reg:
            print(f"- {k}")
        return

    if args.command in {"run", "verify", "unicode", "docs", "notebooks", "canonical"}:
        agent_name = args.agent if args.command == "run" else args.command
        result = run_agent(agent_name, Path(args.root))
        print(result.summary)
        _save_report(agent_name, result)
        if agent_name in {"verify", "docs"} and not result.ok:
            raise SystemExit(1)
        return

    raise SystemExit("Unknown command. Use 'list' or 'run <agent>'.")


if __name__ == "__main__":
    main()



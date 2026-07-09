#!/usr/bin/env python3
"""Check the paper-facing Level Q table against the compact JSON package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PACKAGE_PATH = "certificates/phase4_level_Q_coefficients.json"
PAPER_PATH = "paper/theorem_Q_certified.md"
OUT_PATH = "certificates/phase4_level_Q_table_check.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def add(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})


def compact_decimal(value: str) -> str:
    if "." not in value:
        return value
    value = value.rstrip("0").rstrip(".")
    return value if value else "0"


def compact_interval(interval: dict[str, Any]) -> str:
    return f"[{compact_decimal(interval['lower'])}, {compact_decimal(interval['upper'])}]"


def parse_table(markdown: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    in_table = False
    for line in markdown.splitlines():
        if line.strip() == "| Quantity | Exact value | Paper-facing interval |":
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("|---"):
            continue
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        quantity = cells[0].strip("`")
        rows[quantity] = {"exact": cells[1].strip("`"), "interval": cells[2]}
    return rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package = read_json(repo_root / PACKAGE_PATH)
    paper = (repo_root / PAPER_PATH).read_text()
    table = parse_table(paper)
    checks: list[dict[str, Any]] = []

    add(checks, "package_all_pass", package.get("all_pass") is True, "Level Q package passes")
    add(checks, "table_present", bool(table), "paper-facing table is present")

    expected_quantities = list(package.get("coefficients", {}).keys()) + ["R_threshold"]
    add(
        checks,
        "table_row_set_matches",
        set(table) == set(expected_quantities),
        "table rows match compact package coefficient list plus R_threshold",
    )

    for name, record in package.get("coefficients", {}).items():
        row = table.get(name)
        add(checks, f"{name}_row_present", row is not None, f"{name} row exists")
        if row is None:
            continue
        add(checks, f"{name}_exact_matches", row["exact"] == record["exact"], f"{name} exact value matches")
        add(
            checks,
            f"{name}_interval_matches",
            row["interval"].strip("`") == compact_interval(record["interval"]),
            f"{name} compact interval matches package interval",
        )

    r_row = table.get("R_threshold")
    r_pkg = package.get("R_threshold", {})
    add(checks, "R_threshold_row_present", r_row is not None, "R_threshold row exists")
    if r_row is not None:
        expected_interval = compact_interval(r_pkg["interval"])
        citation = r_pkg.get("citation", {})
        add(checks, "R_threshold_exact_label", r_row["exact"] == "cube-root bracket", "R_threshold exact label is descriptive")
        add(
            checks,
            "R_threshold_interval_mentions_bracket",
            expected_interval in r_row["interval"],
            "R_threshold table row includes machine bracket",
        )
        add(
            checks,
            "R_threshold_interval_mentions_human_round",
            citation.get("human_conservative_round", "") in r_row["interval"],
            "R_threshold table row includes human conservative round",
        )

    all_pass = all(check["pass"] for check in checks)
    payload = {
        "artifact": "phase4_level_Q_table_check",
        "generated_by": "scripts/phase4_level_Q_table_checker.py",
        "scope": "Paper-facing Level Q table consistency check",
        "checked_paper": PAPER_PATH,
        "checked_package": PACKAGE_PATH,
        "checks": checks,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check["pass"]),
            "checks_failed": sum(1 for check in checks if not check["pass"]),
        },
        "all_pass": all_pass,
    }

    out_path = repo_root / OUT_PATH
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    if not all_pass:
        raise SystemExit(f"checker failed; wrote {OUT_PATH}")
    print(f"wrote {out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()

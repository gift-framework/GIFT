#!/usr/bin/env python3
"""
Check synchronization between PDF files and their markdown sources.

Compares modification times and reports any PDFs that may be out of sync.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

# Define PDF to markdown mappings
PDF_MAPPINGS = {
    "publications/pdf/gift-main.pdf": "publications/gift_main.md",
    "publications/pdf/gift_extensions.pdf": "publications/gift_extensions.md",
    "publications/pdf/Supp_A.pdf": "publications/supplements/A_math_foundations.md",
    "publications/pdf/Supp_B.pdf": "publications/supplements/B_rigorous_proofs.md",
    "publications/pdf/Supp_C.pdf": "publications/supplements/C_complete_derivations.md",
    "publications/pdf/Supp_D.pdf": "publications/supplements/D_phenomenology.md",
    "publications/pdf/Supp_E.pdf": "publications/supplements/E_falsification.md",
    "publications/pdf/Supp_F.pdf": "publications/supplements/F_K7_metric.md",
}


def get_file_mtime(filepath: Path) -> float:
    """Get modification time of a file."""
    if filepath.exists():
        return filepath.stat().st_mtime
    return 0.0


def format_time(timestamp: float) -> str:
    """Format timestamp as human-readable string."""
    if timestamp == 0:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def check_pdf_sync(repo_root: Path) -> Dict[str, dict]:
    """Check synchronization between PDFs and markdown sources.

    Args:
        repo_root: Path to repository root

    Returns:
        Dictionary mapping PDF paths to sync status information
    """
    results = {}

    for pdf_rel, md_rel in PDF_MAPPINGS.items():
        pdf_path = repo_root / pdf_rel
        md_path = repo_root / md_rel

        pdf_exists = pdf_path.exists()
        md_exists = md_path.exists()

        pdf_mtime = get_file_mtime(pdf_path)
        md_mtime = get_file_mtime(md_path)

        # Determine sync status
        if not pdf_exists and not md_exists:
            status = "both_missing"
        elif not pdf_exists:
            status = "pdf_missing"
        elif not md_exists:
            status = "md_missing"
        elif md_mtime > pdf_mtime:
            status = "out_of_sync"
            time_diff = md_mtime - pdf_mtime
        else:
            status = "in_sync"
            time_diff = pdf_mtime - md_mtime

        results[pdf_rel] = {
            "pdf_path": str(pdf_path),
            "md_path": str(md_path),
            "pdf_exists": pdf_exists,
            "md_exists": md_exists,
            "pdf_mtime": pdf_mtime,
            "md_mtime": md_mtime,
            "pdf_time": format_time(pdf_mtime),
            "md_time": format_time(md_mtime),
            "status": status,
            "time_diff": time_diff if status in ["out_of_sync", "in_sync"] else 0,
        }

    return results


def print_report(results: Dict[str, dict]):
    """Print a formatted report of sync status."""
    print("\n" + "="*80)
    print("PDF Synchronization Report")
    print("="*80)

    # Count statuses
    in_sync = sum(1 for r in results.values() if r["status"] == "in_sync")
    out_of_sync = sum(1 for r in results.values() if r["status"] == "out_of_sync")
    pdf_missing = sum(1 for r in results.values() if r["status"] == "pdf_missing")
    md_missing = sum(1 for r in results.values() if r["status"] == "md_missing")

    print(f"\nSummary:")
    print(f"  ✓ In sync:        {in_sync}")
    print(f"  ⚠ Out of sync:    {out_of_sync}")
    print(f"  ✗ PDF missing:    {pdf_missing}")
    print(f"  ✗ MD missing:     {md_missing}")
    print()

    # Detailed report
    for pdf_rel, info in sorted(results.items()):
        status = info["status"]

        if status == "in_sync":
            marker = "✓"
            msg = f"In sync (PDF: {info['pdf_time']})"
        elif status == "out_of_sync":
            marker = "⚠"
            days_diff = info['time_diff'] / 86400  # Convert seconds to days
            msg = f"Out of sync! MD newer by {days_diff:.1f} days"
        elif status == "pdf_missing":
            marker = "✗"
            msg = f"PDF missing (MD exists: {info['md_time']})"
        elif status == "md_missing":
            marker = "✗"
            msg = f"MD missing (PDF exists: {info['pdf_time']})"
        else:
            marker = "?"
            msg = "Both missing"

        print(f"{marker} {pdf_rel}")
        print(f"   {msg}")
        if info["md_exists"]:
            print(f"   Source: {info['md_time']}")
        print()

    # Recommendations
    print("="*80)
    print("Recommendations:")
    print("="*80)

    if out_of_sync > 0:
        print("\n⚠ Some PDFs are out of sync with their markdown sources.")
        print("  Action: Regenerate PDFs from markdown using:")
        print("    - Pandoc: pandoc file.md -o file.pdf")
        print("    - Or your preferred markdown-to-PDF tool")

        print("\n  Out of sync files:")
        for pdf_rel, info in results.items():
            if info["status"] == "out_of_sync":
                print(f"    - {pdf_rel} (update from {info['md_path']})")

    if pdf_missing > 0:
        print("\n✗ Some PDFs are missing.")
        print("  Action: Generate PDFs from markdown:")
        for pdf_rel, info in results.items():
            if info["status"] == "pdf_missing":
                print(f"    - Generate {pdf_rel} from {info['md_path']}")

    if md_missing > 0:
        print("\n✗ Some markdown sources are missing.")
        print("  Action: Investigate why PDFs exist without sources")

    if out_of_sync == 0 and pdf_missing == 0 and md_missing == 0:
        print("\n✓ All PDFs are in sync with their sources!")

    print()


def create_sync_status_file(repo_root: Path, results: Dict[str, dict]):
    """Create a PDF_SYNC_STATUS.md file documenting sync status."""
    output_path = repo_root / "publications" / "pdf" / "PDF_SYNC_STATUS.md"

    with output_path.open('w') as f:
        f.write("# PDF Synchronization Status\n\n")
        f.write(f"**Last checked**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Status Summary\n\n")
        f.write("| PDF | Status | PDF Date | Markdown Date |\n")
        f.write("|-----|--------|----------|---------------|\n")

        for pdf_rel, info in sorted(results.items()):
            pdf_name = Path(pdf_rel).name
            status = info["status"]

            if status == "in_sync":
                status_icon = "✓ In sync"
            elif status == "out_of_sync":
                status_icon = "⚠ Out of sync"
            elif status == "pdf_missing":
                status_icon = "✗ Missing"
            else:
                status_icon = "? Unknown"

            pdf_date = info["pdf_time"] if info["pdf_exists"] else "N/A"
            md_date = info["md_time"] if info["md_exists"] else "N/A"

            f.write(f"| {pdf_name} | {status_icon} | {pdf_date} | {md_date} |\n")

        f.write("\n## How to Regenerate PDFs\n\n")
        f.write("If PDFs are out of sync, regenerate using:\n\n")
        f.write("```bash\n")
        f.write("# Using Pandoc\n")
        f.write("pandoc publications/gift_main.md -o publications/pdf/gift-main.pdf\n")
        f.write("pandoc publications/supplements/A_math_foundations.md -o publications/pdf/Supp_A.pdf\n")
        f.write("# ... repeat for other files\n")
        f.write("```\n\n")

        f.write("Or use batch script:\n\n")
        f.write("```bash\n")
        f.write("cd publications\n")
        f.write("for md in *.md; do\n")
        f.write('    pandoc "$md" -o "pdf/${md%.md}.pdf"\n')
        f.write("done\n")
        f.write("```\n")

    print(f"✓ Created sync status file: {output_path}")


def main():
    """Main entry point."""
    # Find repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    # Check sync
    results = check_pdf_sync(repo_root)

    # Print report
    print_report(results)

    # Create status file
    create_sync_status_file(repo_root, results)


if __name__ == "__main__":
    main()

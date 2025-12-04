#!/usr/bin/env python3
"""
GIFT Framework - Quick Start Script
Launches interactive assets and tools
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("GIFT Framework - Quick Start")
    print("=" * 40)
    print()
    print("Available options:")
    print("1. Visualizations (interactive notebooks)")
    print("2. Documentation (web interface)")
    print("3. Agents (verification & maintenance)")
    print("4. Exit")
    print()

    while True:
        choice = input("Select option (1-4): ").strip()

        if choice == "1":
            launch_visualizations()
            break
        elif choice == "2":
            launch_documentation()
            break
        elif choice == "3":
            launch_agents()
            break
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-4.")

def launch_visualizations():
    """Launch Jupyter notebooks for visualizations"""
    print("\nGIFT Visualizations")
    print("-" * 40)
    print("Interactive visualizations are available via Binder:")
    print("  https://mybinder.org/v2/gh/gift-framework/GIFT/main")
    print()
    print("For local notebooks, see G2_ML/ directory.")
    print()

    # Check for G2_ML notebooks
    g2ml_dir = Path("G2_ML")
    if g2ml_dir.exists():
        notebooks = list(g2ml_dir.rglob("*.ipynb"))
        if notebooks:
            print(f"Found {len(notebooks)} notebook(s) in G2_ML/:")
            for nb in notebooks[:5]:
                print(f"  - {nb}")
            if len(notebooks) > 5:
                print(f"  ... and {len(notebooks) - 5} more")

def launch_documentation():
    """Launch documentation web interface"""
    print("\nLaunching GIFT Documentation...")
    docs_file = Path("docs/index.html")
    
    if not docs_file.exists():
        print("Error: Documentation not found")
        return
    
    try:
        import webbrowser
        webbrowser.open(f"file://{docs_file.absolute()}")
        print("Documentation opened in browser")
    except Exception as e:
        print(f"Error opening documentation: {e}")
        print(f"Manual: Open {docs_file.absolute()} in your browser")

def launch_agents():
    """Run maintenance agents"""
    print("\nGIFT Agents:")
    print("1. Verification")
    print("2. Unicode sanitizer (scan)")
    print("3. Docs integrity")
    print("4. Notebook discovery")
    print("5. Canonical monitor")
    choice = input("Select (1-5): ").strip()
    if choice == "1":
        subprocess.run([sys.executable, "-m", "assets.agents.cli", "verify"]) 
    elif choice == "2":
        subprocess.run([sys.executable, "-m", "assets.agents.cli", "unicode"]) 
    elif choice == "3":
        subprocess.run([sys.executable, "-m", "assets.agents.cli", "docs"]) 
    elif choice == "4":
        subprocess.run([sys.executable, "-m", "assets.agents.cli", "notebooks"]) 
    elif choice == "5":
        subprocess.run([sys.executable, "-m", "assets.agents.cli", "canonical"]) 

if __name__ == "__main__":
    main()

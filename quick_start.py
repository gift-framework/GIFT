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
    print("1. Twitter Bot (automated posting)")
    print("2. Visualizations (interactive notebooks)")
    print("3. Documentation (web interface)")
    print("4. Exit")
    print()
    
    while True:
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            launch_twitter_bot()
            break
        elif choice == "2":
            launch_visualizations()
            break
        elif choice == "3":
            launch_documentation()
            break
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-4.")

def launch_twitter_bot():
    """Launch the Twitter bot"""
    print("\nLaunching GIFT Twitter Bot...")
    bot_dir = Path("assets/twitter_bot")
    
    if not bot_dir.exists():
        print("Error: Twitter bot directory not found")
        return
    
    try:
        # Change to bot directory and run
        os.chdir(bot_dir)
        
        print("Options:")
        print("1. Test content generation")
        print("2. Run bot once")
        print("3. Start automated scheduler")
        print("4. Back to main menu")
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            subprocess.run([sys.executable, "content_generator_en.py"])
        elif choice == "2":
            subprocess.run([sys.executable, "twitter_bot_v2.py"])
        elif choice == "3":
            print("Starting scheduler... (Press Ctrl+C to stop)")
            subprocess.run([sys.executable, "scheduler.py"])
        elif choice == "4":
            return
        else:
            print("Invalid choice")
            
    except Exception as e:
        print(f"Error launching Twitter bot: {e}")
    finally:
        os.chdir("../..")

def launch_visualizations():
    """Launch Jupyter notebooks for visualizations"""
    print("\nLaunching GIFT Visualizations...")
    viz_dir = Path("assets/visualizations")
    
    if not viz_dir.exists():
        print("Error: Visualizations directory not found")
        return
    
    try:
        # Check if Jupyter is installed
        result = subprocess.run([sys.executable, "-c", "import jupyter"], 
                              capture_output=True)
        if result.returncode != 0:
            print("Jupyter not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "jupyter"])
        
        # Launch Jupyter
        os.chdir(viz_dir)
        print("Starting Jupyter notebook server...")
        print("Available notebooks:")
        print("- e8_root_system_3d.ipynb")
        print("- precision_dashboard.ipynb") 
        print("- dimensional_reduction_flow.ipynb")
        print("\nOpening in browser...")
        subprocess.run([sys.executable, "-m", "jupyter", "notebook"])
        
    except Exception as e:
        print(f"Error launching visualizations: {e}")
        print("\nAlternative: Use online platforms:")
        print("- Binder: https://mybinder.org/v2/gh/gift-framework/GIFT/main?filepath=assets/visualizations/")
        print("- Colab: https://colab.research.google.com/github/gift-framework/GIFT/blob/main/assets/visualizations/")
    finally:
        os.chdir("../..")

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

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test du générateur de contenu GIFT (sans Twitter)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from content_generator import GIFTContentGenerator

def main():
    print("=== Test du generateur de contenu GIFT ===\n")
    
    generator = GIFTContentGenerator()
    
    print("Contenu quotidien:")
    print("-" * 50)
    content = generator.generate_daily_content()
    print(content)
    print(f"\nLongueur: {len(content)} caracteres")
    print("\n" + "="*70 + "\n")
    
    print("Resume hebdomadaire:")
    print("-" * 50)
    weekly = generator.generate_weekly_summary()
    print(weekly)
    print(f"\nLongueur: {len(weekly)} caracteres")
    print("\n" + "="*70 + "\n")
    
    print("Highlight mensuel:")
    print("-" * 50)
    monthly = generator.generate_monthly_highlight()
    print(monthly)
    print(f"\nLongueur: {len(monthly)} caracteres")

if __name__ == "__main__":
    main()

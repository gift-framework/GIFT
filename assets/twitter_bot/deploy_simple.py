#!/usr/bin/env python3
"""
GIFT Twitter Bot - Script de deploiement (Windows compatible)
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("GIFT Twitter Bot - Script de deploiement")
    print("=" * 50)
    
    # Verifier Python
    python_version = sys.version_info
    if python_version < (3, 11):
        print("ERREUR: Python 3.11+ requis")
        return False
    print(f"OK: Python {python_version.major}.{python_version.minor}")
    
    # Verifier les dependances
    try:
        import tweepy
        import schedule
        print("OK: Dependances installees")
    except ImportError as e:
        print(f"ERREUR: Dependance manquante: {e}")
        print("Installez avec: pip install -r requirements.txt")
        return False
    
    # Verifier la configuration
    config_file = Path("config.py")
    if not config_file.exists():
        print("ERREUR: Fichier config.py manquant")
        print("Copiez config_template.py vers config.py et configurez vos cles API")
        return False
    print("OK: Configuration trouvee")
    
    # Tester le generateur de contenu
    print("\nTest du generateur de contenu...")
    try:
        result = subprocess.run([sys.executable, "content_generator_windows.py"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("OK: Generateur de contenu teste avec succes")
        else:
            print(f"ERREUR lors du test: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"ERREUR lors du test: {e}")
        return False
    
    print("\nTous les tests sont passes!")
    
    # Creer les scripts de deploiement
    print("\nCreation des scripts de deploiement...")
    
    # Script Windows
    windows_script = """@echo off
echo Demarrage du GIFT Twitter Bot...

python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python non trouve
    pause
    exit /b 1
)

python -c "import tweepy, schedule" >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Dependances manquantes
    echo Installez avec: pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "config.py" (
    echo ERREUR: Fichier config.py manquant
    pause
    exit /b 1
)

echo OK: Demarrage du scheduler...
python scheduler.py
pause
"""
    
    Path("start_bot.bat").write_text(windows_script)
    print("OK: Script de demarrage Windows cree: start_bot.bat")
    
    print("\nDeploiement pret!")
    print("\nOptions de deploiement:")
    print("1. Local: python scheduler.py")
    print("2. Script: start_bot.bat")
    print("3. Voir README.md pour plus d'options")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
GIFT Twitter Bot - Script de déploiement
Script pour déployer le bot sur différents environnements
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_requirements():
    """Vérifie que tous les prérequis sont installés"""
    print("Vérification des prérequis...")
    
    # Vérifier Python
    python_version = sys.version_info
    if python_version < (3, 11):
        print("ERREUR: Python 3.11+ requis")
        return False
    print(f"OK: Python {python_version.major}.{python_version.minor}")
    
    # Vérifier les dépendances
    try:
        import tweepy
        import schedule
        print("OK: Dependances installees")
    except ImportError as e:
        print(f"ERREUR: Dependance manquante: {e}")
        print("Installez avec: pip install -r requirements.txt")
        return False
    
    # Vérifier la configuration
    config_file = Path("config.py")
    if not config_file.exists():
        print("ERREUR: Fichier config.py manquant")
        print("Copiez config_template.py vers config.py et configurez vos cles API")
        return False
    print("OK: Configuration trouvee")
    
    return True

def test_bot():
    """Teste le bot en mode dry run"""
    print("\nTest du bot...")
    
    # Modifier temporairement DRY_RUN
    config_content = Path("config.py").read_text()
    if "DRY_RUN = False" in config_content:
        config_content = config_content.replace("DRY_RUN = False", "DRY_RUN = True")
        Path("config.py").write_text(config_content)
        print("Mode dry run activé pour le test")
    
    try:
        result = subprocess.run([sys.executable, "twitter_bot.py"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Bot testé avec succès")
            return True
        else:
            print(f"❌ Erreur lors du test: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Test interrompu (timeout)")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False
    finally:
        # Restaurer DRY_RUN
        if "DRY_RUN = True" in config_content:
            config_content = config_content.replace("DRY_RUN = True", "DRY_RUN = False")
            Path("config.py").write_text(config_content)

def test_content_generator():
    """Teste le générateur de contenu"""
    print("\nTest du générateur de contenu...")
    
    try:
        result = subprocess.run([sys.executable, "content_generator_windows.py"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Générateur de contenu testé avec succès")
            return True
        else:
            print(f"❌ Erreur lors du test: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

def create_deployment_script():
    """Crée un script de déploiement pour différents environnements"""
    
    # Script pour Linux/Mac
    linux_script = """#!/bin/bash
# GIFT Twitter Bot - Script de démarrage Linux/Mac

echo "Démarrage du GIFT Twitter Bot..."

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trouvé"
    exit 1
fi

# Vérifier les dépendances
if ! python3 -c "import tweepy, schedule" &> /dev/null; then
    echo "❌ Dépendances manquantes"
    echo "Installez avec: pip3 install -r requirements.txt"
    exit 1
fi

# Vérifier la configuration
if [ ! -f "config.py" ]; then
    echo "❌ Fichier config.py manquant"
    exit 1
fi

# Démarrer le scheduler
echo "✅ Démarrage du scheduler..."
python3 scheduler.py
"""
    
    Path("start_bot.sh").write_text(linux_script)
    os.chmod("start_bot.sh", 0o755)
    print("✅ Script de démarrage Linux/Mac créé: start_bot.sh")
    
    # Script pour Windows
    windows_script = """@echo off
REM GIFT Twitter Bot - Script de démarrage Windows

echo Démarrage du GIFT Twitter Bot...

REM Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python non trouvé
    pause
    exit /b 1
)

REM Vérifier les dépendances
python -c "import tweepy, schedule" >nul 2>&1
if errorlevel 1 (
    echo ❌ Dépendances manquantes
    echo Installez avec: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Vérifier la configuration
if not exist "config.py" (
    echo ❌ Fichier config.py manquant
    pause
    exit /b 1
)

REM Démarrer le scheduler
echo ✅ Démarrage du scheduler...
python scheduler.py
pause
"""
    
    Path("start_bot.bat").write_text(windows_script)
    print("✅ Script de démarrage Windows créé: start_bot.bat")

def create_dockerfile():
    """Crée un Dockerfile pour le déploiement"""
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \\
    cron \\
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers
COPY twitter_bot/ .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Créer un utilisateur non-root
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# Script de démarrage
CMD ["python", "scheduler.py"]
"""
    
    Path("Dockerfile").write_text(dockerfile_content)
    print("✅ Dockerfile créé")

def create_docker_compose():
    """Crée un docker-compose.yml"""
    compose_content = """version: '3.8'

services:
  gift-twitter-bot:
    build: .
    container_name: gift-twitter-bot
    restart: unless-stopped
    environment:
      - TWITTER_API_KEY=${TWITTER_API_KEY}
      - TWITTER_API_SECRET=${TWITTER_API_SECRET}
      - TWITTER_ACCESS_TOKEN=${TWITTER_ACCESS_TOKEN}
      - TWITTER_ACCESS_TOKEN_SECRET=${TWITTER_ACCESS_TOKEN_SECRET}
      - TWITTER_BEARER_TOKEN=${TWITTER_BEARER_TOKEN}
      - TWITTER_CLIENT_ID=${TWITTER_CLIENT_ID}
      - TWITTER_CLIENT_SECRET=${TWITTER_CLIENT_SECRET}
    volumes:
      - ./logs:/app/logs
    networks:
      - gift-network

networks:
  gift-network:
    driver: bridge
"""
    
    Path("docker-compose.yml").write_text(compose_content)
    print("✅ docker-compose.yml créé")

def main():
    """Fonction principale de déploiement"""
    print("GIFT Twitter Bot - Script de deploiement")
    print("=" * 50)
    
    # Vérifier les prérequis
    if not check_requirements():
        print("\n❌ Prérequis non satisfaits")
        return False
    
    # Tester le générateur de contenu
    if not test_content_generator():
        print("\n❌ Test du générateur de contenu échoué")
        return False
    
    # Tester le bot
    if not test_bot():
        print("\n❌ Test du bot échoué")
        return False
    
    print("\n✅ Tous les tests sont passés!")
    
    # Créer les scripts de déploiement
    print("\nCréation des scripts de déploiement...")
    create_deployment_script()
    create_dockerfile()
    create_docker_compose()
    
    print("\n🎉 Déploiement prêt!")
    print("\nOptions de déploiement:")
    print("1. Local: python scheduler.py")
    print("2. Script: ./start_bot.sh (Linux/Mac) ou start_bot.bat (Windows)")
    print("3. Docker: docker-compose up -d")
    print("4. GitHub Actions: Voir README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

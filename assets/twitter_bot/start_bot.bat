@echo off
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

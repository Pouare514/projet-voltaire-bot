@echo off
echo Demarrage du Projet Voltaire Bot avec surveillance...
echo.

REM Verifier que Python est installe
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe ou non trouve dans le PATH
    pause
    exit /b 1
)

REM Aller dans le repertoire du script
cd /d "%~dp0"

REM Installer les dependances si necessaire
echo Verification des dependances...
python -m pip install -r requirements.txt --quiet

echo.
echo Demarrage du serveur avec monitoring...
echo Pour arreter le serveur, fermez cette fenetre ou appuyez sur Ctrl+C
echo.

REM Demarrer le serveur avec le script de monitoring
python server_monitor.py

echo.
echo Le serveur s'est arrete.
pause

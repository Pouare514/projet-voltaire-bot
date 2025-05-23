#!/usr/bin/env python
"""
Script de démarrage robuste pour le serveur Projet Voltaire Bot.
Ce script redémarrera automatiquement le serveur en cas de plantage.
"""
import subprocess
import time
import sys
import os
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server_restart.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("server_restarter")

def install_dependencies():
    """Installe les dépendances requises"""
    try:
        logger.info("Installation des dépendances...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dépendances installées avec succès")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Erreur lors de l'installation des dépendances: {e}")
        return False

def start_server():
    """Démarre le serveur Flask"""
    logger.info("Démarrage du serveur...")
    server_env = os.environ.copy()
    # Ajoute l'environnement PYTHONUNBUFFERED pour s'assurer que les logs sont écrits immédiatement
    server_env["PYTHONUNBUFFERED"] = "1"
    
    # Démarre le serveur en utilisant le module principal
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=server_env
    )
    
    return process

def monitor_server(max_restarts=5, cooldown_period=300):
    """
    Surveille le serveur et le redémarre en cas de plantage
    
    Args:
        max_restarts: Nombre maximum de redémarrages en période de cooldown
        cooldown_period: Période en secondes pour limiter les redémarrages
    """
    restarts = []
    
    while True:
        # Vérifie si on n'a pas dépassé le nombre max de redémarrages
        current_time = time.time()
        # Ne garde que les redémarrages récents dans la période de cooldown
        restarts = [t for t in restarts if current_time - t < cooldown_period]
        
        if len(restarts) >= max_restarts:
            logger.error(f"Trop de redémarrages ({max_restarts}) en {cooldown_period} secondes. Arrêt du monitoring.")
            print(f"ERREUR: Le serveur a été redémarré {max_restarts} fois en moins de {cooldown_period} secondes.")
            print("Vérifiez les logs pour plus de détails et résolvez les problèmes avant de redémarrer.")
            break
        
        # Installe les dépendances si nécessaire
        if not install_dependencies():
            logger.error("Impossible d'installer les dépendances. Nouvelle tentative dans 30 secondes.")
            time.sleep(30)
            continue
            
        # Démarre le serveur
        process = start_server()
        
        # Lit et affiche les logs en temps réel
        for line in process.stdout:
            print(line, end='')
            
        # Si on arrive ici, c'est que le processus s'est arrêté
        exit_code = process.poll()
        logger.warning(f"Le serveur s'est arrêté avec le code {exit_code}. Redémarrage...")
        print(f"Le serveur s'est arrêté avec le code {exit_code}. Redémarrage...")
        
        # Ajoute un timestamp de redémarrage
        restarts.append(time.time())
        
        # Attente avant redémarrage
        time.sleep(5)

if __name__ == "__main__":
    print("===============================================")
    print("   Projet Voltaire Bot - Serveur Automatique   ")
    print("===============================================")
    print(f"Démarrage du serveur: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Appuyez sur Ctrl+C pour arrêter le serveur")
    print("-----------------------------------------------")
    
    try:
        monitor_server()
    except KeyboardInterrupt:
        print("\nArrêt du serveur demandé par l'utilisateur.")
        logger.info("Arrêt du serveur demandé par l'utilisateur.")
    except Exception as e:
        logger.critical(f"Erreur critique: {e}")
        print(f"\nERREUR CRITIQUE: {e}")
        input("Appuyez sur Entrée pour quitter...")

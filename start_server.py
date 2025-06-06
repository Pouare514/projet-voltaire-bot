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
import signal
import platform
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

# Configuration globale
CONFIG = {
    "MAX_RESTARTS_SHORT_PERIOD": int(os.getenv("MAX_RESTARTS_SHORT_PERIOD", "5")),  # Nombre max de redémarrages dans la période courte
    "SHORT_PERIOD": int(os.getenv("SHORT_PERIOD", "300")),  # Période courte en secondes (5 minutes)
    "RESTART_DELAY": int(os.getenv("RESTART_DELAY", "5")),  # Délai entre les redémarrages en secondes
    "INSTALL_DEPS_RETRY_DELAY": int(os.getenv("INSTALL_DEPS_RETRY_DELAY", "30")),  # Délai avant nouvelle tentative d'installation des dépendances
    "PRE_START_PAUSE": int(os.getenv("PRE_START_PAUSE", "2")),  # Pause avant de démarrer le serveur (pour s'assurer que les ports sont libérés)
}

def is_port_in_use(port):
    """
    Vérifie si un port est déjà utilisé
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """
    Tue le processus utilisant le port spécifié
    """
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
            if output:
                # La dernière colonne est le PID
                pid = output.strip().split()[-1]
                try:
                    subprocess.check_output(f"taskkill /F /PID {pid}", shell=True)
                    logger.info(f"Processus {pid} utilisant le port {port} a été tué")
                    return True
                except:
                    logger.error(f"Impossible de tuer le processus {pid}")
        else:
            output = subprocess.check_output(f"lsof -i :{port} | grep LISTEN", shell=True).decode()
            if output:
                # La deuxième colonne est le PID
                pid = output.strip().split()[1]
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    logger.info(f"Processus {pid} utilisant le port {port} a été tué")
                    return True
                except:
                    logger.error(f"Impossible de tuer le processus {pid}")
        return False
    except:
        logger.warning(f"Aucun processus trouvé sur le port {port}")
        return False

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
    
    # Vérifier si le port 5000 est déjà utilisé
    if is_port_in_use(5000):
        logger.warning("Port 5000 déjà utilisé. Tentative de libération...")
        if kill_process_on_port(5000):
            # Attendre un peu pour s'assurer que le port est libéré
            time.sleep(CONFIG["PRE_START_PAUSE"])
        else:
            logger.error("Impossible de libérer le port 5000. Le serveur pourrait ne pas démarrer correctement.")
    
    server_env = os.environ.copy()
    # Ajoute l'environnement PYTHONUNBUFFERED pour s'assurer que les logs sont écrits immédiatement
    server_env["PYTHONUNBUFFERED"] = "1"
    
    # Configuration pour une meilleure gestion des erreurs et récupération automatique
    server_env["ENABLE_FALLBACK"] = "true"
    server_env["CIRCUIT_FAILURE_THRESHOLD"] = "8"  # Plus tolérant
    server_env["CIRCUIT_RECOVERY_TIMEOUT"] = "25"  # Récupère plus rapidement
    
    # Démarre le serveur en utilisant le module principal
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=server_env
    )
    
    return process

def monitor_server(max_restarts=CONFIG["MAX_RESTARTS_SHORT_PERIOD"], cooldown_period=CONFIG["SHORT_PERIOD"]):
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
            logger.error(f"Trop de redémarrages ({max_restarts}) en {cooldown_period} secondes. Attente de 1 minute avant nouvelle tentative.")
            print(f"ERREUR: Le serveur a été redémarré {max_restarts} fois en moins de {cooldown_period} secondes.")
            print("Attente de 1 minute avant nouvelle tentative...")
            time.sleep(60)  # Attendre 1 minute avant de réessayer
            restarts = []  # Réinitialiser le compteur après l'attente
            continue
        
        # Installe les dépendances si nécessaire
        if not install_dependencies():
            logger.error(f"Impossible d'installer les dépendances. Nouvelle tentative dans {CONFIG['INSTALL_DEPS_RETRY_DELAY']} secondes.")
            time.sleep(CONFIG["INSTALL_DEPS_RETRY_DELAY"])
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
        time.sleep(CONFIG["RESTART_DELAY"])

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

# Script pour redémarrer le serveur en cas d'erreur
# Ce script surveille le processus principal et le redémarre s'il se termine avec une erreur

import os
import sys
import time
import datetime
import subprocess
import signal
import psutil
import logging

# Configuration de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server_restart.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Chemin vers le script principal
MAIN_SCRIPT = "main.py"
MAX_RESTARTS = 10  # Nombre maximal de redémarrages en cas d'échecs répétés
RESTART_COOLDOWN = 5  # Secondes d'attente entre les redémarrages
HEALTH_CHECK_INTERVAL = 30  # Vérifier l'état du serveur toutes les 30 secondes

# Variable pour suivre le processus en cours
current_process = None
restart_count = 0

def get_timestamp():
    """Retourne un timestamp formaté pour les logs."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def signal_handler(sig, frame):
    """Gestionnaire de signaux pour arrêter proprement le serveur."""
    logger.info(f"Signal reçu: {sig}. Arrêt du serveur...")
    if current_process and current_process.poll() is None:
        logger.info("Arrêt du processus fils...")
        try:
            # Envoyer un signal SIGTERM au processus pour arrêt propre
            os.kill(current_process.pid, signal.SIGTERM)
            # Attendre que le processus se termine
            current_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Le processus ne s'est pas terminé proprement, forçage...")
            current_process.kill()
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt du processus: {e}")
    sys.exit(0)

def is_process_healthy(pid):
    """Vérifie si le processus est en bon état."""
    try:
        process = psutil.Process(pid)
        
        # Vérifier l'utilisation CPU (si trop élevée pendant trop longtemps, problème potentiel)
        cpu_percent = process.cpu_percent(interval=1)
        if cpu_percent > 95:  # Si CPU > 95%, considérer comme potentiellement problématique
            logger.warning(f"Utilisation CPU élevée: {cpu_percent}%")
            # On pourrait ajouter un compteur ici pour redémarrer après X périodes de haute utilisation
        
        # Vérifier l'utilisation mémoire (si trop élevée, possible fuite mémoire)
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        if memory_percent > 90:  # Si mémoire > 90%, considérer comme problématique
            logger.warning(f"Utilisation mémoire élevée: {memory_percent}%")
            return False
        
        # Vérifier si le processus répond (pas de freeze)
        if process.status() == psutil.STATUS_ZOMBIE:
            logger.warning(f"Processus en état zombie")
            return False
        
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        logger.warning(f"Processus {pid} non disponible ou inaccessible")
        return False
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de l'état du processus: {e}")
        return False

def start_server():
    """Démarre le serveur et retourne le processus."""
    global restart_count, current_process
    
    restart_count += 1
    logger.info(f"Démarrage du serveur (tentative {restart_count}/{MAX_RESTARTS})...")
    
    try:
        # Lancer le script principal en tant que sous-processus
        process = subprocess.Popen([sys.executable, MAIN_SCRIPT],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)
        
        logger.info(f"Serveur démarré avec PID: {process.pid}")
        current_process = process
        return process
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du serveur: {e}")
        return None

def monitor_server():
    """Surveille le serveur et le redémarre si nécessaire."""
    global restart_count, current_process
    
    # Installer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Démarrage du moniteur de serveur...")
    
    # Démarrer le serveur pour la première fois
    process = start_server()
    if not process:
        logger.error("Impossible de démarrer le serveur initialement")
        return
    
    # Boucle principale de surveillance
    while restart_count <= MAX_RESTARTS:
        # Attendre un peu pour ne pas surcharger le CPU
        time.sleep(HEALTH_CHECK_INTERVAL)
        
        # Vérifier si le processus est toujours en cours d'exécution
        if process.poll() is not None:
            exit_code = process.returncode
            stdout, stderr = process.communicate()
            
            logger.warning(f"Serveur terminé avec code: {exit_code}")
            if stdout:
                logger.info(f"Sortie standard: {stdout[-1000:]}")  # Limiter la taille du log
            if stderr:
                logger.error(f"Erreur standard: {stderr[-1000:]}")  # Limiter la taille du log
            
            # Attendre un peu avant de redémarrer
            time.sleep(RESTART_COOLDOWN)
            
            # Redémarrer le serveur
            process = start_server()
            if not process:
                logger.error("Échec du redémarrage du serveur")
                break
        else:
            # Le processus est en cours d'exécution, vérifier sa santé
            if not is_process_healthy(process.pid):
                logger.warning("Le serveur ne semble pas en bonne santé, redémarrage...")
                
                # Tenter d'arrêter proprement
                try:
                    os.kill(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except:
                    # En cas d'échec, forcer l'arrêt
                    process.kill()
                
                # Attendre un peu avant de redémarrer
                time.sleep(RESTART_COOLDOWN)
                
                # Redémarrer le serveur
                process = start_server()
                if not process:
                    logger.error("Échec du redémarrage du serveur après problème de santé")
                    break
    
    # Si on a atteint le nombre maximal de redémarrages
    if restart_count > MAX_RESTARTS:
        logger.error(f"Nombre maximal de redémarrages ({MAX_RESTARTS}) atteint. Abandon.")

if __name__ == "__main__":
    try:
        monitor_server()
    except Exception as e:
        logger.critical(f"Erreur critique dans le moniteur: {e}")
        if current_process and current_process.poll() is None:
            current_process.kill()

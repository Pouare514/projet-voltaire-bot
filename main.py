from flask import Flask, redirect, request, Response
from g4f.client import Client
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from datetime import datetime
import logging
import json
import os
import sys
import time
import threading
import traceback
import tempfile
import shutil
import atexit
import asyncio
import psutil
import speech_recognition as sr
import requests
from pydub import AudioSegment
from g4f.cookies import set_cookies_dir, read_cookie_files
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
from waitress import serve
from functools import wraps
from circuitbreaker import circuit, CircuitBreakerError
from prometheus_client import Counter, Histogram, start_http_server
from pyrate_limiter import Duration, RequestRate, Limiter

# Configuration du logging avec encodage UTF-8 pour gérer les caractères spéciaux
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration globale
CONFIG = {
    "API_TIMEOUT": int(os.getenv("API_TIMEOUT", "60")),  # augmenté à 60 secondes pour éviter les timeouts
    "API_MAX_RETRIES": int(os.getenv("API_MAX_RETRIES", "5")),  # augmenté à 5 tentatives
    "API_RETRY_DELAY": int(os.getenv("API_RETRY_DELAY", "2")),  # secondes
    "CIRCUIT_FAILURE_THRESHOLD": int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "8")),  # augmenté à 8 échecs
    "CIRCUIT_RECOVERY_TIMEOUT": int(os.getenv("CIRCUIT_RECOVERY_TIMEOUT", "15")),  # réduit à 15 secondes pour réessayer plus rapidement
    "RATE_LIMIT_REQUESTS": int(os.getenv("RATE_LIMIT_REQUESTS", "30")),
    "RATE_LIMIT_PERIOD": int(os.getenv("RATE_LIMIT_PERIOD", "60")),  # secondes
    "MAX_AUDIO_SIZE": int(os.getenv("MAX_AUDIO_SIZE", "10485760")),  # 10MB
    "METRICS_PORT": int(os.getenv("METRICS_PORT", "9090")),
    "ENABLE_METRICS": os.getenv("ENABLE_METRICS", "false").lower() in ("true", "1", "yes"),
    "ENABLE_FALLBACK": os.getenv("ENABLE_FALLBACK", "true").lower() in ("true", "1", "yes"),
    "PROVIDER_ROTATION_ENABLED": True,  # Active la rotation des providers
    "RATE_LIMIT_COOLDOWN": 120,  # Temps d'attente après rate limit en secondes
}

# Liste des providers avec informations de rate limiting
PROVIDER_STATUS = {
    "Blackbox": {"blocked_until": 0, "failures": 0},
    "OpenaiChat": {"blocked_until": 0, "failures": 0},
    "DeepInfra": {"blocked_until": 0, "failures": 0},
    "FreeGpt": {"blocked_until": 0, "failures": 0},
    "default": {"blocked_until": 0, "failures": 0}
}

# Messages d'erreur de rate limiting à détecter
RATE_LIMIT_MESSAGES = [
    "该IP请求过于频繁",  # Ce IP fait trop de requêtes
    "该ip请求过多已被暂时限流",  # Cette IP fait trop de requêtes et a été temporairement limitée
    "请求频繁",  # Requêtes fréquentes
    "rate limit",
    "too many requests",
    "请稍后再试",  # Veuillez réessayer plus tard
    "访问受限",  # Accès limité
    "hourly limit",
    "quota exceeded",
    "过两分钟再试试吧",  # Essayez à nouveau dans deux minutes
    "目前限制了每小时",  # Actuellement limité par heure
    "暂时限流",  # Temporairement limité
    "学校网络和公司网络等同网络下共用额度",  # Partage de quota sur réseaux scolaires/d'entreprise
    "如果限制了可以尝试切换网络使用",  # Si limité, essayez de changer de réseau
    "请关闭代理",  # Veuillez fermer le proxy
    "不要使用公共网络访问",  # N'utilisez pas de réseau public
    "如需合作接口调用请联系",  # Pour une coopération API, contactez
    "本网站正版地址是",  # L'adresse officielle de ce site est
    "如果你在其他网站遇到此报错",  # Si vous rencontrez cette erreur sur d'autres sites
]

# Initialisation des métriques
if CONFIG["ENABLE_METRICS"]:
    try:
        request_counter = Counter('api_requests_total', 'Total count of API requests', ['endpoint'])
        error_counter = Counter('api_errors_total', 'Total count of API errors', ['endpoint', 'error_type'])
        request_latency = Histogram('api_request_latency_seconds', 'API request latency in seconds', ['endpoint'])
        circuit_open_counter = Counter('circuit_open_total', 'Total count of circuit open events', ['circuit'])
        circuit_fallback_counter = Counter('circuit_fallback_total', 'Total count of circuit fallback events', ['circuit'])
        start_http_server(CONFIG["METRICS_PORT"])
        logger.info(f"Serveur de métriques démarré sur le port {CONFIG['METRICS_PORT']}")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des métriques: {e}")
        logger.error(traceback.format_exc())
        CONFIG["ENABLE_METRICS"] = False

# Rate limiter global
limiter = Limiter(RequestRate(CONFIG["RATE_LIMIT_REQUESTS"], Duration.MINUTE))

import g4f
g4f.debug.logging = True

# Configuration de l'event loop pour Windows
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Modèles Pydantic pour la validation des données
class FixSentenceRequest(BaseModel):
    sentence: str

class IntensiveTrainingRequest(BaseModel):
    sentences: List[str]
    rule: str

class PutWordRequest(BaseModel):
    sentence: str
    audio_url: str
    max_size: Optional[int] = CONFIG["MAX_AUDIO_SIZE"]

class NearestWordRequest(BaseModel):
    word: str
    nearest_words: List[str]

# Classes de gestion des erreurs personnalisées
class GPTAPIError(Exception):
    """Exception levée lorsque l'API GPT rencontre une erreur"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class AudioProcessingError(Exception):
    """Exception levée lorsque le traitement audio échoue"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class RateLimitExceededError(Exception):
    """Exception levée lorsque la limite de taux est dépassée"""
    def __init__(self, message: str = "Trop de requêtes"):
        self.message = message
        self.status_code = 429
        super().__init__(self.message)

class ValidationError(Exception):
    """Exception levée lorsque la validation des données échoue"""
    def __init__(self, message: str):
        self.message = message
        self.status_code = 400
        super().__init__(self.message)

# Fonctions utilitaires pour la gestion des providers
def is_rate_limit_error(error_message: str) -> bool:
    """Détecte si l'erreur est due à un rate limiting."""
    if not error_message:
        return False
    
    error_lower = error_message.lower()
    # Vérifier les messages en anglais et en chinois
    for msg in RATE_LIMIT_MESSAGES:
        if msg.lower() in error_lower or msg in error_message:
            return True
    return False

def mark_provider_blocked(provider_name: str = "default"):
    """Marque un provider comme bloqué temporairement."""
    current_time = time.time()
    PROVIDER_STATUS[provider_name]["blocked_until"] = current_time + CONFIG["RATE_LIMIT_COOLDOWN"]
    PROVIDER_STATUS[provider_name]["failures"] += 1
    logger.warning(f"Provider {provider_name} marqué comme bloqué jusqu'à {datetime.fromtimestamp(PROVIDER_STATUS[provider_name]['blocked_until'])}")

def get_available_provider() -> str:
    """Retourne un provider disponible (non bloqué)."""
    current_time = time.time()
    
    for provider, status in PROVIDER_STATUS.items():
        if status["blocked_until"] < current_time:
            return provider
    
    # Si tous les providers sont bloqués, retourner celui avec le moins de blocages
    return min(PROVIDER_STATUS.keys(), key=lambda x: PROVIDER_STATUS[x]["failures"])

def reset_provider_status(provider_name: str = "default"):
    """Remet à zéro le statut d'un provider après succès."""
    PROVIDER_STATUS[provider_name]["blocked_until"] = 0
    PROVIDER_STATUS[provider_name]["failures"] = max(0, PROVIDER_STATUS[provider_name]["failures"] - 1)

# Initialisation de l'application
app = Flask(__name__)
CORS(app, origins="*")  # Allow all origins temporarily for easier testing
client = Client()

# Répertoire temporaire géré pour tous les fichiers temporaires
temp_dir = tempfile.mkdtemp(prefix="voltaire_bot_")
logger.info(f"Répertoire temporaire créé: {temp_dir}")

# Nettoyage des fichiers temporaires à la sortie
def cleanup_temp_files():
    """Nettoie les fichiers temporaires à la fermeture de l'application"""
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Répertoire temporaire supprimé: {temp_dir}")
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des fichiers temporaires: {str(e)}")

atexit.register(cleanup_temp_files)

# Configuration des cookies si disponibles
if os.path.exists("har_and_cookies"):
    cookies_dir = os.path.join(os.path.dirname(__file__), "har_and_cookies")
    set_cookies_dir(cookies_dir)
    read_cookie_files(cookies_dir)

# Initialisation du recognizer pour SpeechRecognition
r = sr.Recognizer()

# Surveillance des ressources système
def check_system_resources():
    """Vérifie les ressources système et log un avertissement si nécessaire"""
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=1)
    
    if memory_usage > 90:
        logger.warning(f"Utilisation mémoire élevée: {memory_usage}%")
    
    if cpu_usage > 90:
        logger.warning(f"Utilisation CPU élevée: {cpu_usage}%")
    
    return memory_usage, cpu_usage

# Démarrer un thread de surveillance des ressources
def monitor_resources():
    while True:
        check_system_resources()
        time.sleep(60)  # Vérifier toutes les minutes

resource_thread = threading.Thread(target=monitor_resources, daemon=True)
resource_thread.start()

# Décorateur de gestion globale des exceptions
def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        endpoint = func.__name__
        
        try:
            if CONFIG["ENABLE_METRICS"]:
                request_counter.labels(endpoint=endpoint).inc()
                
            start_time = time.time()
            result = func(*args, **kwargs)
            
            if CONFIG["ENABLE_METRICS"]:
                request_latency.labels(endpoint=endpoint).observe(time.time() - start_time)
                
            return result
        except HTTPException:
            # Laisse passer les HTTPException pour être gérées par le handler global
            raise
        except RateLimitExceededError as e:
            logger.warning(f"Limite de taux dépassée pour {func.__name__}: {str(e)}")
            if CONFIG["ENABLE_METRICS"]:
                error_counter.labels(endpoint=endpoint, error_type="rate_limit").inc()
            
            return Response(json.dumps({
                "status": e.status_code,
                "message": "Rate Limit Exceeded",
                "description": e.message
            }), status=e.status_code, content_type="application/json")
        except ValidationError as e:
            logger.warning(f"Erreur de validation pour {func.__name__}: {str(e)}")
            if CONFIG["ENABLE_METRICS"]:
                error_counter.labels(endpoint=endpoint, error_type="validation").inc()
            
            return Response(json.dumps({
                "status": e.status_code,
                "message": "Validation Error",
                "description": e.message
            }), status=e.status_code, content_type="application/json")
        except GPTAPIError as e:
            logger.error(f"Erreur API GPT pour {func.__name__}: {str(e)}")
            if CONFIG["ENABLE_METRICS"]:
                error_counter.labels(endpoint=endpoint, error_type="gpt_api").inc()
            
            return Response(json.dumps({
                "status": e.status_code,
                "message": "API Error",
                "description": e.message
            }), status=e.status_code, content_type="application/json")
        except AudioProcessingError as e:
            logger.error(f"Erreur traitement audio pour {func.__name__}: {str(e)}")
            if CONFIG["ENABLE_METRICS"]:
                error_counter.labels(endpoint=endpoint, error_type="audio_processing").inc()
            
            return Response(json.dumps({
                "status": e.status_code,
                "message": "Audio Processing Error",
                "description": e.message
            }), status=e.status_code, content_type="application/json")
        except CircuitBreakerError as e:
            logger.error(f"Circuit breaker ouvert pour {func.__name__}: {str(e)}")
            if CONFIG["ENABLE_METRICS"]:
                circuit_open_counter.labels(circuit=str(e.circuit_breaker)).inc()
            
            # Si fallback activé, on renvoie une réponse par défaut
            if CONFIG["ENABLE_FALLBACK"]:
                logger.info(f"Utilisation du fallback pour {func.__name__}")
                if CONFIG["ENABLE_METRICS"]:
                    circuit_fallback_counter.labels(circuit=str(e.circuit_breaker)).inc()
                
                # Traitement spécifique à chaque endpoint
                if endpoint == "fix_sentence":
                    return Response(json.dumps({
                        "word_to_click": None,
                        "time_taken": 0,
                        "fallback": True
                    }), content_type="application/json")
                elif endpoint == "intensive_training":
                    # Récupérer le nombre de phrases et renvoyer un tableau par défaut
                    sentences = request.json.get("sentences", [])
                    return Response(json.dumps({
                        "phrases": [True] * len(sentences),
                        "fallback": True
                    }), content_type="application/json")
                elif endpoint == "nearest_word":
                    # Renvoyer le premier mot de la liste par défaut
                    nearest_words = request.json.get("nearest_words", [])
                    return Response(json.dumps({
                        "word": nearest_words[0] if nearest_words else "",
                        "fallback": True
                    }), content_type="application/json")
                elif endpoint == "analyze_cod_coi":
                    # Analyse par défaut pour COD/COI
                    return Response(json.dumps({
                        "type": "COD",
                        "fallback": True
                    }), content_type="application/json")
            
            return Response(json.dumps({
                "status": 503,
                "message": "Service Unavailable",
                "description": f"Circuit breaker ouvert: {str(e)}"
            }), status=503, content_type="application/json")
        except Exception as e:
            logger.error(f"Erreur inattendue pour {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            if CONFIG["ENABLE_METRICS"]:
                error_counter.labels(endpoint=endpoint, error_type="unexpected").inc()
            
            return Response(json.dumps({
                "status": 500,
                "message": "Internal Server Error",
                "description": "Une erreur inattendue s'est produite. Veuillez réessayer."
            }), status=500, content_type="application/json")
    return wrapper

# Décorateur pour la limitation de taux
def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            limiter.try_acquire(request.remote_addr)
            return func(*args, **kwargs)
        except:
            raise RateLimitExceededError(f"Limite de taux dépassée pour {request.remote_addr}")
    return wrapper

# Variables pour les contextes memoization
# Ces caches permettent de retourner la même réponse pour des requêtes identiques
# ce qui peut aider à éviter les appels API inutiles
memoization_cache = {
    "fix_sentence": {},
    "intensive_training": {},
    "nearest_word": {},
    "cod_coi": {}
}

# Expiration des entrées du cache (en secondes)
CACHE_EXPIRATION = 3600  # 1 heure

# Fonction utilitaire pour nettoyer le cache
def clean_cache():
    """Nettoie les entrées expirées du cache de memoization"""
    current_time = time.time()
    for cache_type in memoization_cache:
        # Copier les clés pour éviter de modifier le dictionnaire pendant l'itération
        keys_to_check = list(memoization_cache[cache_type].keys())
        for key in keys_to_check:
            timestamp = memoization_cache[cache_type][key].get("timestamp", 0)
            if current_time - timestamp > CACHE_EXPIRATION:
                del memoization_cache[cache_type][key]

# Nettoyer le cache périodiquement
def cache_cleaner():
    while True:
        clean_cache()
        time.sleep(300)  # Toutes les 5 minutes

cache_thread = threading.Thread(target=cache_cleaner, daemon=True)
cache_thread.start()

# Fallback GPT pour les réponses simples sans appel API externe
def fallback_gpt(prompt: str) -> Dict:
    """
    Génère une réponse simple sans appel API externe quand le circuit breaker est ouvert.
    Cette fonction simule simplement une réponse raisonnable.
    
    Args:
        prompt: Le prompt à traiter localement
    
    Returns:
        Une réponse JSON simple
    """
    logger.info(f"Utilisation du fallback GPT local pour: {prompt}")
    
    if "corrige les fautes" in prompt.lower():
        # Cas de fix-sentence
        return {"word_to_click": None}
    elif "sont elles correctes" in prompt.lower():
        # Cas de intensive-training
        sentences_count = prompt.count("\n- ")
        return {"phrases": [True] * sentences_count}
    elif "quel est le mot le plus proche" in prompt.lower():
        # Cas de nearest-word
        words = prompt.split(" parmi : ")[1].split(". ")[0].split(", ")
        return {"word": words[0] if words else ""}
    else:
        # Cas par défaut
        return {"fallback": "Réponse non disponible"}

def is_rate_limited_response(response_content: str) -> bool:
    """
    Détecte si la réponse contient un message de rate limiting
    
    Args:
        response_content: Le contenu de la réponse à vérifier
        
    Returns:
        True si la réponse indique un rate limiting, False sinon
    """
    if not response_content:
        return False
    
    # Convertir en minuscules pour la recherche
    response_lower = response_content.lower()
    
    # Vérifier chaque message de rate limiting
    for message in RATE_LIMIT_MESSAGES:
        if message.lower() in response_lower:
            logger.warning(f"Rate limiting détecté avec le message: '{message}'")
            return True
    
    return False

def handle_rate_limiting(provider_name: str = "default"):
    """
    Gère la détection de rate limiting et met à jour le statut du provider
    
    Args:
        provider_name: Le nom du provider qui a été rate-limité
    """
    current_time = time.time()
    
    # Marquer le provider comme bloqué
    if provider_name in PROVIDER_STATUS:
        PROVIDER_STATUS[provider_name]["blocked_until"] = current_time + CONFIG["RATE_LIMIT_COOLDOWN"]
        PROVIDER_STATUS[provider_name]["failures"] += 1
        logger.warning(f"Provider '{provider_name}' bloqué jusqu'à {datetime.fromtimestamp(PROVIDER_STATUS[provider_name]['blocked_until'])}")
    
    # Si la rotation est activée, essayer de trouver un provider disponible
    if CONFIG["PROVIDER_ROTATION_ENABLED"]:
        available_providers = []
        for provider, status in PROVIDER_STATUS.items():
            if status["blocked_until"] < current_time:
                available_providers.append(provider)
        
        if available_providers:
            logger.info(f"Providers disponibles après rate limiting: {available_providers}")
        else:
            logger.warning("Tous les providers sont bloqués, attente nécessaire")

# Décorateur pour les appels API GPT avec retry
@retry(
    stop=stop_after_attempt(CONFIG["API_MAX_RETRIES"]),
    wait=wait_exponential(multiplier=1, min=CONFIG["API_RETRY_DELAY"], max=CONFIG["API_RETRY_DELAY"]*5),
    retry=retry_if_exception_type((GPTAPIError, TimeoutError)),
    reraise=True
)
@circuit(
    failure_threshold=CONFIG["CIRCUIT_FAILURE_THRESHOLD"],
    recovery_timeout=CONFIG["CIRCUIT_RECOVERY_TIMEOUT"],
    name="call_gpt_api",
    expected_exception=(GPTAPIError, TimeoutError)
)
def call_gpt_api(prompt: str, model: str = "gpt-4", response_format: Dict = {"type": "json_object"}, max_tokens: int = 500) -> Dict:
    """
    Appelle l'API GPT avec retry en cas d'échec et circuit breaker pour éviter les cascades d'erreurs.
    
    Args:
        prompt: Le prompt à envoyer à l'API
        model: Le modèle GPT à utiliser
        response_format: Le format de réponse attendu
        max_tokens: Le nombre maximum de tokens
        
    Returns:
        La réponse JSON de l'API
    
    Raises:
        GPTAPIError: Si l'API rencontre une erreur après plusieurs tentatives
        CircuitBreakerError: Si le circuit breaker est ouvert
    """
    # Vérifier d'abord dans le cache pour éviter des appels API inutiles
    cache_key = f"{prompt}_{model}_{max_tokens}"
    for cache_type in memoization_cache:
        if cache_key in memoization_cache[cache_type]:
            cached_entry = memoization_cache[cache_type][cache_key]
            logger.info(f"Réponse trouvée dans le cache pour le prompt: {prompt[:50]}...")
            return cached_entry["response"]
    
    try:
        logger.info(f"Appel API GPT - Prompt: {prompt[:100]}...")
        
        # Configurer un timeout global
        start_time = time.time()
        response = None
        
        # Créer un événement pour signaler que la fonction a terminé
        api_call_finished = threading.Event()
        api_result = {"response": None, "error": None}
        
        # Utiliser un thread avec timeout
        def api_call():
            try:
                nonlocal response
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=response_format,
                    max_tokens=max_tokens
                )
                api_result["response"] = response
            except Exception as e:
                api_result["error"] = e
            finally:
                api_call_finished.set()
        
        # Exécuter l'appel API dans un thread avec timeout
        api_thread = threading.Thread(target=api_call)
        api_thread.daemon = True
        api_thread.start()
        
        # Attendre que l'appel API termine ou que le timeout soit atteint
        if not api_call_finished.wait(timeout=CONFIG["API_TIMEOUT"]):
            logger.error(f"Timeout lors de l'appel à l'API GPT après {CONFIG['API_TIMEOUT']} secondes")
            raise TimeoutError(f"L'appel à l'API GPT a dépassé le délai d'attente de {CONFIG['API_TIMEOUT']} secondes")
        
        # Si une erreur s'est produite dans le thread
        if api_result["error"]:
            logger.error(f"Erreur dans le thread d'appel API: {str(api_result['error'])}")
            raise GPTAPIError(f"Erreur lors de l'appel à l'API GPT: {str(api_result['error'])}")
        
        # Si aucune réponse n'a été obtenue
        if not response or not hasattr(response, 'choices') or not response.choices:
            logger.error("Réponse API GPT vide ou invalide")
            raise GPTAPIError("Réponse API GPT vide ou invalide")
        
        response_content = response.choices[0].message.content
        logger.info(f"Réponse API GPT reçue en {time.time() - start_time:.2f}s: {response_content[:200]}...")
        
        # Vérifier si la réponse indique un rate limiting
        if is_rate_limited_response(response_content):
            logger.error(f"Rate limiting détecté dans la réponse: {response_content[:500]}...")
            handle_rate_limiting()
            raise GPTAPIError(f"API rate limitée. Contenu: {response_content[:200]}...")
        
        try:
            json_response = json.loads(response_content)
            
            # Correction pour le format de réponse intensive-training
            # Si la réponse est un tableau de booléens sans clé "phrases"
            if isinstance(json_response, list) and all(isinstance(item, bool) for item in json_response):
                json_response = {"phrases": json_response}
                logger.info(f"Reformatage de la réponse en tableau avec clé 'phrases': {json_response}")
            
            # Si la réponse contient une chaîne JSON, tenter de la parser
            elif isinstance(json_response, dict):
                for key, value in json_response.items():
                    if isinstance(value, str) and ('[' in value and ']' in value):
                        try:
                            # Essayer d'extraire un tableau de booléens de la chaîne
                            start_idx = value.find('[')
                            end_idx = value.rfind(']') + 1
                            if start_idx >= 0 and end_idx > start_idx:
                                array_str = value[start_idx:end_idx]
                                array_str = array_str.replace('true', 'True').replace('false', 'False')
                                # Utiliser eval de manière sécurisée pour convertir en tableau Python
                                extracted_array = json.loads(array_str)
                                if isinstance(extracted_array, list):
                                    json_response = {"phrases": [bool(item) for item in extracted_array]}
                                    logger.info(f"Tableau extrait de chaîne JSON: {json_response}")
                                    break
                        except Exception as parse_err:
                            logger.warning(f"Échec de l'extraction du tableau JSON de la chaîne: {parse_err}")
            
            # Mettre en cache la réponse
            cache_type = None
            if "word_to_click" in json_response:
                cache_type = "fix_sentence"
            elif "phrases" in json_response:
                cache_type = "intensive_training"
            elif "word" in json_response:
                cache_type = "nearest_word"
            elif "type" in json_response:
                cache_type = "cod_coi"
            
            if cache_type:
                memoization_cache[cache_type][cache_key] = {
                    "response": json_response,
                    "timestamp": time.time()
                }
            
            return json_response
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON: {e}")
            logger.error(f"Contenu reçu: {response_content}")
            
            # Tenter de créer une structure JSON à partir du texte
            if response_content.strip().startswith("[") and response_content.strip().endswith("]"):
                try:
                    # Essayer de parser une liste directement
                    parsed_list = json.loads(response_content.strip())
                    if all(isinstance(item, bool) for item in parsed_list):
                        return {"phrases": parsed_list}
                except Exception as parse_err:
                    logger.warning(f"Échec du parsing direct de la liste: {parse_err}")
            
            # Rechercher un motif de tableau dans le texte brut
            try:
                # Chercher quelque chose comme [true, false, true]
                import re
                pattern = r'\[((?:true|false)(?:,\s*(?:true|false))*)\]'
                match = re.search(pattern, response_content, re.IGNORECASE)
                if match:
                    array_str = match.group(0)
                    # Convertir en JSON valide et parser
                    array_str = array_str.replace("true", "true").replace("false", "false")
                    parsed_list = json.loads(array_str)
                    if isinstance(parsed_list, list):
                        return {"phrases": [bool(item) for item in parsed_list]}
            except Exception as regex_err:
                logger.warning(f"Échec de l'extraction par regex: {regex_err}")
            
            # En dernier recours, utiliser une heuristique simple
            try:
                true_count = response_content.lower().count('true')
                false_count = response_content.lower().count('false')
                if true_count > 0 or false_count > 0:
                    total = true_count + false_count
                    # Créer un tableau basé sur les occurrences de 'true' et 'false'
                    result = []
                    for _ in range(true_count):
                        result.append(True)
                    for _ in range(false_count):
                        result.append(False)
                    logger.info(f"Tableau généré par heuristique: {result}")
                    return {"phrases": result}
            except Exception as heur_err:
                logger.warning(f"Échec de l'heuristique: {heur_err}")
            
            # Si tout échoue, renvoyer une erreur
            raise GPTAPIError(f"Erreur de décodage JSON et aucune méthode de fallback n'a réussi: {e}")
    except TimeoutError as e:
        logger.error(f"Timeout lors de l'appel à l'API GPT: {str(e)}")
        raise GPTAPIError(f"Timeout lors de l'appel à l'API GPT: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à l'API GPT: {str(e)}")
        logger.error(traceback.format_exc())
        raise GPTAPIError(f"Erreur lors de l'appel à l'API GPT: {str(e)}")

@app.route("/")
@handle_errors
def home():
    """This route redirect to the GitHub repository of the project."""
    logger.info("Redirection vers le repository GitHub")
    return redirect("https://www.youtube.com/watch?v=dQw4w9WgXcQ", code=301)

@app.route("/robots.txt")
@handle_errors
def robots():
    """This route return the robots.txt file."""
    logger.info("Requête robots.txt")
    response = Response("User-agent: *\nDisallow:")
    response.content_type = "text/plain"
    return response

@app.route("/health")
@handle_errors
def health_check():
    """Endpoint de vérification de la santé de l'API."""
    # Vérification des ressources système
    memory_usage, cpu_usage = check_system_resources()
    
    # Déterminer le statut en fonction de l'utilisation des ressources
    status = "ok"
    if memory_usage > 95 or cpu_usage > 95:
        status = "warning"
    
    # Récupérer l'état des circuits breakers
    circuit_breaker_status = {
        "call_gpt_api": "closed",  # Par défaut fermé
    }
    
    return Response(json.dumps({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "system": {
            "memory_usage_percent": memory_usage,
            "cpu_usage_percent": cpu_usage
        },
        "circuit_breakers": circuit_breaker_status,
        "version": "1.2.0"
    }), content_type="application/json")

@app.route("/fix-sentence", methods=["POST"])
@handle_errors
@rate_limit
def fix_sentence():
    """
    This route will fix a sentence, given in the body of the request. Hence, the only method allowed is POST.
    In the body, one parameter need to be given: "sentence", which is the sentence to fix.
    Here, we're using G4F to fix the sentence.
    """
    if not request.json or "sentence" not in request.json:
        response = Response(json.dumps({
            "status": 400,
            "message": "Bad Request",
            "description": "The request must be a JSON with a key \"sentence\"."
        }), status=400, content_type="application/json")
        raise HTTPException("Bad Request", response=response)

    try:
        now = datetime.now()
        sentence = request.json["sentence"]
        
        # Vérifier le cache
        cache_key = f"fix_sentence_{sentence}"
        if cache_key in memoization_cache["fix_sentence"]:
            cached_entry = memoization_cache["fix_sentence"][cache_key]
            logger.info(f"Réponse trouvée dans le cache pour la phrase: {sentence}")
            return Response(json.dumps({
                "word_to_click": cached_entry["response"]["word_to_click"],
                "time_taken": (datetime.now() - now).total_seconds(),
                "cached": True
            }), content_type="application/json")
        
        prompt = ("Corrige les fautes dans cette phrase : \"{}\". Répond avec du JSON avec la clé \"word_to_click\" avec "
                "comme valeur le mot non corrigé qui a été corrigé, ou null s'il n'y a pas de fautes.").format(sentence)
        
        try:
            # Appel à l'API avec retry et circuit breaker
            res_json = call_gpt_api(prompt)
            
            logger.info(f"Réponse API pour fix-sentence: {res_json}")
            
            # Mettre en cache la réponse
            memoization_cache["fix_sentence"][cache_key] = {
                "response": res_json,
                "timestamp": time.time()
            }
            
            return Response(json.dumps({
                "word_to_click": res_json["word_to_click"],
                "time_taken": (datetime.now() - now).total_seconds(),
            }), content_type="application/json")
        
        except CircuitBreakerError as e:
            # Si le circuit est ouvert, utiliser la réponse de fallback
            logger.warning(f"Circuit breaker ouvert pour fix-sentence: {e}")
            
            if CONFIG["ENABLE_METRICS"]:
                circuit_open_counter.labels(circuit="call_gpt_api").inc()
            
            if CONFIG["ENABLE_FALLBACK"]:
                logger.info("Utilisation du fallback pour fix-sentence")
                
                if CONFIG["ENABLE_METRICS"]:
                    circuit_fallback_counter.labels(circuit="call_gpt_api").inc()
                
                # Générer une réponse de fallback
                fallback_response = {"word_to_click": None}
                
                return Response(json.dumps({
                    "word_to_click": fallback_response["word_to_click"],
                    "time_taken": (datetime.now() - now).total_seconds(),
                    "fallback": True
                }), content_type="application/json")
            else:
                # Si le fallback n'est pas activé, relancer l'exception
                raise
    except GPTAPIError as e:
        logger.error(f"Erreur API GPT: {str(e)}")
        return Response(json.dumps({
            "status": 500,
            "message": "API Error",
            "description": str(e)
        }), status=500, content_type="application/json")
    except Exception as e:
        logger.error(f"Erreur dans fix_sentence: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(json.dumps({
            "status": 500,
            "message": "Internal Server Error",
            "description": "Une erreur inattendue s'est produite. Veuillez réessayer."
        }), status=500, content_type="application/json")

@app.route("/intensive-training", methods=["POST"])
@handle_errors
@rate_limit
def intensive_training():
    """
    Perform intensive training using the Projet Voltaire Bot.

    This function takes a JSON request with a list of sentences and a rule, and uses the Projet Voltaire Bot to generate a response in French. The response is a JSON object that contains the correctness of each sentence according to the given rule.

    Returns:
        A JSON response containing the correctness of each sentence.

    Raises:
        HTTPException: If the request is invalid or missing required fields.
    """
    if not request.json or "sentences" not in request.json or "rule" not in request.json:
        response = Response(json.dumps({
            "status": 400,
            "message": "Bad Request",
            "description": "The request must be a JSON with a key \"sentences\" and a key \"rule\"."
        }), status=400, content_type="application/json")
        raise HTTPException("Bad Request", response=response)

    try:
        sentences = request.json["sentences"]
        rule = request.json["rule"]
        
        # Vérifier le cache
        cache_key = f"intensive_training_{hash(str(sentences))}_{hash(rule)}"
        if cache_key in memoization_cache["intensive_training"]:
            cached_entry = memoization_cache["intensive_training"][cache_key]
            logger.info(f"Réponse trouvée dans le cache pour l'entraînement intensif")
            return Response(json.dumps(cached_entry["response"]), content_type="application/json")
        
        prompt = "Les phrases :\n- {}\nSont-elles correctes ? Réponds UNIQUEMENT avec un JSON au format {{\"phrases\": [true, false, true]}} où chaque valeur booléenne indique si la phrase correspondante est correcte.".format("\n- ".join(sentences))
        
        try:
            # Appel à l'API avec retry et circuit breaker
            res_json = call_gpt_api(prompt)
            
            logger.info(f"Réponse API pour intensive-training: {res_json}")
            
            # Extraction du tableau de booléens de la réponse JSON
            phrases_array = None
            
            # Vérifier d'abord s'il y a une clé "phrases" dans la réponse
            if isinstance(res_json, dict) and "phrases" in res_json:
                phrases_array = res_json["phrases"]
                logger.info(f"Tableau trouvé dans la clé 'phrases': {phrases_array}")
            
            # Si non, chercher dans d'autres clés possibles
            if phrases_array is None and isinstance(res_json, dict):
                for key in ["phrases_correctes", "correct", "boolean", "result"]:
                    if key in res_json and isinstance(res_json[key], list):
                        phrases_array = res_json[key]
                        logger.info(f"Tableau trouvé dans la clé '{key}': {phrases_array}")
                        break
            
            # Si toujours pas trouvé, vérifier si la réponse elle-même est un tableau
            if phrases_array is None and isinstance(res_json, list):
                phrases_array = res_json
                logger.info(f"La réponse est directement un tableau: {phrases_array}")
            
            # Si on n'a trouvours rien trouvé, générer un tableau par défaut (tous vrais)
            if phrases_array is None:
                logger.warning(f"Aucun tableau trouvé dans la réponse, génération d'un tableau par défaut")
                phrases_array = [True] * len(sentences)
            
            # Vérifier que le tableau a la bonne longueur
            if len(phrases_array) != len(sentences):
                logger.warning(f"Longueur du tableau ({len(phrases_array)}) différente du nombre de phrases ({len(sentences)}), ajustement")
                # Si trop court, remplir avec des True
                if len(phrases_array) < len(sentences):
                    phrases_array.extend([True] * (len(sentences) - len(phrases_array)))
                # Si trop long, tronquer
                else:
                    phrases_array = phrases_array[:len(sentences)]
            
            # Construire la réponse finale au format attendu par l'extension
            final_response = {"phrases": phrases_array}
            
            # Mettre en cache la réponse
            memoization_cache["intensive_training"][cache_key] = {
                "response": final_response,
                "timestamp": time.time()
            }
            
            logger.info(f"Tableau de booléens final renvoyé: {final_response}")
            
            # Renvoyer la réponse formatée
            return Response(json.dumps(final_response), content_type="application/json")
        
        except CircuitBreakerError as e:
            # Si le circuit est ouvert, utiliser la réponse de fallback
            logger.warning(f"Circuit breaker ouvert pour intensive-training: {e}")
            
            if CONFIG["ENABLE_METRICS"]:
                circuit_open_counter.labels(circuit="call_gpt_api").inc()
            
            if CONFIG["ENABLE_FALLBACK"]:
                logger.info("Utilisation du fallback pour intensive-training")
                
                if CONFIG["ENABLE_METRICS"]:
                    circuit_fallback_counter.labels(circuit="call_gpt_api").inc()
                
                # Générer une réponse de fallback (toutes les phrases sont considérées correctes)
                fallback_response = {"phrases": [True] * len(sentences)}
                
                return Response(json.dumps(fallback_response), content_type="application/json")
            else:
                # Si le fallback n'est pas activé, relancer l'exception
                raise
    
    except GPTAPIError as e:
        logger.error(f"Erreur API GPT: {str(e)}")
        return Response(json.dumps({
            "status": 500,
            "message": "API Error",
            "description": str(e)
        }), status=500, content_type="application/json")
    except Exception as e:
        logger.error(f"Erreur dans intensive_training: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(json.dumps({
            "status": 500,
            "message": "Internal Server Error",
            "description": "Une erreur inattendue s'est produite. Veuillez réessayer."
        }), status=500, content_type="application/json")

@app.route("/put-word", methods=["POST"])
@handle_errors
@rate_limit
def put_word():
    """
    This word will add a missing word to the sentence given in the body of the request.
    The user also needs to provide the audio URL for completing the sentence with the missing word.
    It's using SpeechRecognition to get the missing word from the audio.
    """
    if not request.json or "sentence" not in request.json or "audio_url" not in request.json:
        response = Response(json.dumps({
            "status": 400,
            "message": "Bad Request",
            "description": "The request must be a JSON with a key \"sentence\" and a key \"audio_url\"."
        }), status=400, content_type="application/json")
        raise HTTPException("Bad Request", response=response)

    try:
        sentence: str = request.json["sentence"]
        if "{}" not in sentence:
            raise ValidationError("La phrase doit contenir '{}' pour indiquer l'emplacement du mot manquant")
        
        audio_url: str = request.json["audio_url"]
        logger.info(f"Traitement audio pour la phrase: {sentence}")
        
        if "  " in sentence:
            sentence = sentence.replace("  ", " {} ")

        # Gestion des fichiers temporaires avec timestamp unique
        timestamp = datetime.timestamp(datetime.now())
        audio_filename = os.path.join(temp_dir, f"audio_{timestamp}.mp3")
        audio_wav_filename = audio_filename.replace(".mp3", ".wav")
        
        try:
            # Télécharger le fichier audio
            response = requests.get(audio_url)
            if response.status_code != 200:
                raise AudioProcessingError(f"Impossible de télécharger l'audio: HTTP {response.status_code}")
            
            # Vérifier la taille du fichier
            max_size = request.json.get("max_size", CONFIG["MAX_AUDIO_SIZE"])
            if len(response.content) > max_size:
                raise AudioProcessingError(f"Fichier audio trop volumineux: {len(response.content)} octets (max: {max_size})")
            
            # Sauvegarder le fichier MP3
            with open(audio_filename, "wb") as f:
                f.write(response.content)
            
            # Convertir en WAV
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_mp3(audio_filename)
                audio_segment.export(audio_wav_filename, format="wav")
            except Exception as e:
                raise AudioProcessingError(f"Erreur lors de la conversion audio: {str(e)}")
            
            # Reconnaître la parole
            with sr.AudioFile(audio_wav_filename) as source:
                audio_data = r.record(source)
                try:
                    text = r.recognize_google(audio_data, language="fr-FR")
                    logger.info(f"Texte reconnu: {text}")
                    
                    return Response(json.dumps({
                        "missing_word": text.strip(),
                    }), content_type="application/json")
                except sr.UnknownValueError:
                    raise AudioProcessingError("Impossible de reconnaître la parole dans l'audio")
                except sr.RequestError as e:
                    raise AudioProcessingError(f"Erreur lors de la requête au service de reconnaissance vocale: {e}")
        finally:
            # Nettoyer les fichiers temporaires
            for file in [audio_filename, audio_wav_filename]:
                if os.path.exists(file):
                    os.remove(file)
                    logger.debug(f"Fichier temporaire supprimé: {file}")
    
    except AudioProcessingError as e:
        logger.error(f"Erreur de traitement audio: {str(e)}")
        return Response(json.dumps({
            "status": 500,
            "message": "Audio Processing Error",
            "description": str(e)
        }), status=500, content_type="application/json")
    except Exception as e:
        logger.error(f"Erreur dans put_word: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(json.dumps({
            "status": 500,
            "message": "Internal Server Error",
            "description": "Une erreur inattendue s'est produite. Veuillez réessayer."
        }), status=500, content_type="application/json")

@app.route("/nearest-word", methods=["POST"])
@handle_errors
@rate_limit
def nearest_word():
    """
    Find the nearest word from a list of words.
    """
    if not request.json or "word" not in request.json or "nearest_words" not in request.json:
        response = Response(json.dumps({
            "status": 400,
            "message": "Bad Request",
            "description": "The request must be a JSON with a key \"word\" and a key \"nearest_words\"."
        }), status=400, content_type="application/json")
        raise HTTPException("Bad Request", response=response)

    try:
        word: str = request.json["word"]
        nearest_words: list = request.json["nearest_words"]
        
        # Vérifier le cache
        cache_key = f"nearest_word_{word}_{hash(str(nearest_words))}"
        if cache_key in memoization_cache["nearest_word"]:
            cached_entry = memoization_cache["nearest_word"][cache_key]
            logger.info(f"Réponse trouvée dans le cache pour la recherche de mot proche")
            return Response(json.dumps(cached_entry["response"]), content_type="application/json")

        prompt = f"Quel est le mot le plus proche de \"{word}\" parmi : {', '.join(nearest_words)}. Répond en json avec une clé \"word\"."
        
        try:
            # Appel à l'API avec retry et circuit breaker
            nearest_word_result = call_gpt_api(prompt)
            
            logger.info(f"Réponse API pour nearest-word: {nearest_word_result}")
            
            # S'assurer que la réponse contient bien la clé "word"
            if not "word" in nearest_word_result or not nearest_word_result["word"]:
                logger.warning("La réponse ne contient pas de clé 'word' ou elle est vide")
                # Fallback: prendre le premier mot de la liste
                nearest_word_result = {"word": nearest_words[0] if nearest_words else ""}
            
            # Mettre en cache la réponse
            memoization_cache["nearest_word"][cache_key] = {
                "response": nearest_word_result,
                "timestamp": time.time()
            }
            
            return Response(json.dumps(nearest_word_result), content_type="application/json")
        
        except CircuitBreakerError as e:
            # Si le circuit est ouvert, utiliser la réponse de fallback
            logger.warning(f"Circuit breaker ouvert pour nearest-word: {e}")
            
            if CONFIG["ENABLE_METRICS"]:
                circuit_open_counter.labels(circuit="call_gpt_api").inc()
            
            if CONFIG["ENABLE_FALLBACK"]:
                logger.info("Utilisation du fallback pour nearest-word")
                
                if CONFIG["ENABLE_METRICS"]:
                    circuit_fallback_counter.labels(circuit="call_gpt_api").inc()
                
                # Générer une réponse de fallback (premier mot de la liste)
                fallback_response = {"word": nearest_words[0] if nearest_words else ""}
                
                return Response(json.dumps(fallback_response), content_type="application/json")
            else:
                # Si le fallback n'est pas activé, relancer l'exception
                raise
    
    except GPTAPIError as e:
        logger.error(f"Erreur API GPT: {str(e)}")
        return Response(json.dumps({
            "status": 500,
            "message": "API Error",
            "description": str(e)
        }), status=500, content_type="application/json")
    except Exception as e:
        logger.error(f"Erreur dans nearest_word: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(json.dumps({
            "status": 500,
            "message": "Internal Server Error",
            "description": "Une erreur inattendue s'est produite. Veuillez réessayer."
        }), status=500, content_type="application/json")

@app.route("/analyze-cod-coi", methods=["POST"])
@handle_errors
@rate_limit
def analyze_cod_coi():
    """
    Analyse si un pronom dans une phrase est un COD (Complément d'Objet Direct) ou COI (Complément d'Objet Indirect).
    """
    if not request.json or "sentence" not in request.json or "pronoun" not in request.json:
        response = Response(json.dumps({
            "status": 400,
            "message": "Bad Request",
            "description": "The request must be a JSON with a key \"sentence\" and a key \"pronoun\"."
        }), status=400, content_type="application/json")
        raise HTTPException("Bad Request", response=response)

    try:
        sentence: str = request.json["sentence"]
        pronoun: str = request.json["pronoun"]
        
        # Vérifier le cache
        cache_key = f"cod_coi_{hash(sentence)}_{hash(pronoun)}"
        if cache_key in memoization_cache.get("cod_coi", {}):
            cached_entry = memoization_cache["cod_coi"][cache_key]
            logger.info(f"Réponse trouvée dans le cache pour l'analyse COD/COI")
            return Response(json.dumps(cached_entry["response"]), content_type="application/json")

        prompt = f"""Analyse la phrase suivante et détermine si le pronom souligné "{pronoun}" est un COD (Complément d'Objet Direct) ou un COI (Complément d'Objet Indirect).

Phrase: {sentence}
Pronom à analyser: {pronoun}

Règles de grammaire française:
- COD répond à la question "qui?" ou "quoi?" après le verbe
- COI répond à la question "à qui?", "à quoi?", "de qui?", "de quoi?" après le verbe

Réponds uniquement avec un JSON au format {{"type": "COD"}} ou {{"type": "COI"}}."""

        try:
            # Appel à l'API avec retry et circuit breaker
            result = call_gpt_api(prompt)
            
            logger.info(f"Réponse API pour analyze-cod-coi: {result}")
            
            # Validation de la réponse
            if not isinstance(result, dict) or "type" not in result:
                logger.warning("Réponse API invalide pour COD/COI, fallback vers COD")
                result = {"type": "COD"}
            
            # S'assurer que le type est valide
            if result["type"] not in ["COD", "COI"]:
                logger.warning(f"Type invalide reçu: {result['type']}, fallback vers COD")
                result = {"type": "COD"}
            
            # Initialiser le cache si nécessaire
            if "cod_coi" not in memoization_cache:
                memoization_cache["cod_coi"] = {}
            
            # Mettre en cache la réponse
            memoization_cache["cod_coi"][cache_key] = {
                "response": result,
                "timestamp": time.time()
            }
            
            return Response(json.dumps(result), content_type="application/json")
        
        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker ouvert pour analyze-cod-coi: {e}")
            
            if CONFIG["ENABLE_METRICS"]:
                circuit_open_counter.labels(circuit="call_gpt_api").inc()
            
            if CONFIG["ENABLE_FALLBACK"]:
                logger.info("Utilisation du fallback pour analyze-cod-coi")
                
                if CONFIG["ENABLE_METRICS"]:
                    circuit_fallback_counter.labels(circuit="call_gpt_api").inc()
                
                # Fallback simple : analyser le pronom pour deviner
                fallback_result = {"type": "COD"}  # Par défaut COD
                
                # Heuristiques simples pour le fallback
                pronoun_lower = pronoun.lower()
                sentence_lower = sentence.lower()
                
                # Les pronoms COI courants
                coi_indicators = ["lui", "leur", "y", "en"]
                # Prépositions qui indiquent souvent un COI
                coi_prepositions = ["à", "de", "pour", "avec", "sur", "dans"]
                
                if any(indicator in pronoun_lower for indicator in coi_indicators):
                    fallback_result = {"type": "COI"}
                elif any(prep in sentence_lower for prep in coi_prepositions):
                    # Si la phrase contient des prépositions COI, plus probable que ce soit un COI
                    fallback_result = {"type": "COI"}
                
                return Response(json.dumps(fallback_result), content_type="application/json")
            else:
                raise
    
    except GPTAPIError as e:
        logger.error(f"Erreur API GPT: {str(e)}")
        return Response(json.dumps({
            "status": 500,
            "message": "API Error",
            "description": str(e)
        }), status=500, content_type="application/json")
    except Exception as e:
        logger.error(f"Erreur dans analyze_cod_coi: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(json.dumps({
            "status": 500,
            "message": "Internal Server Error",
            "description": "Une erreur inattendue s'est produite. Veuillez réessayer."
        }), status=500, content_type="application/json")

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    logger.warning(f"HTTPException: {e.code} - {e.name}")
    response = e.get_response()
    response.data = json.dumps({
        "status": e.code,
        "message": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

def run_server(host='0.0.0.0', port=5000):
    """
    Démarre le serveur avec waitress pour une meilleure stabilité en production.
    """
    logger.info(f"Démarrage du serveur sur {host}:{port}")
    serve(app, host=host, port=port)

if __name__ == "__main__":
    # En développement, on peut utiliser le serveur Flask:
    # app.run(host='0.0.0.0', debug=True)
    
    # En production, on utilise waitress:
    run_server(port=5000)

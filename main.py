from flask import Flask, redirect, request, Response
from g4f.client import Client
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import json
from datetime import datetime
import speech_recognition as sr
import requests
import subprocess
import os
import os.path
from g4f.cookies import set_cookies_dir, read_cookie_files
import asyncio
import g4f.debug
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import traceback
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
from waitress import serve
import sys
import tempfile
import shutil
from functools import wraps
from circuitbreaker import circuit
import atexit
from prometheus_client import Counter, Histogram, start_http_server
from pyrate_limiter import Duration, RequestRate, Limiter
import threading
import psutil

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration globale
CONFIG = {
    "API_TIMEOUT": int(os.getenv("API_TIMEOUT", "30")),  # secondes
    "API_MAX_RETRIES": int(os.getenv("API_MAX_RETRIES", "3")),
    "API_RETRY_DELAY": int(os.getenv("API_RETRY_DELAY", "2")),  # secondes
    "CIRCUIT_FAILURE_THRESHOLD": int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "5")),
    "CIRCUIT_RECOVERY_TIMEOUT": int(os.getenv("CIRCUIT_RECOVERY_TIMEOUT", "30")),  # secondes
    "RATE_LIMIT_REQUESTS": int(os.getenv("RATE_LIMIT_REQUESTS", "30")),
    "RATE_LIMIT_PERIOD": int(os.getenv("RATE_LIMIT_PERIOD", "60")),  # secondes
    "MAX_AUDIO_SIZE": int(os.getenv("MAX_AUDIO_SIZE", "10485760")),  # 10MB
    "METRICS_PORT": int(os.getenv("METRICS_PORT", "9090")),
    "ENABLE_METRICS": os.getenv("ENABLE_METRICS", "false").lower() in ("true", "1", "yes"),
}

# Initialisation des métriques
if CONFIG["ENABLE_METRICS"]:
    try:
        request_counter = Counter('api_requests_total', 'Total count of API requests', ['endpoint'])
        error_counter = Counter('api_errors_total', 'Total count of API errors', ['endpoint', 'error_type'])
        request_latency = Histogram('api_request_latency_seconds', 'API request latency in seconds', ['endpoint'])
        start_http_server(CONFIG["METRICS_PORT"])
        logger.info(f"Serveur de métriques démarré sur le port {CONFIG['METRICS_PORT']}")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des métriques: {e}")
        logger.error(traceback.format_exc())
        CONFIG["ENABLE_METRICS"] = False

# Rate limiter global
limiter = Limiter(RequestRate(CONFIG["RATE_LIMIT_REQUESTS"], Duration.MINUTE))

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

# Initialisation de l'application
app = Flask(__name__)
CORS(app, origins="https://www.projet-voltaire.fr")
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
            
            response = Response(json.dumps({
                "status": e.status_code,
                "message": "Rate Limit Exceeded",
                "description": e.message,
                "retry_after": CONFIG["RATE_LIMIT_PERIOD"]
            }), status=e.status_code, content_type="application/json")
            return response
        except ValidationError as e:
            logger.warning(f"Erreur de validation dans {func.__name__}: {str(e)}")
            if CONFIG["ENABLE_METRICS"]:
                error_counter.labels(endpoint=endpoint, error_type="validation").inc()
            
            response = Response(json.dumps({
                "status": e.status_code,
                "message": "Validation Error",
                "description": e.message
            }), status=e.status_code, content_type="application/json")
            return response
        except GPTAPIError as e:
            logger.error(f"Erreur API GPT dans {func.__name__}: {str(e)}")
            if CONFIG["ENABLE_METRICS"]:
                error_counter.labels(endpoint=endpoint, error_type="gpt_api").inc()
            
            response = Response(json.dumps({
                "status": e.status_code,
                "message": "API Error",
                "description": e.message
            }), status=e.status_code, content_type="application/json")
            return response
        except AudioProcessingError as e:
            logger.error(f"Erreur de traitement audio dans {func.__name__}: {str(e)}")
            if CONFIG["ENABLE_METRICS"]:
                error_counter.labels(endpoint=endpoint, error_type="audio_processing").inc()
            
            response = Response(json.dumps({
                "status": e.status_code,
                "message": "Audio Processing Error",
                "description": e.message
            }), status=e.status_code, content_type="application/json")
            return response
        except Exception as e:
            logger.error(f"Erreur non gérée dans {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            if CONFIG["ENABLE_METRICS"]:
                error_counter.labels(endpoint=endpoint, error_type="unhandled").inc()
            
            response = Response(json.dumps({
                "status": 500,
                "message": "Internal Server Error",
                "description": "Une erreur inattendue s'est produite. Veuillez réessayer."
            }), status=500, content_type="application/json")
            return response
    return wrapper

# Décorateur pour la limitation de taux
def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Récupérer l'adresse IP de la requête
            ip = request.remote_addr
            
            # Appliquer le rate limiting
            limiter.try_acquire(ip)
            
            # Si on arrive ici, la requête est autorisée
            return func(*args, **kwargs)
        except:
            raise RateLimitExceededError()
    return wrapper

# Décorateur pour les appels API GPT avec retry
@retry(
    stop=stop_after_attempt(CONFIG["API_MAX_RETRIES"]),
    wait=wait_exponential(multiplier=1, min=CONFIG["API_RETRY_DELAY"], max=CONFIG["API_RETRY_DELAY"]*5),
    retry=retry_if_exception_type(GPTAPIError),
    reraise=True
)
@circuit(
    failure_threshold=CONFIG["CIRCUIT_FAILURE_THRESHOLD"],
    recovery_timeout=CONFIG["CIRCUIT_RECOVERY_TIMEOUT"],
    expected_exception=GPTAPIError
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
    """
    try:
        logger.info(f"Appel API GPT - Prompt: {prompt}")
        
        # Configurer un timeout global
        start_time = time.time()
        response = None
        
        # Utiliser un thread avec timeout
        def api_call():
            nonlocal response
            response = client.chat.completions.create(
                model=model,
                response_format=response_format,
                messages=[{
                    "role": "user", "content": prompt
                }],
                max_tokens=max_tokens,
            )
        
        # Exécuter l'appel API dans un thread avec timeout
        api_thread = threading.Thread(target=api_call)
        api_thread.start()
        api_thread.join(timeout=CONFIG["API_TIMEOUT"])
        
        # Vérifier si le timeout a été atteint
        if api_thread.is_alive():
            raise TimeoutError(f"L'appel à l'API GPT a dépassé le délai d'attente de {CONFIG['API_TIMEOUT']} secondes")
        
        # Vérifier si la réponse est valide
        if not response or not hasattr(response, 'choices') or not response.choices:
            raise GPTAPIError("Réponse invalide de l'API GPT")
        
        response_content = response.choices[0].message.content
        
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            raise GPTAPIError(f"La réponse de l'API n'est pas un JSON valide: {response_content[:100]}...")
        
    except TimeoutError as e:
        logger.error(f"Timeout lors de l'appel à l'API GPT: {str(e)}")
        raise GPTAPIError(f"Timeout lors de l'appel à l'API GPT: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à l'API GPT: {str(e)}")
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
    
    return Response(json.dumps({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "system": {
            "memory_usage_percent": memory_usage,
            "cpu_usage_percent": cpu_usage
        },
        "version": "1.1.0"  # Ajout d'un versionnage
    }), content_type="application/json")

@app.route("/fix-sentence", methods=["POST"])
@handle_errors
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
        prompt = ("Corrige les fautes dans cette phrase : \"{}\". Répond avec du JSON avec la clé \"word_to_click\" avec "
                "comme valeur le mot non corrigé qui a été corrigé, ou null s'il n'y a pas de fautes.").format(sentence)
        
        # Appel à l'API avec retry
        res_json = call_gpt_api(prompt)
        
        logger.info(f"Réponse API pour fix-sentence: {res_json}")
        
        return Response(json.dumps({
            "word_to_click": res_json["word_to_click"],
            "time_taken": (datetime.now() - now).total_seconds(),
        }), content_type="application/json")
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
        prompt = ("Les phrases :\n- {}\nSont elles correctes ? Répond avec un tableau JSON qui prend comme valeur un boolean"
                " si cette dernière est correcte (sous le format [true, false, true]).").format("\n- ".join(sentences))
        
        # Appel à l'API avec retry
        res_json = call_gpt_api(prompt)
        
        logger.info(f"Réponse API pour intensive-training: {res_json}")
        
        # Extraction du tableau de booléens de la réponse JSON
        phrases_array = None
        if isinstance(res_json, dict):
            # Essaie différentes clés possibles
            possible_keys = ['phrases', 'phrases_correctes']
            for key in possible_keys:
                if key in res_json and isinstance(res_json[key], list):
                    phrases_array = res_json[key]
                    break
        
        # Si on n'a pas trouvé de tableau dans les clés connues, retourne la réponse complète
        # Cette partie est pour la rétrocompatibilité
        if phrases_array is None:
            if isinstance(res_json, list):
                phrases_array = res_json
            else:
                # Fallback si on n'a rien trouvé
                logger.warning(f"Format inattendu de la réponse API: {res_json}")
                phrases_array = []
                
        logger.info(f"Tableau de booléens extrait: {phrases_array}")
        
        # Renvoie directement le tableau de booléens
        return Response(json.dumps(phrases_array), content_type="application/json")
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
            response = Response(json.dumps({
                "status": 400,
                "message": "Bad Request",
                "description": "The sentence must contain a \"{}\" to put the missing word."
            }), status=400, content_type="application/json")
            raise HTTPException("Bad Request", response=response)
        
        audio_url: str = request.json["audio_url"]
        logger.info(f"Traitement audio pour la phrase: {sentence}")
        
        if "  " in sentence:
            sentence = sentence.replace("  ", " {} ")

        # Gestion des fichiers temporaires avec timestamp unique
        timestamp = datetime.timestamp(datetime.now())
        audio_filename = os.path.abspath(f"./audio{timestamp}.mp3")
        audio_wav_filename = f"{audio_filename[:-3]}wav"
        
        try:
            # Téléchargement du fichier audio avec retry et timeout
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    audio_file = requests.get(audio_url, timeout=10)
                    audio_file.raise_for_status()  # Raise exception for HTTP errors
                    break
                except (requests.RequestException, requests.Timeout) as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise AudioProcessingError(f"Échec du téléchargement du fichier audio après {max_retries} tentatives: {str(e)}")
                    logger.warning(f"Erreur lors du téléchargement du fichier audio (tentative {retry_count}/{max_retries}): {str(e)}")
                    time.sleep(2)  # Wait before retrying
            
            # Écriture et conversion du fichier audio
            with open(audio_filename, "wb") as f:
                f.write(audio_file.content)
            
            ffmpeg_result = subprocess.run(['ffmpeg', '-i', audio_filename, audio_wav_filename], 
                                        capture_output=True, text=True)
            if ffmpeg_result.returncode != 0:
                raise AudioProcessingError(f"Erreur lors de la conversion audio: {ffmpeg_result.stderr}")

            # Reconnaissance vocale avec retry
            recognition_success = False
            retry_count = 0
            fixed_sentence_stt = ""
            
            while not recognition_success and retry_count < max_retries:
                try:
                    with sr.AudioFile(audio_wav_filename) as source:
                        audio = r.record(source)
                    fixed_sentence_stt = r.recognize_google(audio, language="fr-FR")
                    recognition_success = True
                except sr.UnknownValueError:
                    retry_count += 1
                    logger.warning(f"La reconnaissance vocale n'a pas pu comprendre l'audio (tentative {retry_count}/{max_retries})")
                    if retry_count >= max_retries:
                        raise AudioProcessingError("La reconnaissance vocale n'a pas pu comprendre l'audio après plusieurs tentatives")
                    time.sleep(1)
                except sr.RequestError as e:
                    retry_count += 1
                    logger.warning(f"Erreur du service de reconnaissance vocale (tentative {retry_count}/{max_retries}): {str(e)}")
                    if retry_count >= max_retries:
                        raise AudioProcessingError(f"Erreur du service de reconnaissance vocale après plusieurs tentatives: {str(e)}")
                    time.sleep(2)

            # Extraction du mot manquant
            try:
                missing_word_index = sentence.split(" ").index("{}")
            except ValueError:
                try:
                    missing_word_index = sentence.split(" ").index("{}.")
                except ValueError:
                    # En dernier recours, on cherche n'importe quelle occurrence de {} dans la phrase
                    for i, part in enumerate(sentence.split(" ")):
                        if "{}" in part:
                            missing_word_index = i
                            break
                    else:
                        raise AudioProcessingError("Impossible de trouver la position du mot manquant")

            # Vérification que l'index est valide pour éviter IndexError
            words = fixed_sentence_stt.split()
            if missing_word_index >= len(words):
                logger.warning(f"Index du mot manquant ({missing_word_index}) hors limites. Utilisation du dernier mot.")
                missing_word = words[-1]
            else:
                missing_word = words[missing_word_index]
                
            fixed_sentence = sentence.replace("{}", missing_word)
            
            logger.info(f"Mot manquant trouvé: '{missing_word}'")
            
            return Response(json.dumps({
                "sentence": sentence,
                "fixed_sentence": fixed_sentence,
                "missing_word": missing_word,
            }), content_type="application/json")
            
        finally:
            # Nettoyage des fichiers temporaires
            try:
                if os.path.exists(audio_filename):
                    os.remove(audio_filename)
                if os.path.exists(audio_wav_filename):
                    os.remove(audio_wav_filename)
            except Exception as e:
                logger.warning(f"Erreur lors du nettoyage des fichiers temporaires: {str(e)}")
    
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

        prompt = f"Quel est le mot le plus proche de \"{word}\" parmi : {', '.join(nearest_words)}. Répond en json avec une clé \"word\"."
        
        # Appel à l'API avec retry
        nearest_word_result = call_gpt_api(prompt)
        
        logger.info(f"Réponse API pour nearest-word: {nearest_word_result}")
        
        return Response(json.dumps({
            "word": nearest_word_result['word'],
        }), content_type="application/json")
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

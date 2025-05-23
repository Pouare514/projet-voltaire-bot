(() => {
  // const apiUrl = 'https://projet-voltaire-bot.vercel.app';
  const apiUrl = 'http://localhost:5000';
  
  // Configuration
  const config = {
    maxRetries: 5,               // Nombre maximum de tentatives pour chaque requête
    backoffDelay: 1000,          // Délai initial entre les tentatives (en ms)
    backoffMultiplier: 1.5,      // Multiplicateur pour le délai entre les tentatives
    requestTimeout: 15000,       // Timeout pour les requêtes (en ms)
    healthCheckInterval: 30000,  // Intervalle entre les vérifications de santé du backend (en ms)
  };
  
  // Statistiques et monitoring
  const stats = {
    requests: 0,
    errors: 0,
    retries: 0,
    lastError: null,
    serverStatus: 'unknown',
  };
  
  // Variables pour le state de l'extension
  let isRunning = true;
  let healthCheckIntervalId = null;
  
  // Utilitaires
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  
  // Logging amélioré
  const log = {
    info: (message) => console.log(`[Projet Voltaire Bot] ${message}`),
    warn: (message) => console.warn(`[Projet Voltaire Bot] ⚠️ ${message}`),
    error: (message, error = null) => {
      console.error(`[Projet Voltaire Bot] 🛑 ${message}`);
      if (error) console.error(error);
      stats.lastError = { message, timestamp: Date.now(), error };
      stats.errors++;
    },
    success: (message) => console.log(`[Projet Voltaire Bot] ✅ ${message}`),
  };
  
  // Fonction pour faire des requêtes avec retry et exponential backoff
  async function fetchWithRetry(url, options) {
    let retries = 0;
    let delay = config.backoffDelay;
    
    stats.requests++;
    
    while (retries < config.maxRetries) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), config.requestTimeout);
        
        const fetchOptions = {
          ...options,
          signal: controller.signal,
        };
        
        const response = await fetch(url, fetchOptions);
        clearTimeout(timeoutId);
        
        if (!response.ok) {
          throw new Error(`Status: ${response.status}`);
        }
        
        // Si on arrive ici, la requête a réussi
        return await response.json();
      } catch (error) {
        retries++;
        stats.retries++;
        
        if (error.name === 'AbortError') {
          log.warn(`Timeout dépassé pour la requête à ${url}`);
        } else {
          log.warn(`Erreur lors de la requête à ${url}: ${error.message} (tentative ${retries}/${config.maxRetries})`);
        }
        
        if (retries >= config.maxRetries) {
          log.error(`Échec après ${config.maxRetries} tentatives`, error);
          throw error;
        }
        
        // Attente avant la prochaine tentative avec backoff exponentiel
        log.warn(`Nouvelle tentative dans ${delay}ms...`);
        await wait(delay);
        delay = delay * config.backoffMultiplier;
      }
    }
  }
  
  // Fonction pour vérifier la disponibilité du backend
  async function checkServerHealth() {
    try {
      const response = await fetch(`${apiUrl}/health`, { 
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
      });
      
      if (response.ok) {
        stats.serverStatus = 'ok';
        return true;
      } else {
        stats.serverStatus = `error: ${response.status}`;
        return false;
      }
    } catch (error) {
      stats.serverStatus = `error: ${error.message}`;
      log.error('Le serveur backend est inaccessible', error);
      return false;
    }
  }
  
  // Démarrer la vérification périodique de la santé du serveur
  function startHealthCheck() {
    if (healthCheckIntervalId) {
      clearInterval(healthCheckIntervalId);
    }
    
    healthCheckIntervalId = setInterval(async () => {
      const isHealthy = await checkServerHealth();
      if (!isHealthy && isRunning) {
        log.warn('Le serveur backend semble inaccessible. Les fonctionnalités peuvent être limitées.');
      }
    }, config.healthCheckInterval);
  }
  
  // Fonction pour traiter une phrase
  const processSentence = async () => {
    const sentence = document.querySelector('.sentence').textContent;
    log.info(`Détection d'une nouvelle phrase : ${sentence}`);

    try {
      const res = await fetchWithRetry(`${apiUrl}/fix-sentence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence }),
      });
      
      if (!res) {
        log.warn('Pas de réponse valide du serveur');
        return;
      }
      
      const { word_to_click } = res;
      log.info(`Correction de la phrase : ${sentence}`);
      
      if (word_to_click && word_to_click !== 'null') {
        log.info(`Mot à cliquer : ${word_to_click}`);
        
        const element = Array.from(document.querySelectorAll('.pointAndClickSpan')).find((el) => {
          return el.textContent === word_to_click || 
            word_to_click.split('‑').some((word) => el.textContent === word) ||
            word_to_click.split('-').some((word) => el.textContent === word) ||
            word_to_click.split('\'').some((word) => el.textContent === word);
        });
        
        if (element) {
          element.click();
          log.success(`Clic sur "${word_to_click}" réussi`);
        } else {
          log.warn(`Mot "${word_to_click}" non trouvé dans la phrase`);
        }
      } else {
        log.info('Aucune erreur détectée dans la phrase');
        document.querySelector('.noMistakeButton').click();
      }
      
      await wait(500);
      document.querySelector('.nextButton').click();
      await wait(1000);
      
      if (isRunning) {
        run();
      }
    } catch (error) {
      log.error('Erreur lors de la correction de la phrase', error);
      
      if (isRunning) {
        // En cas d'échec après toutes les tentatives, on attend un peu plus longtemps
        await wait(5000);
        processSentence(); // On réessaie une dernière fois
      }
    }
  };

  // Fonction pour gérer l'entraînement intensif
  const handleIntensiveTrainingPopup = async () => {
    if (document.querySelector('.exitButton')?.style?.display === 'none' && 
        document.querySelector('.understoodButton')?.style?.display === 'none') {
      document.querySelector('.understoodButton').click();
    }
    
    await wait(500);
    
    try {
      const sentences = Array.from(document.querySelectorAll('.intensiveQuestion .sentence')).map(el => el.textContent);
      log.info(`Traitement de ${sentences.length} phrases pour l'entraînement intensif`);
      
      const ruleElement = document.querySelector('.rule-details-description');
      if (!ruleElement) {
        log.warn('Élément de règle non trouvé');
        return;
      }
      
      const rule = new DOMParser().parseFromString(
        ruleElement.innerHTML.split('<br>')[0], 
        'text/html'
      ).body.textContent;
      
      let res = await fetchWithRetry(`${apiUrl}/intensive-training`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rule, sentences }),
      });
      
      // Gestion robuste de la réponse qui peut être soit un tableau directement
      // soit un objet contenant le tableau sous différentes clés possibles
      let correctnessArray = null;
      
      if (res && typeof res === 'object') {
        // Vérifier si la réponse a une clé "phrases"
        if (res.phrases && Array.isArray(res.phrases)) {
          correctnessArray = res.phrases;
          log.info(`Tableau trouvé dans la clé 'phrases': ${JSON.stringify(correctnessArray)}`);
        } 
        // Vérifier d'autres clés possibles
        else {
          const possibleKeys = ['phrases_correctes', 'correct', 'boolean', 'result'];
          for (const key of possibleKeys) {
            if (res[key] && Array.isArray(res[key])) {
              correctnessArray = res[key];
              log.info(`Tableau trouvé dans la clé '${key}': ${JSON.stringify(correctnessArray)}`);
              break;
            }
          }
          
          // Rechercher une clé qui pourrait contenir un tableau sous forme de chaîne
          if (!correctnessArray) {
            for (const key in res) {
              if (typeof res[key] === 'string' && res[key].includes('[') && res[key].includes(']')) {
                try {
                  // Extraction du tableau entre crochets
                  const match = res[key].match(/\[(.*)\]/);
                  if (match && match[1]) {
                    // Conversion en tableau de booléens
                    const values = match[1].split(',').map(v => {
                      const trimmed = v.trim().toLowerCase();
                      return trimmed === 'true' || trimmed === '1';
                    });
                    correctnessArray = values;
                    log.info(`Tableau extrait de la chaîne dans la clé '${key}': ${JSON.stringify(correctnessArray)}`);
                    break;
                  }
                } catch (extractError) {
                  log.warn(`Échec de l'extraction du tableau de la chaîne: ${extractError}`);
                }
              }
            }
          }
        }
      }
      
      // Vérifier si la réponse elle-même est un tableau
      if (correctnessArray === null && Array.isArray(res)) {
        correctnessArray = res;
        log.info(`La réponse est directement un tableau: ${JSON.stringify(correctnessArray)}`);
      }
      
      // Si on n'a toujours pas trouvé, utiliser un fallback
      if (correctnessArray === null) {
        log.error(`Format de réponse inattendu: ${JSON.stringify(res)}`);
        correctnessArray = []; // Fallback en cas de problème
      }
      
      for (let i = 0; i < correctnessArray.length; i++) {
        const correct = correctnessArray[i];
        const questionElement = document.querySelectorAll('.intensiveQuestion')[i];
        if (questionElement) {
          const buttonSelector = `.button${correct ? 'Ok' : 'Ko'}`;
          const button = questionElement.querySelector(buttonSelector);
          if (button) {
            button.click();
            log.info(`Question ${i+1}: Réponse ${correct ? 'correcte' : 'incorrecte'}`);
          }
        }
      }
      
      await wait(500);
      
      const message = document.querySelector('.messageContainer');
      if (message && message.style.visibility !== 'hidden' && 
          message.textContent.includes('Il faut trois bonnes réponses')) {
        log.info('Pas assez de bonnes réponses, nouvel essai');
        document.querySelector('.retryButton').click();
        await wait(500);
        handleIntensiveTrainingPopup();
        return;
      }
      
      const exitButton = document.querySelector('.exitButton.primaryButton');
      if (exitButton) {
        exitButton.click();
        log.success('Entraînement intensif terminé');
        await wait(1000);
        
        if (isRunning) {
          run();
        }
      }
    } catch (error) {
      log.error('Erreur lors de l\'entraînement intensif', error);
      
      if (isRunning) {
        await wait(3000);
        handleIntensiveTrainingPopup();
      }
    }
  };
  
  // Fonction pour traiter un exercice avec audio
  const processVoiceExercise = async (url) => {
    if (!document.querySelector('.sentenceAudioReader')) return;
    
    await wait(500);
    log.info('Exercice avec voix détecté');
    
    try {
      let sentence = document.querySelector('.sentenceOuter .sentence').textContent
        .replace('  ', ' {} ')
        .replace(' .', ' {}.')
        .replace('\',', '\'{},')
        .replace(' -', ' {}-');
        
      if (sentence.startsWith(' ')) sentence = '{} ' + sentence;
      
      const res = await fetchWithRetry(`${apiUrl}/put-word`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence, audio_url: url }),
      });
      
      const { missing_word } = res;
      log.info(`Mot à écrire : ${missing_word}`);
      
      const inputElement = document.querySelector('input.writingExerciseSpan');
      if (inputElement) {
        inputElement.value = missing_word;
        await wait(1000);
        
        const validateButton = document.querySelector('.validateButton');
        if (validateButton) {
          validateButton.click();
          await wait(500);
          
          const nextButton = document.querySelector('.nextButton');
          if (nextButton) {
            nextButton.click();
            log.success('Exercice vocal complété');
            await wait(1000);
            
            if (isRunning) {
              run();
            }
          }
        }
      }
    } catch (error) {
      log.error('Erreur lors du traitement de l\'exercice vocal', error);
      
      if (isRunning) {
        await wait(3000);
        processVoiceExercise(url);
      }
    }
  };
  
  // Fonction pour trouver le mot le plus proche
  const handleNearestWordQuestion = async () => {
    try {
      const wordElement = document.querySelector('.qccv-question');
      if (!wordElement) {
        log.warn('Élément de question non trouvé');
        return;
      }
      
      const nearest_words = Array.from(document.querySelectorAll('.qc-proposal-button'))
        .map(el => el.textContent);
      
      log.info(`Recherche du mot le plus proche de "${wordElement.textContent}"`);
      
      const res = await fetchWithRetry(`${apiUrl}/nearest-word`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          word: wordElement.textContent, 
          nearest_words 
        }),
      });
      
      let closestWord = res.word.replace('-', '‑');
      log.info(`Mot le plus proche : ${closestWord}`);
      
      const element = Array.from(document.querySelectorAll('.qc-proposal-button')).find(el => {
        return el.textContent.toLowerCase() === closestWord.toLowerCase() ||
          el.textContent.toLowerCase().includes(closestWord.toLowerCase());
      });
      
      if (element) {
        element.click();
        log.success(`Sélection du mot "${closestWord}" réussie`);
        await wait(500);
        
        const nextButton = document.querySelector('.qccv-next');
        if (nextButton) {
          nextButton.click();
          await wait(1000);
          
          if (isRunning) {
            run();
          }
        }
      } else {
        log.warn(`Mot "${closestWord}" non trouvé parmi les propositions`);
      }
    } catch (error) {
      log.error('Erreur lors du traitement de la question', error);
      
      if (isRunning) {
        await wait(3000);
        handleNearestWordQuestion();
      }
    }
  };
  
  // Fonction principale pour exécuter le bot
  const run = async () => {
    // Vérifier si on doit s'arrêter
    if (!isRunning) {
      log.info('Bot arrêté');
      return;
    }
    
    try {
      const activityLaunchButton = Array.from(
        document.querySelectorAll('.activity-selector-cell-launch-button, .validation-activity-cell-launch-button')
      ).find(el => el.style.display !== 'none');
      
      if (activityLaunchButton) {
        activityLaunchButton.click();
        log.info('Lancement d\'une nouvelle activité');
        await wait(1000);
        run();
      } else if (document.querySelector('.popupPanelLessonVideo')) {
        await wait(500);
        const closeButton = document.querySelector('.popupButton#btn_fermer');
        if (closeButton) {
          closeButton.click();
          log.info('Fermeture de la vidéo de leçon');
          await wait(500);
          run();
        }
      } else if (document.querySelector('.intensiveTraining')) {
        log.info('Entraînement intensif détecté');
        handleIntensiveTrainingPopup();
      } else if (document.querySelector('.sentence') && document.querySelector('.pointAndClickSpan')) {
        log.info('Exercice de correction détecté');
        processSentence();
      } else if (document.querySelector('.sentenceAudioReader') && document.querySelector('.writingExerciseSpan')) {
        chrome.runtime.sendMessage({ type: 'mute_tab' });
        log.info('Exercice audio détecté');
        processVoiceExercise(document.querySelector('.sentenceAudioReader audio').src);
      } else if (document.querySelector('.qccv-question-container')) {
        log.info('Question à choix multiple détectée');
        handleNearestWordQuestion();
      } else if (document.querySelector('.trainingEndViewCongrate')) {
        const nextLevelButton = document.querySelector('#btn_apprentissage_autres_niveaux');
        if (nextLevelButton) {
          nextLevelButton.click();
          log.success('Niveau terminé, passage au niveau suivant');
          await wait(1000);
          run();
        }
      } else {
        // Si aucun élément connu n'est détecté, on attend et on réessaie
        log.warn('Aucun élément d\'exercice reconnu. Nouvelle tentative dans 3 secondes...');
        await wait(3000);
        run();
      }
    } catch (error) {
      log.error('Erreur dans la fonction principale', error);
      
      // En cas d'erreur, on attend un peu avant de réessayer
      if (isRunning) {
        await wait(5000);
        run();
      }
    }
  };
  
  // Fonction pour démarrer le bot
  const startBot = async () => {
    log.info('Démarrage du Projet Voltaire Bot');
    isRunning = true;
    
    // Vérifier la santé du serveur avant de commencer
    const isServerHealthy = await checkServerHealth();
    if (!isServerHealthy) {
      log.warn('Le serveur backend n\'est pas accessible. Le bot pourrait ne pas fonctionner correctement.');
    }
    
    // Démarrer la vérification périodique de la santé du serveur
    startHealthCheck();
    
    // Démarrer le bot
    run();
  };
  
  // Fonction pour arrêter le bot
  const stopBot = () => {
    log.info('Arrêt du Projet Voltaire Bot');
    isRunning = false;
    
    if (healthCheckIntervalId) {
      clearInterval(healthCheckIntervalId);
      healthCheckIntervalId = null;
    }
  };
  
  // Démarrer le bot
  startBot();
  
  // Exposer les fonctions pour le debugging
  window.ProjetVoltaireBot = {
    start: startBot,
    stop: stopBot,
    stats: () => stats,
    config: config,
  };
})();

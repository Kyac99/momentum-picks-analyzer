"""
Utilitaires pour le projet Momentum Picks Analyzer
Fonctions communes et gestionnaires d'exceptions
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from functools import wraps
from datetime import datetime, timedelta
from config import LOGS_DIR

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Configure un logger pour un module donné
    
    Parameters:
        name (str): Nom du logger
        log_file (str): Nom du fichier de log (optionnel)
        level (int): Niveau de log
        
    Returns:
        logging.Logger: Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Format des messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler pour le fichier si spécifié
    if log_file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Logger par défaut
logger = setup_logger(__name__, "utils.log")

def retry(max_attempts=3, delay=1):
    """
    Décorateur pour réessayer une fonction en cas d'échec
    
    Parameters:
        max_attempts (int): Nombre maximum de tentatives
        delay (int): Délai entre les tentatives (en secondes)
        
    Returns:
        function: Fonction décorée
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            last_exception = None
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    last_exception = e
                    
                    if attempts < max_attempts:
                        wait_time = delay * (2 ** (attempts - 1))  # Délai exponentiel
                        logger.warning(f"Tentative {attempts} échouée pour {func.__name__}: {str(e)}. Nouvelle tentative dans {wait_time}s.")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Échec de la fonction {func.__name__} après {max_attempts} tentatives: {str(e)}")
            
            # Toutes les tentatives ont échoué
            raise last_exception
        
        return wrapper
    
    return decorator

def normalize_data(data, min_percentile=1, max_percentile=99):
    """
    Normalise les données entre 0 et 1, en excluant les valeurs extrêmes
    
    Parameters:
        data (array-like): Données à normaliser
        min_percentile (int): Percentile minimum à considérer
        max_percentile (int): Percentile maximum à considérer
        
    Returns:
        array-like: Données normalisées
    """
    data = np.array(data)
    if len(data) == 0 or np.all(np.isnan(data)):
        return data
    
    # Remplacer les valeurs manquantes par la médiane
    median = np.nanmedian(data)
    data = np.where(np.isnan(data), median, data)
    
    # Obtenir les valeurs aux percentiles spécifiés
    min_val = np.percentile(data, min_percentile)
    max_val = np.percentile(data, max_percentile)
    
    # Écrêter les valeurs extrêmes
    data = np.clip(data, min_val, max_val)
    
    # Normaliser entre 0 et 1
    if max_val > min_val:
        normalized = (data - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(data)
    
    return normalized

def calculate_returns(prices, periods):
    """
    Calcule les rendements des prix sur différentes périodes
    
    Parameters:
        prices (pd.Series): Série de prix
        periods (dict): Dictionnaire des périodes à calculer
        
    Returns:
        dict: Rendements par période
    """
    returns = {}
    
    for period_name, period_length in periods.items():
        if len(prices) > period_length:
            # Calcul du rendement logarithmique
            returns[period_name] = np.log(prices.iloc[-1] / prices.iloc[-period_length-1])
        else:
            returns[period_name] = np.nan
    
    return returns

def rank_percentile(values):
    """
    Attribue un score de percentile à chaque valeur
    
    Parameters:
        values (array-like): Valeurs à classer
        
    Returns:
        array-like: Scores de percentile (0-100)
    """
    values = np.array(values)
    
    # Gestion des valeurs manquantes
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    
    if len(valid_values) == 0:
        return np.zeros_like(values)
    
    # Calcul des rangs
    ranks = np.zeros_like(values)
    ranks[valid_mask] = np.argsort(np.argsort(valid_values)) / (len(valid_values) - 1) * 100
    
    return ranks

def generate_timestamp():
    """
    Génère un timestamp formaté pour les noms de fichiers
    
    Returns:
        str: Timestamp au format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directories():
    """
    S'assure que les répertoires nécessaires existent
    """
    from config import DATA_DIR, RESULTS_DIR, LOGS_DIR
    
    for directory in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Répertoire créé: {directory}")
            
def calculate_weighted_score(values, weights):
    """
    Calcule un score pondéré à partir d'un dictionnaire de valeurs et de poids
    
    Parameters:
        values (dict): Dictionnaire des valeurs
        weights (dict): Dictionnaire des poids
        
    Returns:
        float: Score pondéré
    """
    score = 0.0
    total_weight = 0.0
    
    for key, weight in weights.items():
        if key in values and not np.isnan(values[key]):
            score += values[key] * weight
            total_weight += weight
    
    if total_weight > 0:
        return score / total_weight
    else:
        return np.nan
"""
Configuration du projet Momentum Picks Analyzer
Gère les paramètres et constantes pour l'ensemble du projet
"""

import os
from pathlib import Path

# Répertoires du projet
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Créer les répertoires s'ils n'existent pas
for directory in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Configuration API
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
API_TIMEOUT = 30  # secondes

# Paramètres de l'analyse momentum
MOMENTUM_PERIODS = {
    "short_term": 20,     # 1 mois en jours de trading
    "medium_term": 60,    # 3 mois en jours de trading
    "long_term": 252      # 1 an en jours de trading
}

MOMENTUM_WEIGHTS = {
    "short_term": 0.2,    
    "medium_term": 0.3,   
    "long_term": 0.5      
}

# Paramètres de l'analyse qualité
QUALITY_METRICS = [
    "ROE",               # Return on Equity
    "ProfitMargin",      # Marge bénéficiaire
    "DebtToEquity",      # Ratio dette/capitaux propres
    "OperatingMarginTTM" # Marge opérationnelle
]

QUALITY_WEIGHTS = {
    "ROE": 0.4,
    "ProfitMargin": 0.3,
    "DebtToEquity": 0.1,
    "OperatingMarginTTM": 0.2
}

# Paramètres du screener
SCREENER_CONFIG = {
    "min_market_cap": 1e9,         # Capitalisation boursière minimale ($1B)
    "max_pe_ratio": 50,            # PE ratio maximal
    "min_roe": 0.10,               # ROE minimal (10%)
    "max_debt_to_equity": 2.0,     # Ratio dette/capitaux propres maximal
    "min_operating_margin": 0.05,  # Marge opérationnelle minimale (5%)
    "momentum_threshold": 0.70,    # Percentile momentum minimal
    "quality_threshold": 0.60,     # Percentile qualité minimal
    "top_picks_count": 20          # Nombre d'actions à sélectionner
}

# Indices disponibles
AVAILABLE_INDICES = {
    "SP500": "S&P 500",
    "NASDAQ": "NASDAQ Composite",
    "EUROSTOXX50": "EURO STOXX 50",
    "MSCI_WORLD": "MSCI World",
    "MSCI_TECH": "MSCI World Information Technology"
}

# Paramètres de visualisation
VIZ_CONFIG = {
    "theme": "seaborn-v0_8-darkgrid",
    "figsize": (12, 8),
    "title_fontsize": 16,
    "axis_fontsize": 12,
    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
}
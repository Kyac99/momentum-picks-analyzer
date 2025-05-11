"""
Module de calcul des scores de momentum
Calcule les scores de momentum technique et fondamental
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from config import MOMENTUM_PERIODS, MOMENTUM_WEIGHTS
from utils import setup_logger, normalize_data, calculate_returns, calculate_weighted_score

# Configuration du logger
logger = setup_logger(__name__, "momentum.log")

class MomentumCalculator:
    """
    Classe pour calculer les scores de momentum technique et fondamental
    """
    
    def __init__(self, periods=None, weights=None):
        """
        Initialise l'analyseur de momentum avec les périodes et poids spécifiés
        
        Parameters:
            periods (dict): Dictionnaire des périodes pour le calcul du momentum
            weights (dict): Dictionnaire des poids pour chaque période
        """
        self.periods = periods or MOMENTUM_PERIODS
        self.weights = weights or MOMENTUM_WEIGHTS
        
        # Vérification de la cohérence des périodes et poids
        for period in self.periods:
            if period not in self.weights:
                logger.warning(f"Période '{period}' sans poids correspondant. Poids par défaut à 1.")
                self.weights[period] = 1.0
        
        # Normalisation des poids si nécessaire
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            for period in self.weights:
                self.weights[period] /= total_weight
    
    def calculate_technical_momentum(self, price_data):
        """
        Calcule le score de momentum technique basé sur les prix historiques
        
        Parameters:
            price_data (pd.DataFrame): DataFrame avec les données de prix
            
        Returns:
            dict: Score de momentum technique et scores par période
        """
        if price_data is None or price_data.empty:
            logger.warning("Pas de données de prix disponibles pour le calcul du momentum technique")
            return {
                'score': np.nan,
                'period_scores': {},
                'raw_returns': {}
            }
        
        try:
            # Extraction de la série de prix de clôture ajustés
            close_prices = price_data['adjusted_close'] if 'adjusted_close' in price_data.columns else price_data['close']
            
            # Calcul des rendements pour chaque période
            returns = calculate_returns(close_prices, self.periods)
            
            # Normalisation des rendements (transforme en Z-scores)
            normalized_returns = {}
            for period, value in returns.items():
                if np.isnan(value):
                    normalized_returns[period] = np.nan
                else:
                    # Un rendement positif => momentum positif
                    normalized_returns[period] = value
            
            # Calcul du score pondéré
            momentum_score = calculate_weighted_score(normalized_returns, self.weights)
            
            return {
                'score': momentum_score,
                'period_scores': normalized_returns,
                'raw_returns': returns
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du momentum technique: {str(e)}")
            return {
                'score': np.nan,
                'period_scores': {},
                'raw_returns': {}
            }
    
    def calculate_earnings_momentum(self, earnings_data):
        """
        Calcule le score de momentum des bénéfices basé sur les données de bénéfices trimestriels
        
        Parameters:
            earnings_data (pd.DataFrame): DataFrame avec les données de bénéfices
            
        Returns:
            float: Score de momentum des bénéfices
        """
        if earnings_data is None or earnings_data.empty:
            logger.warning("Pas de données de bénéfices disponibles pour le calcul du momentum des bénéfices")
            return np.nan
        
        try:
            # Tri des données par date
            earnings_data = earnings_data.sort_values('fiscalDateEnding', ascending=False)
            
            if len(earnings_data) < 4:
                logger.warning("Pas assez de données de bénéfices pour calculer le momentum (< 4 trimestres)")
                return np.nan
            
            # Calcul des variations de bénéfices
            eps_values = earnings_data['reportedEPS'].values
            
            # Calcul de la croissance trimestrielle (trimestre actuel vs même trimestre l'année précédente)
            q1_vs_q5 = (eps_values[0] - eps_values[4]) / abs(eps_values[4]) if abs(eps_values[4]) > 0 else np.nan
            q2_vs_q6 = (eps_values[1] - eps_values[5]) / abs(eps_values[5]) if len(eps_values) > 5 and abs(eps_values[5]) > 0 else np.nan
            q3_vs_q7 = (eps_values[2] - eps_values[6]) / abs(eps_values[6]) if len(eps_values) > 6 and abs(eps_values[6]) > 0 else np.nan
            q4_vs_q8 = (eps_values[3] - eps_values[7]) / abs(eps_values[7]) if len(eps_values) > 7 and abs(eps_values[7]) > 0 else np.nan
            
            # Moyenne des croissances sur les 4 derniers trimestres
            growth_rates = [q1_vs_q5, q2_vs_q6, q3_vs_q7, q4_vs_q8]
            valid_rates = [rate for rate in growth_rates if not np.isnan(rate)]
            
            if not valid_rates:
                logger.warning("Pas de taux de croissance valides pour calculer le momentum des bénéfices")
                return np.nan
            
            # Calcul du momentum des bénéfices
            earnings_momentum = np.mean(valid_rates)
            
            return earnings_momentum
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du momentum des bénéfices: {str(e)}")
            return np.nan
    
    def calculate_earnings_surprise_momentum(self, earnings_data):
        """
        Calcule le score de momentum des surprises de bénéfices
        
        Parameters:
            earnings_data (pd.DataFrame): DataFrame avec les données de bénéfices
            
        Returns:
            float: Score de momentum des surprises de bénéfices
        """
        if earnings_data is None or earnings_data.empty:
            logger.warning("Pas de données de bénéfices disponibles pour le calcul du momentum des surprises")
            return np.nan
        
        if 'surprisePercentage' not in earnings_data.columns:
            logger.warning("Pas de données de surprises de bénéfices disponibles")
            return np.nan
        
        try:
            # Tri des données par date
            earnings_data = earnings_data.sort_values('fiscalDateEnding', ascending=False)
            
            if len(earnings_data) < 4:
                logger.warning("Pas assez de données de bénéfices pour calculer le momentum des surprises (< 4 trimestres)")
                return np.nan
            
            # Extraction des pourcentages de surprise pour les 4 derniers trimestres
            surprise_percentages = earnings_data['surprisePercentage'].iloc[:4].values
            
            # Élimination des valeurs manquantes
            valid_surprises = surprise_percentages[~np.isnan(surprise_percentages)]
            
            if len(valid_surprises) == 0:
                logger.warning("Pas de surprises de bénéfices valides")
                return np.nan
            
            # Calcul de la moyenne des surprises
            surprise_momentum = np.mean(valid_surprises) / 100.0  # Conversion en pourcentage décimal
            
            return surprise_momentum
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du momentum des surprises de bénéfices: {str(e)}")
            return np.nan
    
    def calculate_fundamental_momentum(self, fundamental_data):
        """
        Calcule le score de momentum fondamental basé sur les données fondamentales
        
        Parameters:
            fundamental_data (dict): Dictionnaire des données fondamentales
            
        Returns:
            dict: Score de momentum fondamental et ses composantes
        """
        if fundamental_data is None:
            logger.warning("Pas de données fondamentales disponibles pour le calcul du momentum fondamental")
            return {
                'score': np.nan,
                'earnings_momentum': np.nan,
                'surprise_momentum': np.nan
            }
        
        try:
            # Extraction des données de bénéfices trimestriels
            quarterly_earnings = fundamental_data.get('quarterly_earnings', None)
            
            # Calcul du momentum des bénéfices
            earnings_momentum = self.calculate_earnings_momentum(quarterly_earnings)
            
            # Calcul du momentum des surprises de bénéfices
            surprise_momentum = self.calculate_earnings_surprise_momentum(quarterly_earnings)
            
            # Calcul du score combiné (pondération 70% bénéfices, 30% surprises)
            if not np.isnan(earnings_momentum) and not np.isnan(surprise_momentum):
                fundamental_score = 0.7 * earnings_momentum + 0.3 * surprise_momentum
            elif not np.isnan(earnings_momentum):
                fundamental_score = earnings_momentum
            elif not np.isnan(surprise_momentum):
                fundamental_score = surprise_momentum
            else:
                fundamental_score = np.nan
            
            return {
                'score': fundamental_score,
                'earnings_momentum': earnings_momentum,
                'surprise_momentum': surprise_momentum
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du momentum fondamental: {str(e)}")
            return {
                'score': np.nan,
                'earnings_momentum': np.nan,
                'surprise_momentum': np.nan
            }
    
    def calculate_momentum_score(self, stock_data):
        """
        Calcule le score de momentum combiné (technique + fondamental)
        
        Parameters:
            stock_data (dict): Dictionnaire avec les données de l'action
            
        Returns:
            dict: Score de momentum combiné et ses composantes
        """
        # Extraction des données
        price_data = stock_data.get('historical_prices', None)
        fundamental_data = stock_data.get('fundamentals', None)
        
        # Calcul du momentum technique
        technical_momentum = self.calculate_technical_momentum(price_data)
        technical_score = technical_momentum['score']
        
        # Calcul du momentum fondamental si les données sont disponibles
        if fundamental_data is not None:
            fundamental_momentum = self.calculate_fundamental_momentum(fundamental_data)
            fundamental_score = fundamental_momentum['score']
        else:
            fundamental_momentum = {
                'score': np.nan,
                'earnings_momentum': np.nan,
                'surprise_momentum': np.nan
            }
            fundamental_score = np.nan
        
        # Calcul du score combiné (pondération 60% technique, 40% fondamental)
        if not np.isnan(technical_score) and not np.isnan(fundamental_score):
            combined_score = 0.6 * technical_score + 0.4 * fundamental_score
        elif not np.isnan(technical_score):
            combined_score = technical_score
        elif not np.isnan(fundamental_score):
            combined_score = fundamental_score
        else:
            combined_score = np.nan
        
        return {
            'total_score': combined_score,
            'price_momentum': technical_momentum,
            'fundamental_momentum': fundamental_momentum
        }


if __name__ == "__main__":
    # Test simple du module
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Création de données de test
    dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
    
    # Série de prix avec tendance haussière
    prices_up = pd.DataFrame({
        'close': [100 + i * 0.5 + np.random.normal(0, 5) for i in range(300)]
    }, index=dates)
    
    # Série de prix avec tendance baissière
    prices_down = pd.DataFrame({
        'close': [250 - i * 0.3 + np.random.normal(0, 5) for i in range(300)]
    }, index=dates)
    
    # Simulation des données de bénéfices trimestriels (croissance constante)
    quarterly_earnings_up = pd.DataFrame({
        'fiscalDateEnding': pd.date_range(end=datetime.now(), periods=8, freq='3M')[::-1],
        'reportedEPS': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        'estimatedEPS': [0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65],
        'surprise': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        'surprisePercentage': [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    })
    
    # Simulation des données fondamentales
    fundamental_data_up = {
        'quarterly_earnings': quarterly_earnings_up
    }
    
    # Simulation des données complètes d'une action
    stock_data = {
        'historical_prices': prices_up,
        'fundamentals': fundamental_data_up
    }
    
    # Initialisation du calculateur
    calculator = MomentumCalculator()
    
    # Test de la méthode principale
    result = calculator.calculate_momentum_score(stock_data)
    print("Score de momentum total:", result['total_score'])
    print("Score de momentum technique:", result['price_momentum']['score'])
    print("Score de momentum fondamental:", result['fundamental_momentum']['score'])

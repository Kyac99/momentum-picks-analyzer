"""
Module de calcul des scores Momentum
Calcule le score Momentum technique et fondamental pour les actions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MomentumCalculator:
    """
    Classe pour calculer les scores de Momentum des actions
    """
    
    def __init__(self, weights=None):
        """
        Initialise le calculateur de Momentum avec des poids personnalisables
        
        Parameters:
            weights (dict): Dictionnaire des poids pour chaque période
                Format: {'1m': 0.2, '3m': 0.3, '6m': 0.3, '12m': 0.2}
        """
        # Poids par défaut pour les différentes périodes
        self.weights = weights or {
            '1m': 0.2,   # 1 mois
            '3m': 0.3,   # 3 mois
            '6m': 0.3,   # 6 mois
            '12m': 0.2   # 12 mois
        }
        
        # Vérifier que la somme des poids est 1
        if abs(sum(self.weights.values()) - 1.0) > 1e-10:
            raise ValueError("La somme des poids doit être égale à 1")
    
    def calculate_technical_momentum(self, historical_data):
        """
        Calcule le score de Momentum technique basé sur les performances historiques
        
        Parameters:
            historical_data (pd.DataFrame): DataFrame avec les données historiques de prix
            
        Returns:
            dict: Dictionnaire avec les scores de Momentum technique
        """
        # Vérifier que les données sont triées par date
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            raise ValueError("L'index du DataFrame doit être de type DatetimeIndex")
        
        historical_data = historical_data.sort_index()
        
        # Calculer les performances sur différentes périodes
        current_date = historical_data.index[-1]
        latest_close = historical_data['close'].iloc[-1]
        
        # Calculer les dates pour chaque période
        one_month_ago = current_date - timedelta(days=30)
        three_months_ago = current_date - timedelta(days=90)
        six_months_ago = current_date - timedelta(days=180)
        twelve_months_ago = current_date - timedelta(days=365)
        
        # Obtenir les prix au début de chaque période
        try:
            price_1m = historical_data.loc[historical_data.index <= one_month_ago, 'close'].iloc[-1]
            perf_1m = (latest_close / price_1m) - 1
        except (IndexError, KeyError):
            perf_1m = 0
            
        try:
            price_3m = historical_data.loc[historical_data.index <= three_months_ago, 'close'].iloc[-1]
            perf_3m = (latest_close / price_3m) - 1
        except (IndexError, KeyError):
            perf_3m = 0
            
        try:
            price_6m = historical_data.loc[historical_data.index <= six_months_ago, 'close'].iloc[-1]
            perf_6m = (latest_close / price_6m) - 1
        except (IndexError, KeyError):
            perf_6m = 0
            
        try:
            price_12m = historical_data.loc[historical_data.index <= twelve_months_ago, 'close'].iloc[-1]
            perf_12m = (latest_close / price_12m) - 1
        except (IndexError, KeyError):
            perf_12m = 0
        
        # Appliquer les poids pour obtenir le score final
        technical_score = (
            self.weights['1m'] * perf_1m +
            self.weights['3m'] * perf_3m +
            self.weights['6m'] * perf_6m +
            self.weights['12m'] * perf_12m
        )
        
        return {
            'perf_1m': perf_1m,
            'perf_3m': perf_3m,
            'perf_6m': perf_6m,
            'perf_12m': perf_12m,
            'technical_score': technical_score
        }
    
    def calculate_fundamental_momentum(self, earnings_data):
        """
        Calcule le score de Momentum fondamental basé sur les données de bénéfices
        
        Parameters:
            earnings_data (pd.DataFrame): DataFrame avec les données de bénéfices
            
        Returns:
            dict: Dictionnaire avec les scores de Momentum fondamental
        """
        if earnings_data is None or earnings_data.empty:
            return {
                'eps_growth': 0,
                'surprise_score': 0,
                'fundamental_score': 0
            }
        
        # Trier par date
        if 'fiscalDateEnding' in earnings_data.columns:
            earnings_data = earnings_data.sort_values('fiscalDateEnding', ascending=False)
        
        # Calculer la croissance des EPS
        try:
            # Croissance des EPS sur les 4 derniers trimestres
            recent_eps = earnings_data['reportedEPS'].iloc[0:4].sum()
            previous_eps = earnings_data['reportedEPS'].iloc[4:8].sum()
            
            if previous_eps > 0:
                eps_growth = (recent_eps / previous_eps) - 1
            else:
                eps_growth = 0 if recent_eps <= 0 else 1  # Si previous_eps <= 0, mais recent_eps > 0, forte croissance
                
        except (IndexError, KeyError):
            eps_growth = 0
        
        # Calculer le score de surprise
        try:
            # Moyenne des surprises en pourcentage sur les 4 derniers trimestres
            recent_surprises = earnings_data['surprisePercentage'].iloc[0:4].mean()
            surprise_score = recent_surprises / 100  # Normaliser
        except (IndexError, KeyError):
            surprise_score = 0
        
        # Calculer le score fondamental (50% croissance EPS, 50% surprises)
        fundamental_score = 0.5 * eps_growth + 0.5 * surprise_score
        
        return {
            'eps_growth': eps_growth,
            'surprise_score': surprise_score,
            'fundamental_score': fundamental_score
        }
    
    def calculate_combined_momentum(self, technical_momentum, fundamental_momentum, tech_weight=0.7, fund_weight=0.3):
        """
        Combine les scores de Momentum technique et fondamental
        
        Parameters:
            technical_momentum (dict): Résultat de calculate_technical_momentum
            fundamental_momentum (dict): Résultat de calculate_fundamental_momentum
            tech_weight (float): Poids du Momentum technique (défaut: 0.7)
            fund_weight (float): Poids du Momentum fondamental (défaut: 0.3)
            
        Returns:
            float: Score de Momentum combiné
        """
        # Vérifier que la somme des poids est 1
        if abs(tech_weight + fund_weight - 1.0) > 1e-10:
            raise ValueError("La somme des poids tech_weight et fund_weight doit être égale à 1")
        
        # Extraire les scores
        tech_score = technical_momentum['technical_score']
        fund_score = fundamental_momentum['fundamental_score']
        
        # Combiner les scores
        combined_score = tech_weight * tech_score + fund_weight * fund_score
        
        return combined_score
    
    def calculate_momentum_score(self, stock_data):
        """
        Calcule tous les scores de Momentum pour une action
        
        Parameters:
            stock_data (dict): Dictionnaire avec les données historiques et fondamentales
            
        Returns:
            dict: Dictionnaire avec tous les scores de Momentum
        """
        if 'historical' not in stock_data:
            raise ValueError("Les données historiques sont requises pour calculer le Momentum")
        
        # Calculer le Momentum technique
        technical_momentum = self.calculate_technical_momentum(stock_data['historical'])
        
        # Calculer le Momentum fondamental si les données sont disponibles
        if 'earnings' in stock_data:
            fundamental_momentum = self.calculate_fundamental_momentum(stock_data['earnings'])
        else:
            fundamental_momentum = {'eps_growth': 0, 'surprise_score': 0, 'fundamental_score': 0}
        
        # Calculer le score combiné
        combined_score = self.calculate_combined_momentum(
            technical_momentum,
            fundamental_momentum
        )
        
        # Retourner tous les scores
        return {
            'technical': technical_momentum,
            'fundamental': fundamental_momentum,
            'combined_score': combined_score
        }

if __name__ == "__main__":
    # Test simple du module
    import data_loader
    
    # Charger quelques données
    api_key = input("Entrez votre clé API Alpha Vantage: ")
    loader = data_loader.DataLoader(api_key=api_key)
    
    # Récupérer les données pour un symbole
    symbol = "AAPL"
    print(f"Récupération des données pour {symbol}")
    
    stock_data = {
        'historical': loader.get_stock_data(symbol),
        'earnings': loader.get_earnings_data(symbol)
    }
    
    # Calculer les scores de Momentum
    momentum_calc = MomentumCalculator()
    momentum_scores = momentum_calc.calculate_momentum_score(stock_data)
    
    # Afficher les résultats
    print("\nScores de Momentum Technique:")
    for key, value in momentum_scores['technical'].items():
        print(f"{key}: {value:.4f}")
    
    print("\nScores de Momentum Fondamental:")
    for key, value in momentum_scores['fundamental'].items():
        print(f"{key}: {value:.4f}")
    
    print(f"\nScore de Momentum Combiné: {momentum_scores['combined_score']:.4f}")

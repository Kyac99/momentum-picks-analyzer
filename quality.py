"""
Module de calcul des scores Quality
Calcule le score Quality en fonction des indicateurs fondamentaux 
"""

import pandas as pd
import numpy as np

class QualityCalculator:
    """
    Classe pour calculer les scores de Quality des actions
    """
    
    def __init__(self, weights=None):
        """
        Initialise le calculateur de Quality avec des poids personnalisables
        
        Parameters:
            weights (dict): Dictionnaire des poids pour chaque métrique
                Format: {'ROE': 0.3, 'ProfitMargin': 0.3, 'DebtToEquity': 0.2, 'EarningsStability': 0.2}
        """
        # Poids par défaut pour les différentes métriques
        self.weights = weights or {
            'ROE': 0.3,                  # Return on Equity
            'ProfitMargin': 0.3,         # Marge de profit
            'DebtToEquity': 0.2,         # Ratio Dette/Fonds propres
            'EarningsStability': 0.2     # Stabilité des bénéfices
        }
        
        # Vérifier que la somme des poids est 1
        if abs(sum(self.weights.values()) - 1.0) > 1e-10:
            raise ValueError("La somme des poids doit être égale à 1")
    
    def calculate_roe_score(self, roe):
        """
        Calcule un score pour le ROE
        
        Parameters:
            roe (float): Return on Equity
            
        Returns:
            float: Score normalisé entre 0 et 1
        """
        if roe <= 0:
            return 0
        
        # Transformation non-linéaire pour récompenser les ROE élevés tout en pénalisant les valeurs extrêmes
        # ROE de 15% donne un score de 0.75
        # ROE de 25% ou plus donne un score proche de 1
        return min(1.0, roe / 0.25)
    
    def calculate_profit_margin_score(self, profit_margin):
        """
        Calcule un score pour la marge de profit
        
        Parameters:
            profit_margin (float): Marge de profit
            
        Returns:
            float: Score normalisé entre 0 et 1
        """
        if profit_margin <= 0:
            return 0
        
        # Transformation non-linéaire pour récompenser les marges élevées
        # Marge de 10% donne un score de 0.67
        # Marge de 20% ou plus donne un score proche de 1
        return min(1.0, (profit_margin / 0.15) ** 0.8)
    
    def calculate_debt_to_equity_score(self, debt_to_equity):
        """
        Calcule un score pour le ratio Dette/Fonds propres
        Plus le ratio est bas, meilleur est le score
        
        Parameters:
            debt_to_equity (float): Ratio Dette/Fonds propres
            
        Returns:
            float: Score normalisé entre 0 et 1
        """
        if debt_to_equity < 0:
            return 0  # Dette négative, situation anormale
        
        # Transformation non-linéaire pour pénaliser les ratios élevés
        # Un ratio de 0 donne un score de 1 (pas de dette)
        # Un ratio de 1 donne un score de 0.5 (dette = fonds propres)
        # Un ratio de 3+ donne un score proche de 0
        return max(0, 1 - (debt_to_equity / 3) ** 0.7)
    
    def calculate_earnings_stability(self, earnings_data):
        """
        Calcule un score pour la stabilité des bénéfices
        
        Parameters:
            earnings_data (pd.DataFrame): DataFrame avec les données de bénéfices
            
        Returns:
            float: Score normalisé entre 0 et 1
        """
        if earnings_data is None or earnings_data.empty:
            return 0
        
        # Récupérer les EPS des derniers trimestres
        try:
            recent_eps = earnings_data['reportedEPS'].iloc[0:8].to_numpy()
            
            if len(recent_eps) < 4:
                return 0  # Pas assez de données
            
            # Calculer la volatilité (coefficient de variation)
            mean_eps = np.mean(recent_eps)
            
            if mean_eps <= 0:
                return 0  # EPS moyen négatif ou nul
                
            std_eps = np.std(recent_eps)
            cv = std_eps / abs(mean_eps)  # Coefficient de variation
            
            # Transformation en score
            # Un CV de 0 donne un score de 1 (parfaitement stable)
            # Un CV de 0.3 donne un score de 0.5
            # Un CV de 1+ donne un score proche de 0
            stability_score = max(0, 1 - cv)
            
            return stability_score
            
        except (IndexError, KeyError):
            return 0
    
    def calculate_quality_score(self, stock_data):
        """
        Calcule le score de Quality global pour une action
        
        Parameters:
            stock_data (dict): Dictionnaire avec les données fondamentales et de bénéfices
            
        Returns:
            dict: Dictionnaire avec tous les scores de Quality
        """
        # Vérifier si les données fondamentales sont disponibles
        if 'fundamental' not in stock_data:
            return {
                'roe_score': 0,
                'profit_margin_score': 0,
                'debt_to_equity_score': 0,
                'earnings_stability_score': 0,
                'quality_score': 0
            }
        
        fundamental_data = stock_data['fundamental']
        earnings_data = stock_data.get('earnings')
        
        # Calculer les scores individuels
        roe_score = self.calculate_roe_score(fundamental_data.get('ROE', 0))
        profit_margin_score = self.calculate_profit_margin_score(fundamental_data.get('ProfitMargin', 0))
        debt_to_equity_score = self.calculate_debt_to_equity_score(fundamental_data.get('DebtToEquity', 0))
        earnings_stability_score = self.calculate_earnings_stability(earnings_data)
        
        # Calculer le score de Quality global
        quality_score = (
            self.weights['ROE'] * roe_score +
            self.weights['ProfitMargin'] * profit_margin_score +
            self.weights['DebtToEquity'] * debt_to_equity_score +
            self.weights['EarningsStability'] * earnings_stability_score
        )
        
        return {
            'roe_score': roe_score,
            'profit_margin_score': profit_margin_score,
            'debt_to_equity_score': debt_to_equity_score,
            'earnings_stability_score': earnings_stability_score,
            'quality_score': quality_score
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
        'fundamental': loader.get_fundamental_data(symbol),
        'earnings': loader.get_earnings_data(symbol)
    }
    
    # Calculer les scores de Quality
    quality_calc = QualityCalculator()
    quality_scores = quality_calc.calculate_quality_score(stock_data)
    
    # Afficher les résultats
    print("\nScores de Quality:")
    for key, value in quality_scores.items():
        print(f"{key}: {value:.4f}")

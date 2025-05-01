"""
Module de screening des actions
Combine les scores de Momentum et Quality pour sélectionner les meilleures actions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockScreener:
    """
    Classe pour filtrer et sélectionner les actions selon les critères de Momentum et Quality
    """
    
    def __init__(self, momentum_weight=0.6, quality_weight=0.4):
        """
        Initialise le screener avec des poids personnalisables
        
        Parameters:
            momentum_weight (float): Poids du score Momentum (défaut: 0.6)
            quality_weight (float): Poids du score Quality (défaut: 0.4)
        """
        # Vérifier que la somme des poids est 1
        if abs(momentum_weight + quality_weight - 1.0) > 1e-10:
            raise ValueError("La somme des poids doit être égale à 1")
            
        self.momentum_weight = momentum_weight
        self.quality_weight = quality_weight
    
    def normalize_scores(self, scores_dict):
        """
        Normalise les scores entre 0 et 1 pour rendre les valeurs comparables
        
        Parameters:
            scores_dict (dict): Dictionnaire avec les scores par symbole
            
        Returns:
            dict: Dictionnaire avec les scores normalisés
        """
        if not scores_dict:
            return {}
        
        # Préparer les données pour la normalisation
        symbols = list(scores_dict.keys())
        
        # Extraire les scores de Momentum et Quality
        momentum_scores = np.array([scores_dict[symbol].get('momentum', {}).get('combined_score', 0) for symbol in symbols])
        quality_scores = np.array([scores_dict[symbol].get('quality', {}).get('quality_score', 0) for symbol in symbols])
        
        # Gérer les cas où tous les scores sont identiques
        if np.std(momentum_scores) > 1e-10:
            momentum_scaler = MinMaxScaler()
            momentum_scores_normalized = momentum_scaler.fit_transform(momentum_scores.reshape(-1, 1)).flatten()
        else:
            momentum_scores_normalized = momentum_scores
        
        if np.std(quality_scores) > 1e-10:
            quality_scaler = MinMaxScaler()
            quality_scores_normalized = quality_scaler.fit_transform(quality_scores.reshape(-1, 1)).flatten()
        else:
            quality_scores_normalized = quality_scores
        
        # Mettre à jour le dictionnaire avec les scores normalisés
        normalized_dict = {}
        for i, symbol in enumerate(symbols):
            # Copier les scores originaux
            normalized_dict[symbol] = scores_dict[symbol].copy()
            
            # Mettre à jour avec les scores normalisés
            if 'momentum' in normalized_dict[symbol]:
                normalized_dict[symbol]['momentum']['combined_score_normalized'] = momentum_scores_normalized[i]
            
            if 'quality' in normalized_dict[symbol]:
                normalized_dict[symbol]['quality']['quality_score_normalized'] = quality_scores_normalized[i]
        
        return normalized_dict
    
    def calculate_combined_score(self, scores_dict):
        """
        Calcule le score combiné Momentum + Quality
        
        Parameters:
            scores_dict (dict): Dictionnaire avec les scores normalisés
            
        Returns:
            dict: Dictionnaire avec les scores combinés
        """
        combined_dict = {}
        
        for symbol, scores in scores_dict.items():
            momentum_score = scores.get('momentum', {}).get('combined_score_normalized', 0)
            quality_score = scores.get('quality', {}).get('quality_score_normalized', 0)
            
            # Calculer le score combiné
            combined_score = (
                self.momentum_weight * momentum_score +
                self.quality_weight * quality_score
            )
            
            # Stocker le résultat
            combined_dict[symbol] = {
                'momentum_score': momentum_score,
                'quality_score': quality_score,
                'combined_score': combined_score
            }
        
        return combined_dict
    
    def filter_stocks(self, combined_scores, min_momentum=0.0, min_quality=0.0, min_combined=0.0):
        """
        Filtre les actions selon des seuils minimaux
        
        Parameters:
            combined_scores (dict): Dictionnaire avec les scores combinés
            min_momentum (float): Score minimum de Momentum
            min_quality (float): Score minimum de Quality
            min_combined (float): Score minimum combiné
            
        Returns:
            dict: Dictionnaire avec les actions filtrées
        """
        filtered_scores = {}
        
        for symbol, scores in combined_scores.items():
            if (scores['momentum_score'] >= min_momentum and
                scores['quality_score'] >= min_quality and
                scores['combined_score'] >= min_combined):
                filtered_scores[symbol] = scores
        
        return filtered_scores
    
    def rank_stocks(self, combined_scores, limit=None, reverse=True):
        """
        Trie les actions selon leur score combiné
        
        Parameters:
            combined_scores (dict): Dictionnaire avec les scores combinés
            limit (int): Nombre d'actions à retenir (défaut: None = toutes)
            reverse (bool): Si True, trie par ordre décroissant (défaut: True)
            
        Returns:
            list: Liste des symboles triés
        """
        # Trier par score combiné
        sorted_stocks = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=reverse
        )
        
        # Limiter le nombre de résultats si demandé
        if limit is not None:
            sorted_stocks = sorted_stocks[:limit]
        
        return sorted_stocks
    
    def create_results_dataframe(self, stock_data, combined_scores, sorted_stocks):
        """
        Crée un DataFrame avec les résultats détaillés
        
        Parameters:
            stock_data (dict): Dictionnaire avec les données des actions
            combined_scores (dict): Dictionnaire avec les scores combinés
            sorted_stocks (list): Liste des symboles triés
            
        Returns:
            pd.DataFrame: DataFrame avec les résultats
        """
        results = []
        
        for symbol, scores in sorted_stocks:
            # Récupérer les métriques fondamentales
            fundamental_data = {}
            if 'fundamental' in stock_data.get(symbol, {}):
                fundamental_data = stock_data[symbol]['fundamental']
            
            # Récupérer les performances techniques
            technical_data = {}
            if 'momentum' in stock_data.get(symbol, {}) and 'technical' in stock_data[symbol]['momentum']:
                technical_data = stock_data[symbol]['momentum']['technical']
            
            # Construire la ligne de résultat
            result = {
                'Symbol': symbol,
                'Combined_Score': scores['combined_score'],
                'Momentum_Score': scores['momentum_score'],
                'Quality_Score': scores['quality_score'],
                'Perf_1M': technical_data.get('perf_1m', 0) * 100,
                'Perf_3M': technical_data.get('perf_3m', 0) * 100,
                'Perf_6M': technical_data.get('perf_6m', 0) * 100,
                'Perf_12M': technical_data.get('perf_12m', 0) * 100,
                'ROE': fundamental_data.get('ROE', 0) * 100,
                'Profit_Margin': fundamental_data.get('ProfitMargin', 0) * 100,
                'Debt_To_Equity': fundamental_data.get('DebtToEquity', 0),
                'Market_Cap': fundamental_data.get('MarketCap', 0),
                'EPS': fundamental_data.get('EPS', 0),
                'PE_Ratio': fundamental_data.get('PERatio', 0)
            }
            
            results.append(result)
        
        # Créer le DataFrame
        df = pd.DataFrame(results)
        
        # Formater les colonnes numériques
        for col in ['Perf_1M', 'Perf_3M', 'Perf_6M', 'Perf_12M', 'ROE', 'Profit_Margin']:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        for col in ['Combined_Score', 'Momentum_Score', 'Quality_Score', 'Debt_To_Equity', 'PE_Ratio']:
            if col in df.columns:
                df[col] = df[col].round(3)
        
        # Formater les valeurs monétaires
        if 'Market_Cap' in df.columns:
            df['Market_Cap'] = df['Market_Cap'].apply(lambda x: f"{x/1e9:.2f}B" if x >= 1e9 else f"{x/1e6:.2f}M")
        
        if 'EPS' in df.columns:
            df['EPS'] = df['EPS'].round(2)
        
        return df
    
    def screen_stocks(self, stock_data, min_momentum=0.0, min_quality=0.0, min_combined=0.0, top_n=None):
        """
        Exécute le processus complet de screening
        
        Parameters:
            stock_data (dict): Dictionnaire avec les données des actions
            min_momentum (float): Score minimum de Momentum
            min_quality (float): Score minimum de Quality
            min_combined (float): Score minimum combiné
            top_n (int): Nombre d'actions à retenir
            
        Returns:
            tuple: (DataFrame avec les résultats, liste des symboles triés)
        """
        # Préparer un dictionnaire avec les scores
        scores_dict = {}
        for symbol, data in stock_data.items():
            scores_dict[symbol] = {}
            
            if 'momentum' in data:
                scores_dict[symbol]['momentum'] = data['momentum']
            
            if 'quality' in data:
                scores_dict[symbol]['quality'] = data['quality']
        
        # Normaliser les scores
        normalized_scores = self.normalize_scores(scores_dict)
        
        # Calculer les scores combinés
        combined_scores = self.calculate_combined_score(normalized_scores)
        
        # Filtrer les actions
        filtered_scores = self.filter_stocks(
            combined_scores,
            min_momentum=min_momentum,
            min_quality=min_quality,
            min_combined=min_combined
        )
        
        # Trier les actions
        sorted_stocks = self.rank_stocks(filtered_scores, limit=top_n)
        
        # Créer le DataFrame des résultats
        results_df = self.create_results_dataframe(stock_data, combined_scores, sorted_stocks)
        
        return results_df, sorted_stocks

if __name__ == "__main__":
    # Test simple du module
    import data_loader
    import momentum
    import quality
    
    # Charger quelques données
    api_key = input("Entrez votre clé API Alpha Vantage: ")
    loader = data_loader.DataLoader(api_key=api_key)
    
    # Récupérer des données pour quelques symboles
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "FB"]
    stock_data = {}
    
    momentum_calc = momentum.MomentumCalculator()
    quality_calc = quality.QualityCalculator()
    
    for symbol in symbols:
        print(f"Récupération des données pour {symbol}")
        
        # Récupérer les données
        historical = loader.get_stock_data(symbol)
        fundamental = loader.get_fundamental_data(symbol)
        earnings = loader.get_earnings_data(symbol)
        
        if historical is not None:
            stock_data[symbol] = {
                'historical': historical,
                'fundamental': fundamental,
                'earnings': earnings
            }
            
            # Calculer les scores de Momentum
            momentum_scores = momentum_calc.calculate_momentum_score(stock_data[symbol])
            stock_data[symbol]['momentum'] = momentum_scores
            
            # Calculer les scores de Quality
            quality_scores = quality_calc.calculate_quality_score(stock_data[symbol])
            stock_data[symbol]['quality'] = quality_scores
    
    # Exécuter le screening
    screener = StockScreener()
    results_df, sorted_stocks = screener.screen_stocks(stock_data, top_n=3)
    
    # Afficher les résultats
    print("\nRésultats du screening:")
    print(results_df)

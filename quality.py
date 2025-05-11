"""
Module de calcul des scores de qualité
Analyse la qualité des entreprises basée sur des indicateurs fondamentaux
"""

import numpy as np
import pandas as pd
import logging

from config import QUALITY_METRICS, QUALITY_WEIGHTS
from utils import setup_logger, normalize_data, calculate_weighted_score

# Configuration du logger
logger = setup_logger(__name__, "quality.log")

class QualityCalculator:
    """
    Classe pour calculer les scores de qualité des entreprises
    """
    
    def __init__(self, metrics=None, weights=None):
        """
        Initialise l'analyseur de qualité avec les métriques et poids spécifiés
        
        Parameters:
            metrics (list): Liste des métriques à utiliser pour l'analyse de qualité
            weights (dict): Dictionnaire des poids pour chaque métrique
        """
        self.metrics = metrics or QUALITY_METRICS
        self.weights = weights or QUALITY_WEIGHTS
        
        # Vérification de la cohérence des métriques et poids
        for metric in self.metrics:
            if metric not in self.weights:
                logger.warning(f"Métrique '{metric}' sans poids correspondant. Poids par défaut à 1.")
                self.weights[metric] = 1.0
        
        # Normalisation des poids
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            for metric in self.weights:
                self.weights[metric] /= total_weight
    
    def calculate_roe_score(self, overview_data):
        """
        Calcule le score de qualité basé sur le ROE (Return on Equity)
        
        Parameters:
            overview_data (dict): Données d'aperçu de l'entreprise
            
        Returns:
            float: Score ROE (0-1)
        """
        if overview_data is None or 'ROE' not in overview_data:
            return np.nan
        
        roe = overview_data['ROE']
        
        # Logique de scoring: 
        # - ROE < 0 : mauvais (score faible)
        # - 0 <= ROE < 0.05 : médiocre
        # - 0.05 <= ROE < 0.10 : moyen
        # - 0.10 <= ROE < 0.15 : bon
        # - 0.15 <= ROE < 0.25 : très bon
        # - 0.25 <= ROE : excellent (score élevé)
        
        if roe < 0:
            return 0.0
        elif roe < 0.05:
            return 0.2
        elif roe < 0.10:
            return 0.4
        elif roe < 0.15:
            return 0.6
        elif roe < 0.25:
            return 0.8
        else:
            return 1.0
    
    def calculate_profit_margin_score(self, overview_data):
        """
        Calcule le score de qualité basé sur la marge bénéficiaire
        
        Parameters:
            overview_data (dict): Données d'aperçu de l'entreprise
            
        Returns:
            float: Score de marge bénéficiaire (0-1)
        """
        if overview_data is None or 'ProfitMargin' not in overview_data:
            return np.nan
        
        profit_margin = overview_data['ProfitMargin']
        
        # Logique de scoring: 
        # - Marge < 0 : mauvais (score faible)
        # - 0 <= Marge < 0.03 : médiocre
        # - 0.03 <= Marge < 0.07 : moyen
        # - 0.07 <= Marge < 0.12 : bon
        # - 0.12 <= Marge < 0.20 : très bon
        # - 0.20 <= Marge : excellent (score élevé)
        
        if profit_margin < 0:
            return 0.0
        elif profit_margin < 0.03:
            return 0.2
        elif profit_margin < 0.07:
            return 0.4
        elif profit_margin < 0.12:
            return 0.6
        elif profit_margin < 0.20:
            return 0.8
        else:
            return 1.0
    
    def calculate_debt_to_equity_score(self, overview_data):
        """
        Calcule le score de qualité basé sur le ratio dette/capitaux propres
        Note: Plus le ratio est bas, meilleur est le score
        
        Parameters:
            overview_data (dict): Données d'aperçu de l'entreprise
            
        Returns:
            float: Score de ratio dette/capitaux propres (0-1)
        """
        if overview_data is None or 'DebtToEquity' not in overview_data:
            return np.nan
        
        debt_to_equity = overview_data['DebtToEquity']
        
        # Gestion des cas spéciaux
        if debt_to_equity < 0:
            return 0.0  # Capitaux propres négatifs = mauvais signe
        
        # Logique de scoring (inversée car moins de dette = meilleur): 
        # - Ratio > 2.5 : mauvais (score faible)
        # - 1.5 < Ratio <= 2.5 : médiocre
        # - 1.0 < Ratio <= 1.5 : moyen
        # - 0.5 < Ratio <= 1.0 : bon
        # - 0.1 < Ratio <= 0.5 : très bon
        # - 0 <= Ratio <= 0.1 : excellent (score élevé)
        
        if debt_to_equity > 2.5:
            return 0.0
        elif debt_to_equity > 1.5:
            return 0.2
        elif debt_to_equity > 1.0:
            return 0.4
        elif debt_to_equity > 0.5:
            return 0.6
        elif debt_to_equity > 0.1:
            return 0.8
        else:
            return 1.0
    
    def calculate_operating_margin_score(self, overview_data):
        """
        Calcule le score de qualité basé sur la marge opérationnelle
        
        Parameters:
            overview_data (dict): Données d'aperçu de l'entreprise
            
        Returns:
            float: Score de marge opérationnelle (0-1)
        """
        if overview_data is None or 'OperatingMarginTTM' not in overview_data:
            return np.nan
        
        operating_margin = overview_data['OperatingMarginTTM']
        
        # Logique de scoring: 
        # - Marge < 0 : mauvais (score faible)
        # - 0 <= Marge < 0.05 : médiocre
        # - 0.05 <= Marge < 0.10 : moyen
        # - 0.10 <= Marge < 0.15 : bon
        # - 0.15 <= Marge < 0.25 : très bon
        # - 0.25 <= Marge : excellent (score élevé)
        
        if operating_margin < 0:
            return 0.0
        elif operating_margin < 0.05:
            return 0.2
        elif operating_margin < 0.10:
            return 0.4
        elif operating_margin < 0.15:
            return 0.6
        elif operating_margin < 0.25:
            return 0.8
        else:
            return 1.0
    
    def calculate_free_cash_flow_score(self, cash_flow_data):
        """
        Calcule le score de qualité basé sur le Free Cash Flow
        
        Parameters:
            cash_flow_data (pd.DataFrame): Données de flux de trésorerie
            
        Returns:
            float: Score de Free Cash Flow (0-1)
        """
        if cash_flow_data is None or cash_flow_data.empty:
            return np.nan
        
        # Vérification des colonnes nécessaires
        required_columns = ['operatingCashflow', 'capitalExpenditures']
        if not all(col in cash_flow_data.columns for col in required_columns):
            return np.nan
        
        try:
            # Récupération des 2 dernières années
            last_years = cash_flow_data.sort_values('fiscalDateEnding', ascending=False).head(2)
            
            if len(last_years) < 2:
                # Si nous n'avons qu'une seule année, on l'utilise
                fcf = float(last_years['operatingCashflow'].iloc[0]) - float(last_years['capitalExpenditures'].iloc[0])
                
                # Conversion en score basé sur la positivité du FCF
                if fcf < 0:
                    return 0.0
                else:
                    return 0.8  # Bon score, mais pas parfait car nous n'avons qu'une année
            else:
                # Calcul du FCF pour les deux dernières années
                fcf_last = float(last_years['operatingCashflow'].iloc[0]) - float(last_years['capitalExpenditures'].iloc[0])
                fcf_previous = float(last_years['operatingCashflow'].iloc[1]) - float(last_years['capitalExpenditures'].iloc[1])
                
                # Si les deux sont positifs, excellent score
                if fcf_last > 0 and fcf_previous > 0:
                    return 1.0
                # Si le dernier est positif, bon score
                elif fcf_last > 0:
                    return 0.7
                # Si le précédent était positif mais le dernier est négatif, mauvais signe
                elif fcf_previous > 0:
                    return 0.3
                # Les deux sont négatifs, très mauvais signe
                else:
                    return 0.0
                
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score de Free Cash Flow: {str(e)}")
            return np.nan
    
    def calculate_stability_score(self, income_data):
        """
        Calcule le score de stabilité des revenus et bénéfices
        
        Parameters:
            income_data (pd.DataFrame): Données du compte de résultat
            
        Returns:
            float: Score de stabilité (0-1)
        """
        if income_data is None or income_data.empty:
            return np.nan
        
        # Vérification des colonnes nécessaires
        required_columns = ['totalRevenue', 'netIncome']
        if not all(col in income_data.columns for col in required_columns):
            return np.nan
        
        try:
            # Récupération des 3 dernières années
            last_years = income_data.sort_values('fiscalDateEnding', ascending=False).head(3)
            
            if len(last_years) < 3:
                # Pas assez de données pour une analyse de stabilité fiable
                return np.nan
            
            # Calcul des variations annuelles des revenus
            revenue_growth = []
            for i in range(len(last_years) - 1):
                current = float(last_years['totalRevenue'].iloc[i])
                previous = float(last_years['totalRevenue'].iloc[i+1])
                if previous > 0:
                    growth = (current - previous) / previous
                    revenue_growth.append(growth)
            
            # Calcul des variations annuelles des bénéfices nets
            income_growth = []
            for i in range(len(last_years) - 1):
                current = float(last_years['netIncome'].iloc[i])
                previous = float(last_years['netIncome'].iloc[i+1])
                if previous > 0:
                    growth = (current - previous) / previous
                    income_growth.append(growth)
            
            # Calcul de l'écart-type des taux de croissance (mesure de volatilité)
            if len(revenue_growth) > 0:
                revenue_volatility = np.std(revenue_growth)
            else:
                revenue_volatility = np.nan
                
            if len(income_growth) > 0:
                income_volatility = np.std(income_growth)
            else:
                income_volatility = np.nan
            
            # Conversion en score
            if np.isnan(revenue_volatility) and np.isnan(income_volatility):
                return np.nan
            
            # Moyenne des scores de stabilité (inversés car moins de volatilité = meilleur)
            volatility_scores = []
            
            if not np.isnan(revenue_volatility):
                if revenue_volatility > 0.5:
                    volatility_scores.append(0.0)  # Très volatile
                elif revenue_volatility > 0.3:
                    volatility_scores.append(0.3)  # Assez volatile
                elif revenue_volatility > 0.15:
                    volatility_scores.append(0.6)  # Modérément stable
                else:
                    volatility_scores.append(1.0)  # Très stable
            
            if not np.isnan(income_volatility):
                if income_volatility > 0.8:
                    volatility_scores.append(0.0)  # Très volatile
                elif income_volatility > 0.5:
                    volatility_scores.append(0.3)  # Assez volatile
                elif income_volatility > 0.25:
                    volatility_scores.append(0.6)  # Modérément stable
                else:
                    volatility_scores.append(1.0)  # Très stable
            
            return np.mean(volatility_scores)
                
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score de stabilité: {str(e)}")
            return np.nan
    
    def calculate_fundamental_quality(self, fundamental_data):
        """
        Calcule le score de qualité global basé sur les données fondamentales
        
        Parameters:
            fundamental_data (dict): Dictionnaire des données fondamentales
            
        Returns:
            dict: Score de qualité global et scores par métrique
        """
        if fundamental_data is None:
            logger.warning("Pas de données fondamentales disponibles pour le calcul du score de qualité")
            return {
                'score': np.nan,
                'metric_scores': {}
            }
        
        try:
            # Extraction des données pertinentes
            overview_data = fundamental_data.get('overview', None)
            income_data = fundamental_data.get('income_statement', None)
            cash_flow_data = fundamental_data.get('cash_flow', None)
            
            # Initialisation des scores par métrique
            metric_scores = {}
            
            # Calcul des scores pour chaque métrique demandée
            if 'ROE' in self.metrics and overview_data is not None:
                metric_scores['ROE'] = self.calculate_roe_score(overview_data)
                
            if 'ProfitMargin' in self.metrics and overview_data is not None:
                metric_scores['ProfitMargin'] = self.calculate_profit_margin_score(overview_data)
                
            if 'DebtToEquity' in self.metrics and overview_data is not None:
                metric_scores['DebtToEquity'] = self.calculate_debt_to_equity_score(overview_data)
                
            if 'OperatingMarginTTM' in self.metrics and overview_data is not None:
                metric_scores['OperatingMarginTTM'] = self.calculate_operating_margin_score(overview_data)
                
            if 'FreeCashFlow' in self.metrics and cash_flow_data is not None:
                metric_scores['FreeCashFlow'] = self.calculate_free_cash_flow_score(cash_flow_data)
                
            if 'Stability' in self.metrics and income_data is not None:
                metric_scores['Stability'] = self.calculate_stability_score(income_data)
            
            # Calcul du score pondéré
            # On ne tient compte que des métriques disponibles pour le calcul final
            quality_score = calculate_weighted_score(metric_scores, self.weights)
            
            return {
                'score': quality_score,
                'metric_scores': metric_scores
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score de qualité: {str(e)}")
            return {
                'score': np.nan,
                'metric_scores': {}
            }
        
    def calculate_quality_score(self, stock_data):
        """
        Point d'entrée principal pour le calcul du score de qualité d'une action
        
        Parameters:
            stock_data (dict): Dictionnaire contenant les données de l'action
            
        Returns:
            dict: Score de qualité total et détails
        """
        # Extraction des données fondamentales
        fundamental_data = stock_data.get('fundamentals', None)
        
        if fundamental_data is None:
            logger.warning("Pas de données fondamentales disponibles pour le calcul du score de qualité")
            return {
                'total_score': np.nan,
                'metric_scores': {}
            }
        
        # Calcul du score de qualité fondamentale
        quality_results = self.calculate_fundamental_quality(fundamental_data)
        
        # Formatage du résultat pour correspondre à l'interface attendue
        return {
            'total_score': quality_results['score'],
            'metric_scores': quality_results['metric_scores']
        }


if __name__ == "__main__":
    # Test simple du module
    import json
    from datetime import datetime
    
    # Création de données de test
    test_overview = {
        'Symbol': 'TEST',
        'Name': 'Test Company',
        'ROE': 0.18,
        'ProfitMargin': 0.15,
        'OperatingMarginTTM': 0.20,
        'DebtToEquity': 0.8
    }
    
    # Simulation des données du compte de résultat (3 ans)
    test_income = pd.DataFrame({
        'fiscalDateEnding': pd.date_range(end=datetime.now(), periods=3, freq='Y')[::-1],
        'totalRevenue': [1000000, 1100000, 1210000],
        'netIncome': [150000, 165000, 181500]
    })
    
    # Simulation des données de flux de trésorerie (2 ans)
    test_cash_flow = pd.DataFrame({
        'fiscalDateEnding': pd.date_range(end=datetime.now(), periods=2, freq='Y')[::-1],
        'operatingCashflow': [200000, 220000],
        'capitalExpenditures': [50000, 55000]
    })
    
    # Simulation des données fondamentales complètes
    test_fundamental_data = {
        'overview': test_overview,
        'income_statement': test_income,
        'cash_flow': test_cash_flow
    }
    
    # Simulation d'une action complète
    test_stock_data = {
        'fundamentals': test_fundamental_data
    }
    
    # Initialisation de l'analyseur avec des métriques étendues
    extended_metrics = ['ROE', 'ProfitMargin', 'DebtToEquity', 'OperatingMarginTTM', 'FreeCashFlow', 'Stability']
    extended_weights = {
        'ROE': 0.3,
        'ProfitMargin': 0.2,
        'DebtToEquity': 0.1,
        'OperatingMarginTTM': 0.15,
        'FreeCashFlow': 0.15,
        'Stability': 0.1
    }
    
    quality_calculator = QualityCalculator(metrics=extended_metrics, weights=extended_weights)
    
    # Calcul du score de qualité
    quality_result = quality_calculator.calculate_quality_score(test_stock_data)
    
    print("Score de qualité global:", quality_result['total_score'])
    print("\nScores par métrique:")
    for metric, score in quality_result['metric_scores'].items():
        print(f"{metric}: {score}")

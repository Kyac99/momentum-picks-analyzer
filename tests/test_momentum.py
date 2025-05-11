"""
Tests unitaires pour le module de calcul du momentum
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from momentum import MomentumCalculator
from config import MOMENTUM_PERIODS, MOMENTUM_WEIGHTS

class TestMomentumCalculator(unittest.TestCase):
    """Tests pour la classe MomentumCalculator"""

    def setUp(self):
        """Initialisation avant chaque test"""
        self.calculator = MomentumCalculator()
        
        # Créer des données de test
        self.create_test_data()

    def create_test_data(self):
        """Crée des données de test pour les prix historiques"""
        # Créer un DataFrame de prix historiques sur 300 jours
        date_today = datetime.now()
        dates = [date_today - timedelta(days=i) for i in range(300)]
        dates.sort()  # Trier dans l'ordre chronologique
        
        # Simuler une tendance haussière
        prices = np.linspace(100, 200, 300)  # Hausse linéaire de 100 à 200
        
        # Ajouter un peu de bruit aléatoire
        noise = np.random.normal(0, 2, 300)
        prices = prices + noise
        
        # Créer le DataFrame
        self.historical_prices = pd.DataFrame({
            'open': prices - 1,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 300)
        }, index=dates)
        
        # Créer un dictionnaire de données pour une action
        self.stock_data = {
            'historical_prices': self.historical_prices,
            'fundamentals': {
                'EPS': 10.5,
                'RevenueTTM': 350000000000,
                'RevenuePerShareTTM': 21.5,
                'RevenueTTMPreviousYear': 330000000000
            }
        }

    def test_calculate_price_momentum(self):
        """Test du calcul du momentum de prix"""
        # Appeler la méthode testée
        result = self.calculator.calculate_price_momentum(self.historical_prices)
        
        # Vérifier le format des résultats
        self.assertIsInstance(result, dict)
        for period_name in MOMENTUM_PERIODS.keys():
            self.assertIn(f"{period_name}_momentum", result)
            
        # Vérifier les valeurs (croissance => momentum positif)
        for period_name in MOMENTUM_PERIODS.keys():
            momentum_value = result[f"{period_name}_momentum"]
            self.assertGreater(momentum_value, 0)  # Tendance haussière => momentum positif

    def test_calculate_volume_momentum(self):
        """Test du calcul du momentum de volume"""
        # Appeler la méthode testée
        result = self.calculator.calculate_volume_momentum(self.historical_prices)
        
        # Vérifier le format des résultats
        self.assertIsInstance(result, dict)
        for period_name in MOMENTUM_PERIODS.keys():
            self.assertIn(f"{period_name}_volume_ratio", result)

    def test_calculate_fundamental_momentum(self):
        """Test du calcul du momentum fondamental"""
        # Appeler la méthode testée
        result = self.calculator.calculate_fundamental_momentum(self.stock_data)
        
        # Vérifier le format des résultats
        self.assertIsInstance(result, dict)
        self.assertIn("earnings_growth", result)
        self.assertIn("revenue_growth", result)
        
        # Vérifier les valeurs (croissance => momentum positif)
        revenue_growth = result["revenue_growth"]
        self.assertAlmostEqual(revenue_growth, (350000000000 - 330000000000) / 330000000000, places=4)

    def test_calculate_momentum_score(self):
        """Test du calcul du score global de momentum"""
        # Appeler la méthode testée
        result = self.calculator.calculate_momentum_score(self.stock_data)
        
        # Vérifier le format des résultats
        self.assertIsInstance(result, dict)
        self.assertIn("price_momentum", result)
        self.assertIn("volume_momentum", result)
        self.assertIn("fundamental_momentum", result)
        self.assertIn("total_score", result)
        
        # Vérifier que le score total est entre 0 et 1
        self.assertGreaterEqual(result["total_score"], 0)
        self.assertLessEqual(result["total_score"], 1)

    def test_normalize_momentum_values(self):
        """Test de la normalisation des valeurs de momentum"""
        # Créer des données de test
        momentum_values = {
            "short_term_momentum": 0.15,
            "medium_term_momentum": 0.25,
            "long_term_momentum": 0.35
        }
        
        # Appeler la méthode testée
        result = self.calculator._normalize_momentum_values(momentum_values, MOMENTUM_WEIGHTS)
        
        # Vérifier que le résultat est entre 0 et 1
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)
        
        # Vérifier que les poids sont correctement appliqués
        expected = (0.15 * MOMENTUM_WEIGHTS["short_term"] + 
                    0.25 * MOMENTUM_WEIGHTS["medium_term"] + 
                    0.35 * MOMENTUM_WEIGHTS["long_term"])
        self.assertAlmostEqual(result, expected)

    def test_calculate_with_missing_data(self):
        """Test du calcul avec des données manquantes"""
        # Créer des données avec des valeurs fondamentales manquantes
        incomplete_data = {
            'historical_prices': self.historical_prices,
            'fundamentals': {
                'EPS': 10.5
                # Revenue et données précédentes manquantes
            }
        }
        
        # Appeler la méthode testée
        result = self.calculator.calculate_momentum_score(incomplete_data)
        
        # Vérifier que le calcul fonctionne toujours
        self.assertIsInstance(result, dict)
        self.assertIn("price_momentum", result)
        self.assertIn("volume_momentum", result)
        self.assertIn("total_score", result)
        
        # Le momentum fondamental devrait être None ou 0
        if "fundamental_momentum" in result:
            self.assertIn(result["fundamental_momentum"]["earnings_growth"], [None, 0])
            self.assertIn(result["fundamental_momentum"]["revenue_growth"], [None, 0])

if __name__ == '__main__':
    unittest.main()

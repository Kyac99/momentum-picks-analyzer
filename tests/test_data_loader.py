"""
Tests unitaires pour le module de chargement de données
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

from data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Tests pour la classe DataLoader"""

    def setUp(self):
        """Initialisation avant chaque test"""
        self.api_key = "test_api_key"
        self.loader = DataLoader(api_key=self.api_key)

    @patch('data_loader.requests.get')
    def test_get_index_constituents(self, mock_get):
        """Test de récupération des composants d'un indice"""
        # Configurer le mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
                {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology"}
            ]
        }
        mock_get.return_value = mock_response

        # Appeler la méthode testée
        result = self.loader.get_index_constituents("SP500")

        # Vérifier les résultats
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["symbol"], "AAPL")
        self.assertEqual(result[1]["name"], "Microsoft Corp.")

    @patch('data_loader.requests.get')
    def test_get_historical_prices(self, mock_get):
        """Test de récupération des prix historiques"""
        # Configurer le mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2025-05-01": {
                    "1. open": "150.0",
                    "2. high": "152.0",
                    "3. low": "149.0",
                    "4. close": "151.0",
                    "5. volume": "100000"
                },
                "2025-04-30": {
                    "1. open": "148.0",
                    "2. high": "149.0",
                    "3. low": "147.0",
                    "4. close": "148.5",
                    "5. volume": "95000"
                }
            }
        }
        mock_get.return_value = mock_response

        # Appeler la méthode testée
        result = self.loader.get_historical_prices("AAPL")

        # Vérifier les résultats
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result.iloc[0]["close"], 151.0)
        self.assertEqual(result.iloc[1]["volume"], 95000)

    @patch('data_loader.requests.get')
    def test_get_fundamental_data(self, mock_get):
        """Test de récupération des données fondamentales"""
        # Configurer le mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Symbol": "AAPL",
            "AssetType": "Common Stock",
            "Name": "Apple Inc",
            "Description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
            "ROE": "1.789",
            "ProfitMargin": "0.254",
            "DebtToEquity": "2.023",
            "OperatingMarginTTM": "0.305"
        }
        mock_get.return_value = mock_response

        # Appeler la méthode testée
        result = self.loader.get_fundamental_data("AAPL")

        # Vérifier les résultats
        self.assertIsInstance(result, dict)
        self.assertEqual(result["Name"], "Apple Inc")
        self.assertAlmostEqual(result["ROE"], 1.789)
        self.assertAlmostEqual(result["ProfitMargin"], 0.254)

    @patch.object(DataLoader, 'get_index_constituents')
    @patch.object(DataLoader, 'get_historical_prices')
    @patch.object(DataLoader, 'get_fundamental_data')
    def test_load_data_for_index(self, mock_get_fundamental, mock_get_prices, mock_get_constituents):
        """Test du chargement complet des données pour un indice"""
        # Configurer les mocks
        mock_get_constituents.return_value = [
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"}
        ]
        
        mock_prices = pd.DataFrame({
            'open': [150.0], 
            'high': [152.0], 
            'low': [149.0], 
            'close': [151.0], 
            'volume': [100000]
        }, index=[datetime(2025, 5, 1)])
        mock_get_prices.return_value = mock_prices
        
        mock_get_fundamental.return_value = {
            "ROE": 1.789,
            "ProfitMargin": 0.254,
            "DebtToEquity": 2.023,
            "OperatingMarginTTM": 0.305
        }

        # Appeler la méthode testée
        result = self.loader.load_data_for_index("SP500", with_fundamentals=True)

        # Vérifier les résultats
        self.assertIn("AAPL", result)
        self.assertEqual(result["AAPL"]["name"], "Apple Inc.")
        self.assertEqual(result["AAPL"]["sector"], "Technology")
        self.assertIn("historical_prices", result["AAPL"])
        self.assertIn("fundamentals", result["AAPL"])
        self.assertAlmostEqual(result["AAPL"]["fundamentals"]["ROE"], 1.789)

if __name__ == '__main__':
    unittest.main()

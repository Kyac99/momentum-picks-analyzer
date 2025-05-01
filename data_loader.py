"""
Module de chargement des données financières
Utilise l'API Alpha Vantage pour récupérer des données boursières
"""

import os
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

class DataLoader:
    """
    Classe pour charger les données financières depuis Alpha Vantage
    """
    
    def __init__(self, api_key=None):
        """
        Initialise le DataLoader avec la clé API Alpha Vantage
        
        Parameters:
            api_key (str): Clé API Alpha Vantage. Si None, cherche dans les variables d'environnement
        """
        self.api_key = api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Clé API Alpha Vantage non fournie. Utilisez le paramètre api_key ou définissez la variable d'environnement ALPHA_VANTAGE_API_KEY")
        
        self.base_url = 'https://www.alphavantage.co/query'
        self.indices = {
            "SP500": "^GSPC",
            "NASDAQ": "^IXIC",
            "EUROSTOXX50": "^STOXX50E",
            "MSCI_TECH": "XLK"  # ETF représentant la technologie, approximation pour MSCI World Tech
        }
    
    def get_index_components(self, index_name):
        """
        Récupère les composants d'un indice donné
        
        Parameters:
            index_name (str): Nom de l'indice (SP500, NASDAQ, etc.)
            
        Returns:
            list: Liste des symboles des composants de l'indice
        """
        # Cette fonction nécessiterait normalement une source de données plus spécifique
        # Comme spécifié, nous pourrions utiliser une API dédiée ou un dataset préexistant
        
        # Pour l'exemple, on retourne une liste fictive pour chaque indice
        index_components = {
            "SP500": ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "BRK.B", "JNJ", "JPM", "V", "PG"],
            "NASDAQ": ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "PYPL", "ADBE", "CMCSA"],
            "EUROSTOXX50": ["ADS.DE", "AIR.PA", "ALV.DE", "ASML.AS", "CS.PA", "BAS.DE", "BAYN.DE", "BMW.DE", "BNP.PA", "DAI.DE"],
            "MSCI_TECH": ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CSCO", "CRM", "ORCL", "ACN", "IBM"]
        }
        
        # Dans une implémentation réelle, on utiliserait une API ou un dataset
        return index_components.get(index_name, [])

    def get_stock_data(self, symbol, output_size='compact'):
        """
        Récupère les données historiques pour un symbole spécifique
        
        Parameters:
            symbol (str): Symbole de l'action
            output_size (str): 'compact' pour 100 derniers jours, 'full' pour 20 ans
            
        Returns:
            pd.DataFrame: DataFrame avec les données historiques
        """
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': output_size,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Error Message' in data:
                print(f"Erreur lors de la récupération des données pour {symbol}: {data['Error Message']}")
                return None
            
            if 'Time Series (Daily)' not in data:
                print(f"Pas de données disponibles pour {symbol}")
                return None
            
            # Conversion en DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Renommage des colonnes
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })
            
            # Conversion des types
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Conversion de l'index en datetime
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            print(f"Erreur lors de la récupération des données pour {symbol}: {str(e)}")
            return None
    
    def get_fundamental_data(self, symbol):
        """
        Récupère les données fondamentales pour un symbole spécifique
        
        Parameters:
            symbol (str): Symbole de l'action
            
        Returns:
            dict: Dictionnaire contenant les indicateurs fondamentaux
        """
        # Récupération de l'aperçu de l'entreprise
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if not data or 'Symbol' not in data:
                print(f"Pas de données fondamentales disponibles pour {symbol}")
                return None
            
            # Extraction des métriques pertinentes
            fundamental_data = {
                'ROE': float(data.get('ReturnOnEquityTTM', 0)),
                'ProfitMargin': float(data.get('ProfitMargin', 0)),
                'OperatingMarginTTM': float(data.get('OperatingMarginTTM', 0)),
                'EPS': float(data.get('EPS', 0)),
                'PERatio': float(data.get('PERatio', 0)),
                'PEGRatio': float(data.get('PEGRatio', 0)),
                'MarketCap': float(data.get('MarketCapitalization', 0)),
                'DebtToEquity': float(data.get('DebtToEquityRatio', 0) if 'DebtToEquityRatio' in data else 0)
            }
            
            return fundamental_data
            
        except Exception as e:
            print(f"Erreur lors de la récupération des données fondamentales pour {symbol}: {str(e)}")
            return None
    
    def get_earnings_data(self, symbol):
        """
        Récupère les données de bénéfices pour un symbole spécifique
        
        Parameters:
            symbol (str): Symbole de l'action
            
        Returns:
            pd.DataFrame: DataFrame avec les données de bénéfices
        """
        params = {
            'function': 'EARNINGS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'quarterlyEarnings' not in data:
                print(f"Pas de données de bénéfices disponibles pour {symbol}")
                return None
            
            # Conversion en DataFrame
            earnings = data['quarterlyEarnings']
            df = pd.DataFrame(earnings)
            
            # Conversion des types
            for col in ['reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            # Conversion des dates
            if 'fiscalDateEnding' in df.columns:
                df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            
            return df
            
        except Exception as e:
            print(f"Erreur lors de la récupération des données de bénéfices pour {symbol}: {str(e)}")
            return None
    
    def load_data_for_index(self, index_name, with_fundamentals=True):
        """
        Charge les données pour tous les composants d'un indice
        
        Parameters:
            index_name (str): Nom de l'indice
            with_fundamentals (bool): Si True, récupère aussi les données fondamentales
            
        Returns:
            dict: Dictionnaire avec les données pour chaque symbole
        """
        components = self.get_index_components(index_name)
        data = {}
        
        print(f"Chargement des données pour l'indice {index_name} ({len(components)} composants)")
        
        for i, symbol in enumerate(components):
            print(f"[{i+1}/{len(components)}] Chargement des données pour {symbol}")
            
            # Récupération des données historiques
            stock_data = self.get_stock_data(symbol, output_size='full')
            
            if stock_data is not None:
                data[symbol] = {
                    'historical': stock_data
                }
                
                # Récupération des données fondamentales si demandé
                if with_fundamentals:
                    fundamental_data = self.get_fundamental_data(symbol)
                    earnings_data = self.get_earnings_data(symbol)
                    
                    if fundamental_data:
                        data[symbol]['fundamental'] = fundamental_data
                    
                    if earnings_data is not None:
                        data[symbol]['earnings'] = earnings_data
            
            # Pause pour respecter les limites de l'API
            time.sleep(12)  # Alpha Vantage limite à 5 requêtes par minute pour les clés gratuites
        
        return data

if __name__ == "__main__":
    # Test simple du module
    api_key = input("Entrez votre clé API Alpha Vantage: ")
    loader = DataLoader(api_key=api_key)
    
    # Test de récupération de données pour un symbole
    symbol = "AAPL"
    print(f"Récupération des données pour {symbol}")
    
    stock_data = loader.get_stock_data(symbol)
    if stock_data is not None:
        print(stock_data.head())
        
    fundamental_data = loader.get_fundamental_data(symbol)
    if fundamental_data:
        for key, value in fundamental_data.items():
            print(f"{key}: {value}")

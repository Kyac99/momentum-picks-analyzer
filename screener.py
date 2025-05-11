"""
Module de screening d'actions
Combine les scores de momentum et qualité pour sélectionner les meilleures opportunités
"""

import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime

from config import SCREENER_CONFIG
from utils import setup_logger, rank_percentile, generate_timestamp
from momentum import MomentumCalculator
from quality import QualityCalculator

# Configuration du logger
logger = setup_logger(__name__, "screener.log")

class StockScreener:
    """
    Classe pour filtrer et sélectionner les meilleures opportunités d'investissement
    """
    
    def __init__(self, momentum_weight=0.6, quality_weight=0.4, config=None):
        """
        Initialise le screener avec la configuration spécifiée
        
        Parameters:
            momentum_weight (float): Poids du score de momentum dans le score combiné
            quality_weight (float): Poids du score de qualité dans le score combiné
            config (dict): Configuration du screener
        """
        self.config = config or SCREENER_CONFIG
        self.momentum_weight = momentum_weight
        self.quality_weight = quality_weight
        self.momentum_calculator = MomentumCalculator()
        self.quality_calculator = QualityCalculator()
    
    def apply_filters(self, stocks_data):
        """
        Applique les filtres de base aux données des actions
        
        Parameters:
            stocks_data (dict): Dictionnaire des données pour chaque action
            
        Returns:
            dict: Dictionnaire filtré des actions
        """
        filtered_stocks = {}
        
        for symbol, data in stocks_data.items():
            try:
                # Vérification de la présence des données nécessaires
                if 'fundamentals' not in data:
                    logger.warning(f"Données fondamentales manquantes pour {symbol}, ignoré.")
                    continue
                
                fundamentals = data['fundamentals']
                
                # Application des filtres de la configuration
                if 'min_market_cap' in self.config:
                    if 'MarketCapitalization' not in fundamentals or float(fundamentals.get('MarketCapitalization', 0)) < self.config['min_market_cap']:
                        logger.info(f"{symbol} filtré: capitalisation boursière trop faible")
                        continue
                
                if 'max_pe_ratio' in self.config:
                    if 'PERatio' not in fundamentals or float(fundamentals.get('PERatio', 0)) <= 0 or float(fundamentals.get('PERatio', 0)) > self.config['max_pe_ratio']:
                        logger.info(f"{symbol} filtré: PE ratio hors limite")
                        continue
                
                if 'min_roe' in self.config:
                    if 'ROE' not in fundamentals or float(fundamentals.get('ROE', 0)) < self.config['min_roe']:
                        logger.info(f"{symbol} filtré: ROE trop faible")
                        continue
                
                if 'max_debt_to_equity' in self.config:
                    if 'DebtToEquity' not in fundamentals or float(fundamentals.get('DebtToEquity', 0)) > self.config['max_debt_to_equity']:
                        logger.info(f"{symbol} filtré: ratio Dette/Fonds propres trop élevé")
                        continue
                
                if 'min_operating_margin' in self.config:
                    if 'OperatingMarginTTM' not in fundamentals or float(fundamentals.get('OperatingMarginTTM', 0)) < self.config['min_operating_margin']:
                        logger.info(f"{symbol} filtré: marge opérationnelle trop faible")
                        continue
                
                # L'action a passé tous les filtres
                filtered_stocks[symbol] = data
                
            except Exception as e:
                logger.error(f"Erreur lors du filtrage de {symbol}: {str(e)}")
        
        logger.info(f"{len(filtered_stocks)} actions ont passé les filtres sur {len(stocks_data)} au total")
        return filtered_stocks
    
    def screen_stocks(self, stocks_data, min_momentum=0.0, min_quality=0.0, min_combined=0.0, top_n=20):
        """
        Calcule les scores et sélectionne les meilleures actions
        
        Parameters:
            stocks_data (dict): Dictionnaire des données pour chaque action
            min_momentum (float): Score minimum de momentum (0-1)
            min_quality (float): Score minimum de qualité (0-1)
            min_combined (float): Score minimum combiné (0-1)
            top_n (int): Nombre d'actions à sélectionner
            
        Returns:
            tuple: (DataFrame des résultats, liste triée des symboles d'actions)
        """
        results = []
        
        for symbol, data in stocks_data.items():
            try:
                # Calcul du score de momentum
                momentum_result = self.momentum_calculator.calculate_momentum_score(data)
                momentum_score = momentum_result.get('total_score', 0.0)
                
                # Calcul du score de qualité
                quality_result = self.quality_calculator.calculate_quality_score(data)
                quality_score = quality_result.get('total_score', 0.0)
                
                # Calcul du score combiné
                if not np.isnan(momentum_score) and not np.isnan(quality_score):
                    combined_score = (
                        self.momentum_weight * momentum_score + 
                        self.quality_weight * quality_score
                    )
                elif not np.isnan(momentum_score):
                    combined_score = momentum_score
                elif not np.isnan(quality_score):
                    combined_score = quality_score
                else:
                    combined_score = np.nan
                
                # Extraction des informations d'entreprise
                name = data.get('name', symbol)
                sector = data.get('sector', 'Unknown')
                industry = data.get('industry', 'Unknown')
                
                # Création d'une entrée pour les résultats
                result = {
                    'Symbol': symbol,
                    'Name': name,
                    'Sector': sector,
                    'Industry': industry,
                    'Momentum_Score': momentum_score,
                    'Quality_Score': quality_score,
                    'Combined_Score': combined_score
                }
                
                # Ajout des résultats détaillés si disponibles
                if momentum_result and 'price_momentum' in momentum_result:
                    result['Technical_Momentum'] = momentum_result['price_momentum'].get('score', np.nan)
                if momentum_result and 'fundamental_momentum' in momentum_result:
                    result['Fundamental_Momentum'] = momentum_result['fundamental_momentum'].get('score', np.nan)
                
                # Ajout des métriques fondamentales clés
                if 'fundamentals' in data:
                    fundamentals = data['fundamentals']
                    result['Market_Cap'] = fundamentals.get('MarketCapitalization', np.nan)
                    result['PE_Ratio'] = fundamentals.get('PERatio', np.nan)
                    result['ROE'] = fundamentals.get('ROE', np.nan)
                    result['Profit_Margin'] = fundamentals.get('ProfitMargin', np.nan)
                    result['Debt_To_Equity'] = fundamentals.get('DebtToEquity', np.nan)
                    result['Dividend_Yield'] = fundamentals.get('DividendYield', np.nan)
                    result['EPS'] = fundamentals.get('EPS', np.nan)
                    result['Beta'] = fundamentals.get('Beta', np.nan)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Erreur lors du calcul des scores pour {symbol}: {str(e)}")
        
        # Conversion en DataFrame
        if not results:
            return pd.DataFrame(), []
        
        df = pd.DataFrame(results)
        
        # Application des filtres de score minimum
        filtered_df = df.copy()
        if min_momentum > 0:
            filtered_df = filtered_df[filtered_df['Momentum_Score'] >= min_momentum]
        if min_quality > 0:
            filtered_df = filtered_df[filtered_df['Quality_Score'] >= min_quality]
        if min_combined > 0:
            filtered_df = filtered_df[filtered_df['Combined_Score'] >= min_combined]
        
        # Tri par score combiné
        if 'Combined_Score' in filtered_df.columns:
            sorted_df = filtered_df.sort_values('Combined_Score', ascending=False)
        else:
            sorted_df = filtered_df
        
        # Sélection du nombre spécifié d'actions
        top_picks = sorted_df.head(top_n)
        sorted_symbols = top_picks['Symbol'].tolist()
        
        logger.info(f"Sélection de {len(top_picks)} actions sur {len(df)} analysées")
        return top_picks, sorted_symbols
    
    def save_results(self, results_df, output_dir='results'):
        """
        Sauvegarde les résultats du screening dans un fichier CSV
        
        Parameters:
            results_df (pd.DataFrame): DataFrame des résultats
            output_dir (str): Répertoire de sortie
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        # S'assurer que le répertoire existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Timestamp pour le nom de fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"momentum_picks_{timestamp}.csv"
        
        # Chemin complet du fichier
        filepath = os.path.join(output_dir, filename)
        
        # Sauvegarde des résultats
        try:
            results_df.to_csv(filepath, index=False)
            logger.info(f"Résultats sauvegardés dans {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
            return None
    
    def generate_report(self, results_df, sector_breakdown=True, output_format='text'):
        """
        Génère un rapport à partir des résultats du screening
        
        Parameters:
            results_df (pd.DataFrame): DataFrame des résultats
            sector_breakdown (bool): Si True, inclut une analyse par secteur
            output_format (str): Format de sortie ('text', 'html', 'markdown')
            
        Returns:
            str: Rapport formaté
        """
        if results_df.empty:
            return "Aucune action sélectionnée. Veuillez exécuter le screening d'abord."
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if output_format == 'text':
            # Rapport en format texte
            report = []
            report.append("=" * 80)
            report.append(f"RAPPORT DE SCREENING MOMENTUM-QUALITY - {timestamp}")
            report.append("=" * 80)
            report.append("")
            
            report.append(f"Nombre d'actions sélectionnées: {len(results_df)}")
            report.append("")
            
            report.append("TOP PICKS:")
            report.append("-" * 80)
            
            # Formatage des colonnes
            for i, (_, row) in enumerate(results_df.iterrows(), 1):
                symbol = row['Symbol']
                name = row.get('Name', 'N/A')
                sector = row.get('Sector', 'N/A')
                momentum = row.get('Momentum_Score', 0) * 100
                quality = row.get('Quality_Score', 0) * 100
                combined = row.get('Combined_Score', 0) * 100
                
                report.append(f"{i}. {symbol} - {name} ({sector})")
                report.append(f"   Momentum: {momentum:.1f}%, Quality: {quality:.1f}%, Combined: {combined:.1f}%")
                report.append("")
            
            # Analyse par secteur si demandée
            if sector_breakdown and 'Sector' in results_df.columns:
                report.append("")
                report.append("RÉPARTITION PAR SECTEUR:")
                report.append("-" * 80)
                
                sector_counts = results_df['Sector'].value_counts()
                for sector, count in sector_counts.items():
                    percentage = count / len(results_df) * 100
                    report.append(f"{sector}: {count} actions ({percentage:.1f}%)")
                
                report.append("")
            
            return "\n".join(report)
            
        elif output_format == 'markdown':
            # Génération d'un rapport en Markdown
            md = []
            md.append(f"# Rapport de Screening Momentum-Quality - {timestamp}")
            md.append("")
            
            md.append(f"Nombre d'actions sélectionnées: **{len(results_df)}**")
            md.append("")
            
            md.append("## Top Picks")
            md.append("")
            md.append("| Rang | Symbole | Nom | Secteur | Momentum | Qualité | Score Combiné |")
            md.append("|------|---------|-----|---------|----------|---------|---------------|")
            
            for i, (_, row) in enumerate(results_df.iterrows(), 1):
                symbol = row['Symbol']
                name = row.get('Name', 'N/A')
                sector = row.get('Sector', 'N/A')
                momentum = row.get('Momentum_Score', 0) * 100
                quality = row.get('Quality_Score', 0) * 100
                combined = row.get('Combined_Score', 0) * 100
                
                md.append(f"| {i} | {symbol} | {name} | {sector} | {momentum:.1f}% | {quality:.1f}% | {combined:.1f}% |")
            
            md.append("")
            
            # Analyse par secteur si demandée
            if sector_breakdown and 'Sector' in results_df.columns:
                md.append("## Répartition par Secteur")
                md.append("")
                md.append("| Secteur | Nombre d'actions | Pourcentage |")
                md.append("|---------|------------------|-------------|")
                
                sector_counts = results_df['Sector'].value_counts()
                for sector, count in sector_counts.items():
                    percentage = count / len(results_df) * 100
                    md.append(f"| {sector} | {count} | {percentage:.1f}% |")
                
                md.append("")
            
            return "\n".join(md)
            
        elif output_format == 'html':
            # Génération d'un rapport HTML basique
            html = []
            html.append("<html><head><title>Rapport de Screening Momentum-Quality</title></head>")
            html.append("<style>body { font-family: Arial; margin: 20px; } "
                        "table { border-collapse: collapse; width: 100%; } "
                        "th, td { border: 1px solid #ddd; padding: 8px; } "
                        "th { background-color: #f2f2f2; } "
                        "tr:nth-child(even) { background-color: #f9f9f9; } "
                        "h1, h2 { color: #333; }</style>")
            html.append("<body>")
            
            html.append(f"<h1>Rapport de Screening Momentum-Quality - {timestamp}</h1>")
            
            html.append(f"<p>Nombre d'actions sélectionnées: {len(results_df)}</p>")
            
            html.append("<h2>Top Picks</h2>")
            html.append("<table>")
            html.append("<tr><th>Rang</th><th>Symbole</th><th>Nom</th><th>Secteur</th>"
                        "<th>Momentum</th><th>Qualité</th><th>Score Combiné</th></tr>")
            
            for i, (_, row) in enumerate(results_df.iterrows(), 1):
                symbol = row['Symbol']
                name = row.get('Name', 'N/A')
                sector = row.get('Sector', 'N/A')
                momentum = row.get('Momentum_Score', 0) * 100
                quality = row.get('Quality_Score', 0) * 100
                combined = row.get('Combined_Score', 0) * 100
                
                html.append(f"<tr><td>{i}</td><td>{symbol}</td><td>{name}</td><td>{sector}</td>"
                            f"<td>{momentum:.1f}%</td><td>{quality:.1f}%</td><td>{combined:.1f}%</td></tr>")
            
            html.append("</table>")
            
            # Analyse par secteur si demandée
            if sector_breakdown and 'Sector' in results_df.columns:
                html.append("<h2>Répartition par Secteur</h2>")
                html.append("<table>")
                html.append("<tr><th>Secteur</th><th>Nombre d'actions</th><th>Pourcentage</th></tr>")
                
                sector_counts = results_df['Sector'].value_counts()
                for sector, count in sector_counts.items():
                    percentage = count / len(results_df) * 100
                    html.append(f"<tr><td>{sector}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>")
                
                html.append("</table>")
            
            html.append("</body></html>")
            
            return "\n".join(html)
            
        else:
            return "Format de sortie non supporté. Utilisez 'text', 'html' ou 'markdown'."


if __name__ == "__main__":
    # Test simple du module
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # Création de données de test
    test_stocks = {}
    for i, symbol in enumerate(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']):
        # Données historiques
        dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
        
        # Tendance à la hausse avec des variations aléatoires
        close_prices = [100 + i * 0.5 + j * 0.2 + np.random.normal(0, 2) for j in range(300)]
        
        historical_prices = pd.DataFrame({
            'close': close_prices,
            'high': [p + 2 for p in close_prices],
            'low': [p - 2 for p in close_prices],
            'open': [p - 1 for p in close_prices],
            'volume': [1000000 + np.random.randint(0, 500000) for _ in range(300)]
        }, index=dates)
        
        # Données fondamentales
        fundamentals = {
            'Symbol': symbol,
            'Name': f"Test Company {i+1}",
            'Sector': ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Industrial'][i % 5],
            'Industry': 'Software',
            'MarketCapitalization': 1e9 * (10 + i),
            'PERatio': 20 + i,
            'ROE': 0.15 + i * 0.01,
            'ProfitMargin': 0.1 + i * 0.01,
            'OperatingMarginTTM': 0.2 + i * 0.01,
            'DebtToEquity': 0.5 + i * 0.1,
            'DividendYield': 0.02,
            'EPS': 5 + i,
            'Beta': 1.0 + i * 0.1
        }
        
        test_stocks[symbol] = {
            'historical_prices': historical_prices,
            'fundamentals': fundamentals,
            'name': f"Test Company {i+1}",
            'sector': ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Industrial'][i % 5],
            'industry': 'Software'
        }
    
    # Initialisation du screener
    screener = StockScreener()
    
    # Exécution du screening
    results_df, sorted_symbols = screener.screen_stocks(test_stocks)
    
    # Affichage des résultats
    if not results_df.empty:
        print("\nMeilleures opportunités:")
        print(results_df[['Symbol', 'Name', 'Sector', 'Momentum_Score', 'Quality_Score', 'Combined_Score']])
        
        # Génération d'un rapport
        report = screener.generate_report(results_df)
        print("\nRapport:")
        print(report)
    else:
        print("Aucune action sélectionnée.")

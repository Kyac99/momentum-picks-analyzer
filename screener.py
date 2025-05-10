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
from momentum import MomentumAnalyzer
from quality import QualityAnalyzer

# Configuration du logger
logger = setup_logger(__name__, "screener.log")

class StockScreener:
    """
    Classe pour filtrer et sélectionner les meilleures opportunités d'investissement
    """
    
    def __init__(self, config=None):
        """
        Initialise le screener avec la configuration spécifiée
        
        Parameters:
            config (dict): Configuration du screener
        """
        self.config = config or SCREENER_CONFIG
        self.momentum_analyzer = MomentumAnalyzer()
        self.quality_analyzer = QualityAnalyzer()
    
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
                if 'fundamental' not in data or 'overview' not in data['fundamental']:
                    logger.warning(f"Données fondamentales manquantes pour {symbol}, ignoré.")
                    continue
                
                overview = data['fundamental']['overview']
                
                # Application des filtres de la configuration
                if 'min_market_cap' in self.config:
                    if 'MarketCap' not in overview or overview['MarketCap'] < self.config['min_market_cap']:
                        logger.info(f"{symbol} filtré: capitalisation boursière trop faible")
                        continue
                
                if 'max_pe_ratio' in self.config:
                    if 'PERatio' not in overview or overview['PERatio'] <= 0 or overview['PERatio'] > self.config['max_pe_ratio']:
                        logger.info(f"{symbol} filtré: PE ratio hors limite")
                        continue
                
                if 'min_roe' in self.config:
                    if 'ROE' not in overview or overview['ROE'] < self.config['min_roe']:
                        logger.info(f"{symbol} filtré: ROE trop faible")
                        continue
                
                if 'max_debt_to_equity' in self.config:
                    if 'DebtToEquity' not in overview or overview['DebtToEquity'] > self.config['max_debt_to_equity']:
                        logger.info(f"{symbol} filtré: ratio Dette/Fonds propres trop élevé")
                        continue
                
                if 'min_operating_margin' in self.config:
                    if 'OperatingMarginTTM' not in overview or overview['OperatingMarginTTM'] < self.config['min_operating_margin']:
                        logger.info(f"{symbol} filtré: marge opérationnelle trop faible")
                        continue
                
                # L'action a passé tous les filtres
                filtered_stocks[symbol] = data
                
            except Exception as e:
                logger.error(f"Erreur lors du filtrage de {symbol}: {str(e)}")
        
        logger.info(f"{len(filtered_stocks)} actions ont passé les filtres sur {len(stocks_data)} au total")
        return filtered_stocks
    
    def calculate_scores(self, stocks_data):
        """
        Calcule les scores de momentum et qualité pour chaque action
        
        Parameters:
            stocks_data (dict): Dictionnaire des données pour chaque action
            
        Returns:
            pd.DataFrame: DataFrame avec les scores calculés
        """
        results = []
        
        for symbol, data in stocks_data.items():
            try:
                # Extraction des données nécessaires
                historical_data = data.get('historical', None)
                fundamental_data = data.get('fundamental', None)
                
                # Calcul des scores de momentum
                momentum_result = self.momentum_analyzer.calculate_combined_momentum(
                    historical_data, fundamental_data
                )
                
                # Calcul des scores de qualité
                quality_result = self.quality_analyzer.calculate_quality_score(fundamental_data)
                
                # Extraction de l'information d'entreprise
                company_info = {}
                if fundamental_data and 'overview' in fundamental_data:
                    overview = fundamental_data['overview']
                    company_info = {
                        'Name': overview.get('Name', symbol),
                        'Sector': overview.get('Sector', 'Unknown'),
                        'Industry': overview.get('Industry', 'Unknown'),
                        'MarketCap': overview.get('MarketCap', np.nan),
                        'PERatio': overview.get('PERatio', np.nan),
                        'ROE': overview.get('ROE', np.nan),
                        'ProfitMargin': overview.get('ProfitMargin', np.nan),
                        'DebtToEquity': overview.get('DebtToEquity', np.nan),
                        'DividendYield': overview.get('DividendYield', np.nan),
                        'EPS': overview.get('EPS', np.nan),
                        'Beta': overview.get('Beta', np.nan)
                    }
                
                # Création d'une entrée pour les résultats
                result = {
                    'Symbol': symbol,
                    'MomentumScore': momentum_result['score'],
                    'QualityScore': quality_result['score'],
                    'TechnicalMomentum': momentum_result['technical']['score'],
                    'FundamentalMomentum': momentum_result['fundamental']['score'],
                    **company_info
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Erreur lors du calcul des scores pour {symbol}: {str(e)}")
        
        # Conversion en DataFrame
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Calcul des percentiles pour les scores
        if 'MomentumScore' in df.columns and not df['MomentumScore'].isna().all():
            df['MomentumPercentile'] = rank_percentile(df['MomentumScore'].values)
        else:
            df['MomentumPercentile'] = np.nan
            
        if 'QualityScore' in df.columns and not df['QualityScore'].isna().all():
            df['QualityPercentile'] = rank_percentile(df['QualityScore'].values)
        else:
            df['QualityPercentile'] = np.nan
        
        # Calcul du score combiné (60% Momentum, 40% Qualité)
        if 'MomentumPercentile' in df.columns and 'QualityPercentile' in df.columns:
            valid_mask = ~df['MomentumPercentile'].isna() & ~df['QualityPercentile'].isna()
            
            df['CombinedScore'] = np.nan
            df.loc[valid_mask, 'CombinedScore'] = (
                0.6 * df.loc[valid_mask, 'MomentumPercentile'] + 
                0.4 * df.loc[valid_mask, 'QualityPercentile']
            ) / 100.0  # Normalisation entre 0 et 1
        
        return df
    
    def select_top_picks(self, scores_df):
        """
        Sélectionne les meilleures opportunités d'investissement
        
        Parameters:
            scores_df (pd.DataFrame): DataFrame avec les scores calculés
            
        Returns:
            pd.DataFrame: DataFrame avec les meilleures opportunités
        """
        if scores_df.empty:
            logger.warning("Pas de données de scores disponibles pour la sélection")
            return pd.DataFrame()
        
        # Application des seuils de momentum et qualité
        filtered_df = scores_df.copy()
        
        if 'momentum_threshold' in self.config:
            momentum_min = self.config['momentum_threshold'] * 100
            filtered_df = filtered_df[filtered_df['MomentumPercentile'] >= momentum_min]
        
        if 'quality_threshold' in self.config:
            quality_min = self.config['quality_threshold'] * 100
            filtered_df = filtered_df[filtered_df['QualityPercentile'] >= quality_min]
        
        if filtered_df.empty:
            logger.warning("Aucune action ne répond aux critères de seuil de momentum et qualité")
            return pd.DataFrame()
        
        # Tri par score combiné
        if 'CombinedScore' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('CombinedScore', ascending=False)
        
        # Sélection du nombre spécifié d'actions
        top_count = self.config.get('top_picks_count', 20)
        top_picks = filtered_df.head(top_count)
        
        logger.info(f"Sélection de {len(top_picks)} meilleures opportunités sur {len(scores_df)} actions analysées")
        return top_picks
    
    def run_screening(self, stocks_data):
        """
        Exécute le processus complet de screening
        
        Parameters:
            stocks_data (dict): Dictionnaire des données pour chaque action
            
        Returns:
            dict: Résultats du screening incluant les données filtrées et les meilleures opportunités
        """
        logger.info(f"Début du screening sur {len(stocks_data)} actions")
        
        # Étape 1: Application des filtres de base
        filtered_stocks = self.apply_filters(stocks_data)
        
        # Étape 2: Calcul des scores de momentum et qualité
        scores_df = self.calculate_scores(filtered_stocks)
        
        # Étape 3: Sélection des meilleures opportunités
        top_picks = self.select_top_picks(scores_df)
        
        return {
            'filtered_stocks': filtered_stocks,
            'scores': scores_df,
            'top_picks': top_picks,
            'timestamp': generate_timestamp(),
            'config': self.config
        }
    
    def save_results(self, results, filename=None):
        """
        Sauvegarde les résultats du screening
        
        Parameters:
            results (dict): Résultats du screening
            filename (str): Nom du fichier pour la sauvegarde
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        from config import RESULTS_DIR
        
        if filename is None:
            timestamp = results.get('timestamp', generate_timestamp())
            filename = f"screening_results_{timestamp}.pkl"
        
        # S'assurer que le répertoire existe
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Chemin complet du fichier
        filepath = os.path.join(RESULTS_DIR, filename)
        
        # Sauvegarde des résultats
        try:
            # On sauvegarde seulement les DataFrames et la configuration
            save_data = {
                'scores': results.get('scores', pd.DataFrame()),
                'top_picks': results.get('top_picks', pd.DataFrame()),
                'timestamp': results.get('timestamp', generate_timestamp()),
                'config': results.get('config', {})
            }
            
            pd.to_pickle(save_data, filepath)
            logger.info(f"Résultats sauvegardés dans {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats: {str(e)}")
            return None
    
    def load_results(self, filepath):
        """
        Charge les résultats d'un screening précédent
        
        Parameters:
            filepath (str): Chemin du fichier à charger
            
        Returns:
            dict: Résultats du screening
        """
        try:
            results = pd.read_pickle(filepath)
            logger.info(f"Résultats chargés depuis {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des résultats: {str(e)}")
            return None
    
    def generate_report(self, results, sector_breakdown=True, output_format='text'):
        """
        Génère un rapport à partir des résultats du screening
        
        Parameters:
            results (dict): Résultats du screening
            sector_breakdown (bool): Si True, inclut une analyse par secteur
            output_format (str): Format de sortie ('text', 'html', 'markdown')
            
        Returns:
            str: Rapport formaté
        """
        if 'top_picks' not in results or results['top_picks'].empty:
            return "Aucune action sélectionnée. Veuillez exécuter le screening d'abord."
        
        top_picks = results['top_picks']
        timestamp = results.get('timestamp', 'Unknown')
        
        if output_format == 'text':
            # Rapport en format texte
            report = []
            report.append("=" * 80)
            report.append(f"RAPPORT DE SCREENING MOMENTUM-QUALITY - {timestamp}")
            report.append("=" * 80)
            report.append("")
            
            report.append(f"Nombre d'actions sélectionnées: {len(top_picks)}")
            report.append("")
            
            report.append("TOP PICKS:")
            report.append("-" * 80)
            
            # Formatage des colonnes
            for i, (_, row) in enumerate(top_picks.iterrows(), 1):
                symbol = row['Symbol']
                name = row.get('Name', 'N/A')
                sector = row.get('Sector', 'N/A')
                momentum = row.get('MomentumPercentile', 0)
                quality = row.get('QualityPercentile', 0)
                combined = row.get('CombinedScore', 0) * 100
                
                report.append(f"{i}. {symbol} - {name} ({sector})")
                report.append(f"   Momentum: {momentum:.1f}%, Quality: {quality:.1f}%, Combined: {combined:.1f}%")
                report.append("")
            
            # Analyse par secteur si demandée
            if sector_breakdown and 'Sector' in top_picks.columns:
                report.append("")
                report.append("RÉPARTITION PAR SECTEUR:")
                report.append("-" * 80)
                
                sector_counts = top_picks['Sector'].value_counts()
                for sector, count in sector_counts.items():
                    percentage = count / len(top_picks) * 100
                    report.append(f"{sector}: {count} actions ({percentage:.1f}%)")
                
                report.append("")
            
            # Configuration utilisée
            report.append("CONFIGURATION DU SCREENING:")
            report.append("-" * 80)
            
            config = results.get('config', {})
            for key, value in config.items():
                report.append(f"{key}: {value}")
            
            return "\n".join(report)
            
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
            
            html.append(f"<p>Nombre d'actions sélectionnées: {len(top_picks)}</p>")
            
            html.append("<h2>Top Picks</h2>")
            html.append("<table>")
            html.append("<tr><th>Rang</th><th>Symbole</th><th>Nom</th><th>Secteur</th>"
                        "<th>Momentum</th><th>Qualité</th><th>Score Combiné</th></tr>")
            
            for i, (_, row) in enumerate(top_picks.iterrows(), 1):
                symbol = row['Symbol']
                name = row.get('Name', 'N/A')
                sector = row.get('Sector', 'N/A')
                momentum = row.get('MomentumPercentile', 0)
                quality = row.get('QualityPercentile', 0)
                combined = row.get('CombinedScore', 0) * 100
                
                html.append(f"<tr><td>{i}</td><td>{symbol}</td><td>{name}</td><td>{sector}</td>"
                            f"<td>{momentum:.1f}%</td><td>{quality:.1f}%</td><td>{combined:.1f}%</td></tr>")
            
            html.append("</table>")
            
            # Analyse par secteur si demandée
            if sector_breakdown and 'Sector' in top_picks.columns:
                html.append("<h2>Répartition par Secteur</h2>")
                html.append("<table>")
                html.append("<tr><th>Secteur</th><th>Nombre d'actions</th><th>Pourcentage</th></tr>")
                
                sector_counts = top_picks['Sector'].value_counts()
                for sector, count in sector_counts.items():
                    percentage = count / len(top_picks) * 100
                    html.append(f"<tr><td>{sector}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>")
                
                html.append("</table>")
            
            # Configuration
            html.append("<h2>Configuration du Screening</h2>")
            html.append("<table>")
            html.append("<tr><th>Paramètre</th><th>Valeur</th></tr>")
            
            config = results.get('config', {})
            for key, value in config.items():
                html.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
            
            html.append("</table>")
            
            html.append("</body></html>")
            
            return "\n".join(html)
            
        elif output_format == 'markdown':
            # Génération d'un rapport en Markdown
            md = []
            md.append(f"# Rapport de Screening Momentum-Quality - {timestamp}")
            md.append("")
            
            md.append(f"Nombre d'actions sélectionnées: **{len(top_picks)}**")
            md.append("")
            
            md.append("## Top Picks")
            md.append("")
            md.append("| Rang | Symbole | Nom | Secteur | Momentum | Qualité | Score Combiné |")
            md.append("|------|---------|-----|---------|----------|---------|---------------|")
            
            for i, (_, row) in enumerate(top_picks.iterrows(), 1):
                symbol = row['Symbol']
                name = row.get('Name', 'N/A')
                sector = row.get('Sector', 'N/A')
                momentum = row.get('MomentumPercentile', 0)
                quality = row.get('QualityPercentile', 0)
                combined = row.get('CombinedScore', 0) * 100
                
                md.append(f"| {i} | {symbol} | {name} | {sector} | {momentum:.1f}% | {quality:.1f}% | {combined:.1f}% |")
            
            md.append("")
            
            # Analyse par secteur si demandée
            if sector_breakdown and 'Sector' in top_picks.columns:
                md.append("## Répartition par Secteur")
                md.append("")
                md.append("| Secteur | Nombre d'actions | Pourcentage |")
                md.append("|---------|------------------|-------------|")
                
                sector_counts = top_picks['Sector'].value_counts()
                for sector, count in sector_counts.items():
                    percentage = count / len(top_picks) * 100
                    md.append(f"| {sector} | {count} | {percentage:.1f}% |")
                
                md.append("")
            
            # Configuration
            md.append("## Configuration du Screening")
            md.append("")
            md.append("| Paramètre | Valeur |")
            md.append("|-----------|--------|")
            
            config = results.get('config', {})
            for key, value in config.items():
                md.append(f"| {key} | {value} |")
            
            return "\n".join(md)
            
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
        
        historical = pd.DataFrame({
            'close': close_prices,
            'adjusted_close': close_prices,
            'volume': [1000000 + np.random.randint(0, 500000) for _ in range(300)]
        }, index=dates)
        
        # Données fondamentales
        fundamental = {
            'overview': {
                'Symbol': symbol,
                'Name': f"Test Company {i+1}",
                'Sector': ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Industrial'][i % 5],
                'Industry': 'Software',
                'MarketCap': 1e9 * (10 + i),
                'PERatio': 20 + i,
                'ROE': 0.15 + i * 0.01,
                'ProfitMargin': 0.1 + i * 0.01,
                'OperatingMarginTTM': 0.2 + i * 0.01,
                'DebtToEquity': 0.5 + i * 0.1,
                'DividendYield': 0.02,
                'EPS': 5 + i,
                'Beta': 1.0 + i * 0.1
            }
        }
        
        test_stocks[symbol] = {
            'historical': historical,
            'fundamental': fundamental
        }
    
    # Initialisation du screener
    screener = StockScreener()
    
    # Exécution du screening
    results = screener.run_screening(test_stocks)
    
    # Affichage des résultats
    if 'top_picks' in results and not results['top_picks'].empty:
        print("\nMeilleures opportunités:")
        print(results['top_picks'][['Symbol', 'Name', 'Sector', 'MomentumPercentile', 'QualityPercentile', 'CombinedScore']])
        
        # Génération d'un rapport
        report = screener.generate_report(results)
        print("\nRapport:")
        print(report)
    else:
        print("Aucune action sélectionnée.")

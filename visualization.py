"""
Module de visualisation des résultats
Génère des graphiques et visualisations pour l'analyse
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import logging

from config import VIZ_CONFIG, RESULTS_DIR
from utils import setup_logger

# Configuration du logger
logger = setup_logger(__name__, "visualization.log")

class MomentumVisualizer:
    """
    Classe pour générer des visualisations des résultats de l'analyse momentum
    """
    
    def __init__(self, config=None):
        """
        Initialise le visualiseur avec la configuration spécifiée
        
        Parameters:
            config (dict): Configuration de visualisation
        """
        self.config = config or VIZ_CONFIG
        
        # Application du thème
        plt.style.use(self.config.get('theme', 'seaborn-v0_8-darkgrid'))
    
    def plot_price_history(self, historical_data, symbol, name=None, periods=None, save_path=None):
        """
        Trace l'historique des prix et les périodes de momentum
        
        Parameters:
            historical_data (pd.DataFrame): Données historiques de prix
            symbol (str): Symbole de l'action
            name (str): Nom de l'entreprise
            periods (dict): Périodes de momentum à mettre en évidence
            save_path (str): Chemin pour sauvegarder le graphique
            
        Returns:
            plt.Figure: Figure matplotlib générée
        """
        if historical_data is None or historical_data.empty:
            logger.warning(f"Pas de données historiques disponibles pour {symbol}")
            return None
        
        try:
            # Extraction des données de prix
            close_prices = historical_data['adjusted_close'] if 'adjusted_close' in historical_data.columns else historical_data['close']
            
            # Création de la figure
            fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, 8)))
            
            # Traçage de la série de prix
            ax.plot(close_prices.index, close_prices.values, linewidth=2, label='Prix ajusté', color=self.config.get('colors', ['#1f77b4'])[0])
            
            # Formatage des dates sur l'axe x
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            # Mise en évidence des périodes de momentum
            if periods is not None:
                for period_name, period_length in periods.items():
                    if len(close_prices) > period_length:
                        period_start = close_prices.index[-period_length-1]
                        period_end = close_prices.index[-1]
                        
                        # Calcul du rendement sur la période
                        start_price = close_prices.iloc[-period_length-1]
                        end_price = close_prices.iloc[-1]
                        returns = (end_price / start_price - 1) * 100
                        
                        ax.axvspan(period_start, period_end, alpha=0.2, label=f'{period_name}: {returns:.2f}%')
            
            # Ajout des labels et du titre
            display_name = name if name else symbol
            ax.set_title(f"Historique des prix - {display_name} ({symbol})", fontsize=self.config.get('title_fontsize', 16))
            ax.set_xlabel('Date', fontsize=self.config.get('axis_fontsize', 12))
            ax.set_ylabel('Prix ($)', fontsize=self.config.get('axis_fontsize', 12))
            
            # Formatage de l'axe y en dollars
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.2f}'))
            
            # Affichage de la légende
            ax.legend()
            
            # Ajustement des marges
            plt.tight_layout()
            
            # Sauvegarde du graphique si un chemin est spécifié
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Graphique sauvegardé dans {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du traçage de l'historique des prix pour {symbol}: {str(e)}")
            return None
    
    def plot_momentum_breakdown(self, momentum_result, symbol, name=None, save_path=None):
        """
        Trace la répartition des scores de momentum
        
        Parameters:
            momentum_result (dict): Résultat de l'analyse de momentum
            symbol (str): Symbole de l'action
            name (str): Nom de l'entreprise
            save_path (str): Chemin pour sauvegarder le graphique
            
        Returns:
            plt.Figure: Figure matplotlib générée
        """
        if momentum_result is None:
            logger.warning(f"Pas de résultats de momentum disponibles pour {symbol}")
            return None
        
        try:
            # Extraction des composantes du momentum
            technical_score = momentum_result.get('technical', {}).get('score', np.nan)
            fundamental_score = momentum_result.get('fundamental', {}).get('score', np.nan)
            combined_score = momentum_result.get('score', np.nan)
            
            # Extraction des scores par période (momentum technique)
            period_scores = momentum_result.get('technical', {}).get('period_scores', {})
            
            # Création de la figure avec deux sous-graphiques
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.get('figsize', (15, 7)))
            
            # Premier graphique: Répartition des scores de momentum
            labels = ['Technique', 'Fondamental', 'Combiné']
            scores = [technical_score, fundamental_score, combined_score]
            
            # Conversion en pourcentages pour une meilleure lisibilité
            normalized_scores = [score * 100 if not np.isnan(score) else 0 for score in scores]
            
            # Utilisation des couleurs configurées
            colors = self.config.get('colors', ['#1f77b4', '#ff7f0e', '#2ca02c'])[:3]
            
            # Création du graphique en barres
            bars = ax1.bar(labels, normalized_scores, color=colors)
            
            # Ajout des valeurs au-dessus des barres
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            
            # Formatage du premier graphique
            ax1.set_ylim(0, 100)
            ax1.set_ylabel('Score (%)', fontsize=self.config.get('axis_fontsize', 12))
            ax1.set_title('Scores de Momentum', fontsize=self.config.get('title_fontsize', 14))
            ax1.yaxis.set_major_formatter(mtick.PercentFormatter(100))
            
            # Deuxième graphique: Répartition des scores par période
            if period_scores:
                period_labels = list(period_scores.keys())
                period_values = [period_scores[period] * 100 if not np.isnan(period_scores[period]) else 0 for period in period_labels]
                
                ax2.bar(period_labels, period_values, color=colors[0])
                
                # Ajout des valeurs au-dessus des barres
                for i, v in enumerate(period_values):
                    ax2.annotate(f'{v:.1f}%', xy=(i, v), xytext=(0, 3),
                                 textcoords="offset points", ha='center', va='bottom')
                
                # Formatage du deuxième graphique
                ax2.set_ylim(0, 100)
                ax2.set_ylabel('Rendement (%)', fontsize=self.config.get('axis_fontsize', 12))
                ax2.set_title('Rendements par Période', fontsize=self.config.get('title_fontsize', 14))
                ax2.yaxis.set_major_formatter(mtick.PercentFormatter(100))
            
            # Titre global
            display_name = name if name else symbol
            fig.suptitle(f"Analyse Momentum - {display_name} ({symbol})", fontsize=self.config.get('title_fontsize', 16))
            
            # Ajustement des marges
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Sauvegarde du graphique si un chemin est spécifié
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Graphique sauvegardé dans {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du traçage de la répartition des scores de momentum pour {symbol}: {str(e)}")
            return None
    
    def plot_earnings_momentum(self, fundamental_data, symbol, name=None, save_path=None):
        """
        Trace l'évolution des bénéfices trimestriels et le momentum des bénéfices
        
        Parameters:
            fundamental_data (dict): Données fondamentales
            symbol (str): Symbole de l'action
            name (str): Nom de l'entreprise
            save_path (str): Chemin pour sauvegarder le graphique
            
        Returns:
            plt.Figure: Figure matplotlib générée
        """
        if fundamental_data is None or 'quarterly_earnings' not in fundamental_data:
            logger.warning(f"Pas de données de bénéfices disponibles pour {symbol}")
            return None
        
        try:
            # Extraction des données de bénéfices trimestriels
            earnings_data = fundamental_data['quarterly_earnings']
            
            if earnings_data is None or earnings_data.empty or len(earnings_data) < 4:
                logger.warning(f"Pas assez de données de bénéfices disponibles pour {symbol}")
                return None
            
            # Tri des données par date
            earnings_data = earnings_data.sort_values('fiscalDateEnding')
            
            # Création de la figure avec deux sous-graphiques
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.get('figsize', (12, 10)), height_ratios=[3, 1])
            
            # Premier graphique: Évolution des bénéfices
            ax1.plot(earnings_data['fiscalDateEnding'], earnings_data['reportedEPS'], marker='o', linestyle='-',
                    color=self.config.get('colors', ['#1f77b4'])[0], label='EPS Rapporté')
            
            # Si disponible, ajout des bénéfices estimés
            if 'estimatedEPS' in earnings_data.columns:
                ax1.plot(earnings_data['fiscalDateEnding'], earnings_data['estimatedEPS'], marker='x', linestyle='--',
                        color=self.config.get('colors', ['#1f77b4', '#ff7f0e'])[1], label='EPS Estimé')
            
            # Formatage du premier graphique
            ax1.set_ylabel('Bénéfice par Action ($)', fontsize=self.config.get('axis_fontsize', 12))
            ax1.set_title('Évolution des Bénéfices Trimestriels', fontsize=self.config.get('title_fontsize', 14))
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Formatage des dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Deuxième graphique: Surprises de bénéfices
            if 'surprisePercentage' in earnings_data.columns:
                bars = ax2.bar(earnings_data['fiscalDateEnding'], earnings_data['surprisePercentage'],
                              color=earnings_data['surprisePercentage'].apply(lambda x: 
                                self.config.get('colors', ['#2ca02c'])[2] if x >= 0 else 
                                self.config.get('colors', ['#2ca02c', '#d62728'])[3]))
                
                # Ajout des valeurs au-dessus/en-dessous des barres
                for bar in bars:
                    height = bar.get_height()
                    if height >= 0:
                        ax2.annotate(f'+{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    else:
                        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, -12), textcoords="offset points", ha='center', va='bottom')
                
                # Formatage du deuxième graphique
                ax2.set_ylabel('Surprise (%)', fontsize=self.config.get('axis_fontsize', 12))
                ax2.set_title('Surprises des Bénéfices', fontsize=self.config.get('title_fontsize', 14))
                ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax2.grid(True, alpha=0.3)
                
                # Formatage des dates
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Titre global
            display_name = name if name else symbol
            fig.suptitle(f"Momentum des Bénéfices - {display_name} ({symbol})", fontsize=self.config.get('title_fontsize', 16))
            
            # Ajustement des marges
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Sauvegarde du graphique si un chemin est spécifié
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Graphique sauvegardé dans {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du traçage de l'évolution des bénéfices pour {symbol}: {str(e)}")
            return None


class QualityVisualizer:
    """
    Classe pour générer des visualisations des résultats de l'analyse de qualité
    """
    
    def __init__(self, config=None):
        """
        Initialise le visualiseur avec la configuration spécifiée
        
        Parameters:
            config (dict): Configuration de visualisation
        """
        self.config = config or VIZ_CONFIG
        
        # Application du thème
        plt.style.use(self.config.get('theme', 'seaborn-v0_8-darkgrid'))
    
    def plot_quality_breakdown(self, quality_result, symbol, name=None, save_path=None):
        """
        Trace la répartition des scores de qualité
        
        Parameters:
            quality_result (dict): Résultat de l'analyse de qualité
            symbol (str): Symbole de l'action
            name (str): Nom de l'entreprise
            save_path (str): Chemin pour sauvegarder le graphique
            
        Returns:
            plt.Figure: Figure matplotlib générée
        """
        if quality_result is None:
            logger.warning(f"Pas de résultats de qualité disponibles pour {symbol}")
            return None
        
        try:
            # Extraction des scores par métrique
            metric_scores = quality_result.get('metric_scores', {})
            
            if not metric_scores:
                logger.warning(f"Pas de scores de qualité disponibles pour {symbol}")
                return None
            
            # Création de la figure
            fig, ax = plt.subplots(figsize=self.config.get('figsize', (10, 8)))
            
            # Préparation des données pour le graphique en radar
            metrics = list(metric_scores.keys())
            scores = [metric_scores[metric] * 100 if not np.isnan(metric_scores[metric]) else 0 for metric in metrics]
            
            # Calcul des angles pour le graphique en radar
            N = len(metrics)
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            
            # Fermeture du graphique en radar
            scores.append(scores[0])
            angles.append(angles[0])
            metrics.append(metrics[0])
            
            # Création du graphique en radar
            ax.plot(angles, scores, 'o-', linewidth=2, color=self.config.get('colors', ['#1f77b4'])[0])
            ax.fill(angles, scores, alpha=0.25, color=self.config.get('colors', ['#1f77b4'])[0])
            
            # Ajout des axes pour chaque métrique
            ax.set_thetagrids(np.array(angles[:-1]) * 180/np.pi, metrics[:-1])
            
            # Formatage du graphique
            ax.set_ylim(0, 100)
            ax.set_title(f"Scores de Qualité par Métrique - {name if name else symbol}", fontsize=self.config.get('title_fontsize', 16))
            
            # Ajout du score global
            global_score = quality_result.get('score', np.nan)
            if not np.isnan(global_score):
                ax.text(0.5, 0.1, f"Score Global: {global_score*100:.1f}%", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # Ajustement des marges
            plt.tight_layout()
            
            # Sauvegarde du graphique si un chemin est spécifié
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Graphique sauvegardé dans {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du traçage de la répartition des scores de qualité pour {symbol}: {str(e)}")
            return None
    
    def plot_financial_ratios(self, fundamental_data, symbol, name=None, save_path=None):
        """
        Trace les ratios financiers clés
        
        Parameters:
            fundamental_data (dict): Données fondamentales
            symbol (str): Symbole de l'action
            name (str): Nom de l'entreprise
            save_path (str): Chemin pour sauvegarder le graphique
            
        Returns:
            plt.Figure: Figure matplotlib générée
        """
        if fundamental_data is None or 'overview' not in fundamental_data:
            logger.warning(f"Pas de données fondamentales disponibles pour {symbol}")
            return None
        
        try:
            # Extraction des ratios financiers de l'aperçu
            overview = fundamental_data['overview']
            
            # Sélection des ratios à afficher
            ratios = {
                'ROE': overview.get('ROE', np.nan) * 100 if 'ROE' in overview else np.nan,
                'ProfitMargin': overview.get('ProfitMargin', np.nan) * 100 if 'ProfitMargin' in overview else np.nan,
                'OperatingMarginTTM': overview.get('OperatingMarginTTM', np.nan) * 100 if 'OperatingMarginTTM' in overview else np.nan,
                'DebtToEquity': overview.get('DebtToEquity', np.nan) if 'DebtToEquity' in overview else np.nan,
                'PERatio': overview.get('PERatio', np.nan) if 'PERatio' in overview else np.nan,
                'PEGRatio': overview.get('PEGRatio', np.nan) if 'PEGRatio' in overview else np.nan
            }
            
            # Filtrage des ratios disponibles
            valid_ratios = {k: v for k, v in ratios.items() if not np.isnan(v)}
            
            if not valid_ratios:
                logger.warning(f"Pas de ratios financiers disponibles pour {symbol}")
                return None
            
            # Création de la figure
            fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, 8)))
            
            # Création du graphique en barres
            x = range(len(valid_ratios))
            bars = ax.bar(x, valid_ratios.values(), color=self.config.get('colors', ['#1f77b4'])[0])
            
            # Ajout des valeurs au-dessus des barres
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            
            # Formatage du graphique
            ax.set_xticks(x)
            ax.set_xticklabels(valid_ratios.keys(), rotation=45, ha='right')
            ax.set_ylabel('Valeur', fontsize=self.config.get('axis_fontsize', 12))
            
            # Titre
            display_name = name if name else symbol
            ax.set_title(f"Ratios Financiers - {display_name} ({symbol})", fontsize=self.config.get('title_fontsize', 16))
            
            # Ajustement des marges
            plt.tight_layout()
            
            # Sauvegarde du graphique si un chemin est spécifié
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Graphique sauvegardé dans {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du traçage des ratios financiers pour {symbol}: {str(e)}")
            return None


class ScreenerVisualizer:
    """
    Classe pour générer des visualisations des résultats du screening
    """
    
    def __init__(self, config=None):
        """
        Initialise le visualiseur avec la configuration spécifiée
        
        Parameters:
            config (dict): Configuration de visualisation
        """
        self.config = config or VIZ_CONFIG
        
        # Application du thème
        plt.style.use(self.config.get('theme', 'seaborn-v0_8-darkgrid'))
    
    def plot_top_picks(self, top_picks, save_path=None):
        """
        Trace les meilleures opportunités d'investissement
        
        Parameters:
            top_picks (pd.DataFrame): DataFrame avec les meilleures opportunités
            save_path (str): Chemin pour sauvegarder le graphique
            
        Returns:
            plt.Figure: Figure matplotlib générée
        """
        if top_picks is None or top_picks.empty:
            logger.warning("Pas de meilleures opportunités disponibles")
            return None
        
        try:
            # Création de la figure
            fig, ax = plt.subplots(figsize=self.config.get('figsize', (14, 10)))
            
            # Extraction des symboles et scores
            symbols = top_picks['Symbol'].values
            momentum_scores = top_picks['MomentumPercentile'].values
            quality_scores = top_picks['QualityPercentile'].values
            
            # Tri par score combiné
            if 'CombinedScore' in top_picks.columns:
                sorted_indices = top_picks['CombinedScore'].argsort()[::-1]
                symbols = symbols[sorted_indices]
                momentum_scores = momentum_scores[sorted_indices]
                quality_scores = quality_scores[sorted_indices]
            
            # Nombre maximum d'actions à afficher
            max_symbols = min(20, len(symbols))
            
            # Positions des barres
            x = np.arange(max_symbols)
            width = 0.35
            
            # Tracé des barres de momentum et qualité
            bars1 = ax.bar(x - width/2, momentum_scores[:max_symbols], width, label='Momentum', color=self.config.get('colors', ['#1f77b4'])[0])
            bars2 = ax.bar(x + width/2, quality_scores[:max_symbols], width, label='Qualité', color=self.config.get('colors', ['#1f77b4', '#ff7f0e'])[1])
            
            # Formatage du graphique
            ax.set_xlabel('Symbole', fontsize=self.config.get('axis_fontsize', 12))
            ax.set_ylabel('Score (percentile)', fontsize=self.config.get('axis_fontsize', 12))
            ax.set_title('Top Picks: Scores de Momentum et Qualité', fontsize=self.config.get('title_fontsize', 16))
            ax.set_xticks(x)
            ax.set_xticklabels(symbols[:max_symbols], rotation=45, ha='right')
            ax.legend()
            
            # Formatage de l'axe y
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            
            # Ajustement des marges
            plt.tight_layout()
            
            # Sauvegarde du graphique si un chemin est spécifié
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Graphique sauvegardé dans {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du traçage des meilleures opportunités: {str(e)}")
            return None
    
    def plot_sector_breakdown(self, top_picks, save_path=None):
        """
        Trace la répartition sectorielle des meilleures opportunités
        
        Parameters:
            top_picks (pd.DataFrame): DataFrame avec les meilleures opportunités
            save_path (str): Chemin pour sauvegarder le graphique
            
        Returns:
            plt.Figure: Figure matplotlib générée
        """
        if top_picks is None or top_picks.empty or 'Sector' not in top_picks.columns:
            logger.warning("Pas de données sectorielles disponibles")
            return None
        
        try:
            # Création de la figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.get('figsize', (15, 8)))
            
            # Calcul de la répartition sectorielle
            sector_counts = top_picks['Sector'].value_counts()
            
            # Premier graphique: Répartition en nombre d'actions
            sector_counts.plot(kind='bar', ax=ax1, color=self.config.get('colors', ['#1f77b4'])[0])
            
            # Formatage du premier graphique
            ax1.set_title('Répartition par Secteur (Nombre)', fontsize=self.config.get('title_fontsize', 14))
            ax1.set_xlabel('Secteur', fontsize=self.config.get('axis_fontsize', 12))
            ax1.set_ylabel('Nombre d\'actions', fontsize=self.config.get('axis_fontsize', 12))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Ajout des valeurs au-dessus des barres
            for i, v in enumerate(sector_counts):
                ax1.text(i, v + 0.1, str(v), ha='center')
            
            # Deuxième graphique: Répartition en pourcentage
            sector_percentages = sector_counts / sector_counts.sum() * 100
            ax2.pie(sector_percentages, labels=sector_percentages.index, autopct='%1.1f%%',
                   startangle=90, colors=plt.cm.tab10.colors[:len(sector_percentages)])
            
            # Formatage du deuxième graphique
            ax2.set_title('Répartition par Secteur (%)', fontsize=self.config.get('title_fontsize', 14))
            ax2.axis('equal')  # Pour un graphique en camembert circulaire
            
            # Titre global
            fig.suptitle('Analyse Sectorielle des Top Picks', fontsize=self.config.get('title_fontsize', 16))
            
            # Ajustement des marges
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Sauvegarde du graphique si un chemin est spécifié
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Graphique sauvegardé dans {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du traçage de la répartition sectorielle: {str(e)}")
            return None
    
    def plot_scores_scatter(self, scores_df, save_path=None):
        """
        Trace un nuage de points des scores de momentum et qualité
        
        Parameters:
            scores_df (pd.DataFrame): DataFrame avec tous les scores calculés
            save_path (str): Chemin pour sauvegarder le graphique
            
        Returns:
            plt.Figure: Figure matplotlib générée
        """
        if scores_df is None or scores_df.empty:
            logger.warning("Pas de données de scores disponibles")
            return None
        
        # Vérification des colonnes nécessaires
        required_columns = ['MomentumPercentile', 'QualityPercentile', 'Symbol']
        if not all(col in scores_df.columns for col in required_columns):
            logger.warning("Colonnes nécessaires manquantes pour le nuage de points")
            return None
        
        try:
            # Création de la figure
            fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, 10)))
            
            # Extraction des scores
            momentum_scores = scores_df['MomentumPercentile'].values
            quality_scores = scores_df['QualityPercentile'].values
            symbols = scores_df['Symbol'].values
            
            # Calcul du score combiné pour la taille des points
            if 'CombinedScore' in scores_df.columns:
                combined_scores = scores_df['CombinedScore'].values * 500  # Mise à l'échelle pour la taille des points
            else:
                combined_scores = np.ones_like(momentum_scores) * 50
            
            # Secteurs pour les couleurs (si disponible)
            if 'Sector' in scores_df.columns:
                sectors = scores_df['Sector'].values
                unique_sectors = np.unique(sectors)
                sector_to_color = {sector: plt.cm.tab10(i % 10) for i, sector in enumerate(unique_sectors)}
                colors = [sector_to_color[sector] for sector in sectors]
            else:
                colors = self.config.get('colors', ['#1f77b4'])[0]
            
            # Tracé du nuage de points
            scatter = ax.scatter(momentum_scores, quality_scores, s=combined_scores, c=colors, alpha=0.6)
            
            # Ajout des symboles pour les points importants (top 10 combinés)
            if 'CombinedScore' in scores_df.columns:
                top_indices = scores_df['CombinedScore'].argsort()[::-1][:10]
                for idx in top_indices:
                    ax.annotate(symbols[idx], 
                               (momentum_scores[idx], quality_scores[idx]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold')
            
            # Ajout des lignes de division pour les quadrants
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
            
            # Ajouter des étiquettes pour les quadrants
            ax.text(25, 75, "Qualité Élevée\nMomentum Faible", ha='center', va='center', alpha=0.7, fontsize=10)
            ax.text(75, 75, "Qualité Élevée\nMomentum Élevé", ha='center', va='center', alpha=0.7, fontsize=10)
            ax.text(25, 25, "Qualité Faible\nMomentum Faible", ha='center', va='center', alpha=0.7, fontsize=10)
            ax.text(75, 25, "Qualité Faible\nMomentum Élevé", ha='center', va='center', alpha=0.7, fontsize=10)
            
            # Formatage du graphique
            ax.set_xlabel('Score de Momentum (percentile)', fontsize=self.config.get('axis_fontsize', 12))
            ax.set_ylabel('Score de Qualité (percentile)', fontsize=self.config.get('axis_fontsize', 12))
            ax.set_title('Nuage de Points: Momentum vs Qualité', fontsize=self.config.get('title_fontsize', 16))
            
            # Formatage des axes en pourcentage
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.xaxis.set_major_formatter(mtick.PercentFormatter())
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            
            # Légende pour les secteurs (si disponible)
            if 'Sector' in scores_df.columns:
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=sector_to_color[sector], 
                                             markersize=8, label=sector) 
                                  for sector in unique_sectors]
                ax.legend(handles=legend_elements, title='Secteur', loc='upper left', bbox_to_anchor=(1, 1))
            
            # Ajustement des marges
            plt.tight_layout()
            
            # Sauvegarde du graphique si un chemin est spécifié
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Graphique sauvegardé dans {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du traçage du nuage de points: {str(e)}")
            return None
    
    def plot_performance_metrics(self, results, save_path=None):
        """
        Trace les principales métriques de performance des top picks
        
        Parameters:
            results (dict): Résultats du screening
            save_path (str): Chemin pour sauvegarder le graphique
            
        Returns:
            plt.Figure: Figure matplotlib générée
        """
        if results is None or 'top_picks' not in results or results['top_picks'].empty:
            logger.warning("Pas de top picks disponibles pour tracer les métriques de performance")
            return None
        
        top_picks = results['top_picks']
        
        # Vérification des colonnes nécessaires
        performance_metrics = ['PERatio', 'ROE', 'ProfitMargin', 'DebtToEquity']
        available_metrics = [col for col in performance_metrics if col in top_picks.columns]
        
        if not available_metrics:
            logger.warning("Pas de métriques de performance disponibles")
            return None
        
        try:
            # Nombre de métriques à afficher
            n_metrics = len(available_metrics)
            
            # Détermination de la disposition des sous-graphiques
            n_rows = (n_metrics + 1) // 2  # Arrondi supérieur pour avoir au moins une ligne
            n_cols = min(2, n_metrics)  # Maximum 2 colonnes
            
            # Création de la figure avec plusieurs sous-graphiques
            fig, axes = plt.subplots(n_rows, n_cols, figsize=self.config.get('figsize', (15, 10)))
            
            # Si un seul sous-graphique, axes doit être une liste
            if n_metrics == 1:
                axes = np.array([axes])
            
            # Tracé de chaque métrique
            for i, metric in enumerate(available_metrics):
                # Calcul de la position dans la grille
                row = i // n_cols
                col = i % n_cols
                
                # Accès à l'axe correspondant
                if n_rows > 1 and n_cols > 1:
                    ax = axes[row, col]
                else:
                    ax = axes[i]
                
                # Extraction des données pour cette métrique
                data = top_picks[metric].values
                symbols = top_picks['Symbol'].values
                
                # Tri par valeur décroissante (ou croissante pour certaines métriques)
                sort_ascending = metric in ['DebtToEquity', 'PERatio']  # Métriques où plus petit = meilleur
                sorted_indices = data.argsort()
                if not sort_ascending:
                    sorted_indices = sorted_indices[::-1]
                
                # Limitation du nombre de symboles affichés
                max_symbols = min(10, len(symbols))
                indices = sorted_indices[:max_symbols]
                
                # Tracé du graphique en barres
                bars = ax.barh(np.arange(len(indices)), data[indices], 
                              color=self.config.get('colors', ['#1f77b4'])[0])
                
                # Ajout des valeurs à côté des barres
                for j, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.annotate(f'{width:.2f}', xy=(width, bar.get_y() + bar.get_height()/2),
                               xytext=(5, 0), textcoords='offset points',
                               va='center')
                
                # Formatage du graphique
                ax.set_yticks(np.arange(len(indices)))
                ax.set_yticklabels(symbols[indices])
                ax.invert_yaxis()  # Pour que le plus grand soit en haut
                ax.set_title(f"{metric}", fontsize=self.config.get('title_fontsize', 14))
                
                # Ajout d'une grille horizontale pour faciliter la lecture
                ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Si le nombre de métriques est impair, supprimer le dernier sous-graphique vide
            if n_metrics % 2 == 1 and n_rows > 1 and n_cols > 1:
                fig.delaxes(axes[n_rows-1, n_cols-1])
            
            # Titre global
            fig.suptitle('Métriques de Performance des Top Picks', fontsize=self.config.get('title_fontsize', 16))
            
            # Ajustement des marges
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Sauvegarde du graphique si un chemin est spécifié
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Graphique sauvegardé dans {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Erreur lors du traçage des métriques de performance: {str(e)}")
            return None


# Code de test du module si exécuté directement
if __name__ == "__main__":
    # Création de données de test
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Création d'un DataFrame de prix historiques
    dates = pd.date_range(end=datetime.now(), periods=252, freq='B')
    prices = np.linspace(100, 150, len(dates)) + np.random.normal(0, 5, len(dates))
    historical_data = pd.DataFrame({
        'open': prices - 1,
        'high': prices + 2,
        'low': prices - 2,
        'close': prices,
        'adjusted_close': prices,
        'volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # Création de données fondamentales
    fundamental_data = {
        'overview': {
            'Symbol': 'AAPL',
            'Name': 'Apple Inc.',
            'Sector': 'Technology',
            'Industry': 'Consumer Electronics',
            'ROE': 0.15,
            'ProfitMargin': 0.21,
            'OperatingMarginTTM': 0.25,
            'DebtToEquity': 1.2,
            'PERatio': 26.5,
            'PEGRatio': 1.8
        },
        'quarterly_earnings': pd.DataFrame({
            'fiscalDateEnding': pd.date_range(end=datetime.now(), periods=8, freq='3M')[::-1],
            'reportedEPS': [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            'estimatedEPS': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            'surprisePercentage': [9.1, 8.3, 7.7, 7.1, 6.7, 6.3, 5.9, 5.6]
        })
    }
    
    # Création de résultats de screening
    scores_df = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'WMT'],
        'Name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Amazon.com Inc.', 'Meta Platforms Inc.', 
                'Tesla Inc.', 'NVIDIA Corp.', 'JPMorgan Chase & Co.', 'Johnson & Johnson', 'Walmart Inc.'],
        'Sector': ['Technology', 'Technology', 'Communication Services', 'Consumer Discretionary', 'Communication Services',
                  'Consumer Discretionary', 'Technology', 'Financials', 'Healthcare', 'Consumer Staples'],
        'MomentumScore': [0.8, 0.7, 0.6, 0.75, 0.65, 0.85, 0.9, 0.5, 0.4, 0.45],
        'QualityScore': [0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.9, 0.95, 0.7],
        'MomentumPercentile': [80, 70, 60, 75, 65, 85, 90, 50, 40, 45],
        'QualityPercentile': [85, 80, 75, 70, 65, 60, 55, 90, 95, 70],
        'CombinedScore': [0.82, 0.74, 0.66, 0.73, 0.65, 0.75, 0.76, 0.66, 0.62, 0.55],
        'PERatio': [26.5, 30.2, 28.1, 72.4, 22.3, 80.5, 50.2, 15.3, 18.2, 22.8],
        'ROE': [0.15, 0.14, 0.13, 0.10, 0.18, 0.09, 0.20, 0.12, 0.11, 0.08],
        'ProfitMargin': [0.21, 0.19, 0.18, 0.05, 0.15, 0.07, 0.25, 0.22, 0.20, 0.03],
        'DebtToEquity': [1.2, 0.8, 0.5, 1.5, 0.6, 1.8, 0.7, 2.0, 0.9, 1.3]
    })
    
    top_picks = scores_df.sort_values('CombinedScore', ascending=False).head(5)
    
    screening_results = {
        'scores': scores_df,
        'top_picks': top_picks,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': {
            'min_market_cap': 1e9,
            'max_pe_ratio': 50,
            'min_roe': 0.10,
            'momentum_threshold': 0.70,
            'quality_threshold': 0.60
        }
    }
    
    # Test du module MomentumVisualizer
    momentum_viz = MomentumVisualizer()
    price_fig = momentum_viz.plot_price_history(historical_data, 'AAPL', 'Apple Inc.', 
                                               periods={'short_term': 20, 'medium_term': 60, 'long_term': 252})
    plt.close(price_fig)
    
    # Test du module QualityVisualizer
    quality_viz = QualityVisualizer()
    ratios_fig = quality_viz.plot_financial_ratios(fundamental_data, 'AAPL', 'Apple Inc.')
    plt.close(ratios_fig)
    
    # Test du module ScreenerVisualizer
    screener_viz = ScreenerVisualizer()
    top_picks_fig = screener_viz.plot_top_picks(top_picks)
    plt.close(top_picks_fig)
    
    scatter_fig = screener_viz.plot_scores_scatter(scores_df)
    plt.close(scatter_fig)
    
    metrics_fig = screener_viz.plot_performance_metrics(screening_results)
    plt.close(metrics_fig)
    
    print("Tests des visualisations exécutés avec succès !")

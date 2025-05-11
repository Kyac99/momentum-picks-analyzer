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
            
            # Vérification du nom des colonnes pour les scores
            momentum_col = 'Momentum_Score' if 'Momentum_Score' in top_picks.columns else 'MomentumScore'
            quality_col = 'Quality_Score' if 'Quality_Score' in top_picks.columns else 'QualityScore'
            combined_col = 'Combined_Score' if 'Combined_Score' in top_picks.columns else 'CombinedScore'
            
            # Extraction des scores
            momentum_scores = top_picks[momentum_col].values * 100  # Convertir en pourcentage
            quality_scores = top_picks[quality_col].values * 100    # Convertir en pourcentage
            
            # Tri par score combiné si disponible
            if combined_col in top_picks.columns:
                sorted_indices = top_picks[combined_col].argsort()[::-1]
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
            ax.set_ylabel('Score (%)', fontsize=self.config.get('axis_fontsize', 12))
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
        
        # Vérification que les colonnes nécessaires sont présentes
        momentum_col = None
        quality_col = None
        
        # Recherche des colonnes de scores (différentes conventions de nommage possibles)
        for col in ['Momentum_Score', 'MomentumScore', 'MomentumPercentile']:
            if col in scores_df.columns:
                momentum_col = col
                break
                
        for col in ['Quality_Score', 'QualityScore', 'QualityPercentile']:
            if col in scores_df.columns:
                quality_col = col
                break
        
        if not momentum_col or not quality_col or 'Symbol' not in scores_df.columns:
            logger.warning("Données de scores incomplètes ou mal formatées")
            return None
        
        try:
            # Création de la figure
            fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, 10)))
            
            # Normalisation des scores si nécessaire
            momentum_scores = scores_df[momentum_col].values
            quality_scores = scores_df[quality_col].values
            
            # Si les scores sont déjà des percentiles (0-100), les convertir en scores (0-1)
            if momentum_col == 'MomentumPercentile' and np.max(momentum_scores) > 1:
                momentum_scores = momentum_scores / 100.0
            if quality_col == 'QualityPercentile' and np.max(quality_scores) > 1:
                quality_scores = quality_scores / 100.0
            
            # Création du nuage de points
            scatter = ax.scatter(momentum_scores, quality_scores, 
                                alpha=0.7, s=50, 
                                c=momentum_scores * quality_scores,  # Couleur basée sur le produit des scores
                                cmap='viridis')
            
            # Ajout d'une barre de couleur
            cbar = plt.colorbar(scatter)
            cbar.set_label('Score Combiné (Momentum × Qualité)', rotation=270, labelpad=20)
            
            # Ajout des symboles comme annotations
            for i, symbol in enumerate(scores_df['Symbol']):
                ax.annotate(symbol, (momentum_scores[i], quality_scores[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
            
            # Ajout des lignes de référence
            ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)  # Seuil de qualité
            ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)  # Seuil de momentum
            
            # Ajout de zones de quadrant (optionnel)
            ax.fill_between([0.7, 1], 0.6, 1, color='green', alpha=0.1)  # Quadrant optimal
            ax.text(0.85, 0.8, 'Zone Optimale', ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Formatage du graphique
            ax.set_xlabel('Score de Momentum', fontsize=self.config.get('axis_fontsize', 12))
            ax.set_ylabel('Score de Qualité', fontsize=self.config.get('axis_fontsize', 12))
            ax.set_title('Nuage de Points des Scores Momentum vs Qualité', fontsize=self.config.get('title_fontsize', 16))
            
            # Configuration des limites des axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Ajout des grilles
            ax.grid(True, alpha=0.3)
            
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


class ResultVisualizer:
    """
    Classe unifiée pour générer des visualisations des résultats de l'analyse
    Sert d'interface pour les différents visualiseurs
    """
    
    def __init__(self, config=None):
        """
        Initialise le visualiseur avec la configuration spécifiée
        
        Parameters:
            config (dict): Configuration de visualisation
        """
        self.config = config or VIZ_CONFIG
        
        # Initialisation des visualiseurs spécifiques
        self.momentum_viz = MomentumVisualizer(config)
        self.quality_viz = QualityVisualizer(config)
        self.screener_viz = ScreenerVisualizer(config)
    
    def save_all_figures(self, results_df, output_dir=None):
        """
        Génère et sauvegarde toutes les visualisations pour les résultats du screening
        
        Parameters:
            results_df (pd.DataFrame): DataFrame avec les résultats du screening
            output_dir (str): Répertoire de sortie pour les graphiques
            
        Returns:
            dict: Chemins des graphiques générés
        """
        # Utiliser le répertoire de résultats par défaut si non spécifié
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(RESULTS_DIR, f"report_{timestamp}")
        
        # Création du répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Chemins des graphiques à générer
        graph_paths = {}
        
        # Top picks visualization
        top_picks_path = os.path.join(output_dir, "top_picks.png")
        self.screener_viz.plot_top_picks(results_df, save_path=top_picks_path)
        graph_paths['top_picks'] = top_picks_path
        
        # Sector breakdown if available
        if 'Sector' in results_df.columns:
            sector_path = os.path.join(output_dir, "sector_breakdown.png")
            self.screener_viz.plot_sector_breakdown(results_df, save_path=sector_path)
            graph_paths['sector_breakdown'] = sector_path
        
        # Scores scatter visualization
        scatter_path = os.path.join(output_dir, "scores_scatter.png")
        self.screener_viz.plot_scores_scatter(results_df, save_path=scatter_path)
        graph_paths['scores_scatter'] = scatter_path
        
        logger.info(f"Toutes les visualisations ont été sauvegardées dans {output_dir}")
        return graph_paths


if __name__ == "__main__":
    # Test simple du module
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Création de données de test
    dates = pd.date_range(end=datetime.now(), periods=300, freq='D')
    
    # Série de prix avec tendance haussière
    historical_data = pd.DataFrame({
        'close': [100 + i * 0.5 + np.random.normal(0, 5) for i in range(300)],
        'adjusted_close': [100 + i * 0.5 + np.random.normal(0, 5) for i in range(300)],
        'volume': [1000000 + np.random.randint(0, 500000) for _ in range(300)]
    }, index=dates)
    
    # Initialisation du visualiseur
    momentum_viz = MomentumVisualizer()
    
    # Traçage de l'historique des prix
    periods = {'court_terme': 20, 'moyen_terme': 60, 'long_terme': 120}
    fig = momentum_viz.plot_price_history(historical_data, 'AAPL', name='Apple Inc.', periods=periods)
    
    if fig:
        plt.show()
    
    # Test avec le visualiseur de screening
    # Création de données de test pour les scores
    test_scores = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'Name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Amazon.com Inc.', 'Meta Platforms Inc.'],
        'Sector': ['Technology', 'Technology', 'Communication Services', 'Consumer Discretionary', 'Communication Services'],
        'Momentum_Score': [0.85, 0.76, 0.92, 0.68, 0.79],
        'Quality_Score': [0.91, 0.88, 0.73, 0.81, 0.62],
        'Combined_Score': [0.87, 0.82, 0.85, 0.76, 0.70]
    })
    
    # Initialisation du visualiseur unifié
    result_viz = ResultVisualizer()
    
    # Test de sauvegarde de toutes les figures
    test_output_dir = os.path.join(os.getcwd(), "test_output")
    graph_paths = result_viz.save_all_figures(test_scores, output_dir=test_output_dir)
    
    print("Graphiques générés avec succès dans:", test_output_dir)

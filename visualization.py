"""
Module de visualisation des résultats
Génère des graphiques pour analyser les actions sélectionnées
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ResultVisualizer:
    """
    Classe pour visualiser les résultats du screening d'actions
    """
    
    def __init__(self, style='seaborn'):
        """
        Initialise le visualiseur avec un style prédéfini
        
        Parameters:
            style (str): Style de visualisation (défaut: 'seaborn')
        """
        # Configuration du style Matplotlib/Seaborn
        plt.style.use(style)
        sns.set_palette('viridis')
        
        # Configuration du thème Plotly
        self.plotly_template = 'plotly_white'
        
        # Couleurs
        self.colors = {
            'momentum': '#1f77b4',
            'quality': '#ff7f0e',
            'combined': '#2ca02c',
            'performance': '#d62728'
        }
    
    def plot_top_stocks(self, results_df, title='Top Actions Sélectionnées', figsize=(12, 8)):
        """
        Crée un graphique à barres des meilleures actions selon le score combiné
        
        Parameters:
            results_df (pd.DataFrame): DataFrame avec les résultats
            title (str): Titre du graphique
            figsize (tuple): Taille du graphique
            
        Returns:
            plt.Figure: Figure générée
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Préparer les données
        df = results_df.sort_values('Combined_Score', ascending=False).head(10)
        
        # Créer le graphique à barres
        bars = ax.barh(df['Symbol'], df['Combined_Score'], color=self.colors['combined'])
        
        # Ajouter les étiquettes
        ax.set_xlabel('Score Combiné')
        ax.set_title(title)
        ax.set_xlim(0, 1)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                    ha='left', va='center')
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        return fig
    
    def plot_score_components(self, results_df, top_n=10, figsize=(14, 10)):
        """
        Crée un graphique à barres groupées pour comparer les composantes des scores
        
        Parameters:
            results_df (pd.DataFrame): DataFrame avec les résultats
            top_n (int): Nombre d'actions à afficher
            figsize (tuple): Taille du graphique
            
        Returns:
            plt.Figure: Figure générée
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Préparer les données
        df = results_df.sort_values('Combined_Score', ascending=False).head(top_n)
        
        # Définir les positions des barres
        x = np.arange(len(df))
        width = 0.25
        
        # Créer les barres groupées
        ax.bar(x - width, df['Momentum_Score'], width, label='Momentum', color=self.colors['momentum'])
        ax.bar(x, df['Quality_Score'], width, label='Quality', color=self.colors['quality'])
        ax.bar(x + width, df['Combined_Score'], width, label='Combiné', color=self.colors['combined'])
        
        # Ajouter les étiquettes
        ax.set_ylabel('Score')
        ax.set_title('Composantes des Scores par Action')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Symbol'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        return fig
    
    def plot_performance_comparison(self, results_df, figsize=(14, 8)):
        """
        Crée un graphique pour comparer les performances historiques
        
        Parameters:
            results_df (pd.DataFrame): DataFrame avec les résultats
            figsize (tuple): Taille du graphique
            
        Returns:
            plt.Figure: Figure générée
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Préparer les données
        df = results_df.sort_values('Combined_Score', ascending=False).head(10)
        
        # Définir les périodes
        periods = ['Perf_1M', 'Perf_3M', 'Perf_6M', 'Perf_12M']
        period_labels = ['1 Mois', '3 Mois', '6 Mois', '12 Mois']
        
        # Créer le graphique à barres pour chaque action
        bar_width = 0.8 / len(df)
        for i, (_, row) in enumerate(df.iterrows()):
            pos = np.arange(len(periods)) + i * bar_width - 0.4 + bar_width/2
            ax.bar(pos, [row[p] for p in periods], width=bar_width, 
                   label=row['Symbol'], alpha=0.7)
        
        # Ajouter les étiquettes
        ax.set_ylabel('Performance (%)')
        ax.set_title('Comparaison des Performances par Période')
        ax.set_xticks(np.arange(len(periods)))
        ax.set_xticklabels(period_labels)
        ax.legend(title='Symbole')
        
        # Ajouter une ligne de référence à 0%
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        return fig
    
    def plot_quality_metrics(self, results_df, figsize=(14, 8)):
        """
        Crée un graphique pour comparer les métriques de qualité
        
        Parameters:
            results_df (pd.DataFrame): DataFrame avec les résultats
            figsize (tuple): Taille du graphique
            
        Returns:
            plt.Figure: Figure générée
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Préparer les données
        df = results_df.sort_values('Quality_Score', ascending=False).head(10)
        
        # Graphique 1: ROE et Marge de Profit
        df_melt = pd.melt(df, id_vars=['Symbol'], value_vars=['ROE', 'Profit_Margin'],
                          var_name='Metric', value_name='Value')
        
        sns.barplot(x='Symbol', y='Value', hue='Metric', data=df_melt, ax=ax1)
        ax1.set_title('ROE et Marge de Profit')
        ax1.set_ylabel('Pourcentage (%)')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Graphique 2: Ratio Dette/Fonds Propres
        bars = ax2.bar(df['Symbol'], df['Debt_To_Equity'], color='skyblue')
        ax2.set_title('Ratio Dette/Fonds Propres')
        ax2.set_ylabel('Ratio')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        return fig
    
    def plot_interactive_scatter(self, results_df):
        """
        Crée un graphique interactif Plotly pour comparer Momentum et Quality
        
        Parameters:
            results_df (pd.DataFrame): DataFrame avec les résultats
            
        Returns:
            plotly.graph_objects.Figure: Figure interactive générée
        """
        # Préparer les données
        df = results_df.copy()
        
        # Créer le scatter plot
        fig = px.scatter(
            df,
            x='Momentum_Score',
            y='Quality_Score',
            size='Combined_Score',
            color='Perf_6M',
            hover_name='Symbol',
            hover_data=['Perf_1M', 'Perf_3M', 'Perf_12M', 'ROE', 'Profit_Margin'],
            color_continuous_scale='RdYlGn',
            template=self.plotly_template,
            title='Comparaison Momentum vs Quality avec Performance 6 Mois',
            labels={
                'Momentum_Score': 'Score Momentum',
                'Quality_Score': 'Score Quality',
                'Combined_Score': 'Score Combiné',
                'Perf_6M': 'Perf. 6 Mois (%)'
            }
        )
        
        # Ajouter des lignes de quadrant
        fig.add_shape(
            type='line',
            x0=0.5, y0=0,
            x1=0.5, y1=1,
            line=dict(color='gray', dash='dash'),
            name='Quadrant Momentum'
        )
        
        fig.add_shape(
            type='line',
            x0=0, y0=0.5,
            x1=1, y1=0.5,
            line=dict(color='gray', dash='dash'),
            name='Quadrant Quality'
        )
        
        # Ajouter des annotations pour les quadrants
        fig.add_annotation(
            x=0.25, y=0.75,
            text='Strong Quality',
            showarrow=False,
            font=dict(size=12)
        )
        
        fig.add_annotation(
            x=0.75, y=0.75,
            text='Stars',
            showarrow=False,
            font=dict(size=12)
        )
        
        fig.add_annotation(
            x=0.25, y=0.25,
            text='Laggards',
            showarrow=False,
            font=dict(size=12)
        )
        
        fig.add_annotation(
            x=0.75, y=0.25,
            text='Strong Momentum',
            showarrow=False,
            font=dict(size=12)
        )
        
        # Ajuster la mise en page
        fig.update_layout(
            width=900,
            height=700,
            coloraxis_colorbar=dict(
                title='Perf. 6 Mois (%)'
            )
        )
        
        return fig
    
    def plot_interactive_performance(self, results_df, top_n=10):
        """
        Crée un graphique à barres interactif Plotly pour comparer les performances
        
        Parameters:
            results_df (pd.DataFrame): DataFrame avec les résultats
            top_n (int): Nombre d'actions à afficher
            
        Returns:
            plotly.graph_objects.Figure: Figure interactive générée
        """
        # Préparer les données
        df = results_df.sort_values('Combined_Score', ascending=False).head(top_n)
        
        # Définir les périodes
        periods = ['Perf_1M', 'Perf_3M', 'Perf_6M', 'Perf_12M']
        period_labels = ['1 Mois', '3 Mois', '6 Mois', '12 Mois']
        
        # Melt pour format long
        df_melt = pd.melt(
            df,
            id_vars=['Symbol', 'Combined_Score', 'Momentum_Score', 'Quality_Score'],
            value_vars=periods,
            var_name='Period',
            value_name='Performance'
        )
        
        # Mapper les noms des périodes
        period_map = dict(zip(periods, period_labels))
        df_melt['Period'] = df_melt['Period'].map(period_map)
        
        # Créer le graphique à barres
        fig = px.bar(
            df_melt,
            x='Symbol',
            y='Performance',
            color='Period',
            barmode='group',
            hover_data=['Combined_Score', 'Momentum_Score', 'Quality_Score'],
            template=self.plotly_template,
            title='Comparaison des Performances par Période',
            labels={
                'Symbol': 'Symbole',
                'Performance': 'Performance (%)',
                'Period': 'Période',
                'Combined_Score': 'Score Combiné',
                'Momentum_Score': 'Score Momentum',
                'Quality_Score': 'Score Quality'
            }
        )
        
        # Ajouter une ligne de référence à 0%
        fig.add_shape(
            type='line',
            x0=-0.5, y0=0,
            x1=len(df) - 0.5, y1=0,
            line=dict(color='black', dash='solid', width=1),
            name='Référence 0%'
        )
        
        # Ajuster la mise en page
        fig.update_layout(
            width=900,
            height=600,
            xaxis_title='Symbole',
            yaxis_title='Performance (%)'
        )
        
        return fig
    
    def create_momentum_quality_grid(self, results_df, figsize=(16, 12)):
        """
        Crée une grille de graphiques pour analyser Momentum et Quality
        
        Parameters:
            results_df (pd.DataFrame): DataFrame avec les résultats
            figsize (tuple): Taille du graphique
            
        Returns:
            plt.Figure: Figure générée
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Préparer les données
        df = results_df.sort_values('Combined_Score', ascending=False).head(20)
        
        # Graphique 1: Scatter plot Momentum vs Quality
        scatter = axs[0, 0].scatter(
            df['Momentum_Score'],
            df['Quality_Score'],
            s=df['Combined_Score'] * 200,
            c=df['Perf_6M'],
            cmap='RdYlGn',
            alpha=0.7
        )
        
        # Ajouter un colorbar
        cbar = plt.colorbar(scatter, ax=axs[0, 0])
        cbar.set_label('Perf. 6 Mois (%)')
        
        # Ajouter les étiquettes
        for i, row in df.iterrows():
            axs[0, 0].annotate(
                row['Symbol'],
                (row['Momentum_Score'], row['Quality_Score']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        axs[0, 0].set_xlabel('Score Momentum')
        axs[0, 0].set_ylabel('Score Quality')
        axs[0, 0].set_title('Momentum vs Quality')
        axs[0, 0].grid(True, alpha=0.3)
        
        # Ajouter des lignes de quadrant
        axs[0, 0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        axs[0, 0].axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Graphique 2: Top 10 Combined Score
        top_10 = df.head(10)
        bars = axs[0, 1].barh(
            top_10['Symbol'],
            top_10['Combined_Score'],
            color=self.colors['combined']
        )
        
        axs[0, 1].set_xlabel('Score Combiné')
        axs[0, 1].set_title('Top 10 - Score Combiné')
        axs[0, 1].set_xlim(0, 1)
        
        # Graphique 3: Performances sur différentes périodes
        top_5 = df.head(5)
        periods = ['Perf_1M', 'Perf_3M', 'Perf_6M', 'Perf_12M']
        period_labels = ['1 Mois', '3 Mois', '6 Mois', '12 Mois']
        
        for i, symbol in enumerate(top_5['Symbol']):
            row = top_5[top_5['Symbol'] == symbol].iloc[0]
            axs[1, 0].plot(
                period_labels,
                [row[p] for p in periods],
                marker='o',
                label=symbol
            )
        
        axs[1, 0].set_ylabel('Performance (%)')
        axs[1, 0].set_title('Évolution des Performances')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].axhline(0, color='k', linestyle='-', alpha=0.3)
        
        # Graphique 4: Métriques de qualité
        width = 0.35
        top_5_reversed = top_5[::-1]
        
        roe = axs[1, 1].barh(
            np.arange(len(top_5)) - width/2,
            top_5_reversed['ROE'],
            width,
            label='ROE',
            color=self.colors['quality']
        )
        
        profit_margin = axs[1, 1].barh(
            np.arange(len(top_5)) + width/2,
            top_5_reversed['Profit_Margin'],
            width,
            label='Marge de Profit',
            color=self.colors['momentum']
        )
        
        axs[1, 1].set_yticks(np.arange(len(top_5)))
        axs[1, 1].set_yticklabels(top_5_reversed['Symbol'])
        axs[1, 1].set_xlabel('Pourcentage (%)')
        axs[1, 1].set_title('ROE et Marge de Profit')
        axs[1, 1].legend()
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        return fig
    
    def save_all_figures(self, results_df, output_dir='figures'):
        """
        Génère et sauvegarde tous les graphiques
        
        Parameters:
            results_df (pd.DataFrame): DataFrame avec les résultats
            output_dir (str): Répertoire de sortie
            
        Returns:
            list: Liste des chemins des fichiers générés
        """
        import os
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Générer et sauvegarder les graphiques
        file_paths = []
        
        # Graphique 1: Top stocks
        fig1 = self.plot_top_stocks(results_df)
        path1 = os.path.join(output_dir, 'top_stocks.png')
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        file_paths.append(path1)
        
        # Graphique 2: Score components
        fig2 = self.plot_score_components(results_df)
        path2 = os.path.join(output_dir, 'score_components.png')
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        file_paths.append(path2)
        
        # Graphique 3: Performance comparison
        fig3 = self.plot_performance_comparison(results_df)
        path3 = os.path.join(output_dir, 'performance_comparison.png')
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        file_paths.append(path3)
        
        # Graphique 4: Quality metrics
        fig4 = self.plot_quality_metrics(results_df)
        path4 = os.path.join(output_dir, 'quality_metrics.png')
        fig4.savefig(path4, dpi=300, bbox_inches='tight')
        plt.close(fig4)
        file_paths.append(path4)
        
        # Graphique 5: Momentum Quality grid
        fig5 = self.create_momentum_quality_grid(results_df)
        path5 = os.path.join(output_dir, 'momentum_quality_grid.png')
        fig5.savefig(path5, dpi=300, bbox_inches='tight')
        plt.close(fig5)
        file_paths.append(path5)
        
        # Graphiques interactifs Plotly
        fig6 = self.plot_interactive_scatter(results_df)
        path6 = os.path.join(output_dir, 'interactive_scatter.html')
        fig6.write_html(path6)
        file_paths.append(path6)
        
        fig7 = self.plot_interactive_performance(results_df)
        path7 = os.path.join(output_dir, 'interactive_performance.html')
        fig7.write_html(path7)
        file_paths.append(path7)
        
        return file_paths

if __name__ == "__main__":
    # Test simple du module
    import pandas as pd
    
    # Créer un DataFrame de test
    data = {
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V'],
        'Combined_Score': [0.85, 0.82, 0.78, 0.75, 0.72, 0.70, 0.68, 0.65, 0.63, 0.60],
        'Momentum_Score': [0.90, 0.85, 0.80, 0.70, 0.75, 0.80, 0.60, 0.55, 0.50, 0.65],
        'Quality_Score': [0.80, 0.80, 0.75, 0.80, 0.70, 0.60, 0.75, 0.75, 0.75, 0.55],
        'Perf_1M': [5.2, 4.1, 3.5, 2.8, 1.9, 6.7, -1.2, 0.5, 1.8, 2.2],
        'Perf_3M': [12.5, 10.2, 8.7, 9.5, 7.2, 15.4, 3.1, 5.8, 4.2, 6.7],
        'Perf_6M': [22.1, 18.5, 15.2, 17.3, 12.8, 35.6, 8.5, 10.2, 7.8, 11.9],
        'Perf_12M': [45.3, 38.7, 30.1, 42.5, 25.6, 90.2, 15.8, 20.3, 14.2, 18.7],
        'ROE': [35.2, 40.1, 25.8, 30.5, 22.7, 18.5, 45.2, 15.8, 28.4, 32.1],
        'Profit_Margin': [25.8, 30.2, 28.5, 15.2, 35.5, 8.2, 40.1, 32.5, 24.8, 45.2],
        'Debt_To_Equity': [1.2, 0.8, 0.5, 1.5, 0.3, 2.1, 0.1, 1.8, 0.6, 0.4],
        'Market_Cap': ['2.5T', '2.2T', '1.8T', '1.7T', '1.0T', '800B', '750B', '500B', '450B', '430B'],
        'EPS': [5.61, 9.27, 108.42, 64.81, 13.77, 4.90, 3.85, 15.35, 9.21, 7.39],
        'PE_Ratio': [30.2, 25.8, 20.1, 35.5, 22.7, 150.2, 65.8, 12.5, 18.7, 25.4]
    }
    
    df = pd.DataFrame(data)
    
    # Créer le visualiseur
    visualizer = ResultVisualizer()
    
    # Générer et afficher un graphique
    fig = visualizer.plot_top_stocks(df)
    plt.show()
    
    # Sauvegarder tous les graphiques
    file_paths = visualizer.save_all_figures(df)
    print("Graphiques sauvegardés:")
    for path in file_paths:
        print(f"- {path}")

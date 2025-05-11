"""
Interface web Streamlit pour Momentum Picks Analyzer
Permet d'utiliser le screener via une interface graphique simple
"""

import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from data_loader import DataLoader
from momentum import MomentumCalculator
from quality import QualityCalculator
from screener import StockScreener
from visualization import ResultVisualizer
from config import AVAILABLE_INDICES, MOMENTUM_PERIODS, QUALITY_METRICS, SCREENER_CONFIG

# Configuration de la page
st.set_page_config(
    page_title="Momentum Picks Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #26A69A;
    }
    .info-text {
        font-size: 1rem;
        color: #546E7A;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger et traiter les données
@st.cache_data(ttl=3600)  # Cache d'une heure
def load_and_process_data(index, api_key, min_momentum, min_quality, min_combined, momentum_weight, quality_weight, top_n):
    """
    Charge et traite les données pour l'indice spécifié
    
    Returns:
        tuple: (results_df, sorted_stocks, processing_times)
    """
    # Suivi des temps de traitement
    processing_times = {}
    
    # Chargement des données
    start_time = time.time()
    loader = DataLoader(api_key=api_key)
    stock_data = loader.load_data_for_index(index, with_fundamentals=True)
    processing_times['data_loading'] = time.time() - start_time
    
    if not stock_data:
        return None, None, processing_times
    
    # Calculateurs
    momentum_calc = MomentumCalculator()
    quality_calc = QualityCalculator()
    
    # Calcul des scores
    start_time = time.time()
    for symbol, data in stock_data.items():
        try:
            # Calcul du score Momentum
            momentum_scores = momentum_calc.calculate_momentum_score(data)
            stock_data[symbol]['momentum'] = momentum_scores
            
            # Calcul du score Quality
            quality_scores = quality_calc.calculate_quality_score(data)
            stock_data[symbol]['quality'] = quality_scores
        except Exception as e:
            stock_data[symbol]['momentum'] = None
            stock_data[symbol]['quality'] = None
    
    processing_times['score_calculation'] = time.time() - start_time
    
    # Filtrer les actions sans scores
    filtered_data = {symbol: data for symbol, data in stock_data.items() 
                    if data.get('momentum') is not None and data.get('quality') is not None}
    
    # Screening
    start_time = time.time()
    screener = StockScreener(
        momentum_weight=momentum_weight,
        quality_weight=quality_weight
    )
    
    results_df, sorted_stocks = screener.screen_stocks(
        filtered_data,
        min_momentum=min_momentum,
        min_quality=min_quality,
        min_combined=min_combined,
        top_n=top_n
    )
    processing_times['screening'] = time.time() - start_time
    
    return results_df, filtered_data, processing_times

# Titre de l'application
st.markdown('<p class="main-header">Momentum Picks Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    "Un screener d'actions basé sur les facteurs **Momentum** et **Quality** "
    "pour sélectionner les meilleures opportunités d'investissement."
)

# Barre latérale pour les paramètres
st.sidebar.title("Paramètres")

# Clé API
api_key = st.sidebar.text_input(
    "Clé API Alpha Vantage",
    value=os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
    type="password",
    help="Entrez votre clé API Alpha Vantage. Vous pouvez en obtenir une gratuitement sur alphavantage.co"
)

# Sélection de l'indice
selected_index = st.sidebar.selectbox(
    "Indice boursier",
    options=list(AVAILABLE_INDICES.keys()),
    format_func=lambda x: AVAILABLE_INDICES[x],
    help="Sélectionnez l'indice boursier à analyser"
)

# Options avancées
with st.sidebar.expander("Options avancées"):
    # Poids des scores
    momentum_weight = st.slider(
        "Poids du score Momentum",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="Importance relative du facteur Momentum dans le score combiné"
    )
    quality_weight = st.slider(
        "Poids du score Quality",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="Importance relative du facteur Quality dans le score combiné"
    )
    
    # Filtres
    min_momentum = st.slider(
        "Score minimum de Momentum",
        min_value=0.0,
        max_value=1.0,
        value=SCREENER_CONFIG["momentum_threshold"],
        step=0.05,
        help="Score minimal de Momentum pour inclure une action dans les résultats"
    )
    min_quality = st.slider(
        "Score minimum de Quality",
        min_value=0.0,
        max_value=1.0,
        value=SCREENER_CONFIG["quality_threshold"],
        step=0.05,
        help="Score minimal de Quality pour inclure une action dans les résultats"
    )
    min_combined = st.slider(
        "Score combiné minimum",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Score combiné minimal pour inclure une action dans les résultats"
    )
    
    # Nombre d'actions à afficher
    top_n = st.slider(
        "Nombre d'actions à afficher",
        min_value=5,
        max_value=50,
        value=SCREENER_CONFIG["top_picks_count"],
        step=5,
        help="Nombre d'actions à inclure dans les résultats"
    )

# Bouton pour lancer l'analyse
if st.sidebar.button("Lancer l'analyse", type="primary"):
    if not api_key:
        st.error("Veuillez entrer une clé API Alpha Vantage pour continuer.")
    else:
        # Afficher un état de progression
        with st.spinner(f"Analyse de l'indice {AVAILABLE_INDICES[selected_index]} en cours..."):
            # Charger et traiter les données
            results_df, stock_data, processing_times = load_and_process_data(
                selected_index, api_key, min_momentum, min_quality, 
                min_combined, momentum_weight, quality_weight, top_n
            )
            
            if results_df is None or results_df.empty:
                st.error("Aucune donnée n'a pu être chargée. Vérifiez votre clé API et votre connexion.")
            else:
                # Afficher les résultats
                st.success(f"Analyse de {len(stock_data)} actions terminée avec succès!")
                
                # Afficher un résumé
                st.markdown('<p class="sub-header">Résultats de l\'analyse</p>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Nombre d'actions analysées", len(stock_data))
                col2.metric("Nombre d'actions sélectionnées", len(results_df))
                col3.metric("Score combiné moyen", f"{results_df['Combined_Score'].mean():.2f}")
                
                # Tableau des résultats
                st.markdown('<p class="sub-header">Top actions sélectionnées</p>', unsafe_allow_html=True)
                st.dataframe(
                    results_df[['Symbol', 'Name', 'Sector', 'Combined_Score', 'Momentum_Score', 'Quality_Score']].head(top_n),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualisations
                st.markdown('<p class="sub-header">Visualisations</p>', unsafe_allow_html=True)
                tab1, tab2, tab3 = st.tabs(["Momentum vs Quality", "Top actions", "Analyse sectorielle"])
                
                with tab1:
                    # Scatter plot interactif
                    fig = px.scatter(
                        results_df, 
                        x="Momentum_Score", 
                        y="Quality_Score", 
                        color="Combined_Score",
                        size="Combined_Score",
                        hover_name="Symbol",
                        hover_data=["Name", "Sector"],
                        title=f"Momentum vs Quality Scores - {AVAILABLE_INDICES[selected_index]}",
                        color_continuous_scale="viridis"
                    )
                    
                    # Améliorer la mise en page
                    fig.update_layout(
                        height=600,
                        xaxis_title="Score Momentum",
                        yaxis_title="Score Quality",
                        coloraxis_colorbar_title="Score combiné"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Graphique en barres des meilleurs scores
                    top_10 = results_df.sort_values('Combined_Score', ascending=False).head(10)
                    
                    fig = go.Figure()
                    
                    # Ajouter les barres pour le score de Momentum
                    fig.add_trace(go.Bar(
                        x=top_10['Symbol'],
                        y=top_10['Momentum_Score'],
                        name='Score Momentum',
                        marker_color='rgb(55, 83, 109)'
                    ))
                    
                    # Ajouter les barres pour le score de Quality
                    fig.add_trace(go.Bar(
                        x=top_10['Symbol'],
                        y=top_10['Quality_Score'],
                        name='Score Quality',
                        marker_color='rgb(26, 118, 255)'
                    ))
                    
                    # Ajouter les barres pour le score combiné
                    fig.add_trace(go.Bar(
                        x=top_10['Symbol'],
                        y=top_10['Combined_Score'],
                        name='Score Combiné',
                        marker_color='rgb(56, 166, 165)'
                    ))
                    
                    # Mise en page
                    fig.update_layout(
                        title=f"Top 10 actions - {AVAILABLE_INDICES[selected_index]}",
                        xaxis_title="Symbole",
                        yaxis_title="Score",
                        barmode='group',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    if 'Sector' in results_df.columns:
                        # Calculer les statistiques par secteur
                        sector_stats = results_df.groupby('Sector')[
                            ['Momentum_Score', 'Quality_Score', 'Combined_Score']
                        ].agg(['mean', 'count', 'std']).reset_index()
                        
                        # Mettre en forme pour Plotly
                        sector_stats_flat = pd.DataFrame({
                            'Sector': sector_stats['Sector'],
                            'Momentum_Mean': sector_stats['Momentum_Score']['mean'],
                            'Quality_Mean': sector_stats['Quality_Score']['mean'],
                            'Combined_Mean': sector_stats['Combined_Score']['mean'],
                            'Count': sector_stats['Momentum_Score']['count']
                        })
                        
                        # Trier par score combiné moyen
                        sector_stats_flat = sector_stats_flat.sort_values('Combined_Mean', ascending=False)
                        
                        # Créer un graphique en barres par secteur
                        fig = px.bar(
                            sector_stats_flat,
                            x='Sector',
                            y=['Momentum_Mean', 'Quality_Mean', 'Combined_Mean'],
                            title=f"Scores moyens par secteur - {AVAILABLE_INDICES[selected_index]}",
                            barmode='group',
                            color_discrete_sequence=['rgb(55, 83, 109)', 'rgb(26, 118, 255)', 'rgb(56, 166, 165)'],
                            hover_data=['Count']
                        )
                        
                        # Mise en page
                        fig.update_layout(
                            height=500,
                            xaxis_title="Secteur",
                            yaxis_title="Score moyen",
                            legend_title="Métrique"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Données sectorielles non disponibles.")
                
                # Option pour télécharger les résultats
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Télécharger les résultats (CSV)",
                    data=csv,
                    file_name=f"momentum_picks_{selected_index}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
                
                # Afficher les temps de traitement
                with st.expander("Détails du traitement"):
                    st.markdown('<p class="info-text">Temps de traitement (secondes):</p>', unsafe_allow_html=True)
                    st.json(processing_times)
else:
    # Afficher des informations explicatives lorsque l'analyse n'est pas lancée
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("""
    ### Comment utiliser cet outil
    
    1. Entrez votre clé API Alpha Vantage dans la barre latérale
    2. Sélectionnez l'indice boursier que vous souhaitez analyser
    3. Ajustez les paramètres avancés selon vos besoins
    4. Cliquez sur "Lancer l'analyse" pour obtenir les résultats
    
    L'analyse combine deux facteurs principaux:
    
    - **Momentum**: Évalue la dynamique des prix, des volumes et des fondamentaux
    - **Quality**: Évalue la qualité financière des entreprises (ROE, marge bénéficiaire, etc.)
    
    Les actions sont notées et classées en fonction de ces critères pour identifier les meilleures opportunités d'investissement.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Afficher une image ou un graphique exemple
    st.image("https://s3-us-west-2.amazonaws.com/public.model-thinking.com/course_artwork/stock-market.png", use_column_width=True)

# Pied de page
st.sidebar.markdown("""
---
### À propos

Version: 1.0.0  
[Code source](https://github.com/Kyac99/momentum-picks-analyzer)

Développé avec ❤️ par [Kyac99](https://github.com/Kyac99)
""")

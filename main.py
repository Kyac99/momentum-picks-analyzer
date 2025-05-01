"""
Module principal pour exécuter le screener d'actions Momentum Picks
"""

import os
import time
import pandas as pd
import argparse
from datetime import datetime

from data_loader import DataLoader
from momentum import MomentumCalculator
from quality import QualityCalculator
from screener import StockScreener
from visualization import ResultVisualizer

def parse_arguments():
    """
    Parse les arguments de ligne de commande
    
    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(description='Momentum Picks - Screener d\'actions basé sur les facteurs Momentum et Quality')
    
    parser.add_argument('--index', type=str, default='SP500',
                        choices=['SP500', 'NASDAQ', 'EUROSTOXX50', 'MSCI_TECH'],
                        help='Indice à analyser')
    
    parser.add_argument('--top', type=int, default=10,
                        help='Nombre d\'actions à retenir dans le classement final')
    
    parser.add_argument('--min-momentum', type=float, default=0.0,
                        help='Score minimum de Momentum pour le filtrage')
    
    parser.add_argument('--min-quality', type=float, default=0.0,
                        help='Score minimum de Quality pour le filtrage')
    
    parser.add_argument('--min-combined', type=float, default=0.0,
                        help='Score minimum combiné pour le filtrage')
    
    parser.add_argument('--momentum-weight', type=float, default=0.6,
                        help='Poids du score Momentum dans le score combiné')
    
    parser.add_argument('--quality-weight', type=float, default=0.4,
                        help='Poids du score Quality dans le score combiné')
    
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Répertoire de sortie pour les résultats')
    
    parser.add_argument('--api-key', type=str,
                        help='Clé API Alpha Vantage')
    
    parser.add_argument('--no-plots', action='store_true',
                        help='Désactiver la génération des graphiques')
    
    return parser.parse_args()

def main():
    """
    Fonction principale du programme
    """
    # Récupérer les arguments
    args = parse_arguments()
    
    # Récupérer la clé API
    api_key = args.api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        api_key = input("Entrez votre clé API Alpha Vantage: ")
    
    # Créer le répertoire de sortie
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp pour les noms de fichiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialiser le chargeur de données
    print(f"Initialisation du chargeur de données pour l'indice {args.index}...")
    loader = DataLoader(api_key=api_key)
    
    # Charger les données
    print(f"Chargement des données pour l'indice {args.index}...")
    stock_data = loader.load_data_for_index(args.index, with_fundamentals=True)
    
    if not stock_data:
        print("Aucune donnée n'a pu être chargée. Vérifiez votre connexion et votre clé API.")
        return
    
    print(f"Données chargées pour {len(stock_data)} actions.")
    
    # Initialiser les calculateurs
    momentum_calc = MomentumCalculator()
    quality_calc = QualityCalculator()
    
    # Calculer les scores pour chaque action
    print("Calcul des scores Momentum et Quality...")
    for symbol, data in stock_data.items():
        # Calcul du score Momentum
        try:
            momentum_scores = momentum_calc.calculate_momentum_score(data)
            stock_data[symbol]['momentum'] = momentum_scores
        except Exception as e:
            print(f"Erreur lors du calcul du score Momentum pour {symbol}: {str(e)}")
            stock_data[symbol]['momentum'] = None
        
        # Calcul du score Quality
        try:
            quality_scores = quality_calc.calculate_quality_score(data)
            stock_data[symbol]['quality'] = quality_scores
        except Exception as e:
            print(f"Erreur lors du calcul du score Quality pour {symbol}: {str(e)}")
            stock_data[symbol]['quality'] = None
    
    # Filtrer les actions sans scores
    filtered_data = {symbol: data for symbol, data in stock_data.items() 
                    if data.get('momentum') is not None and data.get('quality') is not None}
    
    if not filtered_data:
        print("Aucune action n'a pu être analysée. Vérifiez les données d'entrée.")
        return
    
    print(f"Scores calculés pour {len(filtered_data)} actions sur {len(stock_data)}.")
    
    # Initialiser le screener
    print("Initialisation du screener...")
    screener = StockScreener(
        momentum_weight=args.momentum_weight,
        quality_weight=args.quality_weight
    )
    
    # Exécuter le screening
    print("Exécution du screening...")
    results_df, sorted_stocks = screener.screen_stocks(
        filtered_data,
        min_momentum=args.min_momentum,
        min_quality=args.min_quality,
        min_combined=args.min_combined,
        top_n=args.top
    )
    
    # Enregistrer les résultats
    results_file = os.path.join(output_dir, f"momentum_picks_{args.index}_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    
    print(f"Résultats enregistrés dans {results_file}")
    print("\nTop actions sélectionnées:")
    print(results_df[['Symbol', 'Combined_Score', 'Momentum_Score', 'Quality_Score']].head(args.top))
    
    # Générer les visualisations
    if not args.no_plots:
        print("\nGénération des visualisations...")
        visualizer = ResultVisualizer()
        figures_dir = os.path.join(output_dir, f"figures_{args.index}_{timestamp}")
        
        try:
            file_paths = visualizer.save_all_figures(results_df, output_dir=figures_dir)
            print(f"Visualisations enregistrées dans {figures_dir}")
        except Exception as e:
            print(f"Erreur lors de la génération des visualisations: {str(e)}")
    
    print("\nTraitement terminé avec succès!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraitement interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\nUne erreur s'est produite: {str(e)}")

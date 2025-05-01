# Momentum Picks Analyzer

Un screener d'actions basé sur les facteurs **Momentum** et **Quality** pour sélectionner les meilleures opportunités d'investissement parmi plusieurs indices boursiers mondiaux.

## Objectif

Cet outil Python permet d'analyser des actions à travers différents indices en s'appuyant sur les facteurs Momentum et Quality, dans une approche Evidence-Based Investing.

## Fonctionnalités

- Importation des données de différents indices (Nasdaq, S&P 500, Eurostoxx 50, etc.)
- Calcul des scores Momentum (technique et fondamental)
- Calcul des scores Quality (ROE, ROCE, marge nette, etc.)
- Filtrage et sélection des meilleures opportunités d'investissement
- Visualisation des résultats sous forme de tableaux et graphiques

## Structure du projet

- `data_loader.py` : récupération des données par API
- `momentum.py` : calcul du score de momentum (technique + fondamental)
- `quality.py` : calcul du score de qualité
- `screener.py` : combinaison, filtrage et sélection finale
- `visualization.py` : production des graphiques
- `main.py` ou `notebook.ipynb` : exécution complète

## Installation

```bash
git clone https://github.com/Kyac99/momentum-picks-analyzer.git
cd momentum-picks-analyzer
pip install -r requirements.txt
```

## Utilisation

```bash
python main.py
```

ou ouvrir et exécuter le notebook `notebook.ipynb`.

## Technologies utilisées

- Python 3.9+
- Pandas, NumPy
- Requests, Alpha Vantage API
- Matplotlib, Seaborn, Plotly
- Scikit-learn (pour la normalisation des scores)

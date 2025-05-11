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
- Interface web Streamlit pour une utilisation facile
- Notebook Jupyter pour l'exploration interactive des données
- Tests unitaires pour assurer la fiabilité du code

## Structure du projet

- `data_loader.py` : récupération des données par API
- `momentum.py` : calcul du score de momentum (technique + fondamental)
- `quality.py` : calcul du score de qualité
- `screener.py` : combinaison, filtrage et sélection finale
- `visualization.py` : production des graphiques
- `config.py` : configuration globale du projet
- `main.py` : exécution en ligne de commande
- `notebook.ipynb` : exploration interactive 
- `streamlit_app.py` : interface web
- `tests/` : tests unitaires

## Installation

```bash
git clone https://github.com/Kyac99/momentum-picks-analyzer.git
cd momentum-picks-analyzer
pip install -r requirements.txt
```

## Utilisation

### En ligne de commande

```bash
python main.py --index SP500 --top 20 --momentum-weight 0.6 --quality-weight 0.4
```

Options disponibles:
- `--index` : Indice à analyser (SP500, NASDAQ, EUROSTOXX50, MSCI_TECH)
- `--top` : Nombre d'actions à retenir dans le classement final
- `--min-momentum` : Score minimum de Momentum pour le filtrage
- `--min-quality` : Score minimum de Quality pour le filtrage
- `--min-combined` : Score minimum combiné pour le filtrage
- `--momentum-weight` : Poids du score Momentum dans le score combiné
- `--quality-weight` : Poids du score Quality dans le score combiné
- `--output-dir` : Répertoire de sortie pour les résultats
- `--api-key` : Clé API Alpha Vantage
- `--no-plots` : Désactiver la génération des graphiques

### Interface web Streamlit

```bash
streamlit run streamlit_app.py
```

L'interface web vous permet d'interagir facilement avec l'outil sans ligne de commande, avec des visualisations interactives.

### Notebook Jupyter

```bash
jupyter notebook notebook.ipynb
```

Le notebook vous permet d'explorer les données et les résultats de manière interactive, de personnaliser les analyses et de créer vos propres visualisations.

## Tests

Pour exécuter les tests unitaires:

```bash
pytest
```

Ou pour une sortie plus détaillée:

```bash
pytest -v
```

## Technologies utilisées

- Python 3.9+
- Pandas, NumPy
- Requests, Alpha Vantage API
- Matplotlib, Seaborn, Plotly
- Scikit-learn (pour la normalisation des scores)
- Streamlit (pour l'interface web)
- Jupyter (pour le notebook interactif)
- Pytest (pour les tests unitaires)

## Contribution

Les contributions sont les bienvenues! N'hésitez pas à soumettre des pull requests ou à ouvrir des issues pour signaler des bugs ou proposer des améliorations.

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier LICENSE pour plus de détails.

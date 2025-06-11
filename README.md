# Car Price Estimation LLM

Ce projet vise à développer un modèle de langage (LLM) spécialisé dans l'estimation du prix des voitures d'occasion. Le modèle sera capable d'analyser les caractéristiques d'un véhicule et de fournir une estimation précise de son prix basée sur les données du marché.

## Fonctionnalités

- Estimation du prix des voitures basée sur :
  - Marque et modèle
  - Année
  - Kilométrage
  - Type de carburant
  - Transmission
  - État général
  - Caractéristiques spécifiques
- Analyse des tendances du marché
- Recommandations d'achat/vente
- Interface conversationnelle naturelle

## Structure du projet

```
.
├── data/               # Données brutes et prétraitées
├── models/            # Modèles entraînés et checkpoints
├── notebooks/         # Notebooks Jupyter pour l'analyse et l'expérimentation
├── src/              # Code source
│   ├── preprocessing/ # Scripts de prétraitement des données
│   ├── training/     # Scripts d'entraînement
│   ├── inference/    # Scripts d'inférence
│   └── utils/        # Utilitaires
├── tests/            # Tests unitaires et d'intégration
└── requirements.txt  # Dépendances du projet
```

## Dataset

Le projet utilise plusieurs sources de données :
- Dataset principal de 90 000+ voitures (1970-2024)
- Données de ventes aux enchères
- Données de marché en temps réel

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

[À compléter avec les instructions d'utilisation]

## Licence

[À définir] 
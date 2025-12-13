# Credit Scoring Model

Ce projet implémente un pipeline complet de Credit Scoring, de la préparation des données au déploiement du modèle, en passant par l'entraînement, l'optimisation et l'explicabilité (SHAP).

## Structure du Projet

- `data/` : Données brutes et procesées.
- `notebooks/` :
  - `01_v2_data_preparation.ipynb` : Préparation des données (Feature Engineering).
  - `02_model_training.ipynb` : Entraînement, Optimisation (Optuna) et Tracking (MLflow).
  - `03_explainability.ipynb` : Analyse SHAP (Globale et Locale).
  - `04_mlflow_serving_test.ipynb` : Test de l'API de prédiction.
- `src/` : Code modulaire (`model_utils.py`, `explainability.py`, etc.).
- `models/` : Artefacts des modèles entraînés.
- `mlruns/` : Tracking MLflow local.
- `reports/` : Figures et analyses.

## Installation

1. Cloner le projet.
2. Installer les dépendances :
```bash
pip install -r requirements.txt
```
3. Lancer MLflow UI (optionnel pour visualiser les runs) :
```bash
mlflow ui
```

## Étapes d'Exécution

### 1. Entraînement
Ouvrir et exécuter `notebooks/02_model_training.ipynb`.
- Entraîne plusieurs modèles (Dummy, Logistic Regression, XGBoost, LightGBM).
- Régression logistique via imputation.
- Optimise LightGBM avec Optuna.
- Sauvegarde le meilleur modèle dans `models/best_model.pkl` et `models/final_model` (format MLflow).

### 2. Explicabilité
Ouvrir et exécuter `notebooks/03_explainability.ipynb`.
- Génère les graphiques SHAP (Global Feature Importance, Beeswarm).
- Génère des explications locales pour des clients spécifiques.
- Les figures sont sauvegardées dans `reports/figures`.

### 3. Déploiement (Docker)
Une fois le modèle sauvegardé dans `models/final_model` :

**Build :**
```bash
docker build -t credit-scoring-serving .
```

**Run (Serving) :**
Sur Windows (PowerShell) :
```powershell
docker run -v ${PWD}/models/final_model:/model -p 5000:5000 credit-scoring-serving
```
Sur Linux/Mac :
```bash
docker run -v $(pwd)/models/final_model:/model -p 5000:5000 credit-scoring-serving
```

### 4. Test API
Ouvrir `notebooks/04_mlflow_serving_test.ipynb` pour envoyer des requêtes au conteneur Docker et obtenir des prédictions.

## Auteur
Aubin Hérault, Gael Le Reun, Thomas Bertho

# Credit Scoring Pipeline - Exécution complète du workflow

import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib
import mlflow

# Ajout du dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_prep import load_and_feature_engineering, reduce_mem_usage
from model_utils import (
    get_train_val_test_split,
    train_dummy,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    optimize_lightgbm
)
from explainability import compute_shap, plot_shap_global
from config import RANDOM_SEED


# Étape 1: Chargement et Feature Engineering
def step_1_data_preparation() -> tuple:
    print("\nÉTAPE 1: PRÉPARATION DES DONNÉES")
    
    # Chargement et feature engineering
    df = load_and_feature_engineering()
    df = reduce_mem_usage(df)
    
    # Séparation X / y
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    y = df['TARGET']
    
    # Nettoyage des noms de colonnes pour LightGBM
    X.columns = ["".join(c if c.isalnum() else "_" for c in str(col)) for col in X.columns]
    
    print(f"Dataset final: {X.shape[0]} lignes, {X.shape[1]} colonnes")
    print(f"Distribution TARGET: {y.value_counts().to_dict()}")
    
    return X, y


# Étape 2: Entraînement et Optimisation
def step_2_training(X: pd.DataFrame, y: pd.Series, n_trials: int = 10) -> tuple:
    print("\nÉTAPE 2: ENTRAÎNEMENT ET OPTIMISATION")
    
    # Split Train/Val/Test
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_split(X, y)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Configuration MLflow
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("Credit_Scoring_Pipeline")
    
    # Benchmark des modèles de base
    print("\nBenchmark des modèles")
    results = []
    
    _, metrics_dummy = train_dummy(X_train, y_train, X_val, y_val, X_test, y_test, "pipeline")
    results.append({"model": "Dummy", **metrics_dummy})
    
    _, metrics_rf = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, "pipeline")
    results.append({"model": "RandomForest", **metrics_rf})
    
    _, metrics_xgb = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, "pipeline")
    results.append({"model": "XGBoost", **metrics_xgb})
    
    _, metrics_lgb = train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, "pipeline")
    results.append({"model": "LightGBM", **metrics_lgb})
    
    # Affichage des résultats
    df_results = pd.DataFrame(results)
    print("\nRésultats Benchmark")
    print(df_results[["model", "auc", "recall", "business_cost"]].to_string(index=False))
    
    # Optimisation LightGBM avec Optuna
    print(f"\nOptimisation Optuna ({n_trials} trials)")
    best_params = optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=n_trials)
    
    # Entraînement final avec les meilleurs paramètres
    final_params = best_params.copy()
    final_params.update({
        "metric": "custom",
        "objective": "binary",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
        "n_estimators": 1000
    })
    
    print("\nEntraînement modèle final")
    final_model, final_metrics = train_lightgbm(
        X_train, y_train, X_val, y_val, X_test, y_test,
        dataset_name="final_optimized",
        params=final_params
    )
    
    # Sauvegarde du modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, "models/best_model.pkl")
    mlflow.sklearn.save_model(final_model, "models/final_model")
    print("Modèle sauvegardé dans models/best_model.pkl")
    
    return final_model, X_test, y_test, final_metrics


# Étape 3: Explicabilité SHAP
def step_3_explainability(model, X_test: pd.DataFrame, n_samples: int = 1000):
    print("\nÉTAPE 3: EXPLICABILITÉ SHAP")
    
    # Échantillon pour SHAP (performance)
    X_sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=RANDOM_SEED)
    
    # Calcul SHAP
    print("Calcul des valeurs SHAP...")
    explainer, shap_values = compute_shap(model, X_sample)
    
    # Graphique global
    os.makedirs("reports/figures", exist_ok=True)
    plot_shap_global(shap_values, X_sample, "reports/figures/shap_global.png")
    
    print("Graphiques SHAP générés dans reports/figures/")


def main():
    parser = argparse.ArgumentParser(description="Credit Scoring Pipeline")
    parser.add_argument("--skip-training", action="store_true", help="Charger le modèle existant")
    parser.add_argument("--n-trials", type=int, default=10, help="Nombre de trials Optuna")
    parser.add_argument("--skip-shap", action="store_true", help="Passer l'étape SHAP")
    args = parser.parse_args()
    
    print("\nCREDIT SCORING PIPELINE")
    
    # Étape 1: Préparation des données
    X, y = step_1_data_preparation()
    
    # Étape 2: Entraînement (ou chargement)
    if args.skip_training:
        print("\nChargement du modèle existant")
        model = joblib.load("models/best_model.pkl")
        _, _, X_test, _, _, y_test = get_train_val_test_split(X, y)
        final_metrics = None
    else:
        model, X_test, y_test, final_metrics = step_2_training(X, y, n_trials=args.n_trials)
    
    # Étape 3: Explicabilité
    if not args.skip_shap:
        step_3_explainability(model, X_test)
    
    # Résumé final
    print("\nPIPELINE TERMINÉ")

    if final_metrics:
        print(f"AUC: {final_metrics['auc']:.4f}")
        print(f"Recall: {final_metrics['recall']:.4f}")
        print(f"Business Cost: {final_metrics['business_cost']}")
    print("\nFichiers générés:")
    print("  - models/best_model.pkl")
    print("  - models/final_model/")
    print("  - reports/figures/shap_global.png")
    print("  - mlruns/ (MLflow tracking)")


if __name__ == "__main__":
    main()

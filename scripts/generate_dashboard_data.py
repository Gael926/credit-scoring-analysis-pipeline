# Script de génération des données pour le dashboard Streamlit
import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_prep import load_and_feature_engineering, reduce_mem_usage
from model_utils import get_train_val_test_split
from config import RANDOM_SEED


def main():
    print("Génération des données pour le dashboard...")
    
    # Création du dossier de sortie
    os.makedirs("dashboard_data", exist_ok=True)
    
    # Chargement des données
    print("Chargement et feature engineering...")
    df = load_and_feature_engineering()
    df = reduce_mem_usage(df)
    
    # Séparation X / y
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    y = df['TARGET']
    
    # Récupération des SK_ID_CURR avant nettoyage
    client_ids = df['SK_ID_CURR'].values
    
    # Nettoyage des noms de colonnes pour LightGBM
    X.columns = ["".join(c if c.isalnum() else "_" for c in str(col)) for col in X.columns]
    feature_names = X.columns.tolist()
    
    # Split pour récupérer le Test Set
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_split(X, y)
    
    # Calcul des médianes sur le Train Set (pas de fuite)
    print("Calcul des médianes...")
    medians = X_train.median().to_dict()
    
    # Conversion des valeurs numpy en types Python natifs pour JSON
    medians_clean = {}
    for k, v in medians.items():
        if pd.isna(v):
            medians_clean[k] = 0.0
        elif isinstance(v, (np.floating, np.integer)):
            medians_clean[k] = float(v)
        else:
            medians_clean[k] = v
    
    # Sauvegarde des défauts
    defaults = {
        "features": feature_names,
        "medians": medians_clean
    }
    
    with open("dashboard_data/feature_defaults.json", "w") as f:
        json.dump(defaults, f, indent=2)
    print(f"Fichier feature_defaults.json créé ({len(feature_names)} features)")
    
    # Échantillon de clients du Test Set
    print("Création de l'échantillon de clients...")
    test_indices = X_test.index.tolist()
    sample_size = min(50, len(test_indices))
    sample_indices = np.random.RandomState(RANDOM_SEED).choice(test_indices, sample_size, replace=False)
    
    # Création du DataFrame échantillon avec SK_ID_CURR
    sample_df = X.loc[sample_indices].copy()
    sample_df.insert(0, 'SK_ID_CURR', client_ids[sample_indices])
    sample_df['TARGET'] = y.loc[sample_indices].values
    
    sample_df.to_csv("dashboard_data/sample_clients.csv", index=False)
    print(f"Fichier sample_clients.csv créé ({sample_size} clients)")
    
    print("\nDonnées générées avec succès dans dashboard_data/")


if __name__ == "__main__":
    main()

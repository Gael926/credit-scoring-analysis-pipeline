import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import joblib
import os

try:
    from metrics import calculate_auc, calculate_recall, calculate_f1, business_cost_metric
except ImportError:
    from src.metrics import calculate_auc, calculate_recall, calculate_f1, business_cost_metric

RANDOM_SEED = 42

def get_train_val_test_split(X, y):
    """
    Split les données en Train (70%), Validation (15%), Test (15%) de manière stratifiée.
    """
    # Split 1: Train (70%) vs Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    
    # Split 2: Temp en Val (50% de 30% = 15%) et Test (50% de 30% = 15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(model, X, y, threshold=0.5):
    """Calcule les métriques pour un modèle donné."""
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        # Pour les modèles qui ne supportent pas predict_proba (rare en classif)
        y_prob = model.predict(X)
        
    y_pred = (y_prob > threshold).astype(int)
    
    metrics = {
        "auc": calculate_auc(y, y_prob),
        "recall": calculate_recall(y, y_pred),
        "f1": calculate_f1(y, y_pred),
        "business_cost": business_cost_metric(y, y_prob, threshold)
    }
    return metrics

def train_and_log_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, params=None):
    """
    Entraîne un modèle, loggue les paramètres et métriques dans MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        print(f"Entraînement de {model_name}...")
        
        if params:
            mlflow.log_params(params)
        
        # Entraînement
        # Gestion spécifique pour LightGBM et XGBoost pour utiliser l'early stopping si dispo
        if "LightGBM" in model_name:
             model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
            )
        elif "XGBoost" in model_name:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
            
        # Évaluation sur le Test Set (ou Val selon besoin, ici Test pour score final)
        metrics = evaluate_model(model, X_test, y_test)
        
        # Logging MLflow
        mlflow.log_metrics(metrics)
        
        # Sauvegarde du modèle
        if "LightGBM" in model_name:
            mlflow.lightgbm.log_model(model, "model")
        elif "XGBoost" in model_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
            
        print(f"Metrics {model_name}: {metrics}")
        return model, metrics

def train_dummy(X_train, y_train, X_val, y_val, X_test, y_test):
    model = DummyClassifier(strategy='most_frequent')
    return train_and_log_model(model, "Dummy", X_train, y_train, X_val, y_val, X_test, y_test)

def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    params = {
        "C": 1.0, 
        "solver": "liblinear", 
        "class_weight": "balanced", 
        "random_state": RANDOM_SEED
        }
    model = LogisticRegression(**params)
    return train_and_log_model(model, "LogisticRegression", X_train, y_train, X_val, y_val, X_test, y_test, params)

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    params = {
        "n_estimators": 100,
        "max_depth": 10, 
        "class_weight": "balanced", 
        "random_state": RANDOM_SEED, 
        "n_jobs": -1
        }
    model = RandomForestClassifier(**params)
    return train_and_log_model(model, "RandomForest", X_train, y_train, X_val, y_val, X_test, y_test, params)

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    # scale_pos_weight pour le déséquilibre
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    params = {
        "n_estimators": 1000, 
        "learning_rate": 0.05, 
        "max_depth": 6, 
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
        "eval_metric": "auc"
    }
    # On retire les params non supportés par le constructeur sklearn wrapper directement si besoin
    # Mais XGBClassifier les accepte souvent. early_stopping_rounds se passe au fit.
    model_params = {k:v for k,v in params.items() if k not in ["early_stopping_rounds", "eval_metric"]}
    model = XGBClassifier(**model_params)
    
    return train_and_log_model(model, "XGBoost", X_train, y_train, X_val, y_val, X_test, y_test, params)

def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1
    }
    model = lgb.LGBMClassifier(**params)
    return train_and_log_model(model, "LightGBM", X_train, y_train, X_val, y_val, X_test, y_test, params)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import joblib
import os
import optuna

try:
    from metrics import calculate_auc, calculate_recall, calculate_f1, business_cost_metric, calculate_accuracy
except ImportError:
    from src.metrics import calculate_auc, calculate_recall, calculate_f1, business_cost_metric, calculate_accuracy

RANDOM_SEED = 42
COST_WEIGHTS = {0: 1, 1: 10}

def lgb_custom_metric(y_true, y_pred):
    """Métrique custom pour LightGBM: Cost (Lower is better)."""
    # y_pred sont les probabilités de la classe 1
    cost = business_cost_metric(y_true, y_pred, threshold=0.5)
    return "business_cost", cost, False

def find_best_threshold(y_true, y_prob):
    """Trouve le seuil qui minimise le coût métier sur un set donné."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = [business_cost_metric(y_true, y_prob, t) for t in thresholds]
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]

def get_train_val_test_split(X, y):
    """Split 70/15/15 standard."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(model, X, y, threshold=0.5):
    """Calcule AUC, Recall, F1, Accuracy et Coût Métier."""
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.predict(X)
        
    y_pred = (y_prob > threshold).astype(int)
    
    return {
        "auc": calculate_auc(y, y_prob),
        "recall": calculate_recall(y, y_pred),
        "f1": calculate_f1(y, y_pred),
        "accuracy": calculate_accuracy(y, y_pred),
        "business_cost": business_cost_metric(y, y_prob, threshold)
    }

def train_and_log(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, params=None, dataset_name="v2"):
    """Entraînement avec recherche de seuil et logging MLflow."""
    run_name = f"{model_name}_{dataset_name}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"Entraînement {run_name}...")
        mlflow.set_tag("dataset", dataset_name)
        if params: mlflow.log_params(params)
        
        # Fit
        if "LightGBM" in model_name:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=lgb_custom_metric,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
            )
        elif "XGBoost" in model_name:
            model.fit(
                X_train, y_train, 
                eval_set=[(X_val, y_val)], 
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
            
        # --- Recherche du meilleur seuil sur le Validation Set ---
        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(X_val)[:, 1]
            best_thresh, min_cost = find_best_threshold(y_val, val_probs)
            print(f"Meilleur seuil trouvé (Val): {best_thresh:.2f} (Coût: {min_cost})")
            mlflow.log_param("best_threshold", best_thresh)
            mlflow.log_metric("val_best_cost", min_cost)
        else:
            best_thresh = 0.5
            
        # --- Evaluation Finale sur le Test Set avec ce seuil ---
        metrics = evaluate_model(model, X_test, y_test, threshold=best_thresh)
        mlflow.log_metrics(metrics)
        
        # Save
        if "LightGBM" in model_name: mlflow.lightgbm.log_model(model, "model")
        elif "XGBoost" in model_name: mlflow.xgboost.log_model(model, "model")
        else: mlflow.sklearn.log_model(model, "model")
            
        print(f"Metrics (Test): {metrics}")
        return model, metrics

# --- Fonctions d'entrainement simples ---

def train_dummy(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name):
    model = DummyClassifier(strategy='most_frequent')
    return train_and_log(model, "Dummy", X_train, y_train, X_val, y_val, X_test, y_test, {}, dataset_name)

def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', LogisticRegression(C=1.0, solver='liblinear', class_weight=COST_WEIGHTS, random_state=RANDOM_SEED))
    ])
    return train_and_log(pipeline, "LogisticRegression", X_train, y_train, X_val, y_val, X_test, y_test, {}, dataset_name)

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name):
    pipeline = Pipeline([
         ('imputer', SimpleImputer(strategy='median')),
         ('model', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight=COST_WEIGHTS, random_state=RANDOM_SEED, n_jobs=-1))
    ])
    return train_and_log(pipeline, "RandomForest", X_train, y_train, X_val, y_val, X_test, y_test, {}, dataset_name)

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name):
    scale_weight = 10 
    model = XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6, 
        scale_pos_weight=scale_weight, random_state=RANDOM_SEED, n_jobs=-1,
        eval_metric="auc", early_stopping_rounds=50
    )
    return train_and_log(model, "XGBoost", X_train, y_train, X_val, y_val, X_test, y_test, {}, dataset_name)

def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name, params=None):
    if params is None:
        params = {
            "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 31,
            "class_weight": COST_WEIGHTS, "random_state": RANDOM_SEED, "n_jobs": -1
        }
    if "class_weight" not in params:
        params["class_weight"] = COST_WEIGHTS
        
    model = lgb.LGBMClassifier(**params)
    return train_and_log(model, "LightGBM", X_train, y_train, X_val, y_val, X_test, y_test, params, dataset_name)

def train_lightgbm_cv(X, y, params=None, n_splits=5):
    """
    Cross-validation simple pour LightGBM.
    """
    print(f"Début Cross-Validation LightGBM ({n_splits} folds)...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    aucs = []
    costs = []
    
    if params is None:
        params = {
            "n_estimators": 1000, "learning_rate": 0.05, "class_weight": COST_WEIGHTS,
            "random_state": RANDOM_SEED, "n_jobs": -1, "verbosity": -1
        }
    if "class_weight" not in params:
        params["class_weight"] = COST_WEIGHTS
    
    params_cv = params.copy()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params_cv)
        model.fit(
            X_tr, y_tr, 
            eval_set=[(X_va, y_va)], 
            eval_metric=lgb_custom_metric,
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        preds = model.predict_proba(X_va)[:, 1]
        
        # Trouver le meilleur seuil pour ce fold
        best_t, min_c = find_best_threshold(y_va, preds)
        auc = calculate_auc(y_va, preds)
        
        aucs.append(auc)
        costs.append(min_c)
        print(f"Fold {fold+1}: BestThresh={best_t:.2f} -> Cost={min_c:.1f}, AUC={auc:.3f}")
        
    print(f"Moyenne CV (Best Thresholds): Cost={np.mean(costs):.1f}, AUC={np.mean(aucs):.3f}")
    return np.mean(aucs), np.mean(costs)

def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=30):
    """
    Optimisation Optuna LightGBM (Minimisation Coût).
    """
    print("Optimisation LightGBM (Optuna) - Minimisation Coût Métier...")
    
    def objective(trial):
        param = {
            "objective": "binary", "metric": "custom", "verbosity": -1,
            "boosting_type": "gbdt", "random_state": RANDOM_SEED, "n_jobs": -1,
            "class_weight": COST_WEIGHTS,
            "n_estimators": 1000,
            
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 50.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 50.0),
        }
        
        model = lgb.LGBMClassifier(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=lgb_custom_metric,
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        
        # Chercher le seuil qui minimise le coût
        _, min_c = find_best_threshold(y_val, preds)
        
        return min_c

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    print("Meilleurs params:", study.best_trial.params)
    return study.best_trial.params
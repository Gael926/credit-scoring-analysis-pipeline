import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.dummy import DummyClassifier
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
from typing import Tuple, Dict, List, Optional, Any
from numpy.typing import ArrayLike

try:
    from config import RANDOM_SEED, DEFAULT_SCALE_POS_WEIGHT
    from metrics import calculate_auc, calculate_recall, calculate_f1, business_cost_metric, calculate_accuracy
except ImportError:
    from src.config import RANDOM_SEED, DEFAULT_SCALE_POS_WEIGHT
    from src.metrics import calculate_auc, calculate_recall, calculate_f1, business_cost_metric, calculate_accuracy

# classe pour l'ensemble des modèles pour la cross validation
class Ensemble:
    def __init__(self, models: List[lgb.LGBMClassifier]) -> None:
        self.models = models
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = [m.predict_proba(X)[:, 1] for m in self.models]
        mean_preds = np.mean(preds, axis=0)
        return np.vstack([1-mean_preds, mean_preds]).T
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

# fonction de métrique custom pour LightGBM
def lgb_custom_metric(y_true: ArrayLike, y_pred: ArrayLike) -> Tuple[str, float, bool]:
    # y_pred sont les probabilités de la classe 1
    cost = business_cost_metric(y_true, y_pred, threshold=0.5)
    return "business_cost", cost, False

# fonction de recherche du meilleur seuil
def find_best_threshold(y_true: ArrayLike, y_prob: ArrayLike) -> Tuple[float, float]:
    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = [business_cost_metric(y_true, y_prob, t) for t in thresholds]
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]

# fonction de split 70/15/15 avec stratification
def get_train_val_test_split(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

#Calcule AUC, Recall, F1, Accuracy et Coût Métier
def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict[str, float]:
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

# fonction d'entraînement et logging MLflow
def train_and_log(model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, params: Optional[Dict[str, Any]] = None, dataset_name: str = "v2") -> Tuple[Any, Dict[str, float]]:
    # nom du run
    run_name = f"{model_name}_{dataset_name}"
    
    # debut du log mlflow
    with mlflow.start_run(run_name=run_name):
        print(f"Entraînement {run_name}...")
        mlflow.set_tag("dataset", dataset_name)
        if params: mlflow.log_params(params)
        
        # entrainement
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
            
        # recherche du meilleur seuil sur le Validation Set
        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(X_val)[:, 1]
            best_thresh, min_cost = find_best_threshold(y_val, val_probs)
            print(f"Meilleur seuil trouvé (Val): {best_thresh:.2f} (Coût: {min_cost})")
            mlflow.log_param("best_threshold", best_thresh)
            mlflow.log_metric("val_best_cost", min_cost)
        else:
            best_thresh = 0.5
            min_cost = None
            
        # evaluation finale sur le Test Set avec ce seuil
        metrics = evaluate_model(model, X_test, y_test, threshold=best_thresh)
        if min_cost is not None:
             metrics["val_best_cost"] = min_cost 
             
        mlflow.log_metrics(metrics)
        
        # enregistement des modèles
        if "LightGBM" in model_name: mlflow.lightgbm.log_model(model, "model")
        elif "XGBoost" in model_name: mlflow.xgboost.log_model(model, "model")
        else: mlflow.sklearn.log_model(model, "model")
            
        print(f"Metrics (Test): {metrics}")
        return model, metrics

# fonctions d'entrainement simples
def train_dummy(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, dataset_name: str) -> Tuple[DummyClassifier, Dict[str, float]]:
    model = DummyClassifier(strategy='most_frequent')
    return train_and_log(model, "Dummy", X_train, y_train, X_val, y_val, X_test, y_test, {}, dataset_name)

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, dataset_name: str) -> Tuple[Pipeline, Dict[str, float]]:
    pipeline = Pipeline([
         ('imputer', SimpleImputer(strategy='median')),
         ('model', RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            class_weight="balanced", 
            random_state=RANDOM_SEED, 
            n_jobs=-1
            )
        )
    ])
    return train_and_log(pipeline, "RandomForest", X_train, y_train, X_val, y_val, X_test, y_test, {}, dataset_name)

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, dataset_name: str) -> Tuple[XGBClassifier, Dict[str, float]]:  
    scale_weight = DEFAULT_SCALE_POS_WEIGHT
    model = XGBClassifier(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=6, 
        scale_pos_weight=scale_weight, 
        random_state=RANDOM_SEED, 
        n_jobs=-1,
        eval_metric="auc", 
        early_stopping_rounds=50
    )
    return train_and_log(model, "XGBoost", X_train, y_train, X_val, y_val, X_test, y_test, {}, dataset_name)

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, dataset_name: str, params: Optional[Dict[str, Any]] = None) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    if params is None:
        params = {
            "n_estimators": 1000, 
            "learning_rate": 0.05, 
            "num_leaves": 31,
            "class_weight": "balanced", 
            "random_state": RANDOM_SEED, 
            "n_jobs": -1
        }
        
    model = lgb.LGBMClassifier(**params)
    return train_and_log(model, "LightGBM", X_train, y_train, X_val, y_val, X_test, y_test, params, dataset_name)


# fonction d'entrainement avec cross-validation
def train_model_cv(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, dataset_name: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Ensemble, Dict[str, float]]:
    
    # Stratified K-Fold sur l'ensemble d'entraînement
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Paramètres par défaut si non fournis
    if params is None:
        params = {
            "n_estimators": 1000, "learning_rate": 0.05, 
            "class_weight": "balanced", "random_state": RANDOM_SEED, 
            "n_jobs": -1, "verbosity": -1
        }
    
    # Initialisation de la liste des modèles
    models: List[lgb.LGBMClassifier] = []
    
    # Run MLflow 
    with mlflow.start_run(run_name=f"LGBM_Ensemble_{dataset_name}"):
        mlflow.log_params(params)
        
        # ENTRAÎNEMENT DES FOLDS
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_va_in = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_va_in = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Utilisation des paramètres optimaux
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr, 
                eval_set=[(X_va_in, y_va_in)],
                eval_metric=lgb_custom_metric,
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            # Sauvegarde du modèle de Fold
            models.append(model)
            mlflow.lightgbm.log_model(model, name="modele_fold")
            print(f"Fold {fold+1} terminé (best_iteration={model.best_iteration_}).")
            


        # CRÉATION ET ÉVALUATION DE L'ENSEMBLE
        ensemble = Ensemble(models)
        
        # Calibration du seuil sur X_val (externe)
        print("\nCalibration seuil Ensemble sur X_val...")
        val_probs = ensemble.predict_proba(X_val)[:, 1]
        best_thresh, min_cost_val = find_best_threshold(y_val, val_probs)
        
        mlflow.log_param("best_threshold", best_thresh)
        mlflow.log_metric("val_best_cost_ensemble", min_cost_val)
        
        # Évaluation finale sur le Test Set avec le meilleur seuil trouvé sur Val
        metrics = evaluate_model(ensemble, X_test, y_test, threshold=best_thresh)
        mlflow.log_metrics(metrics)
        print("Metrics Test (Ensemble):", metrics)

        mlflow.lightgbm.log_model(ensemble, name="modele_ensemble")
        
        return ensemble, metrics

# fonction d'optimisation avec Optuna
def optimize_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 30) -> Dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        param = {
            "objective": "binary", "metric": "custom", "verbosity": -1,
            "boosting_type": "gbdt", "random_state": RANDOM_SEED, "n_jobs": -1,
            "class_weight": "balanced",
            "n_estimators": 1000,
            
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 50.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 50.0),
        }
        
        # apprentissage
        model = lgb.LGBMClassifier(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=lgb_custom_metric,
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        preds = model.predict_proba(X_val)[:, 1]
        _, min_c = find_best_threshold(y_val, preds)
        return min_c

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    print("Meilleurs params:", study.best_trial.params)
    return study.best_trial.params
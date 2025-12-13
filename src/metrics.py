from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, precision_score, f1_score, accuracy_score
import numpy as np

def calculate_auc(y_true, y_pred_proba):
    """Calcule l'AUC."""
    try:
        return roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        return np.nan 

def calculate_recall(y_true, y_pred):
    """Calcule le Rappel."""
    return recall_score(y_true, y_pred)

def calculate_precision(y_true, y_pred):
    """Calcule la Précision."""
    return precision_score(y_true, y_pred)

def calculate_f1(y_true, y_pred):
    """Calcule le F1-score."""
    return f1_score(y_true, y_pred)

def calculate_accuracy(y_true, y_pred):
    """Calcule l'accuracy."""
    return accuracy_score(y_true, y_pred)

def business_cost_metric(y_true, y_pred_proba, threshold=0.5):
    """Calcule le coût métier: 10 * FN + 1 * FP."""
    y_pred = (y_pred_proba > threshold).astype(int)
    
    if len(y_true) == 0:
        return np.nan
        
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        return np.nan

    cost = 10 * fn + 1 * fp
    return cost
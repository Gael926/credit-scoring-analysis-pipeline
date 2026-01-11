import shap
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from numpy.typing import ArrayLike

# focntion pour le calcul des valeurs SHAP
def compute_shap(model, X: pd.DataFrame) -> Tuple[shap.TreeExplainer, np.ndarray]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # LightGBM retourne souvent [0, 1] pour binaire
    if isinstance(shap_values, list):
        return explainer, shap_values[1]
    return explainer, shap_values

# focntion pour l'explication globale
def plot_shap_global(shap_values: ArrayLike, X: pd.DataFrame, save_path: str, max_display: int = 10) -> None:
    plt.figure(figsize=(10, 6))
    # Beeswarm montre l'importance ET la direction (rouge/bleu)
    shap.summary_plot(shap_values, X, plot_type="beeswarm", max_display=max_display, show=False)
    plt.title(f"Top {max_display} Variables (Impact & Direction)")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Graphique global sauvegardé: {save_path}")

# focntion pour l'explication locale
def plot_shap_local(explainer: shap.TreeExplainer, X: pd.DataFrame, index: int, save_path: str) -> None:
    # Création objet Explanation pour le waterfall plot
    explanation = explainer(X)
    
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation[index], max_display=10, show=False)
    plt.title(f"Explication Client {index}")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Graphique local sauvegardé: {save_path}")

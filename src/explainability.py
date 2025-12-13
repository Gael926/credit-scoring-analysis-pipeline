import shap
import matplotlib.pyplot as plt
import os
import numpy as np

def compute_shap(model, X):
    """Calcule les valeurs SHAP."""
    print("Calcul SHAP en cours...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # LightGBM retourne souvent [0, 1] pour binaire
    if isinstance(shap_values, list):
        return explainer, shap_values[1]
    return explainer, shap_values

def plot_shap_global(shap_values, X, save_path=None, max_display=10):
    """
    Affiche l'importance globale (les variables qui impactent le plus).
    Utilise un Beeswarm plot mais limité aux top variables pour la clarté.
    """
    plt.figure(figsize=(10, 6))
    # Beeswarm montre l'importance ET la direction (rouge/bleu)
    shap.summary_plot(shap_values, X, plot_type="beeswarm", max_display=max_display, show=False)
    plt.title(f"Top {max_display} Variables (Impact & Direction)")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Graphique global sauvegardé: {save_path}")
    else:
        plt.show()

def plot_shap_local(explainer, X, index, save_path=None):
    """
    Explication locale pour un client spécifique.
    Montre pourquoi ce client a eu ce score.
    """
    # Création objet Explanation pour le waterfall plot
    explanation = explainer(X)
    
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation[index], max_display=10, show=False)
    plt.title(f"Explication Client {index}")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Graphique local sauvegardé: {save_path}")
    else:
        plt.show()

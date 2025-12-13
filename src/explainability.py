import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def shap_global(model, X, plot_type="bar", max_display=20):
    """
    Calcule et affiche l'importance globale des features avec SHAP.
    
    Args:
        model: Le modèle entraîné (ex: LightGBM).
        X: Le DataFrame des features (peut être un échantillon).
        plot_type: Type de graphique ('bar', 'dot', etc.).
        max_display: Nombre de features à afficher.
    """
    print("Calcul des valeurs SHAP globales...")
    # Pour LightGBM, TreeExplainer est optimisé
    explainer = shap.TreeExplainer(model)
    
    # shap_values peut être une liste (pour classification multiclasse) ou array
    shap_values = explainer.shap_values(X)
    
    # Si classification binaire, shap_values[1] correspond souvent à la classe positive
    if isinstance(shap_values, list):
        vals = shap_values[1]
    else:
        vals = shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(vals, X, plot_type=plot_type, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    # plt.show() # À appeler dans le notebook
    return explainer, vals

def shap_local(model, X, instance_index):
    """
    Affiche l'explication locale pour une instance donnée (Waterfall plot).
    """
    print(f"Calcul SHAP local pour l'index {instance_index}...")
    explainer = shap.TreeExplainer(model)
    
    # On utilise l'objet Explanation pour les nouveaux plots
    shap_explanation = explainer(X)
    
    plt.figure()
    # Waterfall plot pour l'instance spécifique
    shap.plots.waterfall(shap_explanation[instance_index], show=False)
    plt.title(f"Explication Locale (Index {instance_index})")
    plt.tight_layout()
    # plt.show()

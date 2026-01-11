# Analyse comparative des notebooks de préparation des données

L'analyse des fichiers `01_v1_data_preparation.ipynb` et `01_v2_data_preparation.ipynb` révèle les différences fondamentales suivantes dans l'approche de préparation des données pour le projet de Credit Scoring.

## 1. Philosophie et Objectif
- **V1 (Version "Brute")** : Garde quasiment toutes les colonnes originales (159 colonnes après jointures). Elle privilégie la conservation de l'information maximale pour laisser le modèle (LGBM) gérer la complexité.
- **V2 (Version "Optimisée")** : Pratique un nettoyage agressif (92 colonnes après filtrage initial). Elle vise à supprimer le "bruit" et à réduire l'empreinte mémoire pour des itérations plus rapides.

## 2. Nettoyage des Données et Gestion des Outliers
- **V1** : Très peu de nettoyage manuel. Elle se repose sur les fonctions automatiques de `src.data_prep`.
- **V2** :
    - **Suppression thématique** : Retire toutes les colonnes liées à l'habitat (`_AVG`, `_MODE`, `_MEDI`) et les colonnes de documents (`FLAG_DOCUMENT_X`).
    - **Correction métier** : Gère explicitement l'outlier des "365243 jours" (le bug des 1000 ans) dans `DAYS_EMPLOYED` en le remplaçant par `NaN`.
    - **Filtrage des revenus** : Supprime les lignes avec des revenus aberrants (ex: > 100M).

## 3. Encodage des Variables Catégorielles
- **V1** : Utilise massivement `pd.get_dummies` sur toutes les colonnes non-numériques (One-Hot Encoding), ce qui fait grimper le nombre de colonnes de **159 à 297**.
- **V2** : Combine deux stratégies :
    - **Label Encoding** pour les colonnes à haute cardinalité (`ORGANIZATION_TYPE`, `OCCUPATION_TYPE`).
    - **One-Hot Encoding** uniquement pour les colonnes à faible cardinalité.
    - Résultat : Le dataset final reste beaucoup plus compact (**124 colonnes**).

## 4. Visualisations et Analyse Santé
- **V1** : Se concentre sur la distribution de la `TARGET` et les statistiques de valeurs manquantes.
- **V2** : Ajoute un bloc complet d'"Analyse Santé" vérifiant les valeurs infinies (division par zéro dans les ratios) et des boxplots pour les revenus et l'âge.

## 5. Export des données
- **V1** : Ne semble pas finir sur un export formel dans le notebook lui-même (ou utilise des CSV).
- **V2** : Exporte les données au format **Pickle** (`.pkl`), ce qui est beaucoup plus rapide et préserve les types de données (float16/int8) définis par `reduce_mem_usage`.

---
**Verdict** : La **V2** est une évolution plus mature, mieux documentée et plus performante techniquement, tandis que la **V1** servait probablement d'exploration initiale ou de baseline sans filtre.

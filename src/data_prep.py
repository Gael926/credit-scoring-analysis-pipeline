# Fonctions de nettoyage, encodage, jointures

def clean_data(df):
    pass

def encode_features(df):
    pass

import pandas as pd
import numpy as np
import os
import gc

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(CURRENT_DIR), 'data/raw')

def load_data(file_name):
    """Charge un CSV depuis le dossier data de manière robuste."""
    path = os.path.join(DATA_PATH, file_name)
    if not os.path.exists(path):
        print(f"ERREUR: Fichier introuvable {path}")
        return None
    return pd.read_csv(path)

# ratio financers inspirés du kernel kaggle
def create_domain_features(df):
    # 1. Pourcentage de crédit par rapport au revenu
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # 2. Pourcentage de l'annuité par rapport au revenu (Taux d'endettement)
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    
    # 3. Durée du crédit (approximatif)
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    # 4. Pourcentage des jours travaillés par rapport à l'âge (Stabilité pro)
    # Attention: DAYS_EMPLOYED est souvent négatif, on prend l'absolu ou le ratio direct
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    return df

def get_bureau_features():
    bureau = load_data('bureau.csv')
    if bureau is None: return None
    
    # On filtre : on veut savoir ce qui est ACTIF
    bureau['CREDIT_ACTIVE_BINARY'] = (bureau['CREDIT_ACTIVE'] == 'Active').astype(int)
    
    agg = {
        'DAYS_CREDIT': ['max', 'mean'],        # Récence des crédits
        'AMT_CREDIT_SUM': ['sum', 'max'],      # Montant total emprunté ailleurs
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],# DETTE TOTALE ACTUELLE (Crucial)
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],    # Impayés moyens
        'CREDIT_ACTIVE_BINARY': ['mean']       # % de crédits actifs
    }
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg)
    bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    del bureau; gc.collect()
    return bureau_agg

def get_previous_features():
    prev = load_data('previous_application.csv')
    if prev is None: return None
    
    prev['APP_REFUSED'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    
    agg = {
        'AMT_ANNUITY': ['mean'],
        'APP_REFUSED': ['mean'],   # Taux de refus passé chez nous
        'CNT_PAYMENT': ['mean']    # Durée moyenne demandée
    }
    
    prev_agg = prev.groupby('SK_ID_CURR').agg(agg)
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    del prev; gc.collect()
    return prev_agg

def get_pos_cash_features():
    pos = load_data('POS_CASH_balance.csv')
    if pos is None: return None
    
    agg = {
        'SK_DPD': ['max', 'mean'],  # Retards (Days Past Due)
        'CNT_INSTALMENT_FUTURE': ['sum'] # Dette restante
    }
    pos_agg = pos.groupby('SK_ID_CURR').agg(agg)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    del pos; gc.collect()
    return pos_agg

def get_installments_features():
    ins = load_data('installments_payments.csv')
    if ins is None: return None
    
    # Feature Engineering clé : Le sous-paiement
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    
    agg = {
        'PAYMENT_DIFF': ['sum', 'mean'], # Total manquant
        'DAYS_ENTRY_PAYMENT': ['max']    # Dernier paiement vu
    }
    ins_agg = ins.groupby('SK_ID_CURR').agg(agg)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    del ins; gc.collect()
    return ins_agg

# focntion pour les jointure dans la v1
def load_and_feature_engineering():
    # 1. Chargement Train
    df = load_data('application_train.csv')
    print(f"Base Train chargée: {df.shape}")
    
    # 2. Ajout des Ratios Métiers
    df = create_domain_features(df)
    
    # 3. Jointures Externes (Mode Light)
    # Liste des fonctions à appeler
    external_funcs = [
        get_bureau_features,
        get_previous_features,
        get_pos_cash_features,
        get_installments_features
    ]
    
    for func in external_funcs:
        feat_df = func()
        if feat_df is not None:
            df = df.merge(feat_df, on='SK_ID_CURR', how='left')
            print(f"Après {func.__name__}: {df.shape}")
            del feat_df
            gc.collect()
            
    # Nettoyage des noms de colonnes (Espaces etc.)
    df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    
    print("--- Terminé ---")
    return df


# focntion du prof
def reduce_mem_usage(df):
    """
    Itère sur toutes les colonnes d'un DataFrame et réduit la précision
    des types numériques (int et float) pour diminuer la consommation de mémoire.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Usage mémoire initial du DataFrame: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        # Traiter uniquement les colonnes numériques
        if col_type != object and col_type != str and col_type != bool:
            c_min = df[col].min()
            c_max = df[col].max()

            # --- Conversion des entiers (Integers) ---
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            # --- Conversion des décimaux (Floats) ---
            else:
                # La majorité de vos colonnes d'agrégats (mean, var, proportions) sont ici
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    # Conversion principale : float64 -> float32
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64) # Garder float64 si la précision est nécessaire

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Usage mémoire final du DataFrame: {end_mem:.2f} MB")
    print(f"Mémoire réduite de {(start_mem - end_mem) / start_mem * 100:.1f} %")

    return df

#focntion pour retourner les colonnes avec des missing values propre
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

# Dashboard Credit Scoring - Interface Streamlit
import streamlit as st
import pandas as pd
import requests
import json
import os

# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="🏦",
    layout="wide"
)

# URL de l'API (variable d'environnement ou défaut local)
API_URL = os.environ.get("API_URL", "http://localhost:5000/invocations")

# Chargement des données
@st.cache_data
def load_defaults():
    with open("dashboard_data/feature_defaults.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_sample_clients():
    return pd.read_csv("dashboard_data/sample_clients.csv")


def predict_client(features_dict: dict, feature_order: list) -> dict:
    """Envoie une requête à l'API et retourne la prédiction."""
    # Ordonner les features selon l'ordre attendu
    ordered_values = [features_dict.get(f, 0.0) for f in feature_order]
    
    payload = {
        "dataframe_split": {
            "columns": feature_order,
            "data": [ordered_values]
        }
    }
    
    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        # L'API retourne {"predictions": [0.75]} (probabilité)
        predictions = result.get("predictions", [])
        if predictions:
            proba = float(predictions[0])
            classe = 1 if proba > 0.5 else 0
            return {"success": True, "classe": classe, "proba": proba}
        return {"success": False, "error": "Pas de prédiction"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Impossible de contacter l'API"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    st.title("🏦 Credit Scoring Dashboard")
    
    # Chargement des données
    try:
        defaults = load_defaults()
        sample_clients = load_sample_clients()
        feature_order = defaults["features"]
        medians = defaults["medians"]
    except FileNotFoundError:
        st.error("Données manquantes. Exécutez d'abord: python scripts/generate_dashboard_data.py")
        return
    
    # Onglets
    tab1, tab2 = st.tabs(["📋 Audit Client", "🧮 Simulateur"])
    
    # Onglet 1 : Audit
    with tab1:
        st.header("Audit d'un client existant")
        
        # Sélection du client
        client_ids = sample_clients["SK_ID_CURR"].tolist()
        selected_id = st.selectbox("Sélectionner un ID client", client_ids)
        
        if selected_id:
            client_row = sample_clients[sample_clients["SK_ID_CURR"] == selected_id].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Informations du client")
                # Afficher quelques features clés
                info_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", 
                             "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_MEAN"]
                for col in info_cols:
                    if col in client_row.index:
                        value = client_row[col]
                        if "DAYS" in col and not pd.isna(value):
                            value = abs(int(value))
                            if col == "DAYS_BIRTH":
                                st.metric(f"Âge (jours)", f"{value} ({value//365} ans)")
                            else:
                                st.metric(col, f"{value} jours")
                        elif not pd.isna(value):
                            st.metric(col, f"{value:,.2f}")
                
                # Vrai label
                true_label = int(client_row.get("TARGET", -1))
                if true_label == 1:
                    st.warning("⚠️ Client en défaut (TARGET=1)")
                elif true_label == 0:
                    st.success("✅ Client sain (TARGET=0)")
            
            with col2:
                st.subheader("Prédiction du modèle")
                
                if st.button("🔮 Obtenir la prédiction", key="predict_audit"):
                    # Préparation des features
                    features_dict = {}
                    for f in feature_order:
                        if f in client_row.index:
                            val = client_row[f]
                            features_dict[f] = 0.0 if pd.isna(val) else float(val)
                        else:
                            features_dict[f] = medians.get(f, 0.0)
                    
                    with st.spinner("Appel à l'API..."):
                        result = predict_client(features_dict, feature_order)
                    
                    if result["success"]:
                        proba = result["proba"]
                        classe = result["classe"]
                        
                        st.metric("Probabilité de défaut", f"{proba:.2%}")
                        
                        if classe == 1:
                            st.error("🚨 Prédiction: DÉFAUT (Refus recommandé)")
                        else:
                            st.success("✅ Prédiction: PAS DE DÉFAUT (Accord possible)")
                    else:
                        st.error(f"Erreur: {result['error']}")
    
    # Onglet 2 : Simulateur
    with tab2:
        st.header("Simulateur de crédit")
        st.info("Modifiez les valeurs ci-dessous pour simuler une demande de crédit.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input(
                "💰 Revenu annuel",
                min_value=0.0,
                value=float(medians.get("AMT_INCOME_TOTAL", 150000)),
                step=10000.0
            )
            
            credit = st.number_input(
                "💳 Montant du crédit demandé",
                min_value=0.0,
                value=float(medians.get("AMT_CREDIT", 500000)),
                step=50000.0
            )
            
            annuity = st.number_input(
                "📅 Annuité (paiement annuel)",
                min_value=0.0,
                value=float(medians.get("AMT_ANNUITY", 25000)),
                step=1000.0
            )
        
        with col2:
            age_years = st.slider("👤 Âge (années)", 18, 70, 35)
            days_birth = -age_years * 365
            
            employment_years = st.slider("💼 Années d'emploi", 0, 40, 5)
            days_employed = -employment_years * 365
            
            ext_source = st.slider("📊 Score externe (0-1)", 0.0, 1.0, 0.5)
        
        # Calcul des ratios métiers
        credit_income_percent = credit / income if income > 0 else 0
        annuity_income_percent = annuity / income if income > 0 else 0
        credit_term = annuity / credit if credit > 0 else 0
        days_employed_percent = days_employed / days_birth if days_birth != 0 else 0
        
        st.divider()
        
        # Affichage des ratios calculés
        st.subheader("Ratios calculés")
        ratio_col1, ratio_col2, ratio_col3 = st.columns(3)
        with ratio_col1:
            st.metric("Crédit / Revenu", f"{credit_income_percent:.2f}")
        with ratio_col2:
            st.metric("Taux d'endettement", f"{annuity_income_percent:.2%}")
        with ratio_col3:
            st.metric("Durée crédit", f"{credit_term:.4f}")
        
        st.divider()
        
        if st.button("🔮 Simuler la prédiction", key="predict_sim"):
            # Préparation des features avec les médianes par défaut
            features_dict = medians.copy()
            
            # Mise à jour avec les valeurs saisies
            features_dict["AMT_INCOME_TOTAL"] = income
            features_dict["AMT_CREDIT"] = credit
            features_dict["AMT_ANNUITY"] = annuity
            features_dict["DAYS_BIRTH"] = days_birth
            features_dict["DAYS_EMPLOYED"] = days_employed
            
            # Scores externes
            features_dict["EXT_SOURCE_1"] = ext_source
            features_dict["EXT_SOURCE_2"] = ext_source
            features_dict["EXT_SOURCE_3"] = ext_source
            features_dict["EXT_SOURCE_MEAN"] = ext_source
            features_dict["EXT_SOURCE_PROD"] = ext_source ** 3
            
            # Ratios métiers recalculés
            features_dict["CREDIT_INCOME_PERCENT"] = credit_income_percent
            features_dict["ANNUITY_INCOME_PERCENT"] = annuity_income_percent
            features_dict["CREDIT_TERM"] = credit_term
            features_dict["DAYS_EMPLOYED_PERCENT"] = days_employed_percent
            
            with st.spinner("Appel à l'API..."):
                result = predict_client(features_dict, feature_order)
            
            if result["success"]:
                proba = result["proba"]
                classe = result["classe"]
                
                st.metric("Probabilité de défaut", f"{proba:.2%}")
                
                if classe == 1:
                    st.error("🚨 Prédiction: RISQUE ÉLEVÉ - Refus recommandé")
                else:
                    st.success("✅ Prédiction: RISQUE FAIBLE - Accord possible")
            else:
                st.error(f"Erreur: {result['error']}")


if __name__ == "__main__":
    main()

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
    
    # Deux colonnes côte à côte
    col_audit, col_sim = st.columns(2, gap="large")
    
    # Colonne gauche : Audit
    with col_audit:
        st.subheader("Audit Client")
        
        # Sélection du client
        client_ids = sample_clients["SK_ID_CURR"].tolist()
        selected_id = st.selectbox("Sélectionner un ID client", client_ids)
        
        if selected_id:
            client_row = sample_clients[sample_clients["SK_ID_CURR"] == selected_id].iloc[0]
            
            # Vrai label
            true_label = int(client_row.get("TARGET", -1))
            if true_label == 1:
                st.warning("Client en défaut (TARGET=1)")
            elif true_label == 0:
                st.success("Client sain (TARGET=0)")
            
            # Informations clés
            st.caption("📊 Informations du client")
            info_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED"]
            
            m1, m2 = st.columns(2)
            with m1:
                income = client_row.get("AMT_INCOME_TOTAL", 0)
                st.metric("Revenu", f"{income:,.0f} €")
                credit = client_row.get("AMT_CREDIT", 0)
                st.metric("Crédit", f"{credit:,.0f} €")
            with m2:
                days_birth = abs(int(client_row.get("DAYS_BIRTH", 0)))
                st.metric("Âge", f"{days_birth//365} ans")
                days_emp = abs(int(client_row.get("DAYS_EMPLOYED", 0)))
                st.metric("Emploi", f"{days_emp//365} ans")
            
            st.divider()
            
            if st.button("Obtenir la prédiction", key="predict_audit"):
                # Préparation des features
                features_dict = {}
                for f in feature_order:
                    if f in client_row.index:
                        val = client_row[f]
                        features_dict[f] = 0.0 if pd.isna(val) else float(val)
                    else:
                        features_dict[f] = medians.get(f, 0.0)
                
                with st.spinner("Analyse..."):
                    result = predict_client(features_dict, feature_order)
                
                if result["success"]:
                    proba = result["proba"]
                    classe = result["classe"]
                    
                    st.metric("Probabilité de défaut", f"{proba:.1%}")
                    
                    if classe == 1:
                        st.error("Prédiction: DÉFAUT")
                    else:
                        st.success("Prédiction: PAS DE DÉFAUT")
                else:
                    st.error(f"Erreur: {result['error']}")
    
    # Colonne droite : Simulateur
    with col_sim:
        st.subheader("Simulateur de crédit")
        st.info("Simulez votre demande de crédit")
        
        # Inputs en linéaire
        income = st.number_input(
            "Revenu annuel (€)",
            min_value=10000.0,
            max_value=1000000.0,
            value=float(medians.get("AMT_INCOME_TOTAL", 150000)),
            step=5000.0
        )
        
        credit = st.number_input(
            "Montant du crédit (€)",
            min_value=10000.0,
            max_value=2000000.0,
            value=float(medians.get("AMT_CREDIT", 500000)),
            step=10000.0
        )
        
        goods_price = st.number_input(
            "Montant de l'achat (€)",
            min_value=10000.0,
            max_value=2000000.0,
            value=float(medians.get("AMT_GOODS_PRICE", 450000)),
            step=10000.0
        )
        
        age_years = st.slider("Âge", min_value=18, max_value=70, value=35)
        days_birth = -age_years * 365
        
        employment_years = st.slider("Années d'emploi", min_value=0, max_value=45, value=5)
        days_employed = -employment_years * 365
        
        loan_duration = st.slider("Durée du crédit (années)", min_value=1, max_value=30, value=20)
        
        # Calculs
        annuity = credit / loan_duration if loan_duration > 0 else credit
        credit_income_percent = credit / income if income > 0 else 0
        annuity_income_percent = annuity / income if income > 0 else 0
        credit_term = annuity / credit if credit > 0 else 0
        days_employed_percent = days_employed / days_birth if days_birth != 0 else 0
        credit_to_goods = credit / goods_price if goods_price > 0 else 1
        
        st.divider()
        
        # Indicateurs sur 2 colonnes
        st.caption("**📊 Indicateurs**")
        i1, i2 = st.columns(2)
        with i1:
            st.metric("Mensualité", f"{annuity/12:,.0f} €")
            st.metric("Taux endettement", f"{annuity_income_percent*100:.0f}%")
        with i2:
            st.metric("Crédit/Revenu", f"{credit_income_percent:.1f}x")
            apport = max(0, goods_price - credit)
            st.metric("Apport", f"{apport:,.0f} €")
        
        st.divider()
        
        if st.button("Simuler", key="predict_sim", type="primary"):
            # Préparation des features
            features_dict = medians.copy()
            features_dict["AMT_INCOME_TOTAL"] = income
            features_dict["AMT_CREDIT"] = credit
            features_dict["AMT_ANNUITY"] = annuity
            features_dict["AMT_GOODS_PRICE"] = goods_price
            features_dict["DAYS_BIRTH"] = days_birth
            features_dict["DAYS_EMPLOYED"] = days_employed
            features_dict["CREDIT_INCOME_PERCENT"] = credit_income_percent
            features_dict["ANNUITY_INCOME_PERCENT"] = annuity_income_percent
            features_dict["CREDIT_TERM"] = credit_term
            features_dict["DAYS_EMPLOYED_PERCENT"] = days_employed_percent
            features_dict["CREDIT_TO_GOODS_RATIO"] = credit_to_goods
            
            with st.spinner("Analyse..."):
                result = predict_client(features_dict, feature_order)
            
            if result["success"]:
                proba = result["proba"]
                
                st.metric("Probabilité de défaut", f"{proba:.1%}")
                
                if proba > 0.7:
                    st.error("**RISQUE TRÈS ÉLEVÉ**")
                elif proba > 0.5:
                    st.warning("**RISQUE ÉLEVÉ**")
                elif proba > 0.3:
                    st.info("**RISQUE MODÉRÉ**")
                else:
                    st.success("**RISQUE FAIBLE**")
            else:
                st.error(f"Erreur: {result['error']}")


if __name__ == "__main__":
    main()

# Dashboard Credit Scoring - Interface Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="🏦",
    layout="wide"
)

# Mode de prédiction : direct (modèle local) ou API
USE_API = os.environ.get("USE_API", "false").lower() == "true"
API_URL = os.environ.get("API_URL", "http://localhost:5001/invocations")

# Chargement du modèle (pour mode standalone / Streamlit Cloud)
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/best_model.pkl")
    except Exception:
        return None

# Chargement des données
@st.cache_data
def load_defaults():
    with open("dashboard_data/feature_defaults.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_sample_clients():
    return pd.read_csv("dashboard_data/sample_clients.csv")


def predict_client(features_dict: dict, feature_order: list) -> dict:
    """Prédit le risque de défaut. Utilise le modèle local ou l'API selon la config."""
    # Ordonner les features selon l'ordre attendu
    ordered_values = [features_dict.get(f, 0.0) for f in feature_order]
    
    # Mode API (Docker)
    if USE_API:
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
    
    # Mode standalone (Streamlit Cloud)
    else:
        model = load_model()
        if model is None:
            return {"success": False, "error": "Modèle non trouvé"}
        
        try:
            df = pd.DataFrame([ordered_values], columns=feature_order)
            proba = float(model.predict_proba(df)[:, 1][0])
            classe = 1 if proba > 0.5 else 0
            return {"success": True, "classe": classe, "proba": proba}
        except Exception as e:
            return {"success": False, "error": str(e)}



def main():
    st.markdown("""
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        h1 {
            color: #4DA8DA;
            font-family: 'Helvetica Neue', sans-serif;
            text-align: center;
        }
        .stButton>button {
            width: 100%;
            background-color: #4DA8DA;
            color: white;
            border-radius: 5px;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #2E86C1;
            color: white;
        }
        /* Style des métriques ajusté pour le mode sombre */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
            color: #4DA8DA !important;
        }
        [data-testid="stMetricLabel"] {
            font-weight: bold;
            color: #E0E0E0 !important;
        }
        div[data-testid="metric-container"] {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #4DA8DA;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #E0E0E0;
            margin-bottom: 1rem;
            border-bottom: 2px solid #4DA8DA;
            padding-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Credit Scoring Dashboard")
    st.markdown("<br><br>", unsafe_allow_html=True)
    
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
        st.markdown('<div class="section-header">Audit Client</div>', unsafe_allow_html=True)
        
        # Sélection du client
        client_ids = sample_clients["SK_ID_CURR"].tolist()
        selected_id = st.selectbox("Sélectionner un ID client", client_ids)
        
        if selected_id:
            client_row = sample_clients[sample_clients["SK_ID_CURR"] == selected_id].iloc[0]
            
            # Vrai label
            true_label = int(client_row.get("TARGET", -1))
            if true_label == 1:
                st.error("Client en défaut (Historique)")
            elif true_label == 0:
                st.success("Client sain (Historique)")
            
            st.divider()

            # Informations clés
            st.markdown("### 📊 Informations du client")
            
            m1, m2 = st.columns(2)
            with m1:
                income = client_row.get("AMT_INCOME_TOTAL", 0)
                st.metric("Revenu Annuel", f"{income:,.0f} €")
                credit = client_row.get("AMT_CREDIT", 0)
                st.metric("Crédit Total", f"{credit:,.0f} €")
                annuity = client_row.get("AMT_ANNUITY", 0)
                st.metric("Annuité", f"{annuity:,.0f} €")

            with m2:
                days_birth = abs(int(client_row.get("DAYS_BIRTH", 0)))
                st.metric("Âge", f"{days_birth//365} ans")
                days_emp = abs(int(client_row.get("DAYS_EMPLOYED", 0)))
                st.metric("Ancienneté Emploi", f"{days_emp//365} ans")
                ext_mean = client_row.get("EXT_SOURCE_MEAN", 0)
                st.metric("Score Externe", f"{ext_mean:.2f}")
            
            st.divider()
            
            st.markdown("### Analyse de Risque")
            if st.button("Lancer l'analyse du dossier", key="predict_audit"):
                # Préparation des features
                features_dict = {}
                for f in feature_order:
                    if f in client_row.index:
                        val = client_row[f]
                        features_dict[f] = 0.0 if pd.isna(val) else float(val)
                    else:
                        features_dict[f] = medians.get(f, 0.0)
                
                with st.spinner("Analyse du profil en cours..."):
                    result = predict_client(features_dict, feature_order)
                
                if result["success"]:
                    proba = result["proba"]
                    classe = result["classe"]
                    
                    st.metric("Probabilité de défaut", f"{proba:.1%}", delta_color="inverse")
                    
                    if classe == 1:
                        st.error("**Risque Élevé** - Refus Recommandé")
                    else:
                        st.success("**Risque Faible** - Accord Possible")
                else:
                    st.error(f"Erreur: {result['error']}")
    
    # Colonne droite : Simulateur
    with col_sim:
        st.markdown('<div class="section-header">Simulateur de Crédit</div>', unsafe_allow_html=True)
        st.info("Ajustez les paramètres pour une nouvelle simulation")
        
        # Inputs avec des expanders pour gagner de la place si besoin, ou groupés
        with st.container():
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                income = st.number_input("Revenu annuel (€)", 10000.0, 1000000.0, float(medians.get("AMT_INCOME_TOTAL", 150000)), 5000.0)
                credit = st.number_input("Montant Crédit (€)", 10000.0, 2000000.0, float(medians.get("AMT_CREDIT", 500000)), 10000.0)
                goods_price = st.number_input("Montant Achat (€)", 10000.0, 2000000.0, float(medians.get("AMT_GOODS_PRICE", 450000)), 10000.0)
            
            with col_in2:
                age_years = st.slider("Âge", 18, 70, 35)
                employment_years = st.slider("Années d'emploi", 0, 45, 5)
                loan_duration = st.slider("Durée (années)", 1, 30, 20)

        days_birth = -age_years * 365
        days_employed = -employment_years * 365
        
        # Calculs
        annuity = credit / loan_duration if loan_duration > 0 else credit
        credit_income_percent = credit / income if income > 0 else 0
        annuity_income_percent = annuity / income if income > 0 else 0
        credit_term = annuity / credit if credit > 0 else 0
        days_employed_percent = days_employed / days_birth if days_birth != 0 else 0
        credit_to_goods = credit / goods_price if goods_price > 0 else 1
        
        st.divider()
        
        # Indicateurs
        st.markdown("### 📊 Indicateurs Clés")
        i1, i2 = st.columns(2)
        with i1:
            st.metric("Mensualité", f"{annuity/12:,.0f} €")
            st.metric("Taux Endettement", f"{annuity_income_percent*100:.1f} %", delta="High" if annuity_income_percent > 0.33 else "Low")
        with i2:
            st.metric("Ratio Crédit/Revenu", f"{credit_income_percent:.1f}x")
            apport = max(0, goods_price - credit)
            st.metric("Apport Personnel", f"{apport:,.0f} €")
        
        st.divider()
        
        if st.button("Lancer la simulation", key="predict_sim", type="primary"):
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
            
            with st.spinner("Simulation en cours..."):
                result = predict_client(features_dict, feature_order)
            
            if result["success"]:
                proba = result["proba"]
                
                st.metric("Probabilité de défaut", f"{proba:.1%}", delta_color="inverse")
                
                if proba > 0.7:
                    st.error("**Risque Très Élevé**")
                elif proba > 0.5:
                    st.warning("**Risque Élevé**")
                elif proba > 0.3:
                    st.info("**Risque Modéré**")
                else:
                    st.success("**Risque Faible**")
            else:
                st.error(f"Erreur: {result['error']}")
        
        # Espace vide pour l'esthétique
        st.markdown("<br><br><br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

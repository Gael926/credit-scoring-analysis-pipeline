# API Flask pour servir le modèle avec predict_proba
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Charger le modèle au démarrage
model = joblib.load("models/best_model.pkl")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/invocations", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Format MLflow dataframe_split
        if "dataframe_split" in data:
            columns = data["dataframe_split"]["columns"]
            values = data["dataframe_split"]["data"]
            df = pd.DataFrame(values, columns=columns)
        # Format simple
        elif "data" in data:
            df = pd.DataFrame(data["data"], columns=data.get("columns"))
        else:
            return jsonify({"error": "Format non supporté"}), 400
        
        # Prédiction avec probabilités
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[:, 1].tolist()
        else:
            proba = model.predict(df).tolist()
        
        return jsonify({"predictions": proba})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = os.path.join("model", "logistic_regression_New_model.pkl")
SCALER_PATH = os.path.join("model", "scalerNew.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print(
    "Scaler was fitted with feature names:", getattr(scaler, "feature_names_in_", None)
)

# Load feature names
import json

with open("feature_names.json", "r", encoding="utf-8") as f:
    FEATURE_NAMES = json.load(f)


def normalize_name(name):
    # Convert feature name to snake_case for JSON keys
    import re

    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Log incoming data for debugging
    print("Received data:", data)
    normalized_keys = [normalize_name(name) for name in FEATURE_NAMES]
    print("Expected keys:", normalized_keys)
    # Ensure all features are present and in correct order
    try:
        features = [float(data[normalize_name(name)]) for name in FEATURE_NAMES]
    except Exception as e:
        return (
            jsonify(
                {
                    "error": f"Invalid input: {str(e)}",
                    "received_keys": list(data.keys()),
                    "expected_keys": normalized_keys,
                    "received_data": data,
                }
            ),
            400,
        )
    # Build DataFrame with original feature names as columns
    X_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    X_scaled = scaler.transform(X_df)
    print("X_scaled:", X_scaled)
    if hasattr(model, "classes_"):
        print("model.classes_:", model.classes_)
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(X_scaled)
        print("model.predict_proba(X_scaled):", proba_arr)
        pred = model.predict(X_scaled)[0]
        confidence = float(proba_arr[0][int(pred)])
    else:
        decision = model.decision_function(X_scaled)
        print("model.decision_function(X_scaled):", decision)
        pred = model.predict(X_scaled)[0]
        confidence = float(decision[0])
    return jsonify({"prediction": int(pred), "confidence": confidence})


@app.route("/ping")
def ping():
    return "pong", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

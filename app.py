from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback

app = Flask(__name__)
# ✅ Load the trained XGBoost model
model = joblib.load("xgboost_model.pkl")

@app.route("/")
def home():
    return "✅ XGBoost Churn Prediction API is running successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        satisfaction = float(data.get("satisfaction", 0))
        monthly_revenue = float(data.get("monthly_revenue", 0))
        open_tickets = int(data.get("open_tickets", 0))
        churn_history_rate = float(data.get("churn_history_rate", 0))
        tenure_months = int(data.get("tenure_months", 0))
        usage_active_pct = float(data.get("usage_active_pct", 0))

        # Scale and format input
        revenue_scaled = monthly_revenue / 1_000_000
        X = np.array([[satisfaction, revenue_scaled, open_tickets,
                       churn_history_rate, tenure_months, usage_active_pct]])

        # Predict
        y_pred = model.predict(X)[0]
        revenue_drop, workload_change, trust_drop = y_pred

        # Churn logic
        churn_binary = "Yes" if (revenue_drop > 15 or satisfaction < 40) else "No"

        # Narrative
        if satisfaction < 40:
            narrative = "High churn risk detected due to low satisfaction and performance."
            recommendation = "Start retention program immediately."
        elif satisfaction < 70:
            narrative = "Moderate churn risk due to fluctuating satisfaction levels."
            recommendation = "Monitor engagement and improve customer communication."
        else:
            narrative = "Low churn risk. Customer relationship appears stable."
            recommendation = "Continue regular engagement activities."

        result = {
            "revenue_drop": round(revenue_drop, 2),
            "workload_change": round(workload_change, 2),
            "trust_drop": round(trust_drop, 2),
            "narrative": narrative,
            "recommendation": recommendation,
            "churn_prediction": churn_binary,
            "churn_accuracy_percent": 91.6
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
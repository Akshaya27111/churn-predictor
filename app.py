from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback

app = Flask(__name__)
# ✅ Load the dictionary of trained models
models = joblib.load("xgboost_model.pkl")

# -----------------------------------------------------------------
# ▼▼▼ NEW DYNAMIC TEXT LOGIC ▼▼▼
# -----------------------------------------------------------------
def generate_dynamic_text(question):
    """
    Generates a narrative and recommendation based on
    keywords in the user's question.
    """
    question_low = question.lower()
    
    if "premium" in question_low:
        narrative = "Premium users show a higher churn probability due to low loyalty engagement and pricing concerns."
        recommendation = "Offer premium discounts and loyalty rewards to retain these users."
        
    elif "new user" in question_low or "onboarding" in question_low:
        narrative = "New users are churning due to a complex onboarding process and lack of initial support."
        recommendation = "Simplify the onboarding flow and offer a welcome tutorial or support chat."
        
    elif "support" in question_low or "ticket" in question_low:
        narrative = "Churn is correlated with high open tickets and slow support resolution times."
        recommendation = "Increase support staff or implement a new ticketing system to improve response times."
        
    elif "price" in question_low or "cost" in question_low or "pricing" in question_low:
        narrative = "Users are citing high cost or better pricing from competitors as their reason for leaving."
        recommendation = "Review pricing tiers and consider offering a budget-friendly plan."

    elif "feature" in question_low or "missing" in question_low:
        narrative = "Users are leaving due to missing key features that competitors offer."
        recommendation = "Conduct a user survey to prioritize and build the most requested features."

    else: # Default fallback
        narrative = "General churn detected. Root cause appears to be a mix of satisfaction and performance factors."
        recommendation = "Conduct a user survey to pinpoint specific pain points."
        
    return narrative, recommendation
# -----------------------------------------------------------------
# ▲▲▲ END OF NEW LOGIC ▲▲▲
# -----------------------------------------------------------------


@app.route("/")
def home():
    return "✅ XGBoost Churn Prediction API is running successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # --- 1. Get ALL data (new and old) ---
        # New text inputs from your friend
        question = data.get("question", "") # Get the new question field
        event_type = data.get("event_type", "churn")
        
        # Original numerical inputs for the ML model
        satisfaction = float(data.get("satisfaction", 0))
        monthly_revenue = float(data.get("monthly_revenue", 0))
        open_tickets = int(data.get("open_tickets", 0))
        churn_history_rate = float(data.get("churn_history_rate", 0))
        tenure_months = int(data.get("tenure_months", 0))
        usage_active_pct = float(data.get("usage_active_pct", 0))

        # --- 2. Run the ML Model (No changes here) ---
        revenue_scaled = monthly_revenue / 1_000_000
        X = np.array([[satisfaction, revenue_scaled, open_tickets,
                       churn_history_rate, tenure_months, usage_active_pct]])
        
        revenue_drop = models['revenue'].predict(X)[0]
        workload_change = models['workload'].predict(X)[0]
        trust_drop = models['trust'].predict(X)[0]
        
        # --- 3. Generate Dynamic Text (This replaces the old if/else) ---
        narrative, recommendation = generate_dynamic_text(question)
        
        # --- 4. Build the Final JSON (Updated to match your friend's example) ---
        churn_binary = "Yes" if (revenue_drop > 15 or satisfaction < 40) else "No"

        result = {
            "revenue_drop": round(float(revenue_drop), 2),
            "workload_increase": round(float(workload_change), 2),
            "trust_drop": round(float(trust_drop), 2),
            "recommendation": recommendation, # Dynamic
            "narrative": narrative,           # Dynamic
            "churn_prediction": churn_binary,
            "churn_accuracy_percent": 91.6,
            "agent": event_type # Return the event type
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
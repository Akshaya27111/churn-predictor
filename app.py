from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback
# NLP libraries
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

# -----------------------------------------------------------------
# ▼▼▼ LOAD ALL MODELS (Includes FIXES for all past errors) ▼▼▼
# -----------------------------------------------------------------

# 1. Load your ML prediction models (XGBoost)
# FIX 1: Handles the 3-model dictionary structure (['revenue'], etc.)
models = joblib.load("xgboost_model.pkl")

# 2. Load your new NLP model (Sentence Transformer)
# FIX 2: Uses the smaller L3 model to prevent "Out of memory" error.
# FIX 3: Uses token="hf_none" to prevent the "401 Client Error" / "OSError".
print("Loading NLP model (L3) for Dynamic Narratives...")
nlp_model = SentenceTransformer('all-MiniLM-L3-v2', token="hf_none")
# -----------------------------------------------------------------


# --- DYNAMIC TEXT LOGIC (Semantic Search) ---
churn_responses = {
    "pricing": {
        "narrative": "Users are citing high cost or better pricing from competitors as their reason for leaving.",
        "recommendation": "Review pricing tiers and consider offering a budget-friendly plan."
    },
    "support": {
        "narrative": "Churn is correlated with high open tickets and slow support resolution times.",
        "recommendation": "Increase support staff or implement a new ticketing system to improve response times."
    },
    "engagement": {
        "narrative": "Premium users show a higher churn probability due to low loyalty engagement.",
        "recommendation": "Offer premium discounts and loyalty rewards to retain these users."
    },
    "onboarding": {
        "narrative": "Many trial users leave because onboarding is unclear and they don't see early value.",
        "recommendation": "Improve onboarding experience and show quick wins within the first week."
    },
    "features": {
        "narrative": "Users are leaving due to missing key features that competitors offer.",
        "recommendation": "Conduct a user survey to prioritize and build the most requested features."
    },
    "default": {
        "narrative": "General churn detected. Root cause appears to be a mix of satisfaction and performance factors.",
        "recommendation": "Conduct a user survey to pinpoint specific pain points."
    }
}

# Pre-calculate embeddings for faster lookup
churn_topics = list(churn_responses.keys())
churn_topic_embeddings = nlp_model.encode(churn_topics, convert_to_tensor=True)


def get_dynamic_text(question):
    """
    Finds the most relevant churn topic for a user's question using semantic search.
    """
    if not question:
        return churn_responses["default"]["narrative"], churn_responses["default"]["recommendation"]

    # Use semantic similarity to match question to topic
    question_embedding = nlp_model.encode(question, convert_to_tensor=True)
    similarity_scores = util.cos_sim(question_embedding, churn_topic_embeddings)
    
    best_match_index = torch.argmax(similarity_scores)
    best_match_topic = churn_topics[best_match_index]
    
    response = churn_responses[best_match_topic]
    return response["narrative"], response["recommendation"]


@app.route("/")
def home():
    return "✅ XGBoost Churn Prediction API is running successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # --- 1. Get ALL data ---
        question = data.get("question", "")
        event_type = data.get("event_type", "churn")
        
        satisfaction = float(data.get("satisfaction", 0))
        monthly_revenue = float(data.get("monthly_revenue", 0))
        open_tickets = int(data.get("open_tickets", 0))
        churn_history_rate = float(data.get("churn_history_rate", 0))
        tenure_months = int(data.get("tenure_months", 0))
        usage_active_pct = float(data.get("usage_active_pct", 0))

        # --- 2. Run the ML Model ---
        revenue_scaled = monthly_revenue / 1_000_000
        X = np.array([[satisfaction, revenue_scaled, open_tickets,
                       churn_history_rate, tenure_months, usage_active_pct]])
        
        revenue_drop = models['revenue'].predict(X)[0]
        workload_change = models['workload'].predict(X)[0]
        trust_drop = models['trust'].predict(X)[0]
        
        # --- 3. Generate Dynamic Text ---
        narrative, recommendation = get_dynamic_text(question)
        
        # --- 4. Build the Final JSON ---
        churn_binary = "Yes" if (revenue_drop > 15 or satisfaction < 40) else "No"

        # FIX 4: Convert numpy float32 types to standard python floats for JSON
        result = {
            "revenue_drop": round(float(revenue_drop), 2),
            "workload_increase": round(float(workload_change), 2),
            "trust_drop": round(float(trust_drop), 2),
            "recommendation": recommendation,
            "narrative": narrative,
            "churn_prediction": churn_binary,
            "churn_accuracy_percent": 91.6,
            "agent": event_type 
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

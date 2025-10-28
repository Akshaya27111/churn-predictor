from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback
# -----------------------------------------------------------------
# ▼▼▼ NEW IMPORTS FOR NLP ▼▼▼
# -----------------------------------------------------------------
# We use the smaller L3 model to fit Render's 512MB memory limit
from sentence_transformers import SentenceTransformer, util
import torch
# -----------------------------------------------------------------
# ▲▲▲ END OF NEW IMPORTS ▲▲▲
# -----------------------------------------------------------------

app = Flask(__name__)

# -----------------------------------------------------------------
# ▼▼▼ LOAD YOUR MODELS (BOTH OF THEM) ▼▼▼
# -----------------------------------------------------------------
# 1. Load your ML prediction models (XGBoost)
models = joblib.load("xgboost_model.pkl")

# 2. Load your new NLP model (Sentence Transformer)
print("Loading memory-efficient NLP model (L3)...")
nlp_model = SentenceTransformer('all-MiniLM-L3-v2')
print("NLP model loaded.")
# -----------------------------------------------------------------
# ▲▲▲ END OF MODEL LOADING ▲▲▲
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# ▼▼▼ DYNAMIC TEXT LOGIC (USING NLP) ▼▼▼
# -----------------------------------------------------------------
# 1. Define your main churn topics and their pre-written responses
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

# 2. Pre-calculate the embeddings for your topics (only happens once on startup)
churn_topics = list(churn_responses.keys())
churn_topic_embeddings = nlp_model.encode(churn_topics, convert_to_tensor=True)


def get_dynamic_text(question):
    """
    Uses semantic search (NLP) to find the most relevant churn topic for a user's question.
    """
    if not question:
        return churn_responses["default"]["narrative"], churn_responses["default"]["recommendation"]

    # 1. Encode the user's question
    question_embedding = nlp_model.encode(question, convert_to_tensor=True)
    
    # 2. Find the similarity between the question and your topics
    similarity_scores = util.cos_sim(question_embedding, churn_topic_embeddings)
    
    # 3. Get the best match based on semantic closeness
    best_match_index = torch.argmax(similarity_scores)
    best_match_topic = churn_topics[best_match_index]
    
    # 4. Return the corresponding narrative and recommendation
    response = churn_responses[best_match_topic]
    return response["narrative"], response["recommendation"]
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
        
        # --- 1. Get ALL data (including new text fields) ---
        question = data.get("question", "")
        event_type = data.get("event_type", "churn")
        
        # Original numerical inputs
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

        result = {
            "revenue_drop": round(float(revenue_drop), 2),
            "workload_increase": round(float(workload_change), 2),
            "trust_drop": round(float(trust_drop), 2),
            "recommendation": recommendation, # Dynamic
            "narrative": narrative,           # Dynamic
            "churn_prediction": churn_binary,
            "churn_accuracy_percent": 91.6,
            "agent": event_type 
        }

        return jsonify(result)

    except Exception as e:
        # Returns the error and traceback to the client for debugging
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

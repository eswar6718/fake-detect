import joblib
from flask import Request, jsonify

# Load models
log_model = joblib.load("./model/log_model.pkl")
rf_model = joblib.load("./model/rf_model.pkl")
tfidf = joblib.load("./model/tfidf.pkl")

def handler(request):
    try:
        data = request.get_json()
        text = data.get("description", "")

        # Transform input
        X = tfidf.transform([text])

        # Get probabilities
        p1 = log_model.predict_proba(X)[0][1]
        p2 = rf_model.predict_proba(X)[0][1]

        prob = (p1 + p2) / 2  # ensemble

        prediction = "FAKE" if prob < 0.30 else "LEGIT"

        return jsonify({
            "prediction": prediction,
            "probability": round(prob * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

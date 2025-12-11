from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load models once
log_model = joblib.load("./model/log_model.pkl")
rf_model = joblib.load("./model/rf_model.pkl")
tfidf = joblib.load("./model/tfidf.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("description", "")

        X = tfidf.transform([text])
        p1 = log_model.predict_proba(X)[0][1]
        p2 = rf_model.predict_proba(X)[0][1]

        prob = (p1 + p2) / 2
        prediction = "FAKE" if prob < 0.30 else "LEGIT"

        return jsonify({
            "prediction": prediction,
            "probability": round(prob * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

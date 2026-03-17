"""
CineRead · Sentiment Analysis Backend
Recreated from original structure (FastAPI → Flask, same logic preserved)
Original used: FastAPI + CORSMiddleware + pydantic BaseModel + joblib
"""
from flask import Flask, request, jsonify
import joblib
import os

# ── Load models (exactly as original: joblib.load) ───────────────────────────
BASE = os.path.dirname(__file__)
rf = joblib.load(os.path.join(BASE, "rf_model.pkl"))
cv = joblib.load(os.path.join(BASE, "cv.pkl"))

app = Flask(__name__)

# ── CORS (replacing original CORSMiddleware) ─────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        from flask import make_response
        r = make_response()
        r.headers["Access-Control-Allow-Origin"]  = "*"
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return r

# ── /predict (original logic preserved exactly) ──────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    body       = request.get_json(silent=True) or {}
    review     = body.get("text", "")

    vector     = cv.transform([review])
    prediction = rf.predict(vector)

    sentiment  = "Positive 😊" if prediction[0] == 1 else "Negative 😡"

    # Extra: probabilities for the enhanced frontend
    proba = rf.predict_proba(vector)[0]
    score = float(proba[prediction[0]])

    return jsonify({
        "sentiment":             sentiment,
        "score":                 round(score, 4),
        "positive_probability":  round(float(proba[1]), 4),
        "negative_probability":  round(float(proba[0]), 4),
    })

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🎬  CineRead API  →  http://127.0.0.1:8000\n")
    app.run(host="0.0.0.0", port=8000, debug=True)

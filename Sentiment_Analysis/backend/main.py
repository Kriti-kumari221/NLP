from flask import Flask, request, jsonify, make_response
import joblib
import os

# ── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load models ──────────────────────────────────────────
rf = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
cv = joblib.load(os.path.join(BASE_DIR, "cv.pkl"))

app = Flask(__name__)

# ── Home route (optional but nice for testing) ───────────
@app.route("/")
def home():
    return jsonify({
        "message": "🎬 CineRead API is running 🚀",
        "endpoint": "/predict (POST)",
        "status": "OK"
    })

# ── Health check ─────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

# ── CORS headers ─────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        return make_response("", 200)

# ── Main prediction endpoint ─────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        review = data.get("text", "").strip()

        if not review:
            return jsonify({"error": "Text input is required"}), 400

        # Transform and predict
        vector = cv.transform([review])
        prediction = rf.predict(vector)[0]
        proba = rf.predict_proba(vector)[0]

        # 0 → negative, 1 → positive (most common convention)
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😡"
        confidence = float(proba[prediction])   # probability of predicted class

        return jsonify({
            "input": review,
            "sentiment": sentiment,
            "score": round(confidence, 4),
            "positive_probability": round(float(proba[1]), 4),
            "negative_probability": round(float(proba[0]), 4),
        })

    except Exception as e:
        return jsonify({
            "error": "Something went wrong",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    print("\n🎬 CineRead API → http://127.0.0.1:8000\n")
    app.run(host="0.0.0.0", port=8000, debug=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Load model & vectorizer
rf = joblib.load("rf_model.pkl")
cv = joblib.load("cv.pkl")

app = FastAPI()

# Enable CORS (VERY IMPORTANT for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict(review: Review):
    vector = cv.transform([review.text])
    prediction = rf.predict(vector)[0]

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😡"

    return {"sentiment": sentiment}
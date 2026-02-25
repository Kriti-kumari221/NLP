import joblib

rf = joblib.load("rf_model.pkl")
cv = joblib.load("cv.pkl")
sample = ["This movie was absolutely amazing and emotional"]
vector = cv.transform(sample)
prediction = rf.predict(vector)
print("Prediction:", prediction)

# cd backend
# uvicorn main:app --reload
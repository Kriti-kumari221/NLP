# 🎬 CineRead — AI Sentiment Analysis for Movie Reviews

CineRead is a full-stack AI-powered web application that analyzes the sentiment of movie reviews using Machine Learning.
It predicts whether a review is **Positive 😊** or **Negative 😡**, along with confidence scores.

---

## 🚀 Live Demo

👉 https://nlp-one-gilt.vercel.app/

---

## 📌 Features

* 🧠 Machine Learning model (Random Forest)
* 🎯 Real-time sentiment prediction
* 📊 Confidence score + probability breakdown
* 🌐 Flask backend API
* 🎨 Beautiful animated frontend (HTML, CSS, JS)
* 🔗 Full frontend ↔ backend integration

---

## 🏗️ Project Structure

```id="a1"
SENTIMENT_ANALYSIS/
│
├── backend/
│   ├── main.py              # Flask API
│   ├── rf_model.pkl         # Trained ML model
│   ├── cv.pkl               # CountVectorizer
│   ├── IMDB Dataset.csv     # Dataset
│   └── NLP_Sentiment_Analysis.ipynb
│
├── frontend/
│   └── index.html           # UI
│
└── README.md
```

---

## ⚙️ Tech Stack

### 🔹 Backend

* Python
* Flask
* scikit-learn
* joblib

### 🔹 Frontend

* HTML5
* CSS3 (Advanced UI + Animations)
* JavaScript (Fetch API)

---

## 🧠 Model Details

* Algorithm: **Random Forest Classifier**
* Vectorization: **CountVectorizer**
* Dataset: **IMDB Movie Reviews**
* Output:

  * Sentiment (Positive / Negative)
  * Confidence Score
  * Probability distribution

---

## 📡 API Endpoint

### 🔹 POST `/predict`

#### Request:

```json id="a2"
{
  "text": "This movie is amazing!"
}
```

#### Response:

```json id="a3"
{
  "sentiment": "Positive 😊",
  "score": 0.95,
  "positive_probability": 0.95,
  "negative_probability": 0.05
}
```

---

## ⚠️ Important Notes

* Backend must be running for API requests
* Do NOT open frontend using `file://`
* Use a local server (`python -m http.server`) for development
* Ensure CORS is properly configured

---
---

## 🚀 Future Improvements

* 🌐 Deploy backend (Render / Railway)
* ⚛️ Convert frontend to React
* 📄 Add Swagger API docs
* 🤖 Upgrade model (Deep Learning / NLP transformers)

---


---

⭐ If you like this project, give it a star on GitHub!

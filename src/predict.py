import joblib
import re

model = joblib.load("models\logistic_regression_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

def clean_text(text):
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    return text.lower()

def predict_news(text):
    text = clean_text(text)
    X = tfidf.transform([text])
    proba = model.predict_proba(X)[0]
    return {
        "prediction": "Real" if model.predict(X)[0] == 1 else "Fake",
        "confidence": f"{max(proba)*100:.2f}%"
    }
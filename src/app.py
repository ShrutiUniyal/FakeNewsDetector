import streamlit as st
from predict import predict_news

st.title("Fake News Detector")
user_input = st.text_area("Paste news text here:")

if st.button("Check"):
    if user_input:
        result = predict_news(user_input)
        st.success(f"Prediction: **{result['prediction']}** (Confidence: {result['confidence']})")
    else:
        st.warning("Enter text to check!")
import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('Model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def predict_spam_or_not_spam(text):
    text_vectorized = vectorizer.transform([text]).toarray()
    
    prediction = model.predict(text_vectorized)

    if prediction == 1:
        return "Spam"
    else:
        return "Not Spam"

st.title("Spam Detection")


input_text = st.text_area("Enter the message here:")

if st.button("Predict"):
    if input_text.strip():
        with st.spinner('Classifying...'):
            result = predict_spam_or_not_spam(input_text)  
            st.write(f"The message is classified as: **{result}**")
    else:
        st.error("Please enter a non-empty message to classify.")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Save Prediction"):
    if input_text.strip():
        result = predict_spam_or_not_spam(input_text) 
        st.session_state.history.append((input_text, result))
        st.success("Prediction saved!")

st.subheader("Prediction History")
if st.session_state.history:
    for idx, (msg, res) in enumerate(st.session_state.history, 1):
        st.write(f"{idx}. Classified as: {res}")
        st.write(f"Message: {msg}")
        
else:
    st.write("No history yet.")

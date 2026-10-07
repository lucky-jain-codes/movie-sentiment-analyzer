import streamlit as st
import joblib

# Load trained assets
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to evaluate its sentiment using Machine Learning.")

user_input = st.text_area("Review text:", placeholder="Write your review here...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Transform input and get prediction
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]
        confidence = model.predict_proba(vec_input).max() * 100

        if prediction == 'positive':
            st.success(f"Positive Sentiment! 😊 (Confidence: {confidence:.1f}%)")
        else:
            st.error(f"Negative Sentiment! 😞 (Confidence: {confidence:.1f}%)")
    else:
        st.warning("Please enter a review first.")
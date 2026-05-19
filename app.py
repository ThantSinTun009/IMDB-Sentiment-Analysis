# ==========================
# Import Libraries
# ==========================

import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ==========================
# Load Model and Vectorizer
# ==========================

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# ==========================
# Text Preprocessing
# ==========================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)

    # Remove numbers
    text = re.sub(r'\\d+', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned_tokens = []

    for word in tokens:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            cleaned_tokens.append(lemma)

    return ' '.join(cleaned_tokens)

# ==========================
# Streamlit UI
# ==========================

st.title("🎬 IMDB Sentiment Analysis App")

st.write("""
This application predicts whether a movie review is
Positive or Negative using Machine Learning.
""")

# Text Input
review = st.text_area("Enter Movie Review")

# Predict Button
if st.button("Predict Sentiment"):

    if review.strip() != "":

        # Preprocess review
        cleaned_review = preprocess_text(review)

        # Transform text
        vector_input = vectorizer.transform([cleaned_review])

        # Prediction
        prediction = model.predict(vector_input)[0]

        # Probability
        probability = model.predict_proba(vector_input)[0]

        # Display Result
        if prediction == 1:
            st.success("✅ Positive Review")

            st.write(f"Confidence: {max(probability)*100:.2f}%")

        else:
            st.error("❌ Negative Review")

            st.write(f"Confidence: {max(probability)*100:.2f}%")

    else:
        st.warning("Please enter a review.")
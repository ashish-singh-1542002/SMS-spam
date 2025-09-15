import streamlit as st
import pandas as pd
import numpy as np
import string
import time
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import pickle

# Cache NLTK downloads so they run only once
@st.cache_resource
def setup_nltk():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")

setup_nltk()

from nltk.corpus import stopwords

# Streamlit page setup
st.set_page_config(
    page_title="SMS Spam Analyzer",
    page_icon="email.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.warning("⚠️ The SMS spam analyzer model has been trained on diverse datasets to ensure high accuracy. However, it may occasionally produce inaccurate results.")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def transform_text(txt):
    txt = txt.lower()
    tokens = nltk.word_tokenize(txt)
    filtered = [ps.stem(w) for w in tokens if w.isalnum() and w not in stop_words]
    return " ".join(filtered)

# Load pre-trained vectorizer and model
tfidf = pickle.load(open("vector.pkl", "rb"))
model = pickle.load(open("advanced.pkl", "rb"))

# Sidebar info
st.sidebar.title("SMS/Email Spam Analyser")
st.sidebar.caption("Developed by Farneet Singh")

# Main input
st.header("Enter the SMS or Email:")
input_sms = st.text_area("", placeholder="Enter the text here", height=150)

if st.button("Analyze"):
    if input_sms.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms]).toarray()
        result = model.predict(vector_input)[0]
        if result == 1:
            st.error("🚨 Spam Detected!")
        else:
            st.success("✅ This message is Not Spam.")

# FAQ Section
st.markdown("---")
st.header("Frequently Asked Questions")
st.markdown("""
**Q1:** What is an SMS spam analyzer?  
**A:** It's a tool that identifies whether a text message is spam.

**Q2:** How does it work?  
**A:** It uses ML algorithms trained on diverse datasets to detect spam messages.

**Q3:** Is it accurate?  
**A:** Yes! It demonstrates 98% accuracy and 99% precision.

**Q4:** Is my data safe?  
**A:** Absolutely. No personal information is collected or stored.

**Q5:** How to use?  
**A:** Enter a text message above and click "Analyze".
""")

# Feedback Section
st.markdown("---")
st.subheader("Was this application helpful?")
col1, col2 = st.columns(2)

with col1:
    if st.button("Yes 👍"):
        st.success("Thank you for your feedback! We're glad to hear that our tool was useful.")

with col2:
    if st.button("No 👎"):
        st.error("We're sorry to hear that. Please let us know how we can improve.")

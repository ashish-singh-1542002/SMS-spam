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

@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

setup_nltk()

from nltk.corpus import stopwords

st.set_page_config(
    page_title="SMS Spam Analyzer",
    page_icon="email.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.warning("⚠️ The SMS spam analyzer model has been trained on diverse datasets to ensure high accuracy. However, it is important to note that the machine learning model may occasionally produce inaccurate results.")

ps = PorterStemmer()
t = stopwords.words('english')

def transform_text(txt):
    txt = txt.lower()
    tokens = nltk.word_tokenize(txt)
    clean = [ps.stem(i) for i in tokens if i.isalnum() and i not in t]
    return " ".join(clean)

tfidf = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('advanced.pkl', 'rb'))

st.sidebar.title("SMS/Email Spam Analyser")
st.sidebar.text('Developed by Farneet Singh')

st.header('Enter the SMS or Email:')

input_sms = st.text_area('', placeholder='Enter the text here', height=150)

if st.button('Analyze'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms]).toarray()
    result = model.predict(vector_input)[0]
    if result == 1:
        st.error("Spam")
    else:
        st.success("Not Spam")

# FAQ Section
st.write('\n' * 4)
with st.form("key_form"):
    st.header("Frequently Asked Questions")
    st.subheader("What is an SMS spam analyzer?")
    st.write("An SMS spam analyzer is a tool that can help you identify whether a text message is spam or not.")
    st.subheader("How does it work?")
    st.write("It uses machine learning algorithms trained on diverse datasets to detect spam messages accurately.")
    st.subheader("Is it accurate?")
    st.write("It demonstrates 98% accuracy and 99% precision. Continuous improvements are being made.")
    st.subheader("Is my data safe?")
    st.write("Yes. No personal information is collected or stored.")
    st.subheader("How to use?")
    st.write("Enter a text message and click 'Analyze'.")
    if st.form_submit_button(label='Thanks ❤️'):
        st.success("You're welcome!")

# Feedback Section
st.write('\n' * 3)
with st.form('your_feedback'):
    st.subheader('Was this application helpful?')
    if st.form_submit_button("Yes 👍"):
        st.success("Thank you for your feedback! We're glad to hear that our tool was useful.")
    elif st.form_submit_button("No 👎"):
        st.error("We're sorry to hear that. Please let us know how we can improve.")

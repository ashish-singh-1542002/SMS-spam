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

nltk.download('stopwords')
nltk.download('punkt')
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
def transform_text(text):
    text = text.lower() #Lowercase
    print(text)
    
    text = nltk.word_tokenize(text) #Tokenisation
    print(text)
    
    y = []
    for i in text:
        if i.isalnum(): # Removing special characters
            y.append(i)
#     print(y)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in t and i not in string.punctuation: # Removing stop words and punctuation
            y.append(i)
#     print(y) 
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
#     print(y)
    
    return " ".join(y)
    
# tfidf = pickle.load(open('vectorizer2.pkl', 'rb'))
# model = pickle.load(open('model2.pkl', 'rb'))
tfidf = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('advanced.pkl', 'rb'))

st.sidebar.title("SMS/Email Spam Analyser")
st.sidebar.text('Developed by Farneet Singh')



st.header('Enter the SMS or Email:')

input_sms = st.text_area('', placeholder='Enter the text here', height=150)

if st.button('Analyze'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    vector_input = vector_input.toarray()
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.error("Spam")
    else:
        st.success("Not Spam")


# <---------- Spam Detection Functionality Above -------------------------->


# <---------- Other Functionalities Below -------------------------->

# Create a FAQ section
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')

with st.form("key_form"):
    st.header("Frequently Asked Questions")

    # Add questions and answers
    st.subheader("What is an SMS spam analyzer?")
    st.write("An SMS spam analyzer is a tool that can help you identify whether a text message is spam or not.")

    st.subheader("How does the SMS spam analyzer work?")
    st.write("The SMS spam analyzer utilizes a combination of advanced machine learning algorithms to accurately detect spam messages. These algorithms are trained on diverse datasets to learn patterns and characteristics associated with spam.")

    st.subheader("What types of text messages can the SMS spam analyzer analyze?")
    st.write("The SMS spam analyzer can analyze any type of text message, whether it's a promotional message, phishing scam, or other type of spam.")

    st.subheader("Is the SMS spam analyzer accurate?")
    st.write("The SMS spam analyzer demonstrates an impressive 98% accuracy and 99% precision when tested with a diverse dataset. We are committed to continuously improving its performance to provide reliable spam detection and enhance user security.")

    st.subheader("Is my data safe when using the SMS spam analyzer?")
    st.write("Yes, the SMS spam analyzer is designed to protect user data and ensure user privacy. No personal information is collected or stored when using the tool.")

    st.subheader("How can I use the SMS spam analyzer?")
    st.write("To use the SMS spam analyzer, simply enter the text message you want to analyze and click the 'Analyze' button. The tool will then analyze the message and provide a result indicating whether it is spam or not.")
    
    st.write('\n')
    thanks = st.form_submit_button(label='Thanks ❤️')
    if thanks:
        message = st.empty()
        message.write("You're welcome!")
        time.sleep(1.25)
        message.empty()

            

def handle_feedback(feedback):
    if feedback == 'Yes':
        st.write("Thank you for your feedback! We're glad to hear that our tool was useful.")
    elif feedback == 'No':
        st.write("We're sorry to hear that our tool was not useful. Please let us know how we can improve it.")



# Display the feedback section with two buttons
# Define a function to handle feedback
st.write('\n')
st.write('\n')
st.write('\n')
with st.form('your_feedback'):
    st.subheader('Was this application helpful?')
    if st.form_submit_button("Yes 👍"):
        st.write("Thank you for your feedback! We're glad to hear that our tool was useful.")
    elif st.form_submit_button("No 👎"):
        st.markdown('''We're sorry to hear that our tool was not useful. Please let us know how we can improve it.''')

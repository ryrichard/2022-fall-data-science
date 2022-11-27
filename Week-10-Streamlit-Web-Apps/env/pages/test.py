import pickle
import streamlit as st

# Import pandas for data handling
import pandas as pd

# NLTK is our Natural-Language-Took-Kit
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Libraries for helping us with strings
import string
# Regular Expression Library
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
stopwords = stopwords.words('english')

model = pickle.load(open('/app/2022-fall-data-science/Week-10-Streamlit-Web-Apps/env/model/nlp_model.pkl', 'rb')) 
vectorizer = pickle.load(open('/app/2022-fall-data-science/Week-10-Streamlit-Web-Apps/env/model/TfidfVectorizer.pkl', 'rb'))


def lowerWords(txt):
    return txt.lower()

def removePunctuation(txt):
    txt = re.sub(r'[^\w\s]','',txt)
    return txt

def removeStopWords(txt):
    txt = word_tokenize(txt)
    valid_words = []
    
    for word in txt:
        if word not in stopwords:
            valid_words.append(word)
            
    txt = ' '.join(valid_words)
    
    return txt

def textPipeline(txt):
    txt = lowerWords(txt)
    txt = removePunctuation(txt)
    txt = removeStopWords(txt)
    return txt

def PredictParty(text, vectorizer, model):
    pipText = [textPipeline(text)]
    txt = vectorizer.transform(pipText)
    pred = model.predict(txt)
    return pred


st.set_page_config(
    page_title = "Testing NLP",
    # page_icon=""
    layout="centered"
)

c1, c2 = st.columns([1,5])

with c2:
    st.title("NLP Model")
    text = "I have a gun"
    partyPrediction = PredictParty(text, vectorizer, model)
    st.writer(f"This text was written by a {partyPrediction}")
    # text = st.text_input('Sample Text', '')
    # if text is not '':
    #     partyPrediction = PredictParty(text, vectorizer, model)
    #     # st.header(f"This text was written by a {partyPrediction}")
    #     st.write(text)

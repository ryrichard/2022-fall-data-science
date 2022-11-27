import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
stopwords = stopwords.words('english')

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
    txt = textPipeline(text)
    text = vectorizer.transform([txt])
    pred = model.predict(text)
    return pred


model = pickle.load(open('nlp_model.pkl', 'rb'))
vectorizer = pickle.load(open('TfidfVectorizer.pkl', 'rb'))

st.set_page_config(
    page_title = "Testing NLP",
    # page_icon=""
    layout="centered"
)

c1, c2 = st.columns([1,5])

with c2:
    st.title("NLP Model")
    text = st.text_input('Sample Text', '')
    if text is not '':
        partyPrediction = PredictParty(text, vectorizer, model)
        st.header(f"This text was written by a {partyPrediction}")
        st.write(text)

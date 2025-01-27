import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer
import os

st.set_page_config(
    page_title = "Hello",
    page_icon = "👋"
)

#Create a page header
st.header("Welcome to my homepage! 👋")

#create three comlumns
col1, col2, col3, = st.columns([1,1,1]) #gives them even spacing

#inside col1
with col1:
    
    st.write(
        "#col1"
    )
    # display a picutre
    st.image('/app/2022-fall-data-science/Week-10-Streamlit-Web-Apps/env/images/test.png')

    # display the link to that page
    st.write('<a href="/test"> Test Board</a>', unsafe_allow_html=True) #display another streamlit py page


with col2:
    window = st.slider("Slide Test")
    st.write(window)

with col3:
    st.write(
        "# column 3"
        )
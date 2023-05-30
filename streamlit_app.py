import streamlit as st
import pandas as pd

st.write("Hello World")
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

df = pd.read_csv('static/2023_01_to_04.csv')
st.write(df)
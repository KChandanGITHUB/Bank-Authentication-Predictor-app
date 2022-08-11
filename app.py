import numpy as np
import pickle
import pandas as pd
import streamlit as st
import xgboost as xgb 
import pickle
from fileinput import filename

from PIL import Image

pickle_in = open("model.pkl","rb")
model = pickle.load(pickle_in)

def predict(Variance, Skewness, Curtosis, Entrooy):
  # match the name of the columns from the training data.
  prediction = model.predict(pd.DataFrame([[Variance, Skewness, Curtosis, Entrooy]], columns=['Variance', 'Skewness', 'Curtosis', 'Entrooy']))
  return prediction

st.title('Bank Authenticator')
st.header('Enter the characteristics of the Bank Authenticator:')
  
Variance = st.number_input('Variance', min_value=-10.0, max_value=100.0, value=1.0)
Skewness = st.number_input("Skewness", min_value=-10.0, max_value=100.0, value=1.0)
Curtosis = st.number_input("Curtosis", min_value=-10.0, max_value=100.0, value=1.0)
Entropy = st.number_input("Entropy",min_value=-10.0, max_value=100.0, value=1.0)

if st.button("Predict"):
    result = predict(Variance,Skewness,Curtosis,Entropy)
    st.success('The output is {}'.format(result))

import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv("data//Salary_Data.csv", sep = ",")
x = np.array(data['YearsExperience']).reshape(-1, 1)
lr = LinearRegression()
lr.fit(x, np.array(data['Salary']))

st.title("Salary Predictor")
st.image("data//SalPred.jpeg", width = 800)
nav = st.sidebar.radio("Navigation", ["Home", "Prediction", "Contribute"])

if nav == "Home":
    if st.checkbox("Show Table"):
        st.table(data)
        
if nav == "Prediction":
    pass

if nav == "Contribute":
    pass

import streamlit as st
import pandas as pd
import pickle
import numpy as np




model = pickle.load(open('LinearRegressionModel.pkl','rb'))
car = pd.read_csv('cleaningdata.csv')

st.title("Welcome To Car Price Predictor")
st.markdown("### select the company")
company = st.selectbox("select compnay",sorted(car['company'].unique()))
st.markdown("### select the model")
name = st.selectbox("select the model",sorted(car['name'].unique()))
st.markdown("### select Year of Purchase")
year = st.selectbox("select year of purchase",sorted(car['year'].unique(),reverse=True))
st.markdown("### select the Fuel Type")
fuel_type = st.selectbox("select the fuel type",car['fuel_type'].unique())
st.markdown("### Enter the Number of Kilometers that the car has travelled")
kms_driven = st.selectbox("Enter the number of kms that the car has travelled",range(1,50000))

if st.button("Precit Price",use_container_width=True):

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([name,company,year,kms_driven,fuel_type]).reshape(1, 5)))
    ps = str(np.round(prediction[0],2))
    st.header(ps+"Rs")


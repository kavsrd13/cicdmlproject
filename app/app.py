import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("../model/model.pkl")

st.title("💰 Income Prediction App")
st.write("This app predicts whether a person earns >50K or <=50K based on census data.")

# Input fields
age = st.number_input("Age", min_value=17, max_value=90, value=30)
workclass = st.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
])
fnlwgt = st.number_input("Fnlwgt", min_value=10000, max_value=1500000, value=200000)
education = st.selectbox("Education", [
    "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
    "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th",
    "Doctorate", "5th-6th", "Preschool"
])
education_num = st.number_input("Education Num", min_value=1, max_value=16, value=10)
marital_status = st.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])
occupation = st.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
])
relationship = st.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])
race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
sex = st.selectbox("Sex", ["Male", "Female"])
capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
hours_per_week = st.number_input("Hours per week", min_value=1, max_value=100, value=40)
native_country = st.selectbox("Native Country", [
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
    "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South",
    "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland",
    "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France",
    "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia",
    "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
    "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
])

# Collect inputs into dataframe
input_data = pd.DataFrame([{
    "age": age,
    "workclass": workclass,
    "fnlwgt": fnlwgt,
    "education": education,
    "education_num": education_num,
    "marital_status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "sex": sex,
    "capital_gain": capital_gain,
    "capital_loss": capital_loss,
    "hours_per_week": hours_per_week,
    "native_country": native_country
}])

# Prediction button
if st.button("Predict Income"):
    prediction = model.predict(input_data)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income: {result}")

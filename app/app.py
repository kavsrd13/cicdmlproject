import os
import base64
import requests
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/model.pkl")

# GitHub repo details
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
REPO = "kavsrd13/cicdmlproject"
BRANCH = "main"
LOG_FILE_PATH = "logs/inference_log.csv"  # path inside repo
API_URL = f"https://api.github.com/repos/{REPO}/contents/{LOG_FILE_PATH}"

def push_file_to_github(content, message="Update inference log"):
    # Check if file already exists (get its SHA)
    response = requests.get(API_URL, headers={"Authorization": f"token {GITHUB_TOKEN}"})
    sha = None
    if response.status_code == 200:
        sha = response.json()["sha"]

    # Prepare request body
    data = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": BRANCH
    }
    if sha:
        data["sha"] = sha

    # Push file to GitHub
    response = requests.put(
        API_URL,
        headers={"Authorization": f"token {GITHUB_TOKEN}"},
        json=data
    )
    if response.status_code in [200, 201]:
        st.success("✅ Logs pushed to GitHub successfully!")
    else:
        st.error(f"❌ Failed to push logs: {response.json()}")

st.title("💰 Income Prediction App")

# Inputs
age = st.number_input("Age", min_value=17, max_value=90, value=30)
workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc"])
fnlwgt = st.number_input("Fnlwgt", min_value=10000, max_value=1500000, value=200000)

# Collect into DataFrame
input_data = pd.DataFrame([{
    "age": age,
    "workclass": workclass,
    "fnlwgt": fnlwgt
}])

# Predict button
if st.button("Predict Income"):
    prediction = model.predict(input_data)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Income: {result}")

    # Append to log
    input_data["prediction"] = result
    input_data["timestamp"] = pd.Timestamp.now()

    # Convert to CSV string
    csv_content = input_data.to_csv(index=False)

    # Push to GitHub
    push_file_to_github(csv_content, message="Update inference log")

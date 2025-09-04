import streamlit as st
import pandas as pd
import requests
import base64
from evidently.report import Report
from evidently.metrics import DataDriftPreset, PredictionDriftMetric

# GitHub repo info
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
REPO = "kavsrd13/cicdmlproject"
BRANCH = "main"
LOG_FILE_PATH = "logs/inference_log.csv"
API_URL = f"https://api.github.com/repos/{REPO}/contents/{LOG_FILE_PATH}"

def get_logs_from_github():
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(API_URL, headers=headers)
    if response.status_code == 200:
        content = base64.b64decode(response.json()["content"]).decode("utf-8")
        return pd.read_csv(pd.compat.StringIO(content))
    else:
        st.error("❌ Could not fetch logs from GitHub")
        return None

st.title("📊 Drift Monitoring Dashboard")

# Load reference data (training dataset)
reference_data = pd.read_csv("data/adult_income.csv")  # your training data

# Load current (production) data
current_data = get_logs_from_github()

if current_data is not None:
    st.write("✅ Loaded logs from GitHub:")
    st.dataframe(current_data.tail())

    # Evidently report
    report = Report(metrics=[DataDriftPreset(), PredictionDriftMetric()])
    report.run(reference_data=reference_data, current_data=current_data)

    # Save as HTML
    report.save_html("drift_report.html")

    # Display inside Streamlit
    with open("drift_report.html", "r") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=800, scrolling=True)
else:
    st.warning("⚠️ No log data available yet.")

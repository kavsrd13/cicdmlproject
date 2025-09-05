# app/pages/drift_app.py

import streamlit as st
import pandas as pd
import os
import joblib
from datetime import datetime
from streamlit.components.v1 import html as st_html

# Evidently (v0.4.33+)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataSummaryPreset, DataQualityPreset

st.set_page_config(page_title="Model & Data Drift", layout="wide")

st.title("📊 Drift & Monitoring Dashboard (Evidently)")


# ------------------- Reference Data -------------------
def build_reference_dataset():
    ref_path = os.path.join("data", "adult_income.csv")
    if not os.path.exists(ref_path):
        st.error("Reference training data not found at data/adult_income.csv")
        return None

    df = pd.read_csv(ref_path)

    # clean income column if exists
    if "income" in df.columns:
        df["income"] = df["income"].astype(str).str.strip()

    # prepare target column (0/1)
    if "income" in df.columns:
        df["target"] = df["income"].apply(lambda x: 1 if ">50" in str(x) else 0)

    # load trained model
    model_path = os.path.join("model", "model.pkl")
    if not os.path.exists(model_path):
        st.error("Trained model not found at model/model.pkl")
        return None

    model = joblib.load(model_path)
    X = df.drop(columns=[c for c in ["income", "target"] if c in df.columns])

    try:
        preds = model.predict(X)
    except Exception as e:
        st.error(f"Model prediction failed on training data: {e}")
        return None

    df["prediction"] = preds
    return df


# ------------------- Current (inference) Data -------------------
def load_current_dataset():
    log_file = os.path.join("logs", "inference_log.csv")
    if not os.path.exists(log_file):
        st.warning("No inference logs found yet. Make some predictions first.")
        return None
    try:
        df = pd.read_csv(log_file)
        return df
    except Exception as e:
        st.error(f"Error reading inference logs: {e}")
        return None


# ------------------- Main Flow -------------------
with st.spinner("Loading reference and current data..."):
    reference_df = build_reference_dataset()
    current_df = load_current_dataset()

if reference_df is None or current_df is None:
    st.stop()

st.subheader("Reference dataset preview")
st.dataframe(reference_df.sample(min(5, len(reference_df))))

st.subheader("Current inference logs preview")
st.dataframe(current_df.tail(5))


# ------------------- Generate Reports -------------------
st.markdown("### Running Evidently Reports...")

# Data Drift
drift_report = Report(metrics=[DataDriftPreset()])
with st.spinner("Running Data Drift Report..."):
    drift_report.run(reference_data=reference_df, current_data=current_df)

# Data Summary
summary_report = Report(metrics=[DataSummaryPreset()])
with st.spinner("Running Data Summary Report..."):
    summary_report.run(current_data=current_df)

# Data Quality
quality_report = Report(metrics=[DataQualityPreset()])
with st.spinner("Running Data Quality Report..."):
    quality_report.run(current_data=current_df)

# ------------------- Save and Render -------------------
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

drift_html = os.path.join(reports_dir, f"drift_{timestamp}.html")
summary_html = os.path.join(reports_dir, f"summary_{timestamp}.html")
quality_html = os.path.join(reports_dir, f"quality_{timestamp}.html")

drift_report.save_html(drift_html)
summary_report.save_html(summary_html)
quality_report.save_html(quality_html)

st.success("Reports generated successfully ✅")

# Show Drift Report
st.markdown("### 🔄 Data Drift Report")
with open(drift_html, "r", encoding="utf-8") as f:
    st_html(f.read(), height=800, scrolling=True)

# Show Summary Report
st.markdown("### 📑 Data Summary Report")
with open(summary_html, "r", encoding="utf-8") as f:
    st_html(f.read(), height=800, scrolling=True)

# Show Data Quality Report
st.markdown("### ✅ Data Quality Report")
with open(quality_html, "r", encoding="utf-8") as f:
    st_html(f.read(), height=800, scrolling=True)

st.info("Reports compare training data (reference) vs. production logs (current).")

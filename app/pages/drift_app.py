# app/pages/drift_app.py

import streamlit as st
import pandas as pd
import os
import joblib
from datetime import datetime
from streamlit.components.v1 import html as st_html

# Evidently (v0.6.7)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping


# ------------------- Streamlit Config -------------------
st.set_page_config(page_title="Model & Data Drift", layout="wide")
st.title("📊 Drift Monitoring Dashboard (Evidently)")


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


# ------------------- Column Mapping -------------------
def build_column_mapping(df: pd.DataFrame):
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = df.select_dtypes(include=["number"]).columns.tolist()
    column_mapping.categorical_features = df.select_dtypes(exclude=["number"]).columns.tolist()

    if "target" in df.columns:
        column_mapping.target = "target"
    if "prediction" in df.columns:
        column_mapping.prediction = "prediction"
        column_mapping.task = "classification"

    return column_mapping


# ------------------- Align Columns -------------------
def align_columns(reference_df, current_df):
    required_cols = set(reference_df.columns) | set(current_df.columns)

    for col in required_cols:
        if col not in reference_df.columns:
            reference_df[col] = None
        if col not in current_df.columns:
            current_df[col] = None

    return reference_df, current_df


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

# Align schema
reference_df, current_df = align_columns(reference_df, current_df)

# Build column mapping
column_mapping = build_column_mapping(reference_df)

# ------------------- Generate Reports -------------------
st.markdown("### Running Evidently Reports...")

# Data Drift Report
drift_report = Report(metrics=[DataDriftPreset()])
with st.spinner("Running Data Drift Report..."):
    drift_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping
    )

# Model Performance / Classification Drift Report
model_report = Report(metrics=[ClassificationPreset()])
with st.spinner("Running Model Performance Report..."):
    model_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping
    )

# ------------------- Save and Render -------------------
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

drift_html = os.path.join(reports_dir, f"drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html")
model_html = os.path.join(reports_dir, f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html")

drift_report.save_html(drift_html)
model_report.save_html(model_html)

st.success("Reports generated successfully ✅")

# Show Drift Report
st.markdown("### 🔄 Data Drift Report")
with open(drift_html, "r", encoding="utf-8") as f:
    st_html(f.read(), height=800, scrolling=True)

# Show Model Performance Report
st.markdown("### 🤖 Classification Performance Report")
with open(model_html, "r", encoding="utf-8") as f:
    st_html(f.read(), height=800, scrolling=True)

st.info("Reports compare training data (reference) vs. production logs (current).")

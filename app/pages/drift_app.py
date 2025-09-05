# app/pages/drift_app.py

import streamlit as st
import pandas as pd
import requests
import base64
import io
import os
import joblib
from datetime import datetime
from streamlit.components.v1 import html as st_html

# Evidently imports (v0.4.x)
from evidently import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping


# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Model & Data Drift", layout="wide")
st.title("📊 Drift Monitoring Dashboard (Evidently)")

# ----------------------------
# Repository / GitHub Config
# ----------------------------
REPO = "kavsrd13/cicdmlproject"              # replace with your repo
BRANCH = "main"
LOG_FILE_PATH = "logs/inference_log.csv"     # path in repo
GITHUB_API_URL = f"https://api.github.com/repos/{REPO}/contents/{LOG_FILE_PATH}"

# GitHub token from Streamlit secrets
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN") if "GITHUB_TOKEN" in st.secrets else None
if not GITHUB_TOKEN:
    st.warning("⚠️ No GITHUB_TOKEN found in Streamlit secrets. "
               "You can still upload logs manually for local testing.")

# ----------------------------
# Helper Functions
# ----------------------------
def fetch_logs_from_github():
    """Fetch inference logs CSV from GitHub repo via contents API."""
    if not GITHUB_TOKEN:
        return None, "No GITHUB_TOKEN configured."

    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r = requests.get(GITHUB_API_URL, headers=headers)

    if r.status_code == 200:
        content_b64 = r.json().get("content", "")
        content = base64.b64decode(content_b64).decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
        return df, None
    elif r.status_code == 404:
        return None, "Log file not found in repository."
    else:
        return None, f"GitHub API error {r.status_code}: {r.text}"


def build_reference_df():
    """Prepare reference dataset from training data + model predictions."""
    ref_path = os.path.join("data", "adult_income.csv")
    model_path = os.path.join("model", "model.pkl")

    if not os.path.exists(ref_path):
        st.error(f"❌ Reference data not found at {ref_path}")
        return None
    if not os.path.exists(model_path):
        st.error("❌ Trained model not found at model/model.pkl — run training first.")
        return None

    # Load training data
    ref = pd.read_csv(ref_path)
    if "income" in ref.columns:
        ref["income"] = ref["income"].astype(str).str.strip()
        ref["target"] = ref["income"].apply(lambda x: 1 if ">50" in str(x) else 0)
    else:
        ref["target"] = -1

    # Load model & predict
    try:
        model = joblib.load(model_path)
        X_ref = ref.drop(columns=[c for c in ["income", "target"] if c in ref.columns], errors="ignore")
        preds = model.predict(X_ref)
    except Exception as e:
        st.error(f"❌ Model prediction on reference data failed: {e}")
        return None

    # Final reference DF
    ref_for_evidently = X_ref.copy()
    ref_for_evidently["prediction"] = preds
    ref_for_evidently["target"] = ref["target"].values
    return ref_for_evidently


def normalize_current_df(current_df, reference_cols):
    """Clean and align current logs with reference dataset columns."""
    current = current_df.copy()

    # Convert predictions to numeric if needed
    if "prediction" in current.columns:
        def map_pred(x):
            s = str(x).strip()
            if any(substr in s for substr in [">50", "50K"]): return 1
            if any(substr in s for substr in ["<=50", "<50K"]): return 0
            if s in ["1", "0"]: return int(s)
            try: return int(float(s))
            except: return None
        current["prediction"] = current["prediction"].apply(map_pred)

    # Ensure timestamp exists
    if "timestamp" in current.columns:
        current["timestamp"] = pd.to_datetime(current["timestamp"], errors="coerce")
    else:
        current["timestamp"] = pd.Timestamp.now()

    # Align columns
    keep_cols = [c for c in reference_cols if c in current.columns]
    for col in ["prediction", "target"]:
        if col in current.columns and col not in keep_cols:
            keep_cols.append(col)

    return current[keep_cols]


def run_evidently(reference_df, current_df):
    """Run Evidently DataDrift + TargetDrift reports."""
    numeric_cols = reference_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in reference_df.columns if c not in numeric_cols and c not in ("prediction", "target")]

    col_mapping = ColumnMapping(
        target="target" if "target" in reference_df.columns else None,
        prediction="prediction",
        task="classification",
        numerical_features=numeric_cols,
        categorical_features=cat_cols,
    )

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=col_mapping)
    return report

# ----------------------------
# Main App Flow
# ----------------------------
def main():
    st.info("This app monitors **data & model drift** by comparing training data "
            "against inference logs using Evidently.")

    # Load Reference Data
    with st.spinner("Loading reference dataset..."):
        reference_df = build_reference_df()
    if reference_df is None:
        st.stop()

    st.subheader("📂 Reference (Training) Dataset Preview")
    st.dataframe(reference_df.sample(min(5, len(reference_df))))

    # Fetch Current Data
    st.markdown("---")
    st.subheader("📂 Current (Inference) Logs")
    current_df, err = fetch_logs_from_github()

    if err:
        st.warning(err)
        st.info("Upload inference log CSV for local testing:")
        uploaded = st.file_uploader("Upload inference logs", type=["csv"])
        if uploaded:
            current_df = pd.read_csv(uploaded)

    if current_df is None or current_df.empty:
        st.warning("⚠️ No current inference data available. Make some predictions first.")
        st.stop()

    st.dataframe(current_df.tail(10))

    # Normalize & Align
    current = normalize_current_df(current_df, list(reference_df.columns))
    if current.empty:
        st.error("❌ Current data normalization failed.")
        st.stop()

    # Run Evidently
    st.markdown("---")
    st.subheader("📊 Drift Reports")
    with st.spinner("Generating Evidently drift report..."):
        try:
            report = run_evidently(reference_df, current)
        except Exception as e:
            st.error(f"❌ Evidently failed to run: {e}")
            st.stop()

    # Save & Embed Report
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    html_path = os.path.join(reports_dir, f"drift_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html")
    report.save_html(html_path)

    st.success("✅ Evidently drift report generated")

    with open(html_path, "r", encoding="utf-8") as f:
        report_html = f.read()
    st_html(report_html, height=900, scrolling=True)

    # Extra: Show Drift Summary
    st.markdown("---")
    st.subheader("📈 Drift Metrics (Summary)")
    try:
        json_result = report.as_dict()
        drift_summary = pd.json_normalize(json_result["metrics"])
        st.dataframe(drift_summary)
    except Exception:
        st.info("ℹ️ Drift summary table not available for this run.")

    st.markdown("---")
    st.caption("Notes: Reference data = training dataset + model predictions. "
               "Current data = inference logs from GitHub or uploaded CSV.")

# Run the app
if __name__ == "__main__":
    main()

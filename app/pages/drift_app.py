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
from evidently.report import Report
from evidently.report.presets import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

st.set_page_config(page_title="Model & Data Drift", layout="wide")

st.title("📊 Drift Monitoring (Evidently)")

# ---------- User / repo settings ----------
# Put your repo details here or use st.secrets in production
REPO = "kavsrd13/cicdmlproject"          # replace if different
BRANCH = "main"
LOG_FILE_PATH = "logs/inference_log.csv"  # file path in repo
GITHUB_API_URL = f"https://api.github.com/repos/{REPO}/contents/{LOG_FILE_PATH}"

# Get token from Streamlit secrets
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN") if "GITHUB_TOKEN" in st.secrets else None
if not GITHUB_TOKEN:
    st.warning("No GITHUB_TOKEN found in Streamlit secrets. You can still run locally if logs exist.")
# -----------------------------------------

# helper: fetch logs CSV from GitHub (contents API)
def fetch_logs_from_github():
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

# prepare reference dataset (training data + model predictions)
def build_reference_df():
    # load training csv
    ref_path = os.path.join("data", "adult_income.csv")
    if not os.path.exists(ref_path):
        st.error(f"Reference data not found at {ref_path}. Put training CSV at data/adult_income.csv")
        return None

    ref = pd.read_csv(ref_path)
    # make sure the 'income' column is cleaned
    if "income" in ref.columns:
        ref["income"] = ref["income"].astype(str).str.strip()

    # create numeric target column (0/1)
    if "income" in ref.columns:
        ref["target"] = ref["income"].apply(lambda x: 1 if (">50" in str(x)) else 0)
    else:
        # fallback if income not present
        ref["target"] = -1

    # load model and predict on reference inputs
    model_path = os.path.join("model", "model.pkl")
    if not os.path.exists(model_path):
        st.error("Trained model not found at model/model.pkl — run training first.")
        return None

    model = joblib.load(model_path)

    # prepare X_ref: drop 'income' and 'target' (if present) to match model input
    X_ref = ref.drop(columns=[c for c in ["income", "target"] if c in ref.columns], errors='ignore')
    try:
        preds = model.predict(X_ref)
    except Exception as e:
        st.error(f"Model prediction on reference data failed: {e}")
        return None

    ref_for_evidently = X_ref.copy()
    ref_for_evidently["prediction"] = preds
    # keep the numeric target (if available)
    if "target" in ref.columns:
        ref_for_evidently["target"] = ref["target"].values
    return ref_for_evidently

# helper to normalize/clean current logs so columns align with reference
def normalize_current_df(current_df, reference_cols):
    current = current_df.copy()

    # convert prediction column to numeric 0/1 if it's string like '>50K' or similar
    if "prediction" in current.columns:
        # try numeric conversion first
        try:
            current["prediction"] = pd.to_numeric(current["prediction"])
        except Exception:
            # map common string formats to 0/1
            def map_pred(x):
                s = str(x).strip()
                if s == "":
                    return None
                if any(substr in s for substr in [">50", "> 50", "50K", ">50K"]):
                    return 1
                if any(substr in s for substr in ["<=50", "<=50K", "<= 50", "<50", "< 50"]):
                    return 0
                # fallback: if '1' or '0' present
                if s in ["1", "0"]:
                    return int(s)
                # try float
                try:
                    return int(float(s))
                except Exception:
                    return None
            current["prediction"] = current["prediction"].apply(map_pred)

    # ensure timestamp column exists and parse
    if "timestamp" in current.columns:
        try:
            current["timestamp"] = pd.to_datetime(current["timestamp"])
        except Exception:
            pass
    else:
        current["timestamp"] = pd.Timestamp.now()

    # align columns: use intersection of input feature columns + prediction/target if present
    keep_cols = [c for c in reference_cols if c in current.columns]
    # ensure 'prediction' present
    if "prediction" not in keep_cols and "prediction" in current.columns:
        keep_cols.append("prediction")
    # if reference has 'target' include if present
    if "target" in reference_cols and "target" in current.columns:
        keep_cols.append("target")

    current = current[keep_cols]
    return current

# ----------------- Main UI flow -----------------
st.write("Evidently version will be used from the deployed environment.")

with st.spinner("Loading reference dataset and model..."):
    reference_df = build_reference_df()

if reference_df is None:
    st.stop()

st.markdown("### Reference (training) dataset preview")
st.dataframe(reference_df.sample(min(5, len(reference_df))))

st.markdown("---")
st.write("Fetching production inference logs from GitHub...")

current_df, err = fetch_logs_from_github()
if err:
    st.warning(err)
    st.info("If you don't want GitHub logs, you can upload a CSV file of inference logs below for local testing.")
    uploaded = st.file_uploader("Upload inference log CSV", type=["csv"])
    if uploaded:
        current_df = pd.read_csv(uploaded)
    else:
        current_df = None

if current_df is None or current_df.empty:
    st.warning("No current (inference) data available yet. Make some predictions from the app or upload a CSV.")
    st.stop()

st.markdown("### Current (inference) logs preview (last 10 rows)")
st.dataframe(current_df.tail(10))

# Normalize current to align with reference
ref_features = list(reference_df.columns)
current = normalize_current_df(current_df, ref_features)

if current is None or current.empty:
    st.error("Current data could not be normalized or is empty.")
    st.stop()

# Build column mapping for Evidently
numeric_cols = reference_df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in reference_df.columns if c not in numeric_cols and c not in ("prediction", "target")]

col_mapping = ColumnMapping()
# set roles
col_mapping.prediction = "prediction"
if "target" in reference_df.columns:
    col_mapping.target = "target"
# hint task (classification)
col_mapping.task = "classification"
col_mapping.numerical_features = numeric_cols
col_mapping.categorical_features = cat_cols

st.markdown("### Running Evidently reports (DataDriftPreset + TargetDriftPreset). This may take a few seconds...")

report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])


with st.spinner("Generating Evidently report..."):
    try:
        report.run(reference_data=reference_df, current_data=current, column_mapping=col_mapping)
    except Exception as e:
        st.error(f"Evidently failed to run the report: {e}")
        st.stop()

# save HTML report and embed
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)
html_path = os.path.join(reports_dir, f"drift_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html")
report.save_html(html_path)

st.success("Evidently report generated ✅")

st.markdown("### Interactive report")
with open(html_path, "r", encoding="utf-8") as f:
    report_html = f.read()

# embed the HTML (adjust height as needed)
st_html(report_html, height=900, scrolling=True)

st.markdown("---")
st.info("Notes:\n- Reference data is training data + model predictions (used as baseline). \n- Current data is your inference logs from GitHub (or uploaded CSV).")

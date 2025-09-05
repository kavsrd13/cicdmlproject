# app/pages/drift_app.py
import os
import io
import time
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib
import importlib
from streamlit.components.v1 import html as st_html
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model & Data Drift", layout="wide")
st.title("📊 Drift Monitoring (robust Evidently integration)")

# -------------------- Helper: Robust Evidently imports --------------------
# Try a few known import patterns across Evidently versions.
Report = None
DataDriftPreset = None
DataSummaryPreset = None
TargetDriftPreset = None
Dataset = None
DataDefinition = None

# Try Report import
for mod_name in ("evidently.report", "evidently"):
    try:
        mod = importlib.import_module(mod_name)
        Report = getattr(mod, "Report", None)
        break
    except Exception:
        Report = None

# Try presets imports from several possible modules
preset_imports = [
    "from evidently.presets import DataDriftPreset, DataSummaryPreset, TargetDriftPreset",
    "from evidently.metric_preset import DataDriftPreset, DataSummaryPreset, TargetDriftPreset",
    "from evidently.report.presets import DataDriftPreset, DataSummaryPreset, TargetDriftPreset",
    "from evidently.future.presets import DataSummaryPreset, DataDriftPreset, TargetDriftPreset",
]
for stmt in preset_imports:
    try:
        exec(stmt, globals())
        break
    except Exception:
        # continue trying other import locations
        pass

# Try Dataset / DataDefinition (optional)
for name in ("evidently",):
    try:
        mod = importlib.import_module(name)
        Dataset = getattr(mod, "Dataset", None)
        DataDefinition = getattr(mod, "DataDefinition", None)
        break
    except Exception:
        Dataset = None
        DataDefinition = None

# If we still don't have the minimum, stop and show instructions
if Report is None or DataDriftPreset is None:
    st.error(
        "Evidently imports failed in this environment. "
        "Two options:\n\n"
        "1) Pin Evidently to a stable version in requirements.txt (example: evidently==0.4.33) and redeploy. \n"
        "2) If you can't change versions, check the app logs and installed Evidently version and share it.\n\n"
        "You can see the installed Evidently version (if any) below."
    )
    try:
        import evidently
        st.write("Installed evidently:", evidently.__version__)
    except Exception:
        st.write("Evidently not installed in this environment.")
    st.stop()

st.write("Using Evidently Report and presets from the environment.")

# -------------------- Safe CSV reader (avoids UnicodeDecodeError) --------------------
def safe_read_csv(path):
    # try common encodings and fallback options
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    # last resort: engine='python' with on_bad_lines='skip'
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8")
    except Exception as e:
        raise

# -------------------- Load reference (training) dataset --------------------
def build_reference_df():
    ref_path = os.path.join("data", "adult_income.csv")
    if not os.path.exists(ref_path):
        st.error(f"Reference data missing at: {ref_path}. Put your training CSV there.")
        return None
    df = safe_read_csv(ref_path)
    # optional cleaning
    if "income" in df.columns:
        df["income"] = df["income"].astype(str).str.strip()
        df["target"] = df["income"].apply(lambda x: 1 if ">50" in str(x) else 0)
    # ensure no stray object columns breaking presets - minimal cleaning
    return df

# -------------------- Load current (inference) logs --------------------
def load_current_df():
    log_path = os.path.join("logs", "inference_log.csv")
    if not os.path.exists(log_path):
        st.info("No inference log found at logs/inference_log.csv yet. Make some predictions first.")
        return None
    try:
        df = safe_read_csv(log_path)
        return df
    except Exception as e:
        st.error(f"Failed to read logs: {e}")
        return None

# -------------------- Main flow --------------------
with st.spinner("Loading data & model..."):
    reference_df = build_reference_df()
    current_df = load_current_df()

if reference_df is None:
    st.stop()

# Quick preview
st.subheader("Reference preview")
st.dataframe(reference_df.sample(min(5, max(1, len(reference_df)))))

if current_df is None or current_df.empty:
    st.warning("No current (inference) data available. Either make predictions or upload logs below.")
    uploaded = st.file_uploader("Upload inference log CSV for testing", type=["csv"])
    if uploaded:
        try:
            current_df = pd.read_csv(uploaded)
        except Exception:
            current_df = pd.read_csv(uploaded, engine="python", on_bad_lines="skip")
    else:
        st.stop()

st.subheader("Current (inference) preview (last 10 rows)")
st.dataframe(current_df.tail(10))

# Normalize columns: make sure prediction is numeric 0/1 if possible
def normalize_preds(df):
    df = df.copy()
    if "prediction" in df.columns:
        # try numeric conversion, then fallback mapping common strings
        try:
            df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
        except Exception:
            df["prediction"] = df["prediction"].astype(str)
            def map_pred(s):
                s = str(s).strip()
                if any(x in s for x in [">50", "50K"]): return 1
                if any(x in s for x in ["<=50", "<50", "0"]): return 0
                try:
                    return int(float(s))
                except Exception:
                    return None
            df["prediction"] = df["prediction"].apply(map_pred)
    return df

current_df = normalize_preds(current_df)

# Build and run Report
st.markdown("### Running Evidently reports (this may take a few seconds)...")
report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])  # these are the presets we imported above

with st.spinner("Generating report..."):
    try:
        # many Evidently examples accept plain DataFrames for run(current, reference)
        report.run(current_df, reference_df)
    except TypeError:
        # older/newer APIs sometimes pass args differently; try explicit names
        try:
            report.run(current_data=current_df, reference_data=reference_df)
        except Exception as e:
            st.error(f"Report.run failed: {e}")
            st.stop()
    except Exception as e:
        st.error(f"Report.run failed: {e}")
        st.stop()

st.success("Evidently report computed ✅")

# Try to save & embed HTML if available
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
html_path = os.path.join(reports_dir, f"evidently_report_{ts}.html")

html_rendered = False
try:
    # preferred method used in many docs/examples
    if hasattr(report, "save_html"):
        report.save_html(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            st_html(f.read(), height=900, scrolling=True)
        html_rendered = True
    else:
        raise AttributeError("report.save_html unavailable in this Evidently build")
except Exception as e:
    st.warning("Could not export interactive HTML from Evidently: " + str(e))
    st.info("Falling back to JSON/dict visualization. The fallback still shows the numeric results and summary.")
    try:
        # fallback: show the JSON/dict and a small numeric summary plot
        report_dict = report.as_dict() if hasattr(report, "as_dict") else report.json()
        st.subheader("Evidently Report (raw JSON)")
        st.json(report_dict)
        # extract simple numeric metrics to plot (best-effort)
        def extract_numbers(obj, path=""):
            rows = []
            if isinstance(obj, dict):
                for k,v in obj.items():
                    new_path = f"{path}.{k}" if path else k
                    if isinstance(v, (int,float)):
                        rows.append((new_path, v))
                    else:
                        rows.extend(extract_numbers(v, new_path))
            elif isinstance(obj, list):
                for i,v in enumerate(obj):
                    rows.extend(extract_numbers(v, f"{path}[{i}]"))
            return rows

        numeric = extract_numbers(report_dict)
        if numeric:
            df_num = pd.DataFrame(numeric, columns=["metric", "value"]).sort_values("value", ascending=False).head(20)
            st.subheader("Top numeric metrics")
            st.dataframe(df_num)
            # quick bar chart
            fig, ax = plt.subplots(figsize=(10, min(6, len(df_num)/2 + 1)))
            ax.barh(df_num["metric"].iloc[::-1], df_num["value"].iloc[::-1])
            ax.set_xlabel("value")
            st.pyplot(fig)
        else:
            st.info("No numeric metrics found in report JSON to plot.")
    except Exception as e2:
        st.error(f"Fallback display failed: {e2}")

st.markdown("---")
st.info(
    "Notes:\n"
    "- The app tried to use Evidently's interactive HTML output (preferred). If your Evidently version lacks `save_html`, the app shows the report JSON and a small numeric summary instead.\n"
    "- If you want the full interactive HTML in Streamlit, pin Evidently to a version that exposes `save_html` (see instructions below)."
)

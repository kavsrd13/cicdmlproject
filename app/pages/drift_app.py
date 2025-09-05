# app/pages/drift_app.py
import streamlit as st
import pandas as pd
import os
import joblib
from datetime import datetime
from streamlit.components.v1 import html as st_html
import matplotlib.pyplot as plt

# Evidently (targeted to v0.6.7)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping

st.set_page_config(page_title="Model & Data Drift", layout="wide")
st.title("📊 Drift Monitoring Dashboard (Evidently)")

# -------------------- Helpers --------------------
def safe_read_csv(path):
    """Read CSV with fallback encodings to avoid UnicodeDecodeError."""
    encs = ["utf-8", "utf-8-sig", "latin1"]
    for e in encs:
        try:
            return pd.read_csv(path, encoding=e)
        except Exception:
            pass
    # last resort
    return pd.read_csv(path, engine="python", on_bad_lines="skip")

def map_prediction_column(s):
    """Try to convert various prediction string forms to 0/1 numeric."""
    if pd.isna(s):
        return None
    try:
        return int(float(s))
    except Exception:
        s = str(s).strip().lower()
        if any(x in s for x in [">50", "50k", "> 50"]):
            return 1
        if any(x in s for x in ["<=50", "<50", "<=50k", "< 50"]):
            return 0
        if s in ("yes", "y", "true", "1"):
            return 1
        if s in ("no", "n", "false", "0"):
            return 0
    return None

def normalize_predictions_column(df):
    """Ensure 'prediction' column numeric 0/1 where possible."""
    df = df.copy()
    if "prediction" not in df.columns:
        return df
    # try numeric cast
    try:
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
        # if many NaNs produced, map strings
        if df["prediction"].isna().sum() > 0.5 * len(df):
            df["prediction"] = df["prediction"].apply(lambda x: map_prediction_column(x) if not pd.notna(x) else int(x))
    except Exception:
        df["prediction"] = df["prediction"].apply(map_prediction_column)
    return df

def drop_fully_empty_columns(ref_df, cur_df):
    """Drop columns that are fully empty in either dataset (cause Evidently errors)."""
    # only keep columns that are present and have at least one non-null value in both
    common = set(ref_df.columns).intersection(set(cur_df.columns))
    keep = []
    for c in common:
        if not (ref_df[c].isna().all() or cur_df[c].isna().all()):
            keep.append(c)
    ref_clean = ref_df[keep].copy()
    cur_clean = cur_df[keep].copy()
    return ref_clean, cur_clean

def build_column_mapping_from_df(df):
    """Build ColumnMapping using features that remain in df."""
    cm = ColumnMapping()
    # numerical features exclude target/prediction (if present)
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    # exclude target/prediction from feature lists
    features_num = [c for c in numeric if c not in ("target", "prediction")]
    cat = [c for c in df.columns if c not in features_num and c not in ("target", "prediction")]
    cm.numerical_features = features_num
    cm.categorical_features = cat
    if "target" in df.columns and not df["target"].isna().all():
        cm.target = "target"
    if "prediction" in df.columns and not df["prediction"].isna().all():
        cm.prediction = "prediction"
        cm.task = "classification"
    return cm

# -------------------- Load reference & current --------------------
def build_reference_dataset():
    ref_path = os.path.join("data", "adult_income.csv")
    if not os.path.exists(ref_path):
        st.error("Reference training data not found at data/adult_income.csv")
        return None

    ref = safe_read_csv(ref_path)

    # Normalize/clean
    if "income" in ref.columns:
        ref["income"] = ref["income"].astype(str).str.strip()
        ref["target"] = ref["income"].apply(lambda x: 1 if ">50" in str(x) else 0)
    else:
        # no target column in reference — that's okay (we'll handle)
        pass

    model_path = os.path.join("model", "model.pkl")
    if not os.path.exists(model_path):
        st.error("Trained model not found at model/model.pkl")
        return None

    # try to predict on reference rows (if model accepts raw df)
    model = joblib.load(model_path)
    X = ref.drop(columns=[c for c in ["income", "target"] if c in ref.columns], errors="ignore")
    try:
        preds = model.predict(X)
        ref["prediction"] = preds
    except Exception as e:
        st.warning(f"Could not run model.predict on reference data (continuing without reference predictions): {e}")
        # don't inject prediction if failing
    return ref

def load_current_dataset():
    log_file = os.path.join("logs", "inference_log.csv")
    if not os.path.exists(log_file):
        st.warning("No inference logs found at logs/inference_log.csv — run some predictions first.")
        return None
    cur = safe_read_csv(log_file)
    # normalize prediction column into numeric form if present
    cur = normalize_predictions_column(cur)
    return cur

# -------------------- Main --------------------
with st.spinner("Loading reference and current data..."):
    reference_df = build_reference_dataset()
    current_df = load_current_dataset()

if reference_df is None:
    st.stop()
if current_df is None:
    st.stop()

st.subheader("Reference preview (sample)")
st.dataframe(reference_df.sample(min(5, max(1, len(reference_df)))))

st.subheader("Current (inference) preview (last rows)")
st.dataframe(current_df.tail(5))

# Keep only columns that are common and not fully empty in either
reference_df, current_df = drop_fully_empty_columns(reference_df, current_df)

if reference_df.shape[1] == 0 or current_df.shape[1] == 0:
    st.error("No overlapping non-empty columns between reference and current data. Cannot compute drift.")
    st.stop()

# Build mapping from the cleaned reference dataframe
column_mapping = build_column_mapping_from_df(reference_df)

# If prediction exists as strings in current, normalized earlier; ensure numeric dtype if possible
if "prediction" in current_df.columns:
    current_df["prediction"] = pd.to_numeric(current_df["prediction"], errors="coerce")

# Run Data Drift (always)
st.markdown("### Running Data Drift Report")
drift_report = Report(metrics=[DataDriftPreset()])
try:
    drift_report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
except Exception as e:
    st.error(f"DataDrift report failed: {e}")
    st.stop()

# Save or fallback render
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
drift_html = os.path.join(reports_dir, f"drift_{ts}.html")

if hasattr(drift_report, "save_html"):
    try:
        drift_report.save_html(drift_html)
        with open(drift_html, "r", encoding="utf-8") as f:
            st_html(f.read(), height=850, scrolling=True)
    except Exception as e:
        st.warning(f"Could not save/embed HTML for DataDrift: {e}")
        st.json(drift_report.as_dict() if hasattr(drift_report, "as_dict") else {})
else:
    # fallback: show dict
    st.json(drift_report.as_dict() if hasattr(drift_report, "as_dict") else {})

# Run Classification report only if both reference & current have non-empty target & prediction
can_run_classification = (
    ("target" in reference_df.columns and "target" in current_df.columns and
     not reference_df["target"].isna().all() and not current_df["target"].isna().all()) and
    ("prediction" in reference_df.columns and "prediction" in current_df.columns and
     not reference_df["prediction"].isna().all() and not current_df["prediction"].isna().all())
)

if can_run_classification:
    st.markdown("### Running Classification (Performance) Report")
    perf_report = Report(metrics=[ClassificationPreset()])
    try:
        perf_report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
        perf_html = os.path.join(reports_dir, f"perf_{ts}.html")
        if hasattr(perf_report, "save_html"):
            try:
                perf_report.save_html(perf_html)
                with open(perf_html, "r", encoding="utf-8") as f:
                    st_html(f.read(), height=850, scrolling=True)
            except Exception:
                st.json(perf_report.as_dict() if hasattr(perf_report, "as_dict") else {})
        else:
            st.json(perf_report.as_dict() if hasattr(perf_report, "as_dict") else {})
    except Exception as e:
        st.warning(f"Classification report failed: {e}")
else:
    st.info("Classification report skipped: 'target' and/or 'prediction' not present and non-empty in both datasets.")

st.success("Monitoring completed.")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import time
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from datetime import datetime
import pytz

from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

from core.data import generate_synthetic_access_data
from core.model import train, save, load
from core.explain import build_shap_explainer, translate_shap, build_lime_explainer, get_shap_for_instance

# --- Mappings for interpretability ---
ROLEMAP = {0: "Intern", 1: "Engineer", 2: "Manager", 3: "Admin", 4: "Exec"}
AUTHMAP = {0: "Password", 1: "2FA", 2: "Biometric"}
SENSMAP = {0: "Public", 1: "Internal", 2: "Confidential", 3: "Secret"}

st.set_page_config(page_title="XAI for Zero Trust Policy Justifier", layout="wide")

# --- Helper utilities ---
TREE_MODEL_NAMES = {
    "XGBClassifier", "XGBRFClassifier",
    "RandomForestClassifier", "DecisionTreeClassifier",
    "GradientBoostingClassifier", "HistGradientBoostingClassifier",
    "CatBoostClassifier"
}

def is_tree_model(model):
    try:
        return model.__class__.__name__ in TREE_MODEL_NAMES
    except Exception:
        return False

def formattimestamp(ts, zone="UTC"):
    try:
        tz = pytz.timezone(zone)
        return datetime.fromtimestamp(int(ts), tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        try:
            return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ts)

def labelforfeat(feat, val):
    if feat == "user_role":
        return ROLEMAP.get(int(val), str(val))
    if feat == "auth_strength":
        return AUTHMAP.get(int(val), str(val))
    if feat == "resource_sensitivity":
        return SENSMAP.get(int(val), str(val))
    return None

# --- Metrics / evaluation helpers ---
def compute_and_store_metrics(model, X_test, y_test, scaler=None, calibrated=False):
    """
    Compute common metrics (accuracy, precision, recall, f1, roc_auc, pr_auc)
    using the correct representation depending on whether model expects scaled input.
    Stores results in st.session_state['metrics'].
    """
    metrics = {}
    try:
        # choose X representation for predictions
        if (not is_tree_model(model)) and (scaler is not None) and not calibrated:
            Xpred = scaler.transform(X_test)
        elif (not is_tree_model(model)) and (scaler is not None) and calibrated:
            # if calibrated model is a wrapper that expects unscaled data (we used pipeline), prefer pipeline
            Xpred = X_test
        else:
            # tree models accept raw DataFrame/array
            Xpred = X_test

        # get predictions
        ypred = model.predict(Xpred)
        yprob = None
        if hasattr(model, "predict_proba"):
            yprob = model.predict_proba(Xpred)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(Xpred)
            yprob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        metrics['accuracy'] = float(accuracy_score(y_test, ypred))
        metrics['precision'] = float(precision_score(y_test, ypred, zero_division=0))
        metrics['recall'] = float(recall_score(y_test, ypred, zero_division=0))
        metrics['f1'] = float(f1_score(y_test, ypred, zero_division=0))
        if yprob is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_test, yprob))
            except Exception:
                metrics['roc_auc'] = None
            try:
                pr_prec, pr_rec, _ = precision_recall_curve(y_test, yprob)
                metrics['pr_auc'] = float(auc(pr_rec, pr_prec))
            except Exception:
                metrics['pr_auc'] = None
        else:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None

    except Exception as e:
        metrics = {'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'roc_auc': None, 'pr_auc': None}
        st.write("Metric computation failed:", e)

    st.session_state['metrics'] = metrics
    return metrics

def compute_cv_metrics(model, X, y, scaler=None, n_splits=5):
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    if (not is_tree_model(model)) and (scaler is not None):
        estimator = make_pipeline(scaler, model)
    else:
        estimator = model
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    res = cross_validate(estimator, X, y, cv=cv, scoring=scoring, return_train_score=False)
    parsed = {k.replace('test_',''): (float(np.mean(v)), float(np.std(v))) for k,v in res.items() if k.startswith('test_')}
    return parsed

# --- Audit persistence helpers (lightweight CSV persistence) ---
AUDIT_PATH = Path("audit_logs/audit.csv")
AUDIT_PATH.parent.mkdir(exist_ok=True, parents=True)

def append_audit_row(row: dict):
    df_row = pd.DataFrame([row])
    if not AUDIT_PATH.exists():
        df_row.to_csv(AUDIT_PATH, index=False)
    else:
        df_row.to_csv(AUDIT_PATH, index=False, header=False, mode='a')
    st.session_state.setdefault('audit', [])
    st.session_state['audit'].append(row)

# --- Session Initialization ---
def ensuresession():
    if 'model' not in st.session_state:
        df = generate_synthetic_access_data(2000)
        model, acc, X_train, X_test, y_train, y_test, scaler = train(df)

        feature_cols = [c for c in df.columns if c != 'label']

        st.session_state.df = df
        st.session_state.model = model
        st.session_state.acc = acc
        st.session_state.splits = (X_train, X_test, y_train, y_test)
        # scaler may be None for tree models - prefer returning None from train when using trees
        st.session_state.scaler = scaler
        st.session_state.feature_cols = feature_cols

        # restore audit if exists on disk, else start empty
        if AUDIT_PATH.exists():
            try:
                st.session_state['audit'] = pd.read_csv(AUDIT_PATH).to_dict(orient='records')
            except Exception:
                st.session_state['audit'] = []
        else:
            st.session_state['audit'] = []

        # ensure background is raw X_train (DataFrame) for SHAP when using tree models
        try:
            if isinstance(X_train, pd.DataFrame):
                st.session_state['shap_background'] = X_train.copy()
            else:
                st.session_state['shap_background'] = pd.DataFrame(X_train, columns=feature_cols)
        except Exception:
            st.session_state['shap_background'] = X_train

        # compute initial metrics (model may be uncalibrated)
        compute_and_store_metrics(model, X_test, y_test, scaler=scaler, calibrated=False)

st.session_state.setdefault('timezone', 'Asia/Kolkata')
ensuresession()

# optional local CSS
css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

logo_path = Path("assets/logo.svg")
header_html = (
    '<div style="display:flex;align-items:center;gap:14px">'
    f"{f'<img src={logo_path.as_posix()} style=height:56px>' if logo_path.exists() else ''}"
    '<div><h2 style="margin:0">XAI for Zero Trust Policy Justifier</h2>'
    '<div style="color:#9ca3af">Auditable Zero Trust Using Explainable AI to Justify Access Decisions</div></div></div>'
)
st.markdown(header_html, unsafe_allow_html=True)
st.write("---")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    size = st.slider("Synthetic dataset size", 1000, 20000, 5000, step=500)
    retrain = st.button("Generate & Train")
    st.markdown("---")
    st.subheader("Model & Probabilities")
    calib_method = st.selectbox("Calibrate probabilities", ["none", "sigmoid", "isotonic"], index=0)
    if st.button("Apply calibration"):
        # Fit calibrated classifier on training data and store into session as 'calibrated_model'
        base = st.session_state['model']
        Xtr = st.session_state['splits'][0]
        ytr = st.session_state['splits'][2]
        try:
            if (not is_tree_model(base)) and (st.session_state.get('scaler') is not None):
                # wrap base in pipeline so calibration sees proper scaling
                pipeline = make_pipeline(st.session_state['scaler'], base)
                calib = CalibratedClassifierCV(pipeline, method=calib_method, cv=3)
                calib.fit(Xtr, ytr)
            else:
                # tree models: calibrate directly on raw X
                calib = CalibratedClassifierCV(base, method=calib_method, cv=3)
                calib.fit(Xtr, ytr)
            st.session_state['calibrated_model'] = calib
            # recompute metrics using calibrated model (it accepts raw X if we calibrated pipeline)
            compute_and_store_metrics(calib, st.session_state['splits'][1], st.session_state['splits'][3], scaler=st.session_state.get('scaler'), calibrated=True)
            st.success(f"Calibrated with {calib_method}")
        except Exception as e:
            st.error(f"Calibration failed: {e}")

    st.markdown("---")
    st.subheader("Persistence")
    if st.button("Save current model to disk"):
        try:
            timestamp = int(time.time())
            path = f"models/model{timestamp}.pkl"
            Path("models").mkdir(exist_ok=True)
            save(st.session_state.get('calibrated_model', st.session_state['model']), st.session_state['scaler'], path)
            st.success(f"Saved model to {path}")
        except Exception as e:
            st.error(f"Save failed: {e}")
    uploaded_model = st.file_uploader("Load model (.pkl)", type="pkl")
    if uploaded_model is not None:
        try:
            obj = pickle.load(uploaded_model)
            model = obj['model']
            scaler = obj.get('scaler', None)
            st.session_state.model = model
            st.session_state.scaler = scaler
            # ensure shap_background is present
            st.session_state['shap_background'] = st.session_state.get('shap_background', None)
            # recompute metrics if splits exist
            if 'splits' in st.session_state:
                _, X_test, _, y_test = st.session_state['splits']
                compute_and_store_metrics(model, X_test, y_test, scaler=scaler, calibrated=False)
            st.success("Model loaded into session")
        except Exception as e:
            st.error(f"Load failed: {e}")

    st.markdown("---")
    st.subheader("Audit")
    if st.button("Clear Audit Log"):
        st.session_state['audit'] = []
        if AUDIT_PATH.exists():
            try:
                AUDIT_PATH.unlink()
            except Exception:
                pass
        st.success("Audit cleared")

st.subheader("Display timezone")
tz_choice = st.selectbox("Timezone", ["UTC", "Asia/Kolkata", "Asia/Dubai", "Europe/Berlin", "US/Eastern"], index=1)
st.session_state['timezone'] = tz_choice

# Retrain logic
if 'retrain' in locals() and retrain:
    df = generate_synthetic_access_data(size)
    model, acc, X_train, X_test, y_train, y_test, scaler = train(df)
    feature_cols = [c for c in df.columns if c != 'label']
    st.session_state.df = df
    st.session_state.model = model
    st.session_state.acc = acc
    st.session_state.splits = (X_train, X_test, y_train, y_test)
    st.session_state.scaler = scaler
    st.session_state.feature_cols = feature_cols
    try:
        if isinstance(X_train, pd.DataFrame):
            st.session_state['shap_background'] = X_train.copy()
        else:
            st.session_state['shap_background'] = pd.DataFrame(X_train, columns=feature_cols)
    except Exception:
        st.session_state['shap_background'] = X_train
    metrics = compute_and_store_metrics(model, X_test, y_test, scaler=scaler, calibrated=False)
    st.success(f"Model test accuracy {metrics.get('accuracy', float(acc) if acc else 'N/A'):.3f}" if metrics.get('accuracy') is not None else f"Model test accuracy {acc:.3f}")

# load state variables for convenience
df = st.session_state.df
model = st.session_state.get('calibrated_model', st.session_state.model)  # prefer calibrated model if present
# also keep raw_model for explanations if needed
raw_model = st.session_state['model']
scaler = st.session_state.get('scaler', None)
X_train, X_test, y_train, y_test = st.session_state.splits
feature_cols = st.session_state.get('feature_cols', [c for c in df.columns if c != 'label'])

# Build explainers (use raw background for tree models; if non-tree and scaler is used, build explainer on scaled background)
try:
    bg = st.session_state.get('shap_background', None)
    if isinstance(bg, np.ndarray):
        bg = pd.DataFrame(bg, columns=feature_cols)
    # If underlying raw_model is tree-based, build explainer on raw background
    if is_tree_model(raw_model):
        shap_bg = bg
        shap_exp = build_shap_explainer(raw_model, background_data=shap_bg)
    else:
        # non-tree: if scaler exists, build explainer on scaled background to match model pipeline
        if scaler is not None:
            try:
                bg_scaled = pd.DataFrame(scaler.transform(bg), columns=feature_cols)
                shap_exp = build_shap_explainer(raw_model, background_data=bg_scaled)
            except Exception:
                shap_exp = build_shap_explainer(raw_model, background_data=bg)
        else:
            shap_exp = build_shap_explainer(raw_model, background_data=bg)
except Exception:
    shap_exp = build_shap_explainer(raw_model)

try:
    # LIME expects human-readable feature columns; use raw training X
    try:
        lime_exp = build_lime_explainer(pd.DataFrame(X_train, columns=feature_cols) if not isinstance(X_train, pd.DataFrame) else X_train)
    except Exception:
        lime_exp = build_lime_explainer(pd.DataFrame(X_train))
except Exception:
    lime_exp = None

# Metrics for display
metrics = st.session_state.get('metrics', {})
acc = metrics.get('accuracy')
prec = metrics.get('precision')
rec = metrics.get('recall')
f1 = metrics.get('f1')
roc_auc = metrics.get('roc_auc')
pr_auc = metrics.get('pr_auc')

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1,1,1,1,1,1,1,1])
c1.metric("Accuracy", f"{acc:.3f}" if acc is not None else "N/A")
c2.metric("Precision", f"{prec:.3f}" if prec is not None else "N/A")
c3.metric("Recall", f"{rec:.3f}" if rec is not None else "N/A")
c4.metric("F1", f"{f1:.3f}" if f1 is not None else "N/A")
c5.metric("ROC AUC", f"{roc_auc:.3f}" if roc_auc is not None else "N/A")
c6.metric("Dataset Size", len(df))
c7.metric("Feature Count", df.shape[1] - 1)
c8.metric("Audit Entries", len(st.session_state.get('audit', [])))

# ROC & PR curves (only if we have splits and probabilities)
if metrics.get('roc_auc') is not None and 'splits' in st.session_state:
    try:
        # use raw_model for plotting probabilities if calibration exists then `model` might be calibrated (pipeline) which accepts raw X
        model_for_probs = st.session_state.get('calibrated_model', raw_model)
        X_test_plot = st.session_state['splits'][1]
        y_test_plot = st.session_state['splits'][3]

        if is_tree_model(raw_model):
            Xarr = np.array(X_test_plot)
        else:
            # if calibrated_model exists and is a pipeline it expects raw X; else use scaler
            if st.session_state.get('calibrated_model') is not None:
                Xarr = X_test_plot
            elif scaler is not None:
                Xarr = scaler.transform(X_test_plot)
            else:
                Xarr = np.array(X_test_plot)

        if hasattr(model_for_probs, "predict_proba"):
            probs = model_for_probs.predict_proba(Xarr)[:, 1]
        else:
            if hasattr(model_for_probs, "decision_function"):
                scores = model_for_probs.decision_function(Xarr)
                probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            else:
                probs = np.zeros(len(Xarr))

        fpr, tpr, _ = roc_curve(y_test_plot, probs)
        pr_prec, pr_rec, _ = precision_recall_curve(y_test_plot, probs)

        with st.expander("ROC & PR curves"):
            fig, ax = plt.subplots(1,2, figsize=(10,4))
            ax[0].plot(fpr, tpr, label=f"AUC={metrics.get('roc_auc'):.3f}")
            ax[0].plot([0,1],[0,1],'--', color='grey')
            ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC Curve"); ax[0].legend()

            ax[1].plot(pr_rec, pr_prec, label=f"PR AUC={metrics.get('pr_auc'):.3f}" if metrics.get('pr_auc') else "")
            ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall")
            ax[1].legend()

            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.write("Could not plot ROC/PR curves:", e)

st.write("---")
st.subheader("Dataset Preview")
st.dataframe(df.sample(min(8, len(df))).reset_index(drop=True))

st.subheader("Simulate Access Request")
colA, colB = st.columns([2, 1])
with colA:
    user_role = st.selectbox("User Role", list(ROLEMAP.keys()), format_func=lambda x: ROLEMAP[x])
    device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.7)
    geo_risk = st.slider("Geo Risk", 0.0, 1.0, 0.2)
    hour_of_day = st.slider("Hour", 0, 23, 14)
    auth_strength = st.selectbox("Auth Strength", list(AUTHMAP.keys()), format_func=lambda x: AUTHMAP[x])
    resource_sensitivity = st.selectbox("Sensitivity", list(SENSMAP.keys()), format_func=lambda x: SENSMAP[x])
    session_duration_minutes = st.number_input("Session Duration (min)", 1, 240, 15)
    access_history_risk = st.slider("History Risk", 0.0, 1.0, 0.1)
    instance = pd.DataFrame([{
        'user_role': user_role,
        'device_trust_score': float(device_trust_score),
        'geo_risk': float(geo_risk),
        'hour_of_day': int(hour_of_day),
        'auth_strength': int(auth_strength),
        'resource_sensitivity': int(resource_sensitivity),
        'session_duration_minutes': float(session_duration_minutes),
        'access_history_risk': float(access_history_risk)
    }])

if st.button("Evaluate Policy Decision"):
    # Decide which model to call: calibrated_model if exists, else raw_model
    model_for_decision = st.session_state.get('calibrated_model', raw_model)
    try:
        if is_tree_model(raw_model):
            # tree models -> use raw instance for prediction & SHAP
            X_for_pred = instance
            X_for_shap = instance
        else:
            # non-tree: if calibrated_model is present and is a pipeline it expects raw X
            if st.session_state.get('calibrated_model') is not None:
                X_for_pred = instance
            else:
                if scaler is None:
                    st.error("Scaler missing for non-tree model.")
                    X_for_pred = None
                else:
                    X_for_pred = pd.DataFrame(scaler.transform(instance), columns=instance.columns)
            X_for_shap = instance  # show human readable values to users

    except Exception as e:
        st.error("Failed to prepare instance for prediction: " + str(e))
        X_for_pred = None
        X_for_shap = instance

    if X_for_pred is not None:
        try:
            pred = int(model_for_decision.predict(X_for_pred)[0])
            prob = float(model_for_decision.predict_proba(X_for_pred)[0][1]) if hasattr(model_for_decision, "predict_proba") else 0.0
            decision = "ACCESS GRANTED" if pred == 1 else "ACCESS DENIED"
            st.markdown(f"""<div class='card'><b>Decision:</b> {decision} 
            <span style='color:#9ca3af'>Allow Probability:{prob:.2f}</span></div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error("Prediction failed: " + str(e))
            pred = None
            prob = None

        # Record audit (persist to CSV)
        if pred is not None:
            audit_row = {
                'timestamp': int(time.time()),
                **instance.iloc[0].to_dict(),
                'pred': int(pred),
                'prob_allow': float(prob)
            }
            append_audit_row(audit_row)

        # Rebuild SHAP explainer for raw_model (explain base model decisions; calibration is a probability-layer)
        try:
            bg = st.session_state.get('shap_background', None)
            if isinstance(bg, np.ndarray):
                bg = pd.DataFrame(bg, columns=feature_cols)
            if is_tree_model(raw_model):
                shap_exp = build_shap_explainer(raw_model, background_data=bg)
                shap_input = X_for_shap  # raw human-readable
            else:
                # if scaler exists, build explainer on scaled background
                if scaler is not None:
                    bg_scaled = pd.DataFrame(scaler.transform(bg), columns=feature_cols)
                    shap_exp = build_shap_explainer(raw_model, background_data=bg_scaled)
                    shap_input = pd.DataFrame(scaler.transform(X_for_shap), columns=feature_cols)
                else:
                    shap_exp = build_shap_explainer(raw_model, background_data=bg)
                    shap_input = X_for_shap

        except Exception:
            shap_exp = build_shap_explainer(raw_model)
            shap_input = X_for_shap if is_tree_model(raw_model) else (pd.DataFrame(scaler.transform(X_for_shap), columns=feature_cols) if scaler is not None else X_for_shap)

        # Compute SHAP values for the representation the explainer expects
        try:
            shap_1d, base_val, raw_shap = get_shap_for_instance(shap_exp, np.array(shap_input))
        except Exception as e:
            shap_1d, base_val, raw_shap = None, None, None
            st.write("SHAP explanation failed:", str(e))

        if shap_1d is not None:
            st.write("**SHAP Explanation:**")
            try:
                # translate_shap expects human-readable feature values, so pass X_for_shap.iloc[0]
                text, suggestions = translate_shap(X_for_shap.iloc[0], shap_1d)
                st.write(text)
                for s in suggestions:
                    st.write("- " + s)
            except Exception as e:
                st.write("SHAP translate error:", e)

            # Waterfall & summary plots (static)
            try:
                st.write("Waterfall Plot")
                shap_vals = np.array(shap_1d).reshape(-1)
                if base_val is None:
                    base_val = getattr(raw_shap, "expected_value", None) or (getattr(raw_shap, "base_values", None) if hasattr(raw_shap, "base_values") else None)
                fig, ax = plt.subplots(figsize=(8, 4))
                try:
                    shap.plots._waterfall.waterfall_legacy(base_val, shap_vals, X_for_shap.iloc[0], show=False)
                except Exception:
                    contribs = pd.Series(shap_vals, index=X_for_shap.columns)
                    contribs.sort_values().plot(kind='barh', ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.write("Waterfall plot unavailable:", e)

            try:
                st.write("SHAP summary bar")
                fig, ax = plt.subplots(figsize=(7,4))
                shap_vals_for_plot = np.atleast_2d(shap_vals)
                df_inst = pd.DataFrame([X_for_shap.iloc[0]])
                shap.summary_plot(shap_vals_for_plot, df_inst, plot_type="bar", show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.write("SHAP summary bar unavailable:", e)
        else:
            st.write("No SHAP values available.")
    else:
        st.error("Failed to compute decision due to scaling/model issues.")

st.markdown("---")
st.subheader("Batch Audit")
uploadeddf = None
csvupload = st.file_uploader("Upload access log CSV for batch evaluation", type="csv")
if csvupload is not None:
    try:
        uploadeddf = pd.read_csv(csvupload)
        st.dataframe(uploadeddf.head())
        if st.button("Run batch evaluation on uploaded CSV"):
            requiredcols = ['user_role','device_trust_score','geo_risk','hour_of_day','auth_strength','resource_sensitivity','session_duration_minutes','access_history_risk']
            if all(c in uploadeddf.columns for c in requiredcols):
                batchX = uploadeddf[requiredcols]
                # choose representation for prediction
                if is_tree_model(raw_model):
                    batchX_for_pred = batchX
                else:
                    # if calibrated_model exists and is a pipeline: it expects raw X, else scale
                    if st.session_state.get('calibrated_model') is not None:
                        batchX_for_pred = batchX
                    else:
                        try:
                            batchX_for_pred = pd.DataFrame(scaler.transform(batchX), columns=batchX.columns)
                        except Exception:
                            batchX_for_pred = batchX  # fallback

                preds = st.session_state.get('calibrated_model', raw_model).predict(batchX_for_pred)
                probs = st.session_state.get('calibrated_model', raw_model).predict_proba(batchX_for_pred)[:,1] if hasattr(st.session_state.get('calibrated_model', raw_model), "predict_proba") else np.zeros(len(batchX_for_pred))
                uploadeddf['pred'] = preds
                uploadeddf['Allow Probability'] = probs
                st.session_state['lastbatch'] = uploadeddf

                # append to audit via append_audit_row (persist)
                for i, row in uploadeddf.iterrows():
                    audit_row = {
                        'timestamp': int(time.time()),
                        **{c: row.get(c) for c in requiredcols},
                        'pred': int(row['pred']),
                        'prob_allow': float(row['Allow Probability'])
                    }
                    append_audit_row(audit_row)

                st.success("Batch evaluation complete")
                st.dataframe(uploadeddf.head(10))

                # If ground truth 'label' present, compute batch metrics and show confusion matrix
                if 'label' in uploadeddf.columns:
                    y_true = uploadeddf['label']
                    y_pred = uploadeddf['pred']
                    y_prob = uploadeddf['Allow Probability']
                    batch_metrics = {
                        'accuracy': accuracy_score(y_true, y_pred),
                        'precision': precision_score(y_true, y_pred, zero_division=0),
                        'recall': recall_score(y_true, y_pred, zero_division=0),
                        'f1': f1_score(y_true, y_pred, zero_division=0),
                    }
                    st.write("Batch evaluation metrics:")
                    st.json(batch_metrics)

                    # confusion matrix
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(4,4))
                    disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
                    disp.plot(ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.write("Classification report:")
                    st.text(classification_report(y_true, y_pred, digits=4))
            else:
                st.error("Uploaded CSV does not match expected schema. Please map columns.")
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")

if st.button("Export Audit Log CSV"):
    audit = st.session_state.get('audit', [])
    if audit:
        dfaudit = pd.DataFrame(audit)
        if 'timestamp' in dfaudit.columns:
            tz = st.session_state.get('timezone', 'UTC')
            dfaudit['timestamphuman'] = dfaudit['timestamp'].apply(lambda x: formattimestamp(x, tz))
        towrite = io.BytesIO()
        dfaudit.to_csv(towrite, index=False)
        towrite.seek(0)
        st.download_button("Download audit CSV", towrite, file_name="auditlog.csv")
    else:
        st.info("No audit entries to export yet.")

st.markdown("---")
st.subheader("Model Validation & Calibration (k-fold + reliability)")

with st.expander("Run k-fold CV (5) and show mean ± std"):
    if st.button("Run CV now"):
        try:
            raw_X = st.session_state['df'].drop(columns=['label'])
            raw_y = st.session_state['df']['label']
            cv_model = st.session_state['model']
            cv_scaler = st.session_state.get('scaler', None)
            cv_summary = compute_cv_metrics(cv_model, raw_X, raw_y, scaler=cv_scaler, n_splits=5)
            st.write("Cross-validation (5-fold) results (mean ± std):")
            for m, (mean, std) in cv_summary.items():
                st.write(f"{m}: {mean:.4f} ± {std:.4f}")
        except Exception as e:
            st.error("CV failed: " + str(e))

with st.expander("Calibration / Reliability diagram"):
    if st.button("Show reliability diagram on test set"):
        try:
            model_for_cal = st.session_state.get('calibrated_model', raw_model)
            Xtest = st.session_state['splits'][1]
            ytest = st.session_state['splits'][3]
            # choose correct input representation
            if is_tree_model(raw_model):
                X_for_probs = Xtest
            else:
                if st.session_state.get('calibrated_model') is not None:
                    X_for_probs = Xtest
                else:
                    X_for_probs = st.session_state['scaler'].transform(Xtest) if st.session_state.get('scaler') is not None else Xtest

            y_prob = model_for_cal.predict_proba(X_for_probs)[:,1]
            prob_true, prob_pred = calibration_curve(ytest, y_prob, n_bins=10)
            fig, ax = plt.subplots(figsize=(5,5))
            ax.plot(prob_pred, prob_true, marker='o', label='Reliability')
            ax.plot([0,1],[0,1],'--', color='grey', label='Perfect calibration')
            ax.set_xlabel('Predicted probability')
            ax.set_ylabel('Observed frequency')
            ax.set_title('Reliability diagram')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            #st.write("Note: If probabilities are shown to users, consider calibrating (Platt/sigmoid or isotonic).")
        except Exception as e:
            st.error("Reliability diagram failed: " + str(e))

st.markdown("---")
st.subheader("Latest Audit Entries")
last = st.session_state.get('audit', [])[-6:]
if last:
    dflast = pd.DataFrame(last).sort_values('timestamp', ascending=False).head(6).copy()
    if 'timestamp' in dflast.columns:
        tz = st.session_state.get('timezone', 'UTC')
        dflast['time'] = dflast['timestamp'].apply(lambda x: formattimestamp(x, tz))
        cols = ['time'] + [c for c in dflast.columns if c != 'time']
        dflast = dflast[cols]
    st.dataframe(dflast)
else:
    st.write("No audit entries yet.")
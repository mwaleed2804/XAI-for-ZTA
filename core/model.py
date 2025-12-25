# model.py
import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
    auc, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train(df, random_state=42, cv_folds: int = 0):
    """
    Train an XGBoost classifier on dataframe `df` with column 'label' as the target.

    Behavior changes introduced:
    - Tree models (XGBoost) are trained on RAW feature values (no scaling). scaler returned will be None.
      This ensures SHAP explanations and human-readable feature values are in the same units.
    - For non-tree models (not used by default here), StandardScaler is applied and returned.
    - Optionally run k-fold CV by setting cv_folds > 1; CV summary (mean ± std) is attached to model.metrics['cv'].
    - Automatically sets scale_pos_weight for XGBoost based on class imbalance (neg/pos) to help with imbalance.

    Returns:
        model, test_accuracy, X_train_raw, X_test_raw, y_train, y_test, scaler
    Notes:
        - X_train_raw / X_test_raw are returned in RAW (unscaled) form to make it easy to build SHAP explainer
          and display human-readable instances. If a scaler is returned (non-tree models), training used the
          scaled versions internally but the raw splits are still returned (scaler can be used externally).
    """
    # Validate
    if 'label' not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column")

    # Separate features/target
    X_raw = df.drop(columns=['label'])
    y = df['label'].astype(int)

    # train-test split on raw data (keep DataFrame type for convenience downstream)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Determine class imbalance for XGBoost scale_pos_weight
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = 1.0
    try:
        if pos > 0:
            scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0
    except Exception:
        scale_pos_weight = 1.0

    # Build XGBoost model (tree-based) — train on RAW features (no scaling)
    # Note: avoid deprecated use_label_encoder parameter; set verbosity low
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=random_state,
        verbosity=0,
        scale_pos_weight=scale_pos_weight
    )

    # For tree models (XGBoost) --> do not scale; keep scaler = None
    scaler = None

    # Fit model on raw data (pandas DataFrame accepted by XGBoost)
    model.fit(X_train_raw, y_train, eval_set=[(X_test_raw, y_test)], verbose=False)

    # Prepare predictions on the test set (use raw X_test for trees)
    try:
        y_pred = model.predict(X_test_raw)
    except Exception:
        # fallback to numpy array
        y_pred = model.predict(np.array(X_test_raw))

    # Compute probability estimates when available
    try:
        y_prob = model.predict_proba(X_test_raw)[:, 1]
    except Exception:
        y_prob = None

    # Compute metrics
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
    metrics['precision'] = float(precision_score(y_test, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_test, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        if y_prob is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
        else:
            metrics['roc_auc'] = None
    except Exception:
        metrics['roc_auc'] = None

    try:
        if y_prob is not None:
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
            metrics['pr_auc'] = float(auc(recall_vals, precision_vals))
        else:
            metrics['pr_auc'] = None
    except Exception:
        metrics['pr_auc'] = None

    try:
        if y_prob is not None:
            metrics['brier_score'] = float(brier_score_loss(y_test, y_prob))
        else:
            metrics['brier_score'] = None
    except Exception:
        metrics['brier_score'] = None

    # Confusion matrix (JSON-serializable)
    try:
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    except Exception:
        metrics['confusion_matrix'] = None

    # Attach metrics to model
    try:
        model.metrics = metrics
    except Exception:
        pass

    # Optional: k-fold CV (attach mean ± std to model.metrics['cv'])
    if isinstance(cv_folds, int) and cv_folds > 1:
        try:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            # For XGBoost (tree) we train on raw features — cross_validate accepts DataFrame
            estimator_for_cv = model
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_res = cross_validate(estimator_for_cv, X_raw, y, cv=cv, scoring=scoring, return_train_score=False)
            cv_summary = {k.replace('test_',''): (float(np.mean(v)), float(np.std(v))) for k,v in cv_res.items() if k.startswith('test_')}
            metrics['cv'] = cv_summary
            # attach updated metrics
            try:
                model.metrics = metrics
            except Exception:
                pass
        except Exception:
            # do not fail training because CV failed; just omit cv results
            pass

    # Return signature compatible with app.py
    # Keep X_train_raw and X_test_raw as DataFrames (raw units) and return scaler (None for tree models)
    return model, metrics.get('accuracy', None), X_train_raw, X_test_raw, y_train, y_test, scaler


def save(model, scaler, path="model.pkl"):
    """
    Save model, scaler and optional metrics to disk using pickle.
    """
    payload = {
        'model': model,
        'scaler': scaler,
    }
    metrics = getattr(model, 'metrics', None)
    if metrics is not None:
        payload['metrics'] = metrics

    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load(path="model.pkl"):
    """
    Load model and scaler from disk. Restores model.metrics if present in the file.
    Returns: model, scaler
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    model = data.get('model')
    scaler = data.get('scaler')
    metrics = data.get('metrics', None)

    if model is not None and metrics is not None:
        try:
            model.metrics = metrics
        except Exception:
            pass

    return model, scaler

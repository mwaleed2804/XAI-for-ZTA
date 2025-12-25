# core/explain.py
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

def build_shap_explainer(model, background_data=None):
    """
    Build a SHAP explainer robustly:
      - Try TreeExplainer(model, data=background_data)
      - Fallback to shap.Explainer(model, masker=...) if needed
    """
    try:
        if background_data is not None:
            return shap.TreeExplainer(model, data=background_data)
        else:
            return shap.TreeExplainer(model)
    except Exception:
        # fallback to general Explainer, with masker if background provided
        try:
            masker = None
            if background_data is not None:
                try:
                    masker = shap.maskers.Independent(background_data)
                except Exception:
                    masker = None
            return shap.Explainer(model, masker=masker)
        except Exception:
            # last-resort
            return shap.TreeExplainer(model)

def build_lime_explainer(X_train):
    return LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["deny","allow"],
        mode="classification"
    )

def get_shap_for_instance(explainer, instance_array):
    """
    Call a SHAP explainer safely and return (shap_1d, base_value, raw_obj).
    Handles both new API (Explainer(X) -> Explanation) and legacy shap_values API.
    """
    # new API
    try:
        expl = explainer(instance_array)
        vals = getattr(expl, "values", None)
        base = getattr(expl, "base_values", None) or getattr(expl, "expected_value", None)

        if vals is None:
            raise Exception("Explanation object missing .values")

        vals = np.array(vals)
        # vals shape cases:
        #   (n_samples, n_features)
        #   (n_samples, n_classes, n_features)
        if vals.ndim == 2:
            shap_for_display = vals[0]
        elif vals.ndim == 3:
            class_idx = 1 if vals.shape[1] > 1 else 0
            shap_for_display = vals[0, class_idx, :]
        else:
            shap_for_display = vals.reshape(vals.shape[0], -1)[0]

        base_val = None
        if base is not None:
            try:
                b = np.array(base)
                if hasattr(b, "__len__") and len(b) > 1:
                    base_val = float(b[1])
                elif hasattr(b, "__len__") and len(b) == 1:
                    base_val = float(b[0])
                else:
                    base_val = float(b)
            except Exception:
                base_val = None

        return np.asarray(shap_for_display, dtype=float), base_val, expl

    except Exception:
        # legacy shap_values API
        try:
            legacy = explainer.shap_values(instance_array)
        except Exception as e:
            raise RuntimeError(f"SHAP explainer call failed (new + legacy): {e}")

        if isinstance(legacy, (list, tuple)):
            arr = np.array(legacy[1]) if len(legacy) > 1 else np.array(legacy[0])
        else:
            arr = np.array(legacy)

        if arr.ndim == 2:
            shap_for_display = arr[0]
        else:
            shap_for_display = arr.ravel()
        return np.asarray(shap_for_display, dtype=float), None, legacy

def translate_shap(instance, shap_row, top_k=4):
    """
    Convert shap_row (1D array) and instance (pandas.Series) to a human text explanation
    and a small list of remediation suggestions.
    Returns: (text_string, [suggestion_strings])
    """
    vals = np.asarray(shap_row).ravel()
    feature_names = list(instance.index)
    feats = list(zip(feature_names, vals))
    feats_sorted = sorted(feats, key=lambda x: abs(x[1]), reverse=True)[:top_k]

    just_lines = []
    sugg = []

    for feat, val in feats_sorted:
        v = float(val)
        direction = "increased" if v > 0 else "decreased"
        just_lines.append(f"{feat.replace('_',' ').title()} {direction} the allow likelihood by {abs(v):.3f}.")

        fname = feat.lower()
        if "device_trust" in fname and v < 0:
            sugg.append("Improve device posture (apply updates, enable EDR, verify device).")
        if "geo_risk" in fname and v < 0:
            sugg.append("Move to a safer network or use corporate VPN.")
        if "auth_strength" in fname and v < 0:
            sugg.append("Use stronger authentication (2FA or biometric).")
        if "resource_sensitivity" in fname and v < 0:
            sugg.append("Request lower-sensitivity access or escalate authorization.")
        if "access_history" in fname and v < 0:
            sugg.append("Re-authenticate or confirm recent access history.")

    if not sugg:
        sugg.append("Provide stronger identity proofing or retry with higher-trust device.")

    return " ".join(just_lines), sugg

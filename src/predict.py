"""
predict.py
─────────────────────────────────────────────────────────────
Prediction engine for the Student Depression Detection system.

Exposes:
    load_bundle()           → loads model + metadata from .pkl
    build_input_vector()    → converts UI inputs to numeric vector
    predict()               → returns prediction + probabilities
    explain_prediction()    → per-sample feature contribution explanation
"""

import os, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'depression_model.pkl')

# ── BUNDLE CACHE (load once per process) ──────────────────────
_BUNDLE = None


def load_bundle(model_path: str = MODEL_PATH) -> dict:
    """Load and cache the model bundle."""
    global _BUNDLE
    if _BUNDLE is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                "Run  python src/train.py  first."
            )
        with open(model_path, 'rb') as f:
            _BUNDLE = pickle.load(f)
    return _BUNDLE


def get_feature_cols() -> list:
    return load_bundle()['feature_cols']


def get_model():
    return load_bundle()['model']


def get_meta() -> dict:
    b = load_bundle()
    return {
        'model_name':  b['model_name'],
        'test_acc':    b['test_accuracy'],
        'cv_acc':      b['cv_accuracy'],
        'importance':  b['importance_df'],
        'report_dict': b.get('report_dict', {}),
    }


# ── INPUT BUILDER ─────────────────────────────────────────────

def build_input_vector(user_inputs: dict) -> np.ndarray:
    """
    Convert a dict of {field_keyword: value} into a numeric
    feature vector aligned to feature_cols.

    user_inputs keys must include one of these keywords
    (matched by substring against feature column names):
        age, gender, cgpa, academic,
        study_sat, work_study, financial,
        sleep, diet, suicid, family

    sleep  → one of SLEEP_MAP keys (str) or float hours
    diet   → one of DIET_MAP keys  (str) or int 0/1/2
    gender → 'Male' / 'Female'
    binary fields → 'Yes' / 'No'
    """
    bundle      = load_bundle()
    feature_cols = bundle['feature_cols']
    sleep_map    = bundle['sleep_map']
    diet_map     = bundle['diet_map']

    # Start with zeros (safe default)
    vec = {col: 0.0 for col in feature_cols}

    def find_col(keyword: str):
        for col in feature_cols:
            if keyword.lower() in col.lower():
                return col
        return None

    YES_VALS = {'yes', 'male', 'm', '1', 'true'}

    for keyword, raw_val in user_inputs.items():
        col = find_col(keyword)
        if col is None:
            continue

        if keyword == 'sleep':
            if isinstance(raw_val, str):
                vec[col] = sleep_map.get(raw_val.lower().strip(), 6.5)
            else:
                vec[col] = float(raw_val)

        elif keyword == 'diet':
            if isinstance(raw_val, str):
                vec[col] = diet_map.get(raw_val.lower().strip(), 1)
            else:
                vec[col] = float(raw_val)

        elif isinstance(raw_val, str):
            vec[col] = 1.0 if raw_val.lower().strip() in YES_VALS else 0.0

        else:
            vec[col] = float(raw_val)

    return np.array([vec[col] for col in feature_cols]).reshape(1, -1)


# ── PREDICTION ────────────────────────────────────────────────

def predict(user_inputs: dict) -> dict:
    """
    Run inference.

    Returns:
        label      : 'Depressed' | 'Not Depressed'
        prediction : 1 | 0
        confidence : float 0–100  (probability of predicted class)
        prob_dep   : float 0–100  (probability of Depressed)
        prob_not   : float 0–100
    """
    model   = get_model()
    X       = build_input_vector(user_inputs)
    pred    = int(model.predict(X)[0])
    proba   = model.predict_proba(X)[0]

    return {
        'prediction': pred,
        'label':      'Depressed' if pred == 1 else 'Not Depressed',
        'confidence': round(proba[pred]  * 100, 1),
        'prob_dep':   round(proba[1]     * 100, 1),
        'prob_not':   round(proba[0]     * 100, 1),
    }


# ── FEATURE CONTRIBUTION EXPLANATION ─────────────────────────

def explain_prediction(user_inputs: dict,
                        top_n: int = 8) -> pd.DataFrame | None:
    """
    Compute per-sample feature contribution values for the given inputs.

    Returns a DataFrame with columns:
        feature, value, shap_value, direction
    or None if shap is not installed.
    """
    try:
        import shap
    except ImportError:
        return None

    bundle      = load_bundle()
    model       = bundle['model']
    feature_cols = bundle['feature_cols']

    X = build_input_vector(user_inputs)

    try:
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]   # class 1 (Depressed)
            shap_flat = shap_vals[0]
        else:
            background = pd.DataFrame(
                np.zeros((1, len(feature_cols))),
                columns=feature_cols,
            )
            explainer = shap.Explainer(model.predict_proba, background)
            shap_vals = explainer(pd.DataFrame(X, columns=feature_cols))
            shap_flat = shap_vals.values[0, :, 1]

        contrib_df = pd.DataFrame({
            'feature':    feature_cols,
            'value':      X[0],
            'shap_value': shap_flat,
        })
        contrib_df['abs_shap']  = contrib_df['shap_value'].abs()
        contrib_df['direction'] = contrib_df['shap_value'].apply(
            lambda v: '↑ Risk' if v > 0 else '↓ Risk'
        )
        contrib_df = (contrib_df
                      .sort_values('abs_shap', ascending=False)
                      .head(top_n)
                      .reset_index(drop=True))
        return contrib_df

    except Exception:
        return None

"""
preprocess.py
─────────────────────────────────────────────────────────────
Handles all data loading, cleaning, feature engineering, and
encoding for the Student Depression Detection system.

Domain rationale for feature selection is documented inline.
"""

import pandas as pd
import numpy as np
import pandas.api.types as pat
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


# ── CONSTANTS ──────────────────────────────────────────────────────────────

# Domain-driven sleep encoding (ordinal, meaningful scale)
SLEEP_MAP = {
    'less than 5 hours': 4.0,
    '5-6 hours':         5.5,
    '6-7 hours':         6.5,
    '7-8 hours':         7.5,
    'more than 8 hours': 9.0,
    'others':            6.5,   # median fallback
}

# Dietary habits: ordinal encoding (0=worst, 2=best)
DIET_MAP = {
    'unhealthy': 0,
    'moderate':  1,
    'healthy':   2,
    'others':    1,
}

YES_VALS = {'yes', 'male',   'm', '1', 'true'}
NO_VALS  = {'no',  'female', 'f', '0', 'false'}

# ── FEATURE SELECTION RATIONALE ────────────────────────────────────────────
#
# INCLUDED FEATURES (domain + statistical justification):
#   age                  → Developmental risk window; depression peaks 18-25
#   gender               → Well-documented gender differences in depression rates
#   cgpa                 → Academic performance proxy; low CGPA → stress cycle
#   academic_pressure    → Direct stressor; central to student mental health
#   study_satisfaction   → Protective factor; low satisfaction → disengagement
#   work_study_hours     → Overwork is a validated burnout predictor
#   financial_stress     → One of the strongest non-academic stressors in students
#   sleep_duration       → Sleep deprivation directly linked to depression onset
#   dietary_habits       → Gut-brain axis; nutrition affects mood regulation
#   suicidal_thoughts    → Clinical indicator; strongest direct risk marker
#   family_history       → Genetic and environmental heritability ~40%
#
# REMOVED FEATURES:
#   id, city, degree     → Identifiers / too granular / no causal pathway
#   profession           → Mostly 'Student'; near-zero variance in student dataset
#
# ───────────────────────────────────────────────────────────────────────────

# Dataset-specific features intentionally excluded from training because
# they are effectively constant in the student dataset provided here.
EXCLUDED_FEATURES = {
    'work_pressure',
    'job_satisfaction',
}


def load_and_clean(data_path: str) -> pd.DataFrame:
    """Load CSV and standardise column names."""
    df = pd.read_csv(data_path)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('?', '',  regex=False)
        .str.replace('/', '_', regex=False)
    )
    return df


def find_target(df: pd.DataFrame, keyword: str = 'depression') -> str:
    """Locate the target column by exact match first, then substring."""
    exact = [c for c in df.columns if c == keyword]
    if exact:
        return exact[0]
    fuzzy = [c for c in df.columns if keyword in c]
    if fuzzy:
        return fuzzy[0]
    raise ValueError(f"No column containing '{keyword}' found in dataset.")


def encode_binary(series: pd.Series) -> pd.Series:
    """Map yes/no, male/female, true/false → 1.0 / 0.0."""
    s = series.astype(str).str.lower().str.strip()
    return s.map(lambda x: 1.0 if x in YES_VALS else (0.0 if x in NO_VALS else np.nan))


def apply_domain_encodings(feature_df: pd.DataFrame,
                           sleep_map: dict = SLEEP_MAP,
                           diet_map: dict  = DIET_MAP) -> pd.DataFrame:
    """Apply domain-specific ordinal encodings for sleep and diet."""
    df = feature_df.copy()

    sleep_col = next((c for c in df.columns if 'sleep' in c), None)
    if sleep_col:
        df[sleep_col] = (
            df[sleep_col].astype(str).str.lower().str.strip()
            .map(sleep_map)
        )

    diet_col = next((c for c in df.columns if 'diet' in c), None)
    if diet_col:
        df[diet_col] = (
            df[diet_col].astype(str).str.lower().str.strip()
            .map(diet_map)
        )

    return df


def apply_binary_encodings(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect remaining binary string columns and encode them."""
    df = feature_df.copy()
    for col in df.columns:
        if pat.is_string_dtype(df[col]) or df[col].dtype == object:
            uniq = set(df[col].astype(str).str.lower().str.strip().dropna().unique())
            if uniq <= (YES_VALS | NO_VALS):
                df[col] = encode_binary(df[col])
    return df


def build_feature_matrix(df: pd.DataFrame,
                          target_col: str,
                          drop_keywords: list = None) -> tuple[pd.DataFrame, list]:
    """
    Build a clean numeric feature matrix.
    Returns (feature_df, feature_cols).
    """
    if drop_keywords is None:
        drop_keywords = ['id', 'city', 'degree', 'profession']

    drop_cols = [
        c for c in df.columns
        if any(k == c or c.startswith(k + '_') for k in drop_keywords)
        or c == target_col
        or c in EXCLUDED_FEATURES
    ]
    feature_df = df.drop(columns=drop_cols).copy()

    # Apply encodings in order (sleep/diet first, then generic binary)
    feature_df = apply_domain_encodings(feature_df)
    feature_df = apply_binary_encodings(feature_df)

    # Keep only numeric; fill remaining NaN with median
    feature_df = feature_df.select_dtypes(include=[np.number])
    feature_df = feature_df.fillna(feature_df.median())

    feature_cols = feature_df.columns.tolist()
    return feature_df, feature_cols


def compute_mutual_information(X: np.ndarray, y: np.ndarray,
                                feature_cols: list) -> pd.DataFrame:
    """Compute mutual information scores for feature selection analysis."""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False).reset_index(drop=True)
    return mi_df


def compute_correlation_analysis(X: np.ndarray, y: np.ndarray,
                                  feature_cols: list) -> pd.DataFrame:
    """Pearson correlation of each feature with the target."""
    corr = [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
    corr_df = pd.DataFrame({
        'feature': feature_cols,
        'abs_correlation': corr
    }).sort_values('abs_correlation', ascending=False).reset_index(drop=True)
    return corr_df


def preprocess_pipeline(data_path: str) -> dict:
    """
    Full preprocessing pipeline.
    Returns a dict with all artifacts needed for training.
    """
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_and_clean(data_path)
    print(f"  Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")

    target_col = find_target(df)
    print(f"  Target  : '{target_col}'")

    # Clean target
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)
    print(f"  Depressed    : {df[target_col].sum():,}")
    print(f"  Not Depressed: {(df[target_col] == 0).sum():,}")

    feature_df, feature_cols = build_feature_matrix(df, target_col)
    X = feature_df.values
    y = df[target_col].values

    print(f"\n  Features selected ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"    • {c}")

    # Statistical analysis
    mi_df   = compute_mutual_information(X, y, feature_cols)
    corr_df = compute_correlation_analysis(X, y, feature_cols)

    print("\n  Top-5 features by Mutual Information:")
    for _, row in mi_df.head(5).iterrows():
        print(f"    {row['feature']:38s}: MI = {row['mi_score']:.4f}")

    print("=" * 60)

    return {
        'df':           df,
        'X':            X,
        'y':            y,
        'feature_cols': feature_cols,
        'target_col':   target_col,
        'mi_df':        mi_df,
        'corr_df':      corr_df,
        'sleep_map':    SLEEP_MAP,
        'diet_map':     DIET_MAP,
    }

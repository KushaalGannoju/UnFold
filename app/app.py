"""
app.py  ─  Streamlit UI for Student Depression Detection
─────────────────────────────────────────────────────────────
Run:
    streamlit run app/app.py

Requires the model to be trained first:
    python src/train.py
"""

import os, sys, warnings
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'student_depression.csv')

from predict import load_bundle, predict, explain_prediction
from utils   import format_feature_name, risk_category

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Depression Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1d4ed8 100%);
    padding-top: 1rem;
}
section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
section[data-testid="stSidebar"] .stSlider > label,
section[data-testid="stSidebar"] .stRadio > label,
section[data-testid="stSidebar"] .stSelectbox > label,
section[data-testid="stSidebar"] .stTextInput > label { color: #c7d2fe !important; font-size: 0.82rem; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { padding: 0 !important; }
section[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(191, 219, 254, 0.35);
    border-radius: 12px;
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}
section[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="input"] input {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    color: #0f172a !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] input,
section[data-testid="stSidebar"] div[data-baseweb="select"] div {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #475569 !important;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 18px 40px rgba(37, 99, 235, 0.18);
}
.main-header h1 { font-size: 2.1rem; font-weight: 800; margin: 0; }
.main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.4rem 0 0; }

/* Metric pills */
.metric-pill {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    margin: 0.4rem 0.3rem 0 0;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Result cards */
.result-depressed {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border: 2px solid #fca5a5;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-not-depressed {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 2px solid #86efac;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-neutral {
    background: #f8fafc;
    border: 2px dashed #cbd5e1;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: #64748b;
}
.result-emoji  { font-size: 3.5rem; }
.result-label  { font-size: 1.6rem; font-weight: 800; margin: 0.5rem 0; }
.result-sub    { font-size: 0.9rem; opacity: 0.75; }
.prob-text     { font-size: 1rem; font-weight: 600; margin-top: 0.8rem; }

/* Info box */
.info-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #1e40af;
    margin-top: 1rem;
}

.sidebar-note {
    background: rgba(255, 255, 255, 0.11);
    border: 1px solid rgba(191, 219, 254, 0.35);
    border-radius: 12px;
    padding: 0.85rem 0.95rem;
    font-size: 0.82rem;
    color: #dbeafe;
    margin-bottom: 0.8rem;
}

.field-error {
    background: rgba(127, 29, 29, 0.45);
    border: 1px solid rgba(252, 165, 165, 0.45);
    border-radius: 10px;
    padding: 0.7rem 0.85rem;
    font-size: 0.8rem;
    color: #fee2e2;
    margin-top: 0.35rem;
}

.summary-card {
    background: linear-gradient(145deg, #f8fafc 0%, #eef2ff 100%);
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    color: #1e3a8a;
    font-size: 0.9rem;
    margin-top: 0.8rem;
}

.stat-note {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    color: #1d4ed8;
    font-size: 0.87rem;
    margin-top: 0.9rem;
}

/* Feature bar */
.feat-row { display: flex; align-items: center; margin-bottom: 6px; }
.feat-name { width: 200px; font-size: 0.83rem; color: #334155; }
.feat-bar-bg { flex: 1; background: #e2e8f0; border-radius: 6px; height: 10px; }
.feat-bar-fill-up  { height: 10px; border-radius: 6px; background: #ef4444; }
.feat-bar-fill-dn  { height: 10px; border-radius: 6px; background: #22c55e; }
.feat-score { width: 55px; font-size: 0.78rem; text-align: right; color: #64748b; }

/* Section headers */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem;
}

/* Disclaimer */
.disclaimer {
    background: #fefce8;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.82rem;
    color: #92400e;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model...")
def load_model_bundle():
    return load_bundle()

try:
    bundle       = load_model_bundle()
    model_name   = bundle['model_name']
    train_acc    = bundle.get('train_accuracy')
    test_acc     = bundle['test_accuracy']
    cv_acc       = bundle['cv_accuracy']
    cv_std       = bundle.get('cv_std')
    imp_df       = bundle['importance_df']
    model_cmp_df = bundle.get('model_comparison_df')
    report_dict  = bundle.get('report_dict', {})
    cm_values    = bundle.get('confusion_matrix')
    class_labels = bundle.get('class_labels', ['Not Depressed', 'Depressed'])
    MODEL_LOADED = True
except FileNotFoundError as e:
    MODEL_LOADED = False
    load_error   = str(e)

if MODEL_LOADED and model_cmp_df is not None and not isinstance(model_cmp_df, pd.DataFrame):
    model_cmp_df = pd.DataFrame(model_cmp_df)


DEFAULT_INPUT_BOUNDS = {
    'age': {'min': 15, 'max': 60, 'default': 22},
    'cgpa': {'min': 0.0, 'max': 10.0, 'default': 7.0},
}


@st.cache_data(show_spinner=False)
def load_input_bounds() -> dict:
    bounds = {key: value.copy() for key, value in DEFAULT_INPUT_BOUNDS.items()}

    if not os.path.exists(DATA_PATH):
        return bounds

    df = pd.read_csv(DATA_PATH)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('?', '', regex=False)
        .str.replace('/', '_', regex=False)
    )

    if 'age' in df.columns:
        age_values = pd.to_numeric(df['age'], errors='coerce').dropna()
        if not age_values.empty:
            bounds['age']['max'] = max(bounds['age']['min'], int(age_values.max()))
            bounds['age']['default'] = min(
                max(bounds['age']['default'], bounds['age']['min']),
                bounds['age']['max'],
            )

    return bounds


def parse_numeric_text_input(raw_value: str,
                             label: str,
                             min_value: float,
                             max_value: float,
                             cast_type):
    value = (raw_value or '').strip()
    if not value:
        return None, f"{label} is required."

    try:
        parsed = cast_type(value)
    except ValueError:
        kind = "whole number" if cast_type is int else "number"
        return None, f"{label} must be a valid {kind}."

    if parsed < min_value or parsed > max_value:
        if cast_type is int:
            return None, f"{label} must be between {int(min_value)} and {int(max_value)}."
        return None, f"{label} must be between {min_value:.1f} and {max_value:.1f}."

    if cast_type is float:
        parsed = round(float(parsed), 2)

    return parsed, None


# ══════════════════════════════════════════════════════════════
# SIDEBAR  ─  Input Controls
# ══════════════════════════════════════════════════════════════

def render_sidebar(input_bounds: dict) -> tuple[dict, bool, list[str]]:
    st.sidebar.markdown("## 📋 Student Profile")

    inputs = {}
    errors = []

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 🧾 Core Details")

    age_raw = st.sidebar.text_input(
        "Age",
        value=str(input_bounds['age']['default']),
        key='age_input',
        help=f"Enter an age between {input_bounds['age']['min']} and {input_bounds['age']['max']}.",
        placeholder="e.g. 21",
    )
    age_val, age_err = parse_numeric_text_input(
        age_raw,
        "Age",
        input_bounds['age']['min'],
        input_bounds['age']['max'],
        int,
    )
    if age_err:
        errors.append(age_err)
        st.sidebar.markdown(f"<div class='field-error'>{age_err}</div>", unsafe_allow_html=True)
    else:
        inputs['age'] = age_val

    inputs['gender'] = st.sidebar.radio(
        "Gender",
        ["Male", "Female"],
        index=1,
        horizontal=True,
        help="Biological sex used by the trained model as a statistical risk factor.",
    )

    cgpa_raw = st.sidebar.text_input(
        "CGPA",
        value=f"{input_bounds['cgpa']['default']:.1f}",
        key='cgpa_input',
        help="Enter a CGPA between 0.0 and 10.0.",
        placeholder="e.g. 7.8",
    )
    cgpa_val, cgpa_err = parse_numeric_text_input(
        cgpa_raw,
        "CGPA",
        input_bounds['cgpa']['min'],
        input_bounds['cgpa']['max'],
        float,
    )
    if cgpa_err:
        errors.append(cgpa_err)
        st.sidebar.markdown(f"<div class='field-error'>{cgpa_err}</div>", unsafe_allow_html=True)
    else:
        inputs['cgpa'] = cgpa_val

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 🎓 Academic")
    inputs['academic'] = st.sidebar.slider(
        "Academic Pressure  (1 – 5)",
        1, 5, 3,
        help="Self-reported pressure from academic workload.",
    )
    inputs['study_sat'] = st.sidebar.slider(
        "Study Satisfaction  (1 – 5)",
        1, 5, 3,
        help="How satisfied is the student with their studies?",
    )
    inputs['work_study'] = st.sidebar.slider(
        "Work/Study Hours per Day",
        0, 16, 8,
        help="Total hours spent on study-related work across the day.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 🏠 Personal & Health")
    inputs['financial'] = st.sidebar.slider(
        "Financial Stress  (1 – 5)",
        1, 5, 2,
        help="Financial burden or stress level.",
    )
    inputs['sleep'] = st.sidebar.selectbox(
        "Sleep Duration",
        ["Less Than 5 Hours", "5-6 Hours", "6-7 Hours", "7-8 Hours", "More Than 8 Hours"],
        index=2,
        help="Average sleep per night.",
    ).lower()
    inputs['diet'] = st.sidebar.selectbox(
        "Dietary Habits",
        ["Healthy", "Moderate", "Unhealthy"],
        index=1,
        help="General quality of daily diet.",
    ).lower()
    inputs['suicid'] = st.sidebar.radio(
        "Suicidal Thoughts",
        ["Yes", "No"],
        index=1,
        horizontal=True,
        help="Has the student ever had suicidal thoughts?",
    )
    inputs['family'] = st.sidebar.radio(
        "Family History of Mental Illness",
        ["Yes", "No"],
        index=1,
        horizontal=True,
        help="Any immediate family member diagnosed with a mental illness?",
    )

    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button(
        "🔍  Predict Depression Risk",
        use_container_width=True,
        type="primary"
    )
    if predict_btn and errors:
        bullet_list = ''.join(f"<li>{err}</li>" for err in errors)
        st.sidebar.markdown(
            f"<div class='field-error'><b>Fix these before predicting:</b><ul>{bullet_list}</ul></div>",
            unsafe_allow_html=True,
        )

    return inputs, predict_btn, errors


# ══════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════

def format_pct(value) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


# ══════════════════════════════════════════════════════════════
# RESULT CARD
# ══════════════════════════════════════════════════════════════

def render_result(result: dict):
    pred     = result['prediction']
    prob_dep = result['prob_dep']
    prob_not = result['prob_not']
    conf     = result['confidence']
    risk_cat, risk_emoji = risk_category(prob_dep)

    if pred == 1:
        card_class = "result-depressed"
        emoji      = "⚠️"
        label_html = f"<div class='result-label' style='color:#dc2626'>DEPRESSED</div>"
        prob_color = "#dc2626"
    else:
        card_class = "result-not-depressed"
        emoji      = "✅"
        label_html = f"<div class='result-label' style='color:#16a34a'>NOT DEPRESSED</div>"
        prob_color = "#16a34a"

    st.markdown(f"""
    <div class='{card_class}'>
        <div class='result-emoji'>{emoji}</div>
        {label_html}
        <div class='result-sub'>{risk_emoji} Risk Level: <b>{risk_cat}</b></div>
        <div class='prob-text' style='color:{prob_color}'>
            Confidence: {conf}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability gauges
    st.markdown("<div class='section-header'>📊 Probability Breakdown</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🔴 Depressed",     f"{prob_dep:.1f}%")
        st.progress(int(prob_dep))
    with c2:
        st.metric("🟢 Not Depressed", f"{prob_not:.1f}%")
        st.progress(int(prob_not))


# ══════════════════════════════════════════════════════════════
# FEATURE CONTRIBUTION CHART
# ══════════════════════════════════════════════════════════════

def render_shap_chart(contrib_df: pd.DataFrame):
    st.markdown("<div class='section-header'>🔍 Feature Contributions</div>",
                unsafe_allow_html=True)
    st.caption("Positive values (red) push toward 'Depressed'. "
               "Negative values (green) push toward 'Not Depressed'.")

    fig, ax = plt.subplots(figsize=(7, max(3, len(contrib_df) * 0.45)))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')

    feats = contrib_df['feature'].apply(format_feature_name).tolist()
    vals  = contrib_df['shap_value'].tolist()
    colors = ['#ef4444' if v >= 0 else '#22c55e' for v in vals]

    bars = ax.barh(feats[::-1], vals[::-1], color=colors[::-1],
                   edgecolor='white', height=0.65)
    ax.axvline(0, color='#64748b', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Contribution Score (impact on prediction)', fontsize=9)
    ax.set_title('Per-Sample Feature Contribution', fontsize=10, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=8.5)

    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + (0.001 if v >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f'{v:+.3f}', va='center',
                ha='left' if v >= 0 else 'right',
                fontsize=7.5, color='#334155')

    red_patch   = mpatches.Patch(color='#ef4444', label='↑ Increases Depression Risk')
    green_patch = mpatches.Patch(color='#22c55e', label='↓ Decreases Depression Risk')
    ax.legend(handles=[red_patch, green_patch], fontsize=8,
              loc='lower right', framealpha=0.9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# GLOBAL FEATURE IMPORTANCE TAB
# ══════════════════════════════════════════════════════════════

def render_importance_tab():
    st.markdown("### 📂 Training Plots")
    plots_to_show = [
        ('feature_importance.png',         'Feature Importance'),
        ('cv_comparison.png',              'CV Model Comparison'),
        ('confusion_matrix.png',           'Confusion Matrix'),
        ('learning_curve.png',             'Learning Curve'),
        ('cv_distribution.png',            'CV Score Distribution'),
        ('feature_selection_analysis.png', 'Feature Selection Analysis'),
    ]
    plots_dir = os.path.join(BASE_DIR, 'plots')
    cols2 = st.columns(2)
    for i, (fname, caption) in enumerate(plots_to_show):
        path = os.path.join(plots_dir, fname)
        if os.path.exists(path):
            with cols2[i % 2]:
                st.image(path, caption=caption, width="stretch")


# ══════════════════════════════════════════════════════════════
# MODEL COMPARISON HELPERS
# ══════════════════════════════════════════════════════════════

def render_model_comparison():
    if model_cmp_df is None or getattr(model_cmp_df, 'empty', True):
        st.info("Retrain the model with the updated pipeline to see train/test/CV comparison across models.")
        return

    cmp_df = model_cmp_df.copy()
    display_df = cmp_df.drop(columns=['features_used', 'fit_time_s'], errors='ignore').copy()
    percent_cols = ['train_accuracy', 'cv_accuracy', 'test_accuracy', 'precision', 'recall', 'f1_score']
    for col in percent_cols:
        display_df[col] = display_df[col].apply(format_pct)
    display_df['cv_std'] = display_df['cv_std'].apply(lambda v: f"{v*100:.2f}%")
    display_df = display_df.rename(columns={
        'model': 'Model',
        'train_accuracy': 'Train Accuracy',
        'cv_accuracy': 'CV Accuracy',
        'cv_std': 'CV Std Dev',
        'test_accuracy': 'Test Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score',
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, max(3.8, len(cmp_df) * 0.8)))
    y_pos = list(range(len(cmp_df)))
    width = 0.22
    ax.barh([y + width for y in y_pos], cmp_df['train_accuracy'] * 100, height=width, color='#2563eb', label='Train')
    ax.barh(y_pos, cmp_df['cv_accuracy'] * 100, height=width, color='#0f766e', label='CV')
    ax.barh([y - width for y in y_pos], cmp_df['test_accuracy'] * 100, height=width, color='#f59e0b', label='Test')
    ax.set_yticks(y_pos, cmp_df['model'])
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Candidate Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 100)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# MODEL METRICS TAB
# ══════════════════════════════════════════════════════════════

def render_metrics_tab():
    st.markdown("### 🏆 Model Performance")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Training Accuracy", format_pct(train_acc))
    mc2.metric("Test Accuracy", format_pct(test_acc))
    cv_label = format_pct(cv_acc)
    if cv_std is not None:
        cv_label = f"{format_pct(cv_acc)} ± {cv_std*100:.2f}%"
    mc3.metric("CV Accuracy", cv_label)

    dep = report_dict.get('Depressed', {})
    mc4.metric("Recall (Dep.)", format_pct(dep.get('recall', 0)))
    st.caption(f"Final model used in the application: {model_name}")

    st.markdown("---")
    st.markdown("### ⚖️ Candidate Model Comparison")
    st.caption("Comparison across the tuned candidate models used in the final training workflow.")
    render_model_comparison()

    st.markdown("---")
    st.markdown("### 📋 Classification Report")
    if report_dict:
        rows = []
        for cls in ['Not Depressed', 'Depressed', 'macro avg', 'weighted avg']:
            d = report_dict.get(cls, {})
            if d:
                rows.append({
                    'Class':     cls,
                    'Precision': f"{d.get('precision', 0)*100:.2f}%",
                    'Recall':    f"{d.get('recall', 0)*100:.2f}%",
                    'F1-Score':  f"{d.get('f1-score', 0)*100:.2f}%",
                    'Support':   int(d.get('support', 0)),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔬 Feature Selection Rationale")
    rationale = {
        'Suicidal Thoughts':     'Strongest single clinical indicator. Direct risk marker.',
        'Academic Pressure':     'Core student-specific stressor. Highly predictive.',
        'Financial Stress':      'Non-academic stressor with large effect size in studies.',
        'CGPA':                  'Academic performance proxy; low CGPA creates a stress cycle.',
        'Sleep Duration':        'Disrupted sleep is both a symptom and a cause of depression.',
        'Family History':        '~40% heritability; strong genetic and environmental predictor.',
        'Work/Study Hours':      'Overwork validated as a burnout and depression predictor.',
        'Dietary Habits':        'Gut-brain axis; poor nutrition impairs mood regulation.',
        'Study Satisfaction':    'Protective factor; dissatisfaction → disengagement loop.',
        'Age':                   'Peak depression onset 18–25 — the student demographic.',
        'Gender':                'Documented differences in prevalence and symptom expression.',
    }
    rows_r = [{'Feature': k, 'Domain Rationale': v} for k, v in rationale.items()]
    st.dataframe(pd.DataFrame(rows_r), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# ABOUT TAB
# ══════════════════════════════════════════════════════════════

def render_about_tab():
    st.markdown("### ℹ️ About This System")
    st.markdown("""
This is a **professional, research-grade** Student Depression Detection system built with:

| Component         | Details |
|-------------------|---------|
| **Dataset**       | Kaggle Student Depression Dataset — 27,901 students |
| **Models**        | Tuned Gradient Boosting / Random Forest / Logistic Regression |
| **Tuning**        | RandomizedSearchCV with 5-Fold Stratified CV |
| **Interpretability** | Feature contribution analysis + Permutation Importance |
| **Feature Selection** | Domain reasoning + Mutual Information + Correlation |
| **UI**            | Streamlit (replaces Tkinter) |

---
### 🏗️ Project Structure
```
student_depression/
├── data/                  ← Dataset
├── models/                ← Saved model (.pkl)
├── plots/                 ← Training visualisations
├── src/
│   ├── preprocess.py      ← Data loading & encoding
│   ├── train.py           ← Training pipeline
│   ├── predict.py         ← Inference + feature contributions
│   └── utils.py           ← Shared helpers
├── app/
│   └── app.py             ← This Streamlit app
├── notebooks/             ← Jupyter analysis
├── requirements.txt
└── README.md
```

---
### ⚠️ Disclaimer
This tool is intended for **research and educational purposes only**.
It is **not a clinical diagnostic tool** and must not replace professional mental health evaluation.

If you or someone you know is struggling, please reach out to:
- **iCall (India)**: 9152987821
- **Vandrevala Foundation**: 1860-2662-345
- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/
    """)


# ══════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════

def main():
    if not MODEL_LOADED:
        st.error(f"⚠️ Model not found. Please run `python src/train.py` first.\n\n{load_error}")
        st.stop()

    input_bounds = load_input_bounds()

    # Header
    prec = report_dict.get('Depressed', {}).get('precision', 0)
    rec  = report_dict.get('Depressed', {}).get('recall', 0)
    f1   = report_dict.get('Depressed', {}).get('f1-score', 0)

    st.markdown(f"""
    <div class='main-header'>
        <h1>🧠 Student Depression Detection</h1>
        <p>AI-powered mental health risk screening for students</p>
        <span class='metric-pill'>Model: {model_name}</span>
        <span class='metric-pill'>Test Acc: {format_pct(test_acc)}</span>
        <span class='metric-pill'>CV Acc: {format_pct(cv_acc)}</span>
        <span class='metric-pill'>F1: {format_pct(f1)}</span>
        <span class='metric-pill'>Dataset: 27,901 students</span>
    </div>
    """, unsafe_allow_html=True)

    # ===== ADDED: Recall Display =====
    # Extract recall from existing report_dict (already loaded above)
    recall = report_dict.get('Depressed', {}).get('recall', 0)
    st.markdown(
        f"<div style='background:#fefce8;border-left:4px solid #f59e0b;"
        f"border-radius:6px;padding:0.5rem 1rem;margin-bottom:0.8rem;"
        f"font-size:0.88rem;color:#78350f'>"
        f"⚡ <b>Recall (Depressed class): {format_pct(recall)}</b> &nbsp;|&nbsp; "
        f"Precision: {format_pct(prec)} &nbsp;|&nbsp; F1: {format_pct(f1)} &nbsp;&nbsp;"
        f"<span style='opacity:0.75'>"
        f"Recall is prioritized to minimize missed depression cases.</span></div>",
        unsafe_allow_html=True,
    )
    # ===== END: Recall Display =====

    # Sidebar inputs
    user_inputs, predict_btn, input_errors = render_sidebar(input_bounds)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Prediction",
        "📈 Feature Importance",
        "🏆 Model Metrics",
        "ℹ️ About",
    ])

    with tab1:
        col_main, col_side = st.columns([1.1, 0.9])

        with col_main:
            st.markdown("<div class='section-header'>🎯 Prediction Result</div>",
                        unsafe_allow_html=True)

            if predict_btn and input_errors:
                st.markdown("""
                <div class='result-neutral'>
                    <div class='result-emoji'>🛠️</div>
                    <div style='font-size:1.2rem;font-weight:700;margin:0.5rem 0;'>
                        Input Needs Attention
                    </div>
                    <div style='color:#64748b'>
                        Please correct the highlighted sidebar fields before running the assessment.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif predict_btn or st.session_state.get('last_result'):
                if predict_btn:
                    with st.spinner("Analysing student profile..."):
                        result = predict(user_inputs)
                        contrib = explain_prediction(user_inputs)
                    current_inputs = user_inputs.copy()
                    st.session_state['last_result']  = result
                    st.session_state['last_contrib']  = contrib
                    st.session_state['last_inputs']   = current_inputs
                else:
                    result  = st.session_state['last_result']
                    contrib = st.session_state['last_contrib']
                    current_inputs = st.session_state.get('last_inputs', user_inputs)

                render_result(result)

                if contrib is not None:
                    render_shap_chart(contrib)
                else:
                    # Fallback: show global feature importance as proxy
                    st.markdown("<div class='section-header'>📊 Top Contributing Features</div>",
                                unsafe_allow_html=True)
                    top_f = imp_df.head(8)
                    for _, row in top_f.iterrows():
                        fname = format_feature_name(row['feature'])
                        pct   = int(row['perm_imp'] / imp_df['perm_imp'].max() * 100)
                        st.markdown(f"**{fname}** — permutation importance: `{row['perm_imp']:.4f}`")
                        st.progress(pct)

            else:
                st.markdown("""
                <div class='result-neutral'>
                    <div class='result-emoji'>🎓</div>
                    <div style='font-size:1.2rem;font-weight:700;margin:0.5rem 0;'>
                        Ready for Assessment
                    </div>
                    <div style='color:#64748b'>
                        Fill in the student profile in the sidebar,<br>
                        then click <b>Predict Depression Risk</b>.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_side:
            st.markdown("<div class='section-header'>📌 Assessment Notes</div>",
                        unsafe_allow_html=True)
            st.markdown("""
1. Enter **Age** and **CGPA** directly as typed values.
2. Complete the academic, personal, and health inputs.
3. Click **Predict Depression Risk** to generate the result and explanation.
            """)

            st.markdown(
                f"""
                <div class='summary-card'>
                    <b>Current input summary</b><br>
                    Age range allowed: <b>{input_bounds['age']['min']}–{input_bounds['age']['max']}</b><br>
                    CGPA range allowed: <b>{input_bounds['cgpa']['min']:.1f}–{input_bounds['cgpa']['max']:.1f}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("""
            <div class='disclaimer'>
            ⚠️ <b>Disclaimer:</b> This is a research tool only. 
            It is not a clinical diagnostic instrument. 
            Always consult a qualified mental health professional.
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        render_importance_tab()

    with tab3:
        render_metrics_tab()

    with tab4:
        render_about_tab()


if __name__ == '__main__':
    main()

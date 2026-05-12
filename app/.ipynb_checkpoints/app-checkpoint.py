"""
app.py  ─  Streamlit UI for Student Depression Detection
─────────────────────────────────────────────────────────────
Run:
    streamlit run app/app.py

Requires the model to be trained first:
    python src/train.py
"""

import os, sys, warnings
import numpy as np
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

from predict import load_bundle, predict, explain_prediction
from utils   import format_feature_name, risk_category, FIELD_DEFINITIONS, get_plot_path

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
    background: linear-gradient(160deg, #1e1b4b 0%, #312e81 100%);
    padding-top: 1rem;
}
section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
section[data-testid="stSidebar"] .stSlider > label,
section[data-testid="stSidebar"] .stRadio > label,
section[data-testid="stSidebar"] .stSelectbox > label { color: #c7d2fe !important; font-size: 0.82rem; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* Main header */
.main-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
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
    test_acc     = bundle['test_accuracy']
    cv_acc       = bundle['cv_accuracy']
    imp_df       = bundle['importance_df']
    report_dict  = bundle.get('report_dict', {})
    MODEL_LOADED = True
except FileNotFoundError as e:
    MODEL_LOADED = False
    load_error   = str(e)


# ══════════════════════════════════════════════════════════════
# SIDEBAR  ─  Input Controls
# ══════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    st.sidebar.markdown("## 📋 Student Profile")
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 🎓 Academic")

    inputs = {}

    for label, kw, widget, opts, help_txt in FIELD_DEFINITIONS:
        if widget == 'slider':
            lo, hi, default = opts
            step = 0.1 if isinstance(default, float) else 1.0
            val  = st.sidebar.slider(label, float(lo), float(hi),
                                     float(default), step=step, help=help_txt)
            inputs[kw] = val

        elif widget == 'radio':
            val = st.sidebar.radio(label, opts, index=len(opts) - 1,
                                   horizontal=True, help=help_txt)
            inputs[kw] = val

        elif widget == 'select':
            display = [o.title() for o in opts]
            idx = st.sidebar.selectbox(label, display, index=2, help=help_txt)
            inputs[kw] = opts[display.index(idx)]   # back to raw key

        # Section breaks
        if kw == 'cgpa':
            st.sidebar.markdown("---")
            st.sidebar.markdown("##### 💼 Work & Lifestyle")
        if kw == 'financial':
            st.sidebar.markdown("---")
            st.sidebar.markdown("##### 🏠 Personal & Health")

    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button(
        "🔍  Predict Depression Risk",
        use_container_width=True,
        type="primary"
    )
    return inputs, predict_btn


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
# SHAP CONTRIBUTION CHART
# ══════════════════════════════════════════════════════════════

def render_shap_chart(contrib_df: pd.DataFrame):
    st.markdown("<div class='section-header'>🔍 Feature Contributions (SHAP)</div>",
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
    ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=9)
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
    st.markdown("### 📈 Model Feature Importance (Global)")
    st.caption(
        "Three importance methods are shown. Permutation Importance and SHAP "
        "are more reliable than native GB importance for correlated features."
    )

    top = imp_df.head(12).copy()
    top['label'] = top['feature'].apply(format_feature_name)

    methods = [
        ('native_imp', 'GB Native Importance', '#5c6bc0'),
        ('perm_imp',   'Permutation Importance', '#26a69a'),
    ]
    if imp_df['shap_imp'].sum() > 0:
        methods.append(('shap_imp', 'SHAP Mean |φ|', '#ef5350'))

    ncols = len(methods)
    cols  = st.columns(ncols)
    for col_st, (metric, title, color) in zip(cols, methods):
        with col_st:
            st.markdown(f"**{title}**")
            fig, ax = plt.subplots(figsize=(4, 5))
            fig.patch.set_facecolor('#f8fafc')
            ax.set_facecolor('#f8fafc')
            vals  = top[metric].values
            feats = top['label'].tolist()
            ax.barh(feats[::-1], vals[::-1], color=color,
                    edgecolor='white', alpha=0.9)
            ax.set_xlabel('Score', fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(axis='y', labelsize=7.5)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Saved plots
    st.markdown("---")
    st.markdown("### 📂 Training Plots")
    plots_to_show = [
        ('feature_importance.png',         'Feature Importance — 3 Methods'),
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
                st.image(path, caption=caption, use_column_width=True)


# ══════════════════════════════════════════════════════════════
# MODEL METRICS TAB
# ══════════════════════════════════════════════════════════════

def render_metrics_tab():
    st.markdown("### 🏆 Model Performance")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Test Accuracy",  f"{test_acc*100:.2f}%")
    mc2.metric("CV Accuracy",    f"{cv_acc*100:.2f}%")

    dep = report_dict.get('Depressed', {})
    mc3.metric("Recall (Dep.)",   f"{dep.get('recall', 0)*100:.2f}%")
    mc4.metric("F1-Score (Dep.)", f"{dep.get('f1-score', 0)*100:.2f}%")

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
        'Work Pressure':         'Dual-burden (study + work) increases risk significantly.',
        'Job Satisfaction':      'Meaningful work is a protective factor against burnout.',
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
| **Model**         | Tuned Gradient Boosting / Random Forest / XGBoost |
| **Tuning**        | RandomizedSearchCV with 5-Fold Stratified CV |
| **Interpretability** | SHAP values + Permutation Importance |
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
│   ├── predict.py         ← Inference + SHAP
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

    # Header
    prec = report_dict.get('Depressed', {}).get('precision', 0)
    rec  = report_dict.get('Depressed', {}).get('recall', 0)
    f1   = report_dict.get('Depressed', {}).get('f1-score', 0)

    st.markdown(f"""
    <div class='main-header'>
        <h1>🧠 Student Depression Detection</h1>
        <p>AI-powered mental health risk screening for students</p>
        <span class='metric-pill'>Model: {model_name}</span>
        <span class='metric-pill'>Test Acc: {test_acc*100:.1f}%</span>
        <span class='metric-pill'>CV Acc: {cv_acc*100:.1f}%</span>
        <span class='metric-pill'>F1: {f1*100:.1f}%</span>
        <span class='metric-pill'>Dataset: 27,901 students</span>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar inputs
    user_inputs, predict_btn = render_sidebar()

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

            if predict_btn or st.session_state.get('last_result'):
                if predict_btn:
                    with st.spinner("Analysing student profile..."):
                        result = predict(user_inputs)
                        contrib = explain_prediction(user_inputs)
                    st.session_state['last_result']  = result
                    st.session_state['last_contrib']  = contrib
                    st.session_state['last_inputs']   = user_inputs.copy()
                else:
                    result  = st.session_state['last_result']
                    contrib = st.session_state['last_contrib']

                render_result(result)

                if contrib is not None:
                    render_shap_chart(contrib)
                else:
                    # Fallback: show global feature importance as proxy
                    st.markdown("<div class='section-header'>📊 Top Contributing Features (Global)</div>",
                                unsafe_allow_html=True)
                    st.caption("Install `shap` for per-prediction explanations: `pip install shap`")
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
            st.markdown("<div class='section-header'>📌 How to Use</div>",
                        unsafe_allow_html=True)
            st.markdown("""
1. **Fill in the sidebar** with the student's profile details
2. Use **sliders** for numeric values
3. Use **dropdowns** for sleep and diet categories
4. Click **Predict Depression Risk**
5. Review the **colour-coded result** and **confidence score**
6. Check **SHAP values** to understand *why* the model gave this prediction

---
**Colour Guide**
- 🟢 **Green** — Not Depressed
- 🔴 **Red** — Depressed
- 🟡 **Amber** — Borderline / Moderate risk
            """)

            st.markdown("<div class='section-header'>🔑 Key Features</div>",
                        unsafe_allow_html=True)
            top5 = imp_df.head(5)
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                fname = format_feature_name(row['feature'])
                st.markdown(f"**{i}. {fname}**")
                st.progress(int(row['perm_imp'] / imp_df['perm_imp'].max() * 100))

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

# 🧠 Student Depression Detection — Production ML System

A professional, research-grade machine learning system for detecting depression risk in students, featuring feature contribution analysis, multi-model comparison, hyperparameter tuning, and a modern Streamlit UI.

---

## 📁 Project Structure

```
student_depression/
├── data/
│   └── student_depression.csv       ← Kaggle dataset (add here)
│
├── models/
│   └── depression_model.pkl         ← Auto-generated after training
│
├── plots/                           ← Auto-generated training plots
│   ├── cv_comparison.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   ├── learning_curve.png
│   ├── cv_distribution.png
│   └── feature_selection_analysis.png
│
├── src/
│   ├── preprocess.py                ← Data loading, cleaning, encoding
│   ├── train.py                     ← Full training pipeline
│   ├── predict.py                   ← Inference + feature contribution explanation
│   └── utils.py                     ← Shared helpers & field definitions
│
├── app/
│   └── app.py                       ← Streamlit web application
│
├── notebooks/
│   └── analysis.ipynb               ← Exploratory analysis (optional)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your dataset
Place `student_depression.csv` inside the `data/` folder.

### 3. Train the model
```bash
python3 src/train.py
```

This will:
- Preprocess data with domain-driven feature selection
- Tune Gradient Boosting, Random Forest, and Logistic Regression
- Run 5-Fold Stratified Cross-Validation
- Compute contribution-based + permutation importance
- Save model to `models/depression_model.pkl`
- Save 6 plots to `plots/`

### 4. Launch the Streamlit app
```bash
streamlit run app/app.py
```

---

## 🔬 Feature Selection Rationale

Features were chosen using **three complementary methods**:

| Method | Purpose |
|--------|---------|
| **Domain reasoning** | Clinical & research-backed relevance |
| **Mutual Information** | Non-linear statistical dependence with target |
| **Pearson Correlation** | Linear association screening |

### ✅ Included Features

| Feature | Domain Justification |
|---------|---------------------|
| Suicidal Thoughts | Strongest direct clinical risk marker |
| Academic Pressure | Core student stressor — central to mental health research |
| Financial Stress | Validated non-academic stressor with large effect size |
| CGPA | Academic performance proxy; low CGPA creates a stress feedback loop |
| Sleep Duration | Disrupted sleep is both a symptom and cause of depression |
| Family History | ~40% heritability; genetic + environmental risk |
| Work/Study Hours | Overwork is a validated burnout predictor |
| Dietary Habits | Gut-brain axis; poor nutrition impairs mood regulation |
| Study Satisfaction | Protective factor; dissatisfaction leads to disengagement |
| Age | Peak onset window: 18–25 (student demographic) |
| Gender | Documented prevalence differences in clinical literature |
### ❌ Removed Features

| Feature | Reason for Removal |
|---------|-------------------|
| ID | Identifier — zero predictive value |
| City | Too granular; insufficient coverage across cities |
| Degree | Near-zero variance in a student-specific dataset |
| Profession | Mostly 'Student'; redundant with the dataset's scope |

---

## 🤖 Model Architecture

### Candidates
- **Gradient Boosting** (baseline + tuned)
- **Random Forest** (tuned)
- **Logistic Regression** (tuned)

### Tuning
- `RandomizedSearchCV` with 20 iterations
- `StratifiedKFold` (5 splits) — preserves class balance in each fold

### Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Learning Curve (bias-variance tradeoff)

---

## 📊 Interpretability

### Feature Importance Methods

1. **GB Native `feature_importances_`** — fast but biased toward high-cardinality features
2. **Permutation Importance** — model-agnostic, evaluated on held-out test set (more reliable)
3. **Contribution analysis** — per-sample explanations for the final Gradient Boosting model

> Why academic pressure might be *overestimated* by native importance:
> Native GB importance counts how often a feature is used for splitting, not how much it actually *affects the outcome*. Permutation and contribution analysis show the true causal picture — suicidal thoughts and financial stress are typically stronger.

---

## 🖥️ Streamlit UI Features

| Feature | Detail |
|---------|--------|
| Sidebar inputs | Sliders, dropdowns, radio buttons |
| Colour-coded result | 🟢 Green = Not Depressed, 🔴 Red = Depressed |
| Confidence score | Predicted class probability (%) |
| Probability gauge | Both class probabilities displayed |
| Feature contribution chart | Per-prediction feature contributions |
| Feature importance | All 3 methods visualised |
| Model metrics | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Plots gallery | All training plots embedded in app |

---

## 🔮 Future Extensions

- **Text-based sentiment analysis**: Add a free-text input field; pass text through a sentiment/NLP model (e.g. `transformers` BERT) and use the score as an additional feature
- **Longitudinal tracking**: Allow repeated assessments over time to track risk trajectory
- **Multi-modal**: Integrate sleep tracker / activity data APIs
- **Federated learning**: Train across institutions without sharing raw student data
- **Calibration**: Platt scaling / isotonic regression for better-calibrated probabilities

---

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**.

It is **not a clinical diagnostic instrument** and must not replace professional mental health evaluation. If you or someone you know is at risk, please contact a qualified mental health professional or a crisis helpline.

**India helplines:**
- iCall: 9152987821
- Vandrevala Foundation: 1860-2662-345
- AASRA: 9820466627

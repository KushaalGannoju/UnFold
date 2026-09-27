# 🧠 UnFold — Multimodal Student Mental Health Intelligence System (Minor II)
## Dual-Engine Multimodal Architecture & GenAI Integration

**Department of Computer Science and Engineering**  
**Maulana Azad National Institute of Technology (MANIT), Bhopal**  

**Project Guides:** Dr. R.K. Pateriya, Dr. Archana Balmik  
**Project Team:** Himanshu Verma, Kushaal Sai Gannoju, Ritika Rani Ujjainia, Rayyan Zameer Baikadi  

---

## 1. Executive Summary & Problem Formulation

In the previous semester (Minor Project I), a tabular machine learning system was implemented using the Kaggle Student Depression Dataset (27,901 records). While the resulting Gradient Boosting model achieved **84.70% test accuracy** and an **87.13% F1-score**, single-modality tabular systems suffer from three fundamental limitations:

1. **Absence of Semantic Nuance**: Students rarely express depression as structured numbers ($1$ to $5$ ratings). They express it in natural language—feelings of exhaustion, panic over deadlines, insomnia, and self-doubt.
2. **Emotional Masking**: A student might report acceptable CGPA ($8.2$) and moderate sleep ($6.5\text{h}$), but their personal reflection may reveal acute panic, loneliness, or crisis.
3. **Descriptive vs. Actionable Explainability**: Raw SHAP feature attributions tell stakeholders *which* features contributed to the prediction (e.g. $+0.18$ on Academic Pressure), but cannot offer personalized, empathetic Cognitive Behavioral Therapy (CBT) guidance or structured intake briefings for campus counselors.

To solve this, **Minor Project II introduces a Dual-Engine Multimodal Hybrid Architecture**:
- **Engine A (Tabular ML)**: Tuned Gradient Boosting on demographic and lifestyle indicators with SHAP local explanations.
- **Engine B (Clinical NLP)**: Fine-tuned `roberta-base` transformer on **30,000 balanced mental health statements**, reaching **86.20% accuracy and 86.28% F1-score**.
- **Multimodal Fusion Engine**: Dynamic, calibrated late fusion layer computing consensus scores and detecting secondary comorbid flags (High Anxiety, Academic Burnout, Sleep Disturbance).
- **GenAI Clinical Layer**: Google Gemini 1.5/2.5 Flash integration translating model outputs into empathetic CBT coping pathways and downloadable 1-page Clinical Counselor Dossiers (PDF).

---

## 2. Dataset Architecture & Preprocessing

The system leverages two large-scale benchmarks:

| Dataset | Modality | Records | Target / Classes | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Kaggle Student Depression Dataset** | Tabular (11 Features) | 27,901 rows | Binary ($0$: Not Depressed, $1$: Depressed) | Trains Engine A on lifestyle, CGPA, study hours, sleep, and financial strain. |
| **Mental Health Text Classification Dataset** (`btwitssayan`) | Natural Language Text | 53,043 rows (Sampled: 30,000) | Binary ($0$: Normal, $1$: Depressed/Distress) | Trains Engine B on authentic personal expressions, anxiety, and stress statements. |

### Tabular Preprocessing Pipeline (`src/preprocess.py`)
- **Domain Ordinal Encoding**: 
  - `Sleep Duration`: $\text{Less than 5h} \to 4.0$, $\text{5-6h} \to 5.5$, $\text{6-7h} \to 6.5$, $\text{7-8h} \to 7.5$, $\text{>8h} \to 9.0$.
  - `Dietary Habits`: $\text{Unhealthy} \to 0$, $\text{Moderate} \to 1$, $\text{Healthy} \to 2$.
- **Statistical Feature Screening**: Mutual Information ($I(X; Y)$) and Pearson correlation screening to retain high-signal features while pruning noise (`City`, `Degree`, `Profession`).

### NLP Preprocessing Pipeline (`src/nlp_engine.py`)
- Byte-pair tokenization using `roberta-base` AutoTokenizer ($\text{max\_length}=128$).
- Automated regex-based secondary comorbidity detectors extracting Anxiety, Academic Burnout, Chronic Insomnia, Social Alienation, and Acute Self-Harm risk flags.

---

## 3. Model Architecture & Mathematical Formulation

```
[Tabular Vector X_tab] ────────► [Gradient Boosting] ────► P_tab ∈ [0, 1] ──┐
                                                                            ├──► [Fusion Engine] ──► P_hybrid
[Text Statement X_text] ───────► [RoBERTa Classifier] ───► P_text ∈ [0, 1] ─┘
```

### Engine A: Gradient Boosting Classifier
Given feature vector $\mathbf{x}_{\text{tab}} \in \mathbb{R}^{11}$, Engine A computes:
$$P_{\text{tab}}(y=1 \mid \mathbf{x}_{\text{tab}}) = \frac{1}{1 + e^{-F_M(\mathbf{x}_{\text{tab}})}}$$
where $F_M(\mathbf{x})$ is the ensemble of $M$ decision trees optimized with deviance loss.

### Engine B: Fine-Tuned RoBERTa Transformer
Given token sequence $\mathbf{w} = [w_1, w_2, \dots, w_L]$, RoBERTa extracts contextualized representations:
$$\mathbf{H} = \text{TransformerEncoder}(\mathbf{w}) \in \mathbb{R}^{L \times 768}$$
$$\mathbf{h}_{\text{CLS}} = \mathbf{H}[0] \in \mathbb{R}^{768}$$
$$\mathbf{z} = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \mathbf{h}_{\text{CLS}} + \mathbf{b}_1) + \mathbf{b}_2 \in \mathbb{R}^2$$
$$P_{\text{text}}(y=1 \mid \mathbf{w}) = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$$

### Multimodal Fusion Formula (`src/fusion.py`)
The unified prediction is calculated via calibrated late fusion:
$$P_{\text{hybrid}} = \alpha \cdot P_{\text{tab}} + (1 - \alpha) \cdot P_{\text{text}}$$
where $\alpha = 0.45$ and $(1 - \alpha) = 0.55$.

**Clinical Safety Override**: If either the student's text contains direct self-harm markers or tabular `suicidal_thoughts == 'Yes'`, the fusion engine enforces a safety minimum threshold:
$$P_{\text{hybrid}} = \max(P_{\text{hybrid}}, P_{\text{tab}}, P_{\text{text}}, 82.0\%)$$
ensuring that high-risk presentations are never diluted.

---

## 4. Experimental Results & Benchmark Comparison

Evaluation conducted on a held-out test split of 6,000 samples for RoBERTa (trained on Kaggle GPU) and 5,580 held-out samples for Gradient Boosting:

| Model Architecture | Input Modality | Accuracy | Precision | Recall (Depressed) | F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | Tabular Only | 78.20% | 79.10% | 80.40% | 79.74% | Minor I Baseline |
| **Random Forest** | Tabular Only | 82.40% | 83.10% | 85.20% | 84.14% | Minor I Tuned |
| **Gradient Boosting** | Tabular Only | 84.70% | 85.84% | 88.46% | 87.13% | Minor I Deployed |
| **Clinical RoBERTa** | Text Only | **86.20%** | **85.77%** | **86.80%** | **86.28%** | **Minor II Engine B** |
| **Dual-Engine Hybrid Fusion** | Tabular + Text | **88.60%** | **88.10%** | **89.20%** | **88.65%** | **⚡ Minor II Flagship** |

### RoBERTa Test Confusion Matrix (6,000 Test Samples):
- **True Negatives (Not Depressed)**: 2,568 (85.6%)
- **False Positives**: 432 (14.4%)
- **False Negatives**: 396 (13.2%)
- **True Positives (Depressed)**: 2,604 (86.8%)

---

## 5. Explainable AI (XAI) & GenAI Integration

### Multimodal Interpretability
1. **Tabular SHAP TreeExplainer**: Quantifies each lifestyle factor's exact marginal contribution to depression probability.
2. **Gradient-Based Token Saliency**: Computes the gradient of the predicted depression logit with respect to input embeddings:
   $$\text{Attribution}(w_i) = \|\nabla_{\mathbf{e}_i} z_{\text{depressed}}\|_2$$
   Highlights distress-inducing words (*"exhausted"*, *"failing"*, *"anxious"*) in soft red within the Streamlit UI.

### GenAI Advisor Layer (`src/genai_advisor.py`)
- Utilizes **Google Gemini 1.5/2.5 Flash** (via `google-genai` SDK).
- Converts raw SHAP numbers into an empathetic **3-Step CBT Action Roadmap** tailored directly to the student.
- Drafts structured **Clinical Intake Briefings** for college counseling centers.
- Features a **100% offline rule-based fallback engine** ensuring zero downtime if no API key is provided.

### Automated Clinical Report Generation (`src/report_generator.py`)
- Generates a clinical-grade 1-page **Counselor Assessment Dossier (PDF)** using `fpdf2`.
- Formatted with institutional header, multimodal risk gauges, lifestyle audits, verbatim text excerpts, and recommended exploratory questions for counseling sessions.

---

## 6. How to Run the System

```bash
# 1. Activate environment
pip install -r requirements.txt

# 2. Run automated integration test suite
python src/test_hybrid_pipeline.py

# 3. Launch Streamlit UI 2.0
streamlit run app/app.py
```

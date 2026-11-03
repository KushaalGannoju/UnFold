"""
nlp_engine.py
─────────────────────────────────────────────────────────────
Engine B: Clinical NLP inference, Token Saliency, and
Secondary Comorbidity Detection using fine-tuned RoBERTa.
"""

import os
import re
from html import escape
import warnings
import threading
from support_flags import support_mentions
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "roberta_mental_health")

_MODEL = None
_TOKENIZER = None
_ENGINE_LOCK = threading.RLock()
MAX_TOKENS = 256


def load_nlp_engine(model_dir: str = MODEL_DIR):
    """Load and cache the fine-tuned RoBERTa model and tokenizer."""
    global _MODEL, _TOKENIZER
    with _ENGINE_LOCK:
        return _load_nlp_engine(model_dir)


def _load_nlp_engine(model_dir):
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"RoBERTa weights not found at '{model_dir}'.\n"
                "Please place the unzipped roberta_mental_health folder in models/."
            )
        _TOKENIZER = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        _MODEL.eval()
    return _MODEL, _TOKENIZER


def predict_text(text: str) -> dict:
    """
    Run semantic classification on input text reflection.
    Returns:
        prediction : 0 (Not Depressed) | 1 (Depressed)
        label      : 'Not Depressed' | 'Depressed'
        confidence : float 0–100%
        prob_dep   : float 0–100% (Depressed probability)
        prob_not   : float 0–100% (Not Depressed probability)
    """
    if not text or not text.strip():
        # Neutral fallback for empty input
        return {
            "prediction": None, "label": "No journal submitted", "available": False,
            "confidence": None, "prob_dep": None, "prob_not": None,
        }

    model, tokenizer = load_nlp_engine()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
    )

    with _ENGINE_LOCK, torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

    raw_prob_not = probs[0] * 100
    raw_prob_dep = probs[1] * 100
    pred = 1 if raw_prob_dep >= 50.0 else 0

    prob_dep = round(raw_prob_dep, 1)
    prob_not = round(100.0 - prob_dep, 1)

    return {
        "prediction": pred,
        "label": "Depressed" if pred == 1 else "Not Depressed",
        "confidence": round(max(prob_dep, prob_not), 1),
        "prob_dep": prob_dep,
        "prob_not": prob_not,
        "available": True,
        "truncated": len(tokenizer.encode(text)) > MAX_TOKENS,
        "max_tokens": MAX_TOKENS,
        "calibration": "Uncalibrated softmax score",
    }



def detect_comorbid_tags(text: str) -> list[dict]:
    """
    Extract secondary psychiatric and lifestyle distress indicators
    from student text (Anxiety, Academic Burnout, Suicidal Ideation, Insomnia, Isolation).
    """
    if not text or not text.strip():
        return []

    lower_text = text.lower()
    tags = support_mentions(text)

    # 2. High Anxiety / Panic Indicators
    anxiety_keywords = [
        "anxiety", "anxious", "panic", "worried", "nervous", "dread",
        "shaking", "fearful", "overthinking", "racing heart", "restless",
        "trembling", "paralyzed with fear", "freaking out", "hyperventilating"
    ]
    anxiety_matches = [w for w in anxiety_keywords if re.search(rf"\b{re.escape(w)}\b", lower_text)]
    if len(anxiety_matches) >= 2 or any(k in anxiety_matches for k in ["panic", "anxiety", "anxious"]):
        tags.append({
            "category": "Anxiety",
            "label": "⚡ High Anxiety & Panic Tendencies",
            "severity": "High" if len(anxiety_matches) >= 2 else "Moderate",
            "badge_color": "#f59e0b",
            "text_color": "#ffffff",
            "description": f"Detected heightened autonomic stress & anxiety markers ({', '.join(anxiety_matches[:3])}).",
            "is_critical": False,
        })

    # 3. Academic Burnout & Performance Stress
    academic_keywords = [
        "exam", "exams", "gpa", "cgpa", "assignment", "assignments", "grades",
        "deadline", "deadlines", "failing", "syllabus", "coursework", "professor",
        "studying", "academic", "semester", "attendance", "backlog", "backlogs"
    ]
    burnout_keywords = ["burnout", "burnt out", "overwhelmed", "exhausted", "can't focus", "no motivation", "pressure"]
    acad_matches = [w for w in academic_keywords if re.search(rf"\b{re.escape(w)}\b", lower_text)]
    burn_matches = [w for w in burnout_keywords if re.search(rf"\b{re.escape(w)}\b", lower_text)]

    if acad_matches and burn_matches:
        tags.append({
            "category": "Academic Burnout",
            "label": "📚 Severe Academic Burnout",
            "severity": "High",
            "badge_color": "#6366f1",
            "text_color": "#ffffff",
            "description": f"Curriculum stress combined with emotional exhaustion ({', '.join((acad_matches + burn_matches)[:3])}).",
            "is_critical": False,
        })

    # 4. Severe Insomnia / Sleep Disruption
    sleep_keywords = [
        "insomnia", "can't sleep", "cannot sleep", "haven't slept", "trouble sleeping",
        "awake all night", "sleep deprived", "nightmares", "staying up", "wake up tired"
    ]
    sleep_matches = [w for w in sleep_keywords if w in lower_text]
    if sleep_matches:
        tags.append({
            "category": "Sleep Disturbance",
            "label": "🌙 Chronic Sleep Deprivation",
            "severity": "Moderate",
            "badge_color": "#8b5cf6",
            "text_color": "#ffffff",
            "description": f"Sleep-wake cycle disturbance identified ({', '.join(sleep_matches[:2])}).",
            "is_critical": False,
        })

    # 5. Social Withdrawal & Alienation
    isolation_keywords = [
        "alone", "lonely", "isolated", "no friends", "nobody cares", "withdrawn",
        "nobody understands", "left out", "disconnected", "avoiding people"
    ]
    iso_matches = [w for w in isolation_keywords if re.search(rf"\b{re.escape(w)}\b", lower_text)]
    if len(iso_matches) >= 1:
        tags.append({
            "category": "Social Isolation",
            "label": "👥 Social Withdrawal & Alienation",
            "severity": "Moderate",
            "badge_color": "#0ea5e9",
            "text_color": "#ffffff",
            "description": f"Feelings of loneliness or social disconnection ({', '.join(iso_matches[:2])}).",
            "is_critical": False,
        })

    return tags


def compute_token_saliency(text: str, top_k: int = 15) -> tuple[list[dict], str]:
    """
    Compute gradient-based token attribution saliency scores for RoBERTa.
    Returns:
        token_scores : list of {'token': str, 'score': float, 'is_distress': bool}
        highlighted_html : HTML string with color-coded highlighting
    """
    if not text or not text.strip():
        return [], "<p>No text provided.</p>"

    model, tokenizer = load_nlp_engine()
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Compute word embeddings with gradient tracking
    with _ENGINE_LOCK, torch.enable_grad():
        embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        grad, = torch.autograd.grad(outputs.logits[0, 1], embeddings)
        attributions = torch.norm(grad[0], dim=-1).detach().cpu().numpy()

    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]

    # Filter out special tokens (<s>, </s>, <pad>)
    filtered_tokens = []
    filtered_scores = []
    for tok, score in zip(tokens, attributions):
        if tok and tok not in ["<s>", "</s>", "<pad>", "", " "]:
            filtered_tokens.append(tok)
            filtered_scores.append(float(score))

    if not filtered_scores:
        return [], escape(text)

    # Normalize attribution scores to 0-1
    max_score = max(filtered_scores) if max(filtered_scores) > 0 else 1.0
    min_score = min(filtered_scores)
    range_score = max_score - min_score if max_score > min_score else 1.0

    normalized_scores = [(s - min_score) / range_score for s in filtered_scores]

    token_data = []
    for tok, n_score in zip(filtered_tokens, normalized_scores):
        token_data.append({
            "token": tok,
            "score": round(n_score, 3),
            "high_influence": n_score > 0.65,
        })

    # Build HTML highlighted string
    html_spans = []
    for item in token_data:
        tok = escape(item["token"])
        score = item["score"]
        if score > 0.65:
            # High distress attribution
            span = f"<span style='background:rgba(239, 68, 68, 0.45); color:#7f1d1d; padding:2px 5px; border-radius:4px; font-weight:600;' title='Attribution: {score:.2f}'>{tok}</span>"
        elif score > 0.40:
            # Moderate distress attribution
            span = f"<span style='background:rgba(251, 146, 60, 0.35); color:#9a3412; padding:2px 4px; border-radius:4px;' title='Attribution: {score:.2f}'>{tok}</span>"
        else:
            # Baseline/neutral
            span = f"<span style='color:#334155;' title='Attribution: {score:.2f}'>{tok}</span>"
        html_spans.append(span)

    highlighted_html = (
        "<div style='white-space:pre-wrap; line-height:2.2; font-size:0.95rem;'>"
        + "".join(html_spans)
        + "</div>"
    )

    return token_data, highlighted_html

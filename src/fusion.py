"""Combine model scores without allowing support rules to alter predictions."""

import math
from utils import risk_category, risk_color


def fuse_predictions(tabular_result, text_result, comorbid_tags=None,
                     user_inputs=None, alpha_tab=0.45, alpha_text=0.55):
    tags = comorbid_tags or []
    profile = user_inputs or {}
    weights = (float(alpha_tab), float(alpha_text))
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Fusion weights must be finite, non-negative, and have a positive sum.")
    p_tab = float(tabular_result["prob_dep"])
    available = text_result.get("available", True)
    p_text = float(text_result["prob_dep"]) if available else None
    if any(not math.isfinite(p) or not 0 <= p <= 100
           for p in (p_tab, p_text) if p is not None):
        raise ValueError("Model scores must be between 0 and 100.")
    w_tab, w_text = (weights[0] / sum(weights), weights[1] / sum(weights)) if available else (1.0, 0.0)
    score = round(w_tab * p_tab + w_text * (p_text or 0), 1)
    category, emoji = risk_category(score)
    if not available:
        status, note = "Profile only", "No journal was submitted. The result uses the profile model only."
    elif abs(p_text - p_tab) <= 20:
        status, note = "Similar model scores", "Both models produced similar scores; agreement is not a measure of clinical certainty."
    elif p_text < p_tab:
        status, note = "Profile score is higher", "Lifestyle and academic inputs scored higher than the journal. Consider both sources during a conversation."
    else:
        status, note = "Journal score is higher", "The journal scored higher than the profile. Consider the student's current experience alongside their routine."
    current = any(t.get("is_critical", False) for t in tags)
    history = str(profile.get("suicid", "No")).lower() in {"yes", "1", "true"}
    return {
        "prediction": int(score >= 50), "label": "Elevated model signal" if score >= 50 else "Lower model signal",
        "hybrid_prob_dep": score, "hybrid_prob_not": round(100 - score, 1),
        "confidence": round(max(score, 100 - score), 1),
        "risk_category": category, "risk_emoji": emoji, "risk_color": risk_color(score),
        "tabular_prob_dep": p_tab, "text_prob_dep": p_text,
        "weights": {"tabular": w_tab, "text": w_text},
        "consensus_status": status, "consensus_note": note,
        "comorbid_tags": tags, "is_critical": current, "history_reported": history,
        "support_review": current or history or any(t.get("context_review") for t in tags),
        "calibration": "Uncalibrated model score; not an individual probability of depression.",
    }

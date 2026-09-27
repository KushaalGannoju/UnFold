"""
fusion.py
─────────────────────────────────────────────────────────────
Hybrid Multimodal Fusion Engine.
Merges Engine A (Tabular GBDT) and Engine B (Clinical NLP RoBERTa).
Provides consensus analysis and clinical risk categorization.
"""

from utils import risk_category, risk_color


def fuse_predictions(
    tabular_result: dict,
    text_result: dict,
    comorbid_tags: list = None,
    user_inputs: dict = None,
    alpha_tab: float = 0.45,
    alpha_text: float = 0.55,
) -> dict:
    """
    Fuse predictions from Engine A (Tabular) and Engine B (Text).

    Parameters:
        tabular_result : dict with 'prob_dep', 'prob_not', 'confidence'
        text_result    : dict with 'prob_dep', 'prob_not', 'confidence'
        comorbid_tags  : list of secondary tags from nlp_engine
        user_inputs    : raw tabular inputs (to inspect suicidal thoughts)
        alpha_tab      : weight for tabular model (default: 0.45)
        alpha_text     : weight for clinical text model (default: 0.55)

    Returns:
        Structured multimodal diagnostic dictionary.
    """
    comorbid_tags = comorbid_tags or []
    user_inputs = user_inputs or {}

    p_tab_dep = float(tabular_result.get("prob_dep", 50.0))
    p_text_dep = float(text_result.get("prob_dep", 50.0))

    # Check for critical safety triggers
    has_text_crisis = any(t.get("is_critical", False) for t in comorbid_tags)
    suicide_input = str(user_inputs.get("suicid", "")).strip().lower()
    has_input_crisis = suicide_input in ["yes", "1", "true"]
    is_critical_flag = has_text_crisis or has_input_crisis

    # Dynamic Weight Adjustment
    # If text is empty or neutral default (50.0 / 50.0)
    if text_result.get("confidence", 0) == 50.0 and p_text_dep == 50.0:
        # User didn't enter a text reflection; rely 95% on tabular
        w_tab, w_text = 0.95, 0.05
    else:
        w_tab, w_text = alpha_tab, alpha_text

    # Compute fused probability
    fused_dep = (w_tab * p_tab_dep) + (w_text * p_text_dep)

    # Clinical Escalation Override: If direct suicidal thoughts or severe distress indicated,
    # ensure final risk is not diluted by a mild text or tabular score
    if is_critical_flag:
        fused_dep = max(fused_dep, p_tab_dep, p_text_dep, 82.0)

    fused_dep = round(min(max(fused_dep, 0.0), 100.0), 1)
    fused_not = round(100.0 - fused_dep, 1)

    prediction = 1 if fused_dep >= 50.0 else 0
    cat, emoji = risk_category(fused_dep)
    color = risk_color(fused_dep)

    # Multimodal Consensus & Alignment Analysis
    diff = p_text_dep - p_tab_dep
    if abs(diff) <= 20.0:
        consensus_status = "High Concordance"
        consensus_note = (
            "Both lifestyle indicators and subjective text expression align strongly, "
            "providing high clinical confidence in the assessment."
        )
    elif diff < -25.0:
        consensus_status = "Potential Emotional Masking"
        consensus_note = (
            "Lifestyle and academic stressors (e.g. disrupted sleep, high pressure) "
            "indicate significant strain despite reserved or minimal text expression. "
            "Proactive, empathetic check-in is recommended."
        )
    else:
        consensus_status = "Acute Situational Distress"
        consensus_note = (
            "The student's written reflection reveals heightened emotional vulnerability or anxiety "
            "that exceeds what standard academic/lifestyle numbers capture."
        )

    return {
        "prediction": prediction,
        "label": "Depressed" if prediction == 1 else "Not Depressed",
        "hybrid_prob_dep": fused_dep,
        "hybrid_prob_not": fused_not,
        "confidence": round(max(fused_dep, fused_not), 1),
        "risk_category": cat,
        "risk_emoji": emoji,
        "risk_color": color,
        "tabular_prob_dep": p_tab_dep,
        "text_prob_dep": p_text_dep,
        "weights": {"tabular": w_tab, "text": w_text},
        "consensus_status": consensus_status,
        "consensus_note": consensus_note,
        "comorbid_tags": comorbid_tags,
        "is_critical": is_critical_flag,
    }

"""
test_hybrid_pipeline.py
─────────────────────────────────────────────────────────────
Comprehensive integration test for the Dual-Engine Hybrid Pipeline:
Tabular GBDT + Clinical RoBERTa + Fusion + GenAI + PDF Generation.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from predict import predict, explain_prediction, predict_hybrid
from nlp_engine import predict_text, detect_comorbid_tags, compute_token_saliency
from fusion import fuse_predictions
from genai_advisor import generate_student_coping_plan, generate_counselor_briefing
from report_generator import generate_counselor_pdf


def run_tests():
    print("=" * 65)
    print("  RUNNING DUAL-ENGINE HYBRID PIPELINE INTEGRATION TESTS")
    print("=" * 65)

    sample_inputs = {
        "age": 21,
        "gender": "Female",
        "cgpa": 7.4,
        "academic": 4,
        "study_sat": 2,
        "work_study": 7,
        "financial": 4,
        "sleep": "5-6 hours",
        "diet": "moderate",
        "suicid": "No",
        "family": "Yes",
    }

    sample_text = (
        "I've been feeling completely overwhelmed by upcoming project submissions and exams. "
        "I can't sleep properly at night and wake up feeling terrified and panicked about my future."
    )

    # Test 1: Tabular Inference
    print("\n[Test 1] Engine A (Tabular GBDT)...")
    tab_res = predict(sample_inputs)
    print(f"  • Prediction: {tab_res['label']} ({tab_res['prob_dep']}%)")
    tab_contrib = explain_prediction(sample_inputs)
    print(f"  • SHAP contributors computed: {len(tab_contrib)} features")

    # Test 2: RoBERTa NLP Inference
    print("\n[Test 2] Engine B (Clinical RoBERTa)...")
    text_res = predict_text(sample_text)
    print(f"  • Prediction: {text_res['label']} ({text_res['prob_dep']}%)")
    comorbids = detect_comorbid_tags(sample_text)
    print(f"  • Comorbid tags detected: {[t['category'] for t in comorbids]}")
    tokens, html = compute_token_saliency(sample_text)
    print(f"  • Token saliency computed: {len(tokens)} tokens, HTML length: {len(html)}")

    # Test 3: Multimodal Fusion
    print("\n[Test 3] Multimodal Fusion Engine...")
    fusion_res = fuse_predictions(tab_res, text_res, comorbids, sample_inputs)
    print(f"  • Hybrid Consensus Score: {fusion_res['hybrid_prob_dep']}%")
    print(f"  • Risk Category: {fusion_res['risk_category']}")
    print(f"  • Consensus Status: {fusion_res['consensus_status']}")

    # Test 4: End-to-End predict_hybrid()
    print("\n[Test 4] Unified predict_hybrid() API...")
    unified_res = predict_hybrid(sample_inputs, sample_text)
    assert "tabular_result" in unified_res
    assert "text_result" in unified_res
    assert "fusion_result" in unified_res
    print("  • Unified predict_hybrid returned valid payload.")

    # Test 5: GenAI Advisor
    print("\n[Test 5] GenAI Clinical Advisor...")
    shap_list = tab_contrib.to_dict("records") if tab_contrib is not None else []
    plan = generate_student_coping_plan(sample_inputs, shap_list, sample_text, fusion_res)
    assert len(plan) > 200
    print(f"  • Coping plan generated ({len(plan)} chars).")

    briefing = generate_counselor_briefing(sample_inputs, shap_list, sample_text, fusion_res)
    assert "executive_summary" in briefing
    assert "interview_prompts" in briefing
    print(f"  • Counselor briefing generated ({len(briefing['raw_text'])} chars).")

    # Test 6: PDF Dossier Export
    print("\n[Test 6] PDF Counselor Dossier Generation...")
    pdf_path = generate_counselor_pdf(sample_inputs, sample_text, fusion_res, briefing)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000
    print(f"  • PDF file created: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

    print("\n" + "=" * 65)
    print("  [SUCCESS] ALL 6 INTEGRATION TESTS PASSED PERFECTLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()

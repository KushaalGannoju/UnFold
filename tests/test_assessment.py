"""Regression checks for privacy, support context, scores, and immutable submissions."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assessment import DEFAULT_PROFILE, DEMO_CASES, assessment_signature, validate_profile, run_assessment
from fusion import fuse_predictions
from genai_advisor import generate_guidance, local_guidance
from report_generator import generate_counselor_pdf
from support_flags import support_mentions
from research import load_paired_evaluation


class AssessmentTests(unittest.TestCase):
    def test_explicit_denials(self):
        for text in ["I am not suicidal.", "I don't want to die.", "I have no thoughts of self-harm.",
                     "I am not planning to kill myself."]:
            with self.subTest(text=text):
                self.assertEqual(support_mentions(text), [])

    def test_current_statement_after_denial(self):
        self.assertTrue(support_mentions("I was not suicidal before, but I want to die now.")[0]["is_critical"])

    def test_historical_and_third_person_context(self):
        for text in ["I used to be suicidal last year.", "My friend is suicidal.", "I read an article about suicide prevention."]:
            with self.subTest(text=text):
                self.assertFalse(support_mentions(text)[0]["is_critical"])
                self.assertTrue(support_mentions(text)[0]["context_review"])

    def test_alerts_do_not_change_scores(self):
        baseline = fuse_predictions({"prob_dep": 10}, {"prob_dep": 20})
        flagged = fuse_predictions({"prob_dep": 10}, {"prob_dep": 20},
                                   [{"is_critical": True}], {"suicid": "Yes"})
        self.assertEqual(baseline["hybrid_prob_dep"], flagged["hybrid_prob_dep"])
        self.assertTrue(flagged["is_critical"])
        self.assertFalse(fuse_predictions({"prob_dep": 10}, {"prob_dep": 20}, user_inputs={"suicid": "Yes"})["is_critical"])

    def test_negative_disagreement(self):
        result = fuse_predictions({"prob_dep": 60}, {"prob_dep": 38})
        self.assertEqual(result["consensus_status"], "Profile score is higher")

    def test_missing_journal_has_no_invented_score(self):
        result = fuse_predictions({"prob_dep": 12}, {"available": False, "prob_dep": None})
        self.assertEqual(result["hybrid_prob_dep"], 12)
        self.assertIsNone(result["text_prob_dep"])
        self.assertEqual(result["weights"], {"tabular": 1, "text": 0})

    def test_weights_validate(self):
        for weights in [(0, 0), (-1, 2), (float("nan"), 1)]:
            with self.assertRaises(ValueError):
                fuse_predictions({"prob_dep": 30}, {"prob_dep": 60}, alpha_tab=weights[0], alpha_text=weights[1])

    def test_no_online_access_without_opt_in(self):
        with patch("genai_advisor.get_default_api_key", side_effect=AssertionError("Key access without consent")):
            result = generate_guidance(DEFAULT_PROFILE, [], "I feel tired.", self.fusion())
        self.assertEqual(result["source"], "Local guidance")

    def test_missing_key_reports_actual_fallback(self):
        with patch("genai_advisor.get_default_api_key", return_value=None):
            result = generate_guidance(DEFAULT_PROFILE, [], "I feel tired.", self.fusion(), allow_online=True)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["source"], "Local guidance")

    def test_current_support_uses_local_guidance_without_google_request(self):
        fusion = self.fusion()
        fusion["is_critical"] = True
        with patch("google.genai.Client", side_effect=AssertionError("Unexpected Google request")):
            result = generate_guidance(DEFAULT_PROFILE, [], "Synthetic support text", fusion,
                                       allow_online=True, api_key="test-key")
        self.assertEqual(result["source"], "Local guidance")
        self.assertTrue(result["fallback"])

    def test_online_response_and_failure_are_labeled(self):
        from unittest.mock import MagicMock
        response = {"summary": "Supportive summary", "actions": ["First step", "Second step", "Third step"],
                    "executive_summary": "Intake summary", "primary_concerns": "Topics", "interview_prompts": "Questions"}
        with patch("google.genai.Client") as client:
            model = client.return_value.__enter__.return_value.models
            model.generate_content.return_value = MagicMock(text=json.dumps(response))
            result = generate_guidance(DEFAULT_PROFILE, [], "Synthetic text", self.fusion(),
                                       allow_online=True, api_key="test-key")
            self.assertEqual(result["source"], "Google Gemini")
            self.assertFalse(result["fallback"])
            self.assertEqual(model.generate_content.call_count, 1)
            model.generate_content.side_effect = RuntimeError("Synthetic provider failure")
            failed = generate_guidance(DEFAULT_PROFILE, [], "Synthetic text", self.fusion(),
                                       allow_online=True, api_key="test-key")
            self.assertTrue(failed["fallback"])
            self.assertEqual(failed["source"], "Local guidance")

    def test_paired_evaluation_uses_matching_rows(self):
        import pandas as pd
        from evaluate_hybrid import evaluate_paired
        frame = pd.DataFrame([{**DEFAULT_PROFILE, "reflection": "Synthetic A", "depression": 0},
                              {**DEFAULT_PROFILE, "reflection": "Synthetic B", "depression": 1}])
        with patch("evaluate_hybrid.predict", side_effect=[{"prob_dep": 10}, {"prob_dep": 90}]), \
             patch("evaluate_hybrid.predict_text", side_effect=[{"prob_dep": 20}, {"prob_dep": 80}]):
            artifact = evaluate_paired(frame)
        self.assertEqual(artifact["sample_count"], 2)
        self.assertEqual(artifact["metrics"]["Combined"]["accuracy"], 1)
        self.assertNotIn("reflection", artifact)
        with self.assertRaises(ValueError):
            evaluate_paired(frame.drop(columns="reflection"))

    def test_short_journal_preserves_profile_in_brief(self):
        result = local_guidance(DEFAULT_PROFILE, "I feel tired.", self.fusion())
        self.assertIn("Academic pressure:", result["briefing"]["primary_concerns"])
        self.assertIn("Sleep:", result["briefing"]["primary_concerns"])

    def test_signature_tracks_inputs_and_privacy(self):
        p = copy.deepcopy(DEFAULT_PROFILE)
        original = assessment_signature(p, "test", False)
        self.assertNotEqual(original, assessment_signature(p, "test", True))
        p["academic"] = 5
        self.assertNotEqual(original, assessment_signature(p, "test", False))

    def test_snapshot_does_not_alias_input(self):
        profile = copy.deepcopy(DEFAULT_PROFILE)
        result = {"tabular_contrib": None, "fusion_result": self.fusion()}
        with patch("predict.predict_hybrid", return_value=result):
            snapshot = run_assessment(profile, "A quiet week.")
        profile["academic"] = 5
        self.assertEqual(snapshot["profile"]["academic"], 3)
        self.assertIsNone(snapshot["pdf"])

    def test_pdf_defaults_to_bytes_and_preserves_full_reflection(self):
        result = local_guidance(DEFAULT_PROFILE, "A quiet week.", self.fusion())
        with patch("pathlib.Path.write_bytes", side_effect=AssertionError("Unexpected disk write")):
            data = generate_counselor_pdf(DEFAULT_PROFILE, "A long reflection. " * 100,
                                          self.fusion(), result["briefing"])
        self.assertTrue(data.startswith(b"%PDF"))
        self.assertGreater(len(data), 1000)

    def test_demo_cases_are_complete(self):
        for case in DEMO_CASES.values():
            self.assertEqual(set(DEFAULT_PROFILE), set(case["profile"]))
            validate_profile(case["profile"])
            self.assertTrue(case["journal"])

    def test_invalid_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps({"schema_version": 1}))
            with self.assertRaises(ValueError):
                load_paired_evaluation(path)

    @staticmethod
    def fusion():
        return fuse_predictions({"prob_dep": 30}, {"prob_dep": 40})


if __name__ == "__main__":
    unittest.main()

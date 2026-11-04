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
from genai_advisor import generate_guidance, local_guidance, get_setting, guidance_error, student_plan_markdown
from report_generator import generate_counselor_pdf
from support_flags import support_mentions
from research import load_paired_evaluation


def rich_response():
    guidance = local_guidance(DEFAULT_PROFILE, "I feel tired after a demanding week.",
                              fuse_predictions({"prob_dep": 30}, {"prob_dep": 40}))
    return {**{key: guidance[key] for key in ["summary", "perspective", "grounding", "action_details", "support_message", "closing"]},
            **{key: guidance["briefing"][key] for key in ["executive_summary", "primary_concerns", "interview_prompts"]}}


class AssessmentTests(unittest.TestCase):
    def test_cloud_secrets_override_environment(self):
        with patch("streamlit.secrets", {"GEMINI_API_KEY": " cloud-test-key ", "GEMINI_MODEL": "gemini-test"}), \
             patch.dict("os.environ", {"GEMINI_API_KEY": "old-test-key"}):
            self.assertEqual(get_setting("GEMINI_API_KEY"), "cloud-test-key")
            self.assertEqual(get_setting("GEMINI_MODEL"), "gemini-test")

    def test_provider_errors_are_specific_and_never_echo_details(self):
        for status, category in [(400, "request"), (401, "access"), (403, "access"),
                                 (404, "model"), (429, "quota"), (503, "provider")]:
            exc = RuntimeError("private-key-and-journal-must-not-appear")
            exc.code = status
            actual, message, code = guidance_error(exc)
            self.assertEqual(actual, category)
            self.assertEqual(code, status)
            self.assertNotIn("private-key", message)
        self.assertEqual(guidance_error(TimeoutError())[0], "timeout")

    def test_failed_request_logs_only_safe_metadata(self):
        with patch("google.genai.Client") as client, patch("genai_advisor.get_setting", return_value="gemini-test"):
            exc = RuntimeError("private-key-and-journal-must-not-appear")
            exc.code = 429
            client.return_value.__enter__.return_value.models.generate_content.side_effect = exc
            with self.assertLogs("genai_advisor", level="WARNING") as captured:
                result = generate_guidance(DEFAULT_PROFILE, [], "Synthetic text", self.fusion(),
                                           allow_online=True, api_key="test-key")
        self.assertEqual(result["failure_category"], "quota")
        self.assertIn("429", result["fallback_reason"])
        self.assertNotIn("private-key", " ".join(captured.output))

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
        response = rich_response()
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

    def test_server_failure_switches_to_one_fallback_model(self):
        from unittest.mock import MagicMock
        payload = rich_response()
        unavailable = RuntimeError("synthetic unavailable")
        unavailable.code = 503
        with patch("google.genai.Client") as client, patch("genai_advisor.get_setting", side_effect=["gemini-primary", "gemini-backup"]):
            models = client.return_value.__enter__.return_value.models
            models.generate_content.side_effect = [unavailable, MagicMock(text=json.dumps(payload))]
            result = generate_guidance(DEFAULT_PROFILE, [], "Synthetic text", self.fusion(),
                                       allow_online=True, api_key="test-key")
            self.assertEqual(result["source"], "Google Gemini")
            self.assertEqual(result["model"], "gemini-backup")
            self.assertEqual([c.kwargs["model"] for c in models.generate_content.call_args_list],
                             ["gemini-primary", "gemini-backup"])
            retry = client.call_args.kwargs["http_options"].retry_options
            self.assertEqual(retry.attempts, 2)
            self.assertNotIn(429, retry.http_status_codes)

    def test_permanent_errors_never_switch_models(self):
        for status in [400, 401, 403, 429]:
            with self.subTest(status=status), patch("google.genai.Client") as client:
                error = RuntimeError("synthetic error")
                error.code = status
                models = client.return_value.__enter__.return_value.models
                models.generate_content.side_effect = error
                result = generate_guidance(DEFAULT_PROFILE, [], "Synthetic text", self.fusion(),
                                           allow_online=True, api_key="test-key")
                self.assertEqual(models.generate_content.call_count, 1)
                self.assertEqual(result["source"], "Local guidance")

    def test_duplicate_or_disabled_fallback_does_not_repeat_model(self):
        for fallback in ["gemini-primary", "none"]:
            with self.subTest(fallback=fallback), patch("google.genai.Client") as client, \
                 patch("genai_advisor.get_setting", side_effect=["gemini-primary", fallback]):
                error = RuntimeError("synthetic unavailable")
                error.code = 503
                models = client.return_value.__enter__.return_value.models
                models.generate_content.side_effect = error
                result = generate_guidance(DEFAULT_PROFILE, [], "Synthetic text", self.fusion(),
                                           allow_online=True, api_key="test-key")
                self.assertEqual(models.generate_content.call_count, 1)
                self.assertEqual(result["failure_category"], "provider")

    def test_real_sdk_retries_503_without_network(self):
        import httpx
        from google import genai
        from google.genai import types
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(503, json={"error": {"code": 503, "status": "UNAVAILABLE", "message": "Synthetic busy response"}})
            return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": "Synthetic success"}]}, "finishReason": "STOP"}]})

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        options = types.HttpOptions(httpx_client=http_client, retry_options=types.HttpRetryOptions(
            attempts=2, initial_delay=0.001, max_delay=0.001, jitter=0, http_status_codes=[503]))
        with genai.Client(api_key="test-key", http_options=options) as client:
            response = client.models.generate_content(model="gemini-test", contents="Synthetic test")
        self.assertEqual(len(calls), 2)
        self.assertEqual(response.text, "Synthetic success")

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

    def test_guidance_contains_substantive_student_sections(self):
        profile = {**DEFAULT_PROFILE, "academic": 5, "sleep": "5-6 hours", "financial": 5}
        guidance = local_guidance(profile, "I feel exhausted.", self.fusion())
        plan = student_plan_markdown(guidance)
        self.assertGreater(len(plan.split()), 400)
        self.assertIn("academic pressure", guidance["perspective"])
        self.assertIn("shorter sleep", guidance["perspective"])
        self.assertIn("Financial stress", guidance["perspective"])
        self.assertEqual(guidance["actions"], [a["title"] for a in guidance["action_details"]])
        self.assertTrue(all(len(a["steps"]) >= 2 and a["when"] and a["why"] for a in guidance["action_details"]))
        self.assertIn("Finding the words", plan)
        self.assertIn(guidance["support_message"], plan)

    def test_sparse_and_current_support_guidance_remain_appropriate(self):
        sparse = local_guidance(DEFAULT_PROFILE, "", self.fusion())
        self.assertIn("without writing a reflection", sparse["summary"])
        support = local_guidance(DEFAULT_PROFILE, "Synthetic text", {**self.fusion(), "is_critical": True})
        self.assertIn("Right now", support["action_details"][0]["when"])
        self.assertIn("don't feel safe", support["support_message"])
        self.assertNotIn("coursework", support["actions"][0])

    def test_incomplete_or_thin_online_guidance_is_not_published(self):
        from unittest.mock import MagicMock
        for invalid in [{"summary": "A short response"},
                        {**rich_response(), **{f: "Too short" for f in ["summary", "perspective", "grounding", "closing"]},
                         "action_details": [{"title": "Rest", "when": "Today", "why": "Helpful", "steps": ["Pause", "Rest"]}] * 3}]:
            with patch("google.genai.Client") as client:
                models = client.return_value.__enter__.return_value.models
                models.generate_content.return_value = MagicMock(text=json.dumps(invalid))
                guidance = generate_guidance(DEFAULT_PROFILE, [], "Synthetic text", self.fusion(),
                                             allow_online=True, api_key="test-key")
                self.assertEqual(guidance["source"], "Local guidance")
                self.assertEqual(guidance["failure_category"], "response")
                self.assertGreater(len(student_plan_markdown(guidance).split()), 400)

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

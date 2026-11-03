"""Exercise Streamlit reruns using synthetic submissions and no external AI calls."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from streamlit.testing.v1 import AppTest
from assessment import assessment_signature
from fusion import fuse_predictions
from genai_advisor import local_guidance

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


def synthetic_assessment(profile, journal, online=False):
    fusion = fuse_predictions({"prob_dep": 30}, {"prob_dep": 40, "available": bool(journal)})
    return {"id": "synthetic", "assessed_at": "2026-11-04T12:00:00+00:00",
            "signature": assessment_signature(profile, journal, online), "profile": dict(profile),
            "journal": journal, "result": {"fusion_result": fusion, "tabular_contrib": None,
            "token_html": "<p>Sample reflection</p>", "text_result": {}, "comorbid_tags": []},
            "guidance": local_guidance(profile, journal, fusion), "pdf": None,
            "completed_actions": [False, False, False]}


def button(app, label):
    return next(b for b in app.button if b.label == label)


class UITests(unittest.TestCase):
    def test_submission_edit_checklist_report_clear_and_navigation(self):
        with patch("assessment.run_assessment", side_effect=synthetic_assessment) as assess:
            app = AppTest.from_file(APP, default_timeout=20).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.text_area[0].value, "")
            self.assertFalse(app.toggle[0].value)
            app.text_area[0].set_value("A quiet week.").run()
            button(app, "Run assessment").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(assess.call_count, 1)
            app.checkbox[0].check().run()
            self.assertTrue(app.session_state["uf_snapshot"]["completed_actions"][0])
            self.assertEqual(assess.call_count, 1)
            app.slider(key="uf_input_academic").set_value(5).run()
            self.assertTrue(any("Results need updating" in w.value for w in app.warning))
            self.assertEqual(assess.call_count, 1)
            app.session_state["uf_view"] = "Counselor"
            app.run()
            self.assertTrue(button(app, "Prepare counselor report").disabled)
            button(app, "Run assessment").click().run()
            self.assertEqual(assess.call_count, 2)
            with patch("report_generator.generate_counselor_pdf", return_value=b"%PDF-test") as report:
                button(app, "Prepare counselor report").click().run()
                self.assertEqual(report.call_count, 1)
                app.run()
                self.assertEqual(report.call_count, 1)
            self.assertEqual(assess.call_count, 2)
            app.radio(key="uf_nav").set_value("About").run()
            app.radio(key="uf_nav").set_value("Assessment").run()
            self.assertEqual(app.text_area[0].value, "A quiet week.")
            self.assertTrue(app.session_state["uf_snapshot"]["pdf"].startswith(b"%PDF"))
            button(app, "Clear").click().run()
            self.assertNotIn("uf_snapshot", app.session_state)
            self.assertEqual(app.text_area[0].value, "")
            self.assertFalse(app.toggle[0].value)
            self.assertEqual(len(app.exception), 0)

    def test_demo_loads_matching_profile_and_journal_without_submitting(self):
        with patch("assessment.run_assessment") as assess:
            app = AppTest.from_file(APP, default_timeout=20).run()
            app.selectbox(key="uf_demo").set_value("Academic pressure").run()
            button(app, "Load scenario").click().run()
            self.assertEqual(app.slider(key="uf_input_academic").value, 5)
            self.assertIn("exams", app.text_area[0].value)
            self.assertFalse(app.toggle[0].value)
            assess.assert_not_called()
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()

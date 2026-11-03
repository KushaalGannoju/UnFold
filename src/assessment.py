"""Assessment snapshots and demo fixtures, independent of the Streamlit render loop."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from uuid import uuid4

DEFAULT_PROFILE = {"age": 21, "gender": "Female", "cgpa": 7.8, "academic": 3,
                   "study_sat": 3, "work_study": 6, "financial": 3, "sleep": "7-8 hours",
                   "diet": "moderate", "suicid": "No", "family": "No"}
SLEEP_OPTIONS = ["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"]
DEMO_CASES = {
    "Balanced week": {"profile": {**DEFAULT_PROFILE, "academic": 2, "study_sat": 4, "financial": 1},
                      "journal": "I enjoyed classes this week, spent time with friends, and finished my assignments. I feel rested and supported."},
    "Academic pressure": {"profile": {**DEFAULT_PROFILE, "academic": 5, "study_sat": 2, "work_study": 10, "sleep": "5-6 hours"},
                          "journal": "I feel overwhelmed by exams and deadlines. I am exhausted and finding it difficult to concentrate on coursework."},
    "Anxious week": {"profile": {**DEFAULT_PROFILE, "academic": 4, "financial": 4, "sleep": "5-6 hours"},
                     "journal": "I feel anxious and nervous about placements. I keep overthinking what could go wrong and struggle to relax."},
    "Support check-in": {"profile": {**DEFAULT_PROFILE, "academic": 5, "suicid": "Yes", "study_sat": 1, "sleep": "Less than 5 hours"},
                         "journal": "I feel hopeless and alone. I want to end my life and need someone to help me right now."},
}


def validate_profile(profile):
    for field, low, high in [("age", 16, 60), ("cgpa", 0, 10), ("academic", 1, 5),
                             ("study_sat", 1, 5), ("work_study", 0, 16), ("financial", 1, 5)]:
        value = float(profile[field])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{field} must be between {low} and {high}.")
    for field, options in {"gender": ["Female", "Male"], "sleep": SLEEP_OPTIONS,
                           "diet": ["healthy", "moderate", "unhealthy"], "suicid": ["Yes", "No"],
                           "family": ["Yes", "No"]}.items():
        if profile.get(field) not in options:
            raise ValueError(f"Unsupported value for {field}.")


def assessment_signature(profile, journal, online):
    payload = json.dumps({"profile": profile, "journal": journal.strip(), "online": bool(online)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def run_assessment(profile, journal, online=False):
    from predict import predict_hybrid
    from genai_advisor import generate_guidance

    validate_profile(profile)
    if len(journal) > 10000:
        raise ValueError("The journal must be at most 10,000 characters.")
    result = predict_hybrid(profile, journal.strip())
    contributions = result["tabular_contrib"]
    guidance = generate_guidance(profile, contributions.to_dict("records") if contributions is not None else [],
                                 journal.strip(), result["fusion_result"], allow_online=online)
    return {"id": uuid4().hex, "assessed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "signature": assessment_signature(profile, journal, online), "profile": deepcopy(profile),
            "journal": journal.strip(), "result": result, "guidance": guidance,
            "pdf": None, "completed_actions": [False] * len(guidance["actions"])}

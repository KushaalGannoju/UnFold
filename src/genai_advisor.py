"""
genai_advisor.py
─────────────────────────────────────────────────────────────
GenAI Clinical Advisor & Safety Sentinel powered by Gemini API.
Transforms multimodal predictions & SHAP values into personalized,
empathetic student coping plans and clinical counselor dossiers.
Includes an intelligent offline fallback engine.
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()


def get_default_api_key() -> str | None:
    """Retrieve a Gemini key from the environment or Streamlit Cloud secrets."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key

    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None



def generate_student_coping_plan(
    student_profile: dict,
    shap_contributors: list,
    text_reflection: str,
    fusion_result: dict,
    api_key: str = None,
) -> str:
    """
    Generate an empathetic, actionable, CBT-informed coping plan for the student.
    Uses Gemini (3.8 Flash / fallback) if API key is configured, otherwise uses the clinical rule engine.
    """
    key = api_key or get_default_api_key()
    if key and key.strip():
        try:
            return _call_gemini_student_plan(
                student_profile, shap_contributors, text_reflection, fusion_result, key.strip()
            )
        except Exception as e:
            print(f"[GenAI Advisor] Notice: Gemini API call failed ({e}). Utilizing dynamic clinical engine.")

    return _generate_local_student_plan(student_profile, shap_contributors, text_reflection, fusion_result)


def generate_counselor_briefing(
    student_profile: dict,
    shap_contributors: list,
    text_reflection: str,
    fusion_result: dict,
    api_key: str = None,
) -> dict:
    """
    Generate a clinical dossier summary for campus counseling professionals.
    """
    key = api_key or get_default_api_key()
    if key and key.strip():
        try:
            return _call_gemini_counselor_briefing(
                student_profile, shap_contributors, text_reflection, fusion_result, key.strip()
            )
        except Exception as e:
            print(f"[GenAI Advisor] Notice: Gemini briefing call failed ({e}). Utilizing dynamic clinical engine.")

    return _generate_local_counselor_briefing(student_profile, shap_contributors, text_reflection, fusion_result)


# ══════════════════════════════════════════════════════════════
# GEMINI API IMPLEMENTATIONS (with Multi-Model Fallback)
# ══════════════════════════════════════════════════════════════

def _generate_with_gemini(client, prompt: str, max_tokens: int, temperature: float = 0.4) -> str:
    """Try verified production flash models with fallback."""
    from google.genai import types
    candidate_models = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.7-flash",
        "gemini-2.5-flash",
    ]
    last_err = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("Failed to generate content with Gemini.")


def _call_gemini_student_plan(profile, shap_items, text, fusion, api_key: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)

    risk_cat = fusion.get("risk_category", "Moderate Risk")
    risk_score = fusion.get("hybrid_prob_dep", 50.0)
    comorbids = [t["label"] for t in fusion.get("comorbid_tags", [])]

    top_drivers = []
    if shap_items:
        for item in shap_items[:4]:
            top_drivers.append(f"{item.get('feature', '')} ({item.get('direction', '')})")
    drivers_str = ", ".join(top_drivers) if top_drivers else "Academic pressure and routine changes"

    prompt = f"""
You are an expert, compassionate university mental health advisor trained in Cognitive Behavioral Therapy (CBT).
A student has completed a mental health screening combining lifestyle metrics and their personal journal reflection:

--- Student Assessment Data ---
• Overall Estimated Depression Risk: {risk_score}% ({risk_cat})
• Key Contributing Factors (from ML analysis): {drivers_str}
• Secondary Flags: {', '.join(comorbids) if comorbids else 'None'}
• Student's Personal Reflection: "{text}"
• Profile Summary: Age {profile.get('age', 'N/A')}, CGPA {profile.get('cgpa', 'N/A')}, Sleep: {profile.get('sleep', 'N/A')}, Academic Pressure: {profile.get('academic', 'N/A')}/5, Financial Stress: {profile.get('financial', 'N/A')}/5.

--- Task ---
Write a supportive, non-judgmental, and practical coping guide addressed directly to the student ("You").
Structure your response cleanly using Markdown:

### 1. 💭 Validating How You Feel
Acknowledge the specific stressors mentioned in their reflection and metrics empathetically. Normalize seeking balance without minimizing their challenges.

### 2. 🌱 Actionable 3-Step Pathway
Provide 3 concrete, achievable micro-actions tailored to their primary drivers:
- **Step 1 (Immediate - Today)**: A gentle grounding or routine adjustment (e.g. sleep wind-down or boundary setting).
- **Step 2 (Academic & Routine)**: A practical pacing or study adjustment to alleviate pressure without hurting grades.
- **Step 3 (Cognitive Reframing)**: A brief CBT reflection to counter negative self-talk or overwhelm.

### 3. 🤝 Supportive Campus Resources
Encourage reaching out to friends, university counseling centers, or trusted faculty. If the risk is High or Critical, emphasize that asking for help is a sign of strength and provide gentle encouragement to connect with support services.

Tone: Warm, encouraging, grounded, professional. Avoid clinical jargon or definitive diagnoses.
"""
    return _generate_with_gemini(client, prompt, max_tokens=1200, temperature=0.4)


def _call_gemini_counselor_briefing(profile, shap_items, text, fusion, api_key: str) -> dict:
    from google import genai

    client = genai.Client(api_key=api_key)

    risk_cat = fusion.get("risk_category", "Moderate Risk")
    risk_score = fusion.get("hybrid_prob_dep", 50.0)
    comorbids = [t["label"] for t in fusion.get("comorbid_tags", [])]

    prompt = f"""
You are a senior clinical supervisor creating a 1-page structured briefing for a university counselor preparing for an intake session.

Student Metrics:
• Fused Risk: {risk_score}% ({risk_cat})
• Consensus: {fusion.get('consensus_status', 'N/A')}
• Comorbid Indicators: {', '.join(comorbids) if comorbids else 'None'}
• Student Narrative: "{text}"
• Tabular Metrics: Age: {profile.get('age')}, CGPA: {profile.get('cgpa')}, Sleep: {profile.get('sleep')}, Academic Pressure: {profile.get('academic')}/5, Financial Stress: {profile.get('financial')}/5, Suicidal Ideation: {profile.get('suicid')}.

Return a structured text with these exact section headers:
[EXECUTIVE SUMMARY]
(2-3 sentences summarizing presentation)
[PRIMARY CLINICAL CONCERNS]
(Bullet points of main risk drivers from both text and lifestyle data)
[SUGGESTED EXPLORATORY QUESTIONS]
(3-4 open-ended, non-threatening questions for the counselor to open the session)
"""
    raw = _generate_with_gemini(client, prompt, max_tokens=800, temperature=0.2)

    # Parse sections
    def extract_section(header, default_text):
        m = re.search(rf"\[{header}\]\s*(.*?)(?=\[\w|\Z)", raw, re.DOTALL)
        return m.group(1).strip() if m else default_text

    return {
        "executive_summary": extract_section("EXECUTIVE SUMMARY", "Student exhibits noticeable academic and emotional strain."),
        "primary_concerns": extract_section("PRIMARY CLINICAL CONCERNS", "• Academic pressure\n• Sleep disruption"),
        "interview_prompts": extract_section("SUGGESTED EXPLORATORY QUESTIONS", "• How have your energy levels been throughout the week?\n• What parts of coursework feel most overwhelming right now?"),
        "raw_text": raw,
    }


# ══════════════════════════════════════════════════════════════
# DYNAMIC LOCAL CLINICAL FALLBACK ENGINE (100% OFFLINE)
# ══════════════════════════════════════════════════════════════

def _generate_local_student_plan(profile, shap_items, text, fusion) -> str:
    risk_cat = fusion.get("risk_category", "Moderate Risk")
    risk_score = fusion.get("hybrid_prob_dep", 50.0)
    comorbids = [t["label"] for t in fusion.get("comorbid_tags", [])]
    tag_str = f" (**{', '.join(comorbids)}**)" if comorbids else ""

    acad_press = float(profile.get("academic", 3) or 3)
    fin_stress = float(profile.get("financial", 2) or 2)
    sleep_str = str(profile.get("sleep", "5-6 hours"))

    # Tailor based on risk category
    if "Low" in risk_cat or "Minimal" in risk_cat:
        val_text = f"Your current assessment reflects a **{risk_cat} ({risk_score}%)** profile{tag_str}. Your current habits show positive stability and resilience. Maintaining this equilibrium requires conscious pacing so you stay energized throughout the academic term."
        step1 = "- **Celebrate Small Wins**: Acknowledge your consistent routine and give yourself credit for navigating college demands calmly."
        step2 = "- **Sustained Boundary Management**: Protect your downtime even during peak project cycles so stress doesn't accumulate unnoticed."
        step3 = "- **Preventive Buffer**: Keep up healthy sleep and social interactions before exam schedules tighten."
    else:
        # Dynamic steps based on actual student lifestyle flags
        val_text = f"Your assessment indicates a **{risk_cat} ({risk_score}%)** mental health profile{tag_str}. Navigating rigorous academic expectations alongside personal health is deeply challenging. Your feelings are completely valid, and support is available."

        if "Less than 5" in sleep_str or "5-6" in sleep_str:
            step1 = f"- **Circadian Reset (Sleep reported: {sleep_str})**: Sleep deprivation amplifies cortisol and anxiety. Establish a firm 45-minute digital curfew before bed to allow your nervous system to wind down."
        else:
            step1 = "- **Micro-Decompression Breaks**: Take two 10-minute pauses during the day away from screens, focusing on slow diaphragmatic breathing (4s inhale, 6s exhale)."

        if acad_press >= 4:
            step2 = f"- **Academic Task Chunking (Pressure rated {int(acad_press)}/5)**: The sheer volume of work can feel paralyzing. Break study units into 25-minute Pomodoro sprints and focus strictly on today's single highest priority."
        elif fin_stress >= 4:
            step2 = f"- **Financial & Resource Pacing (Stress rated {int(fin_stress)}/5)**: Financial worries take a heavy cognitive toll. Explore university emergency grants or student aid, and speak with a campus advisor about available resources."
        else:
            step2 = "- **The 'Rule of Three'**: Select only three manageable tasks to accomplish each day. De-couple your self-worth from temporary grades or feedback."

        step3 = "- **Cognitive Reframing (CBT)**: When feelings of inadequacy arise, remind yourself: *'I am facing a heavy workload right now, but my capacity is built step by step, not all at once.'*"

    markdown = f"""
### 💭 Understanding Your Current State
{val_text}

---
### 🌱 Actionable 3-Step Pathway

#### 1. Immediate Reset (Restoring Autonomic Balance)
{step1}

#### 2. Academic & Workload Pacing (Cognitive De-escalation)
{step2}

#### 3. Cognitive Reframing & Behavioral Support
{step3}

---
### 🤝 Campus & Support Resources
If you find yourself feeling persistently overwhelmed or losing motivation, reaching out to a professional counselor or trusted mentor can offer tremendous relief.

> **Helplines (India):**
> • **Tele-MANAS (Govt of India 24/7):** 14416 / 1800-891-4416
> • **iCall:** 9152987821
> • **Vandrevala Foundation:** 9999 666 555
"""
    return markdown


def _generate_local_counselor_briefing(profile, shap_items, text, fusion) -> dict:
    risk_cat = fusion.get("risk_category", "Moderate Risk")
    risk_score = fusion.get("hybrid_prob_dep", 50.0)
    comorbids = [t["label"] for t in fusion.get("comorbid_tags", [])]

    acad_press = profile.get("academic", "N/A")
    fin_stress = profile.get("financial", "N/A")
    sleep_str = profile.get("sleep", "N/A")

    exec_summary = (
        f"Student presents with an estimated multimodal depression risk of {risk_score}% ({risk_cat}). "
        f"Consensus status: {fusion.get('consensus_status', 'Concordant')}. "
        f"Secondary indicators: {', '.join(comorbids) if comorbids else 'None highlighted'}."
    )

    primary_concerns = (
        f"• Academic Load: Pressure rating {acad_press}/5, Satisfaction rating {profile.get('study_sat', 'N/A')}/5\n"
        f"• Sleep Architecture: {sleep_str} reported\n"
        f"• Financial & Environmental Stressors: Rating {fin_stress}/5\n"
        f"• Narrative Text Snippet: \"{text[:140]}...\"" if len(text) > 140 else f"• Narrative Text Snippet: \"{text}\""
    )

    # Dynamic exploratory questions based on student data
    q_sleep = f"\"You noted sleeping {sleep_str} — how has fatigue been impacting your concentration and mood lately?\""
    q_acad = f"\"With academic pressure at {acad_press}/5, which specific subject or deadline is feeling the most daunting right now?\""
    q_coping = "\"When you feel the weight of these demands, what has historically helped you recharge?\""
    q_text = f"\"In your journal reflection, you mentioned feelings around '{text[:45]}...' — could you elaborate on what led up to that?\"" if len(text) > 15 else "\"What is one small change that would take the most pressure off your week?\""

    interview_prompts = f"1. {q_acad}\n2. {q_sleep}\n3. {q_coping}\n4. {q_text}"

    return {
        "executive_summary": exec_summary,
        "primary_concerns": primary_concerns,
        "interview_prompts": interview_prompts,
        "raw_text": f"{exec_summary}\n\n{primary_concerns}\n\n{interview_prompts}",
    }

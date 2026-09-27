"""
report_generator.py
─────────────────────────────────────────────────────────────
Generates a downloadable, professional 1-page Clinical Counselor
Assessment Dossier (PDF) using fpdf2.
"""

import os
import re
from datetime import datetime
from fpdf import FPDF


def sanitize_pdf_text(text: str) -> str:
    """Sanitize Unicode characters for standard PDF core fonts (Latin-1 safe)."""
    if not text:
        return ""
    replacements = {
        "\u2014": "--",   # em-dash
        "\u2013": "-",    # en-dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2022": "*",    # bullet
        "\u2026": "...",  # ellipsis
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    # Strip non-latin1 characters (like emojis)
    return text.encode("latin-1", "ignore").decode("latin-1")


class ClinicalDossierPDF(FPDF):
    def header(self):
        # Header banner (UnFold Deep Blue)
        self.set_fill_color(15, 35, 75)
        self.rect(0, 0, 210, 24, "F")

        # Try embedding UnFold logo
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logo_path = os.path.join(base_dir, "assets", "unfold_logo_white.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(base_dir, "assets", "unfold_icon.png")

        if os.path.exists(logo_path):
            try:
                self.image(logo_path, x=175, y=3, h=18)
            except Exception:
                pass

        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 5)
        self.cell(160, 8, "UnFold -- STUDENT CLINICAL ASSESSMENT DOSSIER", ln=True, align="L")

        self.set_font("Helvetica", "", 9)
        self.set_text_color(219, 234, 254)
        self.set_xy(10, 13)
        self.cell(160, 6, "Multimodal Mental Health Intelligence & Clinical Triage System (Dual-Engine)", ln=True, align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-18)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(
            0,
            5,
            sanitize_pdf_text("UnFold Confidential Clinical Intake Aid -- For authorized counseling staff only. Not a diagnostic tool."),
            ln=True,
            align="C",
        )
        self.cell(0, 4, f"Page {self.page_no()}", align="C")


def generate_counselor_pdf(
    student_profile: dict,
    text_reflection: str,
    fusion_result: dict,
    counselor_briefing: dict,
    output_path: str = None,
) -> str:
    """
    Build and export a structured 1-page PDF dossier for university counselors.
    Returns:
        output_path: Absolute path to the generated PDF file.
    """
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reports_dir = os.path.join(base_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(reports_dir, f"Counselor_Dossier_{timestamp}.pdf")

    pdf = ClinicalDossierPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Metadata Row
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.set_xy(10, 27)
    now_str = datetime.now().strftime("%d %b %Y, %I:%M %p")
    pdf.cell(100, 5, f"Date of Screening: {now_str}")
    pdf.cell(90, 5, "Institution: MANIT Student Health Support", align="R", ln=True)

    # Divider line
    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.4)
    pdf.line(10, 33, 200, 33)

    pdf.set_y(36)

    # 1. MULTIMODAL RISK SUMMARY BOX
    risk_cat = fusion_result.get("risk_category", "Moderate Risk")
    hybrid_score = fusion_result.get("hybrid_prob_dep", 50.0)
    tab_score = fusion_result.get("tabular_prob_dep", 50.0)
    text_score = fusion_result.get("text_prob_dep", 50.0)

    # Box background
    if hybrid_score >= 75:
        pdf.set_fill_color(254, 242, 242)
        pdf.set_draw_color(248, 113, 113)
        tag_color = (220, 38, 38)
    elif hybrid_score >= 50:
        pdf.set_fill_color(255, 247, 237)
        pdf.set_draw_color(251, 146, 60)
        tag_color = (234, 88, 12)
    else:
        pdf.set_fill_color(240, 253, 244)
        pdf.set_draw_color(134, 239, 172)
        tag_color = (22, 163, 74)

    pdf.rect(10, 36, 190, 26, "DF")

    pdf.set_xy(15, 39)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*tag_color)
    pdf.cell(90, 6, sanitize_pdf_text(f"HYBRID CONSENSUS: {risk_cat.upper()} ({hybrid_score:.1f}%)"))

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    pdf.set_xy(15, 46)
    status_str = sanitize_pdf_text(fusion_result.get("consensus_status", "Concordant"))
    pdf.cell(
        180,
        5,
        f"Engine A (Tabular Lifestyle): {tab_score:.1f}%  |  Engine B (Semantic NLP): {text_score:.1f}%  |  Status: {status_str}",
        ln=True,
    )

    comorbids = [t["label"] for t in fusion_result.get("comorbid_tags", [])]
    comorbid_str = sanitize_pdf_text(", ".join(comorbids)) if comorbids else "None Highlighted"
    pdf.set_xy(15, 52)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(180, 5, f"Secondary Comorbid Indicators: {comorbid_str}", ln=True)

    # 2. STUDENT LIFESTYLE & ACADEMIC ATTRIBUTES
    pdf.set_y(66)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, "1. Student Profile & Core Lifestyle Markers", ln=True)

    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_fill_color(248, 250, 252)
    pdf.rect(10, 72, 190, 20, "F")

    pdf.set_xy(13, 74)
    c1 = (
        f"Age: {student_profile.get('age', 'N/A')}  |  "
        f"Gender: {student_profile.get('gender', 'N/A')}  |  "
        f"CGPA: {student_profile.get('cgpa', 'N/A')}  |  "
        f"Sleep Duration: {student_profile.get('sleep', 'N/A')}"
    )
    pdf.cell(180, 5, sanitize_pdf_text(c1), ln=True)

    pdf.set_xy(13, 80)
    c2 = (
        f"Academic Pressure: {student_profile.get('academic', 'N/A')}/5  |  "
        f"Financial Stress: {student_profile.get('financial', 'N/A')}/5  |  "
        f"Study Satisfaction: {student_profile.get('study_sat', 'N/A')}/5  |  "
        f"Diet: {str(student_profile.get('diet', 'N/A')).title()}"
    )
    pdf.cell(180, 5, sanitize_pdf_text(c2), ln=True)

    pdf.set_xy(13, 86)
    suicide_val = student_profile.get("suicid", "No")
    suicide_str = "ALERT: YES" if str(suicide_val).lower() in ["yes", "1"] else "Reported No"
    c3 = f"Suicidal Ideation History: {suicide_str}  |  Family History of Mental Illness: {student_profile.get('family', 'No')}"
    pdf.cell(180, 5, sanitize_pdf_text(c3), ln=True)

    # 3. VERBATIM STUDENT NARRATIVE
    pdf.set_y(96)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, "2. Student Verbatim Reflection & Psychological Narrative", ln=True)

    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(71, 85, 105)
    pdf.set_fill_color(255, 255, 255)
    pdf.set_draw_color(203, 213, 225)

    clean_reflection = sanitize_pdf_text(text_reflection.strip()) if text_reflection else "No self-reflection submitted."
    if len(clean_reflection) > 360:
        clean_reflection = clean_reflection[:360] + "..."

    pdf.set_xy(10, 102)
    pdf.multi_cell(190, 4.8, f'"{clean_reflection}"', border=1, fill=True)

    # 4. CLINICAL INTAKE RECOMMENDATIONS & EXPLORATORY PROMPTS
    pdf.set_y(pdf.get_y() + 4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, "3. Clinical Assessment & Recommended Intake Prompts", ln=True)

    # Executive Summary text
    exec_summary = sanitize_pdf_text(counselor_briefing.get("executive_summary", ""))
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(190, 4.5, exec_summary)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 5, "Suggested Exploratory Questions for 1-on-1 Session:", ln=True)

    pdf.set_font("Helvetica", "", 8.2)
    pdf.set_text_color(51, 65, 85)
    prompts = sanitize_pdf_text(counselor_briefing.get("interview_prompts", ""))
    for line in prompts.split("\n"):
        if line.strip():
            pdf.cell(5)
            pdf.multi_cell(185, 4.2, line.strip())

    # 5. EMERGENCY CONTACT FOOTNOTE IF HIGH/CRITICAL RISK
    if hybrid_score >= 50 or fusion_result.get("is_critical"):
        pdf.set_y(pdf.get_y() + 3)
        pdf.set_fill_color(254, 242, 242)
        pdf.set_draw_color(239, 68, 68)
        curr_y = pdf.get_y()
        if curr_y > 270:
            pdf.add_page()
            curr_y = 30
        pdf.rect(10, curr_y, 190, 13, "DF")
        pdf.set_xy(13, curr_y + 2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(185, 28, 28)
        pdf.cell(180, 4, "PRIORITY PROTOCOL: Elevated Risk Detected", ln=True)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(127, 29, 29)
        pdf.cell(180, 4, "Follow standard institutional triage guidelines. Tele-MANAS: 14416 | iCall: 9152987821 | Vandrevala: 9999 666 555", ln=True)

    pdf.output(output_path)
    return output_path

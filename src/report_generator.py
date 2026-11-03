"""Generate counselor reports in memory; disk output is an explicit caller choice."""

from datetime import datetime, timezone
from pathlib import Path
from fpdf import FPDF
import matplotlib


def generate_counselor_pdf(student_profile, text_reflection, fusion_result, counselor_briefing,
                           output_path=None, assessed_at=None):
    pdf = FPDF()
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdf.add_font("DejaVu", fname=str(font_dir / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", style="B", fname=str(font_dir / "DejaVuSans-Bold.ttf"))
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(16, 16, 16)
    pdf.add_page()

    def body(text, bold=False, size=10):
        pdf.set_font("DejaVu", "B" if bold else "", size)
        pdf.set_text_color(35, 48, 58)
        pdf.multi_cell(0, 6, str(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    def section(title, text):
        pdf.ln(3)
        body(title, bold=True, size=11)
        body(text)

    logo = Path(__file__).resolve().parents[1] / "assets" / "unfold_icon.png"
    if logo.exists():
        pdf.image(str(logo), x=177, y=15, w=15)
    body("UnFold", bold=True, size=22)
    body("Counselor intake brief", bold=True, size=12)
    body("Assessment: " + (assessed_at or datetime.now(timezone.utc).isoformat()))
    body("Educational screening aid. Model scores are uncalibrated signals, not diagnoses or personal risk probabilities.", size=9)
    f = fusion_result
    journal_score = f.get("text_prob_dep")
    section("Screening summary", f"Combined model score: {f['hybrid_prob_dep']:.1f}/100\n"
            f"Profile model: {f['tabular_prob_dep']:.1f}/100\n"
            + (f"Journal model: {journal_score:.1f}/100\n" if journal_score is not None else "Journal model: not assessed\n")
            + f["consensus_note"])
    if f.get("is_critical"):
        section("Support check-in", "Potential current support-related language was flagged. Check in directly and follow institutional support procedures. The flag does not alter the model score.")
    elif f.get("support_review"):
        section("Support check-in", "Past history or contextual language was reported. Clarify the student's current circumstances; this is not evidence of an immediate crisis.")
    section("Student profile", "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in student_profile.items()))
    section("Submitted reflection", text_reflection.strip() or "No journal submitted.")
    for key, title in [("executive_summary", "Intake summary"), ("primary_concerns", "Conversation topics"),
                       ("interview_prompts", "Exploratory questions")]:
        section(title, counselor_briefing.get(key, ""))
    body("Confidential student-provided information. Share only with the student's permission.", size=8)
    data = bytes(pdf.output())
    if output_path is not None:
        Path(output_path).write_bytes(data)
        return str(output_path)
    return data

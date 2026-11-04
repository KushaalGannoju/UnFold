"""UnFold session-based screening workspace. Run: streamlit run app/app.py"""

import base64
from datetime import datetime
from html import escape
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))
from assessment import DEFAULT_PROFILE, SLEEP_OPTIONS, DEMO_CASES, assessment_signature, run_assessment
from genai_advisor import get_default_api_key, student_plan_markdown
from predict import load_bundle
from report_generator import generate_counselor_pdf
from research import benchmark_rows, load_paired_evaluation
from resources import SUPPORT_RESOURCE, SUPPORT_WEBSITE
from support_flags import support_mentions
from utils import format_feature_name

st.set_page_config(page_title="UnFold | Student wellbeing", page_icon=str(BASE_DIR / "assets/unfold_icon.png"),
                   layout="wide", initial_sidebar_state="auto")
st.markdown("""<style>
html, body, button, input, textarea {letter-spacing:0 !important;}
.block-container {max-width:1160px; padding-top:4.5rem; padding-bottom:3rem;}
[data-testid="stSidebar"] {min-width:290px; max-width:310px; background:#f6f2fb; border-right:1px solid #e7def0;}
[data-testid="stSidebar"] .block-container {padding-top:1.5rem;}
h1 {font-size:1.8rem !important; padding-top:0 !important;}
h2 {font-size:1.25rem !important;}
h3 {font-size:1.05rem !important;}
.brand {display:flex; align-items:center; gap:14px; margin-bottom:1.4rem;}
.brand img {height:46px; width:46px; object-fit:contain;}
.brand h1 {margin:0; padding:0; line-height:1.2;}
.brand p {margin:4px 0 0; color:#71667f; font-size:0.92rem;}
.brand > div {min-width:0;}
.session-note {margin-left:auto; color:#71667f; font-size:0.78rem; white-space:nowrap;}
.guidance-copy {max-width:760px; font-size:1rem; line-height:1.85; color:#443c50;}
.guidance-copy p {margin:0 0 1rem; line-height:1.85; overflow-wrap:anywhere;}
.guidance-opening {border-left:3px solid #c9b5e5; padding-left:20px; margin:18px 0 28px;}
.st-key-uf_grounding {background:#f0f6f2; border-color:#d8e6dc !important;}
.st-key-uf_grounding p {color:#3e5b49;}
.action-timing {font-size:0.78rem; color:#7955ad; font-weight:600; margin-top:18px; margin-bottom:6px;}
[data-testid="stCheckbox"] label p {font-weight:600;}
[data-testid="stMetricValue"] {font-size:1.5rem;}
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {border-radius:6px; min-height:40px;}
[data-testid="stTextArea"] textarea {line-height:1.65; border-radius:6px;}
[data-testid="stExpander"] {border-radius:6px; box-shadow:none;}
[data-testid="stCode"] {border:1px solid #e7def0; border-radius:6px;}
@media(max-width:768px) {
 .block-container {padding:4.5rem 1rem 2rem;}
 .brand {gap:10px; flex-wrap:nowrap; margin-bottom:0.7rem;}
 .brand img {height:36px; width:36px;}
 .brand h1 {font-size:1.5rem !important;}
 .session-note {display:none;}
 [data-testid="stSidebar"] {min-width:270px; max-width:290px;}
}
</style>
""", unsafe_allow_html=True)


def init_state():
    for field, value in DEFAULT_PROFILE.items():
        st.session_state.setdefault("uf_input_" + field, value)
    st.session_state.setdefault("uf_journal", st.session_state.get("uf_draft_journal", ""))
    st.session_state.setdefault("uf_online", st.session_state.get("uf_draft_online", False))
    st.session_state.setdefault("uf_view", st.session_state.get("uf_saved_view", "Student"))


def save_draft():
    st.session_state["uf_draft_journal"] = st.session_state["uf_journal"]
    st.session_state["uf_draft_online"] = st.session_state["uf_online"]


def save_view():
    st.session_state["uf_saved_view"] = st.session_state.get("uf_view") or "Student"


def clear_assessment():
    for key in list(st.session_state):
        if key.startswith("uf_") and key not in {"uf_nav", "uf_view"}:
            del st.session_state[key]
    init_state()


def load_demo():
    choice = st.session_state.get("uf_demo")
    if choice in DEMO_CASES:
        case = DEMO_CASES[choice]
        for key, value in case["profile"].items():
            st.session_state["uf_input_" + key] = value
        st.session_state["uf_journal"] = case["journal"]
        st.session_state["uf_draft_journal"] = case["journal"]
        st.session_state["uf_online"] = False
        st.session_state["uf_draft_online"] = False
        st.session_state["uf_loaded_demo"] = choice


@st.cache_resource(show_spinner=False)
def model_bundle():
    return load_bundle()


def render_profile():
    with st.sidebar:
        st.image(str(BASE_DIR / "assets/unfold_logo_transparent.png"), width=110)
        st.subheader("Student profile")
        with st.expander("Demo examples"):
            st.selectbox("Scenario", list(DEMO_CASES), key="uf_demo")
            st.button("Load scenario", icon=":material/science:", on_click=load_demo, width="stretch")
        st.markdown("**Personal details**")
        a, b = st.columns(2)
        a.number_input("Age", min_value=16, max_value=60, step=1, key="uf_input_age")
        b.selectbox("Gender", ["Female", "Male"], key="uf_input_gender")
        st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1, key="uf_input_cgpa")
        st.markdown("**Study & routine**")
        st.slider("Academic pressure", 1, 5, key="uf_input_academic", help="1: Low; 5: Very high")
        st.slider("Study satisfaction", 1, 5, key="uf_input_study_sat", help="1: Very dissatisfied; 5: Very satisfied")
        st.slider("Daily study / work hours", 0, 16, key="uf_input_work_study")
        st.slider("Financial stress", 1, 5, key="uf_input_financial", help="1: Low; 5: Very high")
        st.selectbox("Sleep duration", SLEEP_OPTIONS, key="uf_input_sleep")
        st.selectbox("Dietary habits", ["healthy", "moderate", "unhealthy"], format_func=str.title, key="uf_input_diet")
        with st.expander("Personal history", expanded=True):
            st.radio("Past suicidal thoughts", ["No", "Yes"], horizontal=True, key="uf_input_suicid",
                     help="Past history is recorded separately from a current support flag.")
            st.radio("Family mental-health history", ["No", "Yes"], horizontal=True, key="uf_input_family")
    return {k: st.session_state["uf_input_" + k] for k in DEFAULT_PROFILE}


def render_support(profile, journal):
    flags = support_mentions(journal)
    if any(t["is_critical"] for t in flags):
        st.error("Your reflection may describe a current need for support. Reach out to someone you trust or a qualified professional. If you feel unable to stay safe, seek immediate in-person emergency help.")
        st.markdown(SUPPORT_RESOURCE)
    elif flags or profile["suicid"] == "Yes":
        st.info("Past history or support-related context is present. A direct conversation can help clarify how you are feeling now.")


def render_explanations(snapshot):
    result = snapshot["result"]
    f = result["fusion_result"]
    combined, a, b = st.columns(3)
    combined.metric("Combined model", f"{f['hybrid_prob_dep']:.1f}/100")
    a.metric("Profile model", f"{f['tabular_prob_dep']:.1f}/100")
    b.metric("Journal model", f"{f['text_prob_dep']:.1f}/100" if f["text_prob_dep"] is not None else "Not assessed")
    st.caption("A screening signal, not a diagnosis or an individual probability of depression.")
    st.write(f["consensus_note"])
    st.caption(f"Profile weight {f['weights']['tabular']:.0%} · Journal weight {f['weights']['text']:.0%}. Scores are uncalibrated.")
    st.markdown("#### Profile contributions")
    contrib = result["tabular_contrib"]
    if contrib is not None and not contrib.empty:
        values = contrib.sort_values("shap_value")
        fig, ax = plt.subplots(figsize=(8, 3.6))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.barh(values.feature.map(format_feature_name), values.shap_value,
                color=["#9180b2" if v < 0 else "#bc895d" for v in values.shap_value])
        ax.axvline(0, color="#8b9aa2", linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlabel("SHAP contribution to the model output (log-odds)", fontsize=9)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    else:
        st.caption("A contribution explanation is unavailable for this assessment.")
    st.markdown("#### Journal influence")
    st.caption("Highlights show sensitivity of the model output to tokens, not evidence that a word indicates distress.")
    st.markdown(result["token_html"], unsafe_allow_html=True)
    if result["text_result"].get("truncated"):
        st.warning("The journal exceeded the model's 256-token window. Prediction and highlighting use the same initial window; the full reflection remains in the report.")
    tags = result["comorbid_tags"]
    if tags:
        st.markdown("#### Language flags for review")
        for tag in tags:
            st.write(f"**{tag['category']}** — {tag['description']}")
        st.caption("Phrase rules can miss context. These are conversation prompts, not diagnoses.")


def render_prose(text, opening=False):
    paragraphs = "".join("<p>" + escape(paragraph).replace("\n", "<br/>") + "</p>"
                         for paragraph in text.split("\n\n") if paragraph.strip())
    css_class = "guidance-copy guidance-opening" if opening else "guidance-copy"
    st.markdown(f"<div class='{css_class}'>{paragraphs}</div>", unsafe_allow_html=True)


def render_results(snapshot, stale, counselor):
    result, guidance = snapshot["result"], snapshot["guidance"]
    fusion = result["fusion_result"]
    st.divider()
    st.subheader("A little space for you")
    if stale:
        st.warning("Results need updating. The summary below belongs to your last submitted assessment.")
    submitted = datetime.fromisoformat(snapshot["assessed_at"]).astimezone(ZoneInfo("Asia/Kolkata"))
    st.caption("Submitted " + submitted.strftime("%d %b %Y, %I:%M %p IST") + " · " + guidance["source"])
    if guidance.get("fallback"):
        st.caption(guidance["fallback_reason"])
    render_prose(guidance["summary"], opening=True)
    if guidance.get("perspective"):
        st.markdown("#### Making sense of this moment")
        render_prose(guidance["perspective"])
    if guidance.get("grounding"):
        with st.container(border=True, key="uf_grounding"):
            st.markdown("#### A moment to pause")
            render_prose(guidance["grounding"])
    st.markdown("#### A few steps to choose from")
    st.caption("One small step is enough for today. This is not a measure of your progress or worth.")
    details = guidance.get("action_details", [])
    for i, action in enumerate(guidance["actions"]):
        if i < len(details):
            st.markdown(f"<div class='action-timing'>{escape(details[i]['when'])}</div>", unsafe_allow_html=True)
        key = f"uf_action_{snapshot['id']}_{i}"
        st.session_state.setdefault(key, snapshot["completed_actions"][i])
        snapshot["completed_actions"][i] = st.checkbox(action, key=key, disabled=stale)
        if i < len(details):
            render_prose(details[i]["why"])
            for step in details[i]["steps"]:
                render_prose("• " + step)
    completed = sum(snapshot["completed_actions"])
    if completed:
        st.caption(f"{completed} step{'s' if completed != 1 else ''} tried. You can leave the others for another day.")
    if guidance.get("support_message"):
        st.markdown("#### Finding the words")
        st.caption("A message you can adapt for someone you trust.")
        st.code(guidance["support_message"], language=None, wrap_lines=True)
    if guidance.get("closing"):
        st.markdown("#### Take this at your pace")
        render_prose(guidance["closing"])
    st.download_button("Keep a copy of your plan", student_plan_markdown(guidance).encode("utf-8"),
                       file_name=f"UnFold_plan_{snapshot['id'][:8]}.md", mime="text/markdown",
                       icon=":material/download:", on_click="ignore", disabled=stale)
    with st.expander("Model evidence & explanations", expanded=counselor):
        render_explanations(snapshot)
    if counselor:
        st.subheader("Counselor intake")
        briefing = guidance["briefing"]
        for key, title in [("executive_summary", "Summary"), ("primary_concerns", "Conversation topics"),
                           ("interview_prompts", "Exploratory questions")]:
            st.markdown("**" + title + "**")
            st.text(briefing[key])
        with st.expander("Submitted profile & journal"):
            st.dataframe(pd.DataFrame([{"Field": k.replace("_", " ").title(), "Value": str(v)}
                                       for k, v in snapshot["profile"].items()]), hide_index=True, width="stretch")
            st.text(snapshot["journal"] or "No journal submitted.")
        if snapshot["pdf"] is None:
            if st.button("Prepare counselor report", icon=":material/picture_as_pdf:", disabled=stale):
                try:
                    snapshot["pdf"] = generate_counselor_pdf(snapshot["profile"], snapshot["journal"], fusion,
                                                             briefing, assessed_at=snapshot["assessed_at"])
                except Exception:
                    st.error("The report could not be prepared. Your assessment is still available.")
        if snapshot["pdf"] is not None:
            st.download_button("Download counselor report", snapshot["pdf"],
                               file_name=f"UnFold_{snapshot['id'][:8]}.pdf", mime="application/pdf",
                               icon=":material/download:", on_click="ignore", disabled=stale)


def render_assessment(profile):
    heading, mode = st.columns([2, 1])
    heading.subheader("How has your week been?")
    with mode:
        view = st.segmented_control("View", ["Student", "Counselor"], key="uf_view",
                                    label_visibility="collapsed", on_change=save_view) or "Student"
    if st.session_state.get("uf_loaded_demo"):
        st.caption("Demo scenario: " + st.session_state["uf_loaded_demo"])
    journal = st.text_area("Your reflection (optional)", key="uf_journal", height=150, max_chars=10000,
                           placeholder="What has been on your mind?", help="Leave blank to use your profile only.", on_change=save_draft)
    online = st.toggle("Use online AI guidance", key="uf_online",
                       help="When enabled, submitting sends your profile, journal, model results, and contributions to Google Gemini.", on_change=save_draft)
    if online:
        st.caption("Your profile and reflection will be sent to Google for guidance. Google's processing and retention policies apply.")
        if not get_default_api_key():
            st.info("No online AI key is configured. This assessment will use local guidance.")
    else:
        st.caption("Local guidance · Profile and journal stay in this app's server session; they are not sent to Google.")
    run, clear = st.columns([3, 1])
    clicked = run.button("Run assessment", icon=":material/arrow_forward:", type="primary", width="stretch")
    clear.button("Clear", icon=":material/restart_alt:", on_click=clear_assessment,
                 width="stretch", help="Clear this session's inputs, result, checklist, and report.")
    render_support(profile, journal)
    if clicked:
        try:
            with st.spinner("Assessing your profile and reflection..."):
                snapshot = run_assessment(profile, journal, online)
            for key in list(st.session_state):
                if key.startswith("uf_action_"):
                    del st.session_state[key]
            st.session_state["uf_snapshot"] = snapshot
        except Exception:
            st.error("The assessment could not finish. Check that the model files and dependencies are installed, then try again.")
    snapshot = st.session_state.get("uf_snapshot")
    if snapshot:
        stale = snapshot["signature"] != assessment_signature(profile, journal, online)
        render_results(snapshot, stale, view == "Counselor")


def render_research():
    st.subheader("Research & model evidence")
    bundle = model_bundle()
    st.markdown("#### Saved tabular evaluation")
    st.caption("Metrics below come from the deployed model bundle. Baselines used a smaller training subset; these are not equal-budget comparisons.")
    rows = benchmark_rows(bundle)
    if rows:
        table = pd.DataFrame(rows)
        for col in ["Accuracy", "Precision", "Recall", "F1"]:
            table[col] = table[col].map(lambda v: f"{v:.2%}")
        st.dataframe(table, hide_index=True, width="stretch")
    st.markdown("#### Paired multimodal evaluation")
    try:
        evaluation = load_paired_evaluation()
    except (ValueError, OSError):
        evaluation = None
        st.warning("The paired evaluation artifact could not be read.")
    if evaluation:
        st.caption(f"{evaluation['sample_count']} paired samples · {evaluation.get('evaluated_at', '')}")
        st.write(evaluation.get("split_description", ""))
        st.dataframe(pd.DataFrame(evaluation["metrics"]).T, width="stretch")
    else:
        st.info("Journal and combined benchmark scores are not yet verified on a paired held-out dataset. No hybrid accuracy or recall claim is published here.")
    st.caption("Fusion uses fixed 45% profile / 55% journal weights. Without a journal it uses the profile only. Scores have not been fitted to a probability calibrator.")
    st.markdown("#### Evaluation gallery")
    plots = [("confusion_matrix.png", "Saved tabular confusion matrix"),
             ("cv_comparison.png", "Tabular cross-validation"),
             ("roberta_confusion_matrix.png", "Historical journal evaluation image; not a paired validation"),
             ("roberta_training_epochs.png", "Historical journal training run")]
    for index in range(0, len(plots), 2):
        for column, (name, caption) in zip(st.columns(2), plots[index:index + 2]):
            path = BASE_DIR / "plots" / name
            if path.exists():
                column.image(str(path), caption=caption, width="stretch")


def render_about():
    st.subheader("About UnFold")
    st.write("UnFold combines a student profile model and a journal language model to support reflection and counselor preparation. It is an academic project from MANIT Bhopal, not a clinical diagnostic service.")
    st.markdown("#### Your data")
    st.write("Inputs, results, action choices, and prepared reports are held in this browser's server session. The application does not write new counselor PDFs to disk. Clear removes the assessment from session state; it does not retract information already sent to Google or remove downloads from your device.")
    st.write("Online guidance is off by default. If enabled for an assessment, the app sends the submitted profile, journal, model summary and leading contributions to Google Gemini. If unavailable, local guidance is used and labeled accordingly.")
    st.markdown("#### Model limits")
    st.write("The journal model uses an initial 256-token window. Phrase flags are context-sensitive heuristics that may miss or misread statements. Model agreement is not clinical certainty. A higher or lower score cannot establish a diagnosis or exclude a need for support.")
    st.markdown("#### Support")
    st.markdown(SUPPORT_RESOURCE + " · " + SUPPORT_WEBSITE)
    st.markdown("#### Project team")
    st.write("Kushaal Sai Gannoju · Himanshu Verma · Ritika Rani Ujjainia · Rayyan Zameer Baikadi")
    st.caption("Department of Computer Science & Engineering, MANIT Bhopal · Minor Project II")


def main():
    init_state()
    profile = render_profile()
    logo = base64.b64encode((BASE_DIR / "assets/unfold_icon.png").read_bytes()).decode()
    st.markdown(f"<div class='brand'><img alt='UnFold' src='data:image/png;base64,{logo}'/>"
                "<div><h1>UnFold</h1><p>A space to reflect. A starting point for support.</p></div>"
                "<span class='session-note'>Session only</span></div>", unsafe_allow_html=True)
    nav = st.radio("Navigation", ["Assessment", "Research", "About"], horizontal=True,
                    label_visibility="collapsed", key="uf_nav")
    if nav == "Assessment":
        render_assessment(profile)
    elif nav == "Research":
        render_research()
    else:
        render_about()


if __name__ == "__main__":
    main()

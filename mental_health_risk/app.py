import json
import os
from datetime import datetime

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from utils import (
    MODEL_DIR, RISK_MAP, HELPLINES, SUPPORT_TIPS, LOW_CONFIDENCE,
    KEYWORD_MIN_SCORE, SUICIDAL_THRESHOLD, clean_text, keyword_hit,
    risk_score, decide_label, highlight_html, top_words,
)

st.set_page_config(page_title="MindScope | Text Insights", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

DISCLAIMER = (
    "Educational prototype only. This tool classifies text patterns; it does not diagnose a person, "
    "measure their true mental state, or replace support from a qualified professional."
)

# Visual styling only; prediction and training logic remains in utils.py/train.py.
st.markdown("""
<style>
:root { --ink:#182238; --muted:#64748b; --line:#e5eaf2; --soft:#f5f7fb; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
.hero { padding: 1.6rem 1.8rem; border-radius: 22px; background: linear-gradient(125deg,#eef2ff 0%,#f4f0ff 52%,#eafaf6 100%); border: 1px solid #e4e7fb; margin-bottom: 1rem; }
.hero h1 { color: #182238; font-size: 2rem; margin: 0 0 .4rem 0; }
.hero p { color: #526078; margin: 0; font-size: 1rem; }
.metric-card { padding: 1rem 1.1rem; border: 1px solid #e5eaf2; border-radius: 16px; background: white; min-height: 110px; }
.metric-label { color: #64748b; font-size: .86rem; margin-bottom: .35rem; }
.metric-value { color: #182238; font-size: 1.65rem; font-weight: 750; line-height: 1.2; }
.metric-note { color: #64748b; font-size: .8rem; margin-top: .35rem; }
.section-card { padding: 1rem 1.1rem; border: 1px solid #e5eaf2; border-radius: 16px; background: #fff; }
.small-muted { color: #64748b; font-size: .88rem; }
div.stButton > button { border-radius: 10px; font-weight: 650; }
</style>
""", unsafe_allow_html=True)


def go_to_page(target):
    st.session_state["page_nav"] = target


def metric_card(label, value, note=""):
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
    model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    return vectorizer, model


def analyze(text, vectorizer, model):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    values = model.predict_proba(X)[0]
    probs = {c: float(p) for c, p in zip(model.classes_, values)}
    label = decide_label(probs)
    score = risk_score(probs)
    hits = keyword_hit(text)
    overridden = False
    if hits and label != "Suicidal":
        label = "Suicidal"
        overridden = True
    if hits:
        score = max(score, KEYWORD_MIN_SCORE)
    low_conf = max(probs.values()) < LOW_CONFIDENCE and not hits
    return {"probs": probs, "label": label, "score": score, "overridden": overridden,
            "keyword_hit": bool(hits), "low_conf": low_conf}


def show_helplines():
    st.warning("If someone may be in immediate danger, contact local emergency services or a trusted person now.")
    for item in HELPLINES:
        st.markdown(f"**{item['name']}:** {item['number']}")


def show_result(user_text, result):
    info = RISK_MAP[result["label"]]
    st.markdown(
        f"<div style='padding:20px;border-radius:16px;background:{info['color']}18;"
        f"border:1px solid {info['color']}66;border-left:7px solid {info['color']}'>"
        f"<div style='font-size:1.25rem;font-weight:750'>{info['emoji']} {info['level']}</div>"
        f"<div style='margin-top:7px;color:#334155'>Model category: <b>{result['label']}</b> &nbsp; · &nbsp; "
        f"Prototype score: <b>{result['score']:.0f}/100</b></div></div>", unsafe_allow_html=True)
    st.caption(DISCLAIMER)
    if result["low_conf"]:
        st.warning("Low confidence: the model is uncertain about this text. The result may be unreliable.")
    if result["overridden"]:
        st.info("A configured safety phrase was detected, so the rule-based safeguard changed the displayed category.")
    elif result["keyword_hit"]:
        st.info("A configured safety phrase was detected. Context still matters, and this is not a clinical assessment.")

    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Class probabilities")
        for cls, probability in sorted(result["probs"].items(), key=lambda item: item[1], reverse=True):
            st.write(f"{RISK_MAP[cls]['emoji']} **{cls}** · {probability * 100:.1f}%")
            st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"The Suicidal class threshold is configured at {SUICIDAL_THRESHOLD:.0%}; this is a model rule, not a clinical threshold.")
    with b:
        st.subheader("Words the model weighted")
        st.markdown(highlight_html(user_text, vectorizer, model, result["label"]), unsafe_allow_html=True)
        words = top_words(user_text, vectorizer, model, result["label"])
        if words:
            st.write("**Higher positive model weights:** " + ", ".join(f"`{word}` ({weight:.2f})" for word, weight in words))
        else:
            st.caption("No known influential words were found in the model vocabulary.")
        st.caption("Highlighted words are model associations, not proof of meaning or cause.")
    if result["label"] == "Suicidal":
        st.divider()
        st.subheader("Support options")
        for tip in SUPPORT_TIPS["Suicidal"]:
            st.write(f"- {tip}")
        show_helplines()


if "history" not in st.session_state:
    st.session_state.history = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

with st.sidebar:
    st.markdown("# 🧠 MindScope")
    st.caption("Text insight prototype")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Analyze Text", "Reflection Prompts", "Trend Tracker", "Support Library", "Model Insights", "About & Privacy"],
        label_visibility="visible", key="page_nav",
    )
    st.divider()
    st.markdown("**A gentle reminder**")
    st.caption("A model output is only one imperfect signal. Your experiences deserve human understanding.")
    st.caption("Version 2.0 · Educational prototype")

model_path = os.path.join(MODEL_DIR, "model.joblib")
vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
model_ready = os.path.exists(model_path) and os.path.exists(vectorizer_path)
vectorizer = model = None
if model_ready:
    vectorizer, model = load_artifacts()

if page == "Dashboard":
    st.markdown("<div class='hero'><h1>Make space to understand the words.</h1><p>Explore text patterns, reflect on your own context, and view model output in one calm workspace.</p></div>", unsafe_allow_html=True)
    st.caption(DISCLAIMER)
    total = len(st.session_state.history)
    latest = st.session_state.history[-1] if total else None
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1: metric_card("Saved in this session", total, "Trend entries in current session")
    with c2: metric_card("Latest category", latest["category"] if latest else "—", "No diagnosis is made")
    with c3: metric_card("Latest prototype score", f"{latest['score']:.0f}/100" if latest else "—", "Not a clinical measurement")
    st.write("")
    st.subheader("Choose your next step")
    cards = st.columns(3, gap="medium")
    with cards[0]:
        st.markdown("<div class='section-card'><h3>✍️ Analyze text</h3><p>Submit a journal-style entry or a social post and inspect the model's category probabilities.</p></div>", unsafe_allow_html=True)
        st.button("Open text analysis", use_container_width=True, on_click=go_to_page, args=("Analyze Text",))
    with cards[1]:
        st.markdown("<div class='section-card'><h3>📈 Track entries</h3><p>Review scores you chose to add to this session and download a CSV copy.</p></div>", unsafe_allow_html=True)
        st.button("Open trend tracker", use_container_width=True, on_click=go_to_page, args=("Trend Tracker",))
    with cards[2]:
        st.markdown("<div class='section-card'><h3>🌱 Support library</h3><p>Browse general grounding ideas and ways to reach a trusted person or support service.</p></div>", unsafe_allow_html=True)
        st.button("Explore support", use_container_width=True, on_click=go_to_page, args=("Support Library",))
    st.subheader("How it works")
    st.write("**Text input → cleaning → TF-IDF features → trained classifier → safeguards → readable results.**")
    st.caption("Self-reported mood, stress and sleep fields on the analysis page are optional context only. They do not change the machine-learning prediction.")

elif page == "Analyze Text":
    st.markdown("<div class='hero'><h1>Analyze a piece of text</h1><p>Use text you have permission to analyze. Avoid including names or identifying details.</p></div>", unsafe_allow_html=True)
    if not model_ready:
        st.error("The trained model files are missing. From the project folder, run `python train.py` first; this creates the model files used here.")
        st.stop()
    st.caption(DISCLAIMER)
    with st.form("analysis_form"):
        text_col, context_col = st.columns([2.1, 1], gap="large")
        with text_col:
            text_type = st.selectbox("Entry type", ["Journal", "Social Media", "Reflection", "Other"])
            user_text = st.text_area("Text to analyze", height=230, max_chars=10000,
                                     placeholder="Write or paste your text here…")
            st.caption(f"Character count: {len(user_text)} / 10,000")
            with st.expander("Need a starting point? Try one of these prompts"):
                st.markdown("""- What has been on my mind today?
- What felt difficult, and what helped?
- What would I tell a trusted friend?
- What is one small thing I need right now?""")
                st.caption("Prompts are optional. Copy one into the text box if useful; they are not diagnostic questions.")
        with context_col:
            st.subheader("Optional context")
            mood = st.select_slider("How are you feeling today?", options=["Not selected", "Very low", "Low", "Mixed", "Okay", "Good"], value="Not selected")
            stress = st.select_slider("Perceived stress", options=["Not selected", "Low", "Moderate", "High"], value="Not selected")
            sleep = st.selectbox("Sleep lately", ["Not selected", "Poor", "Mixed", "Okay", "Restful"])
            context_tags = st.multiselect("Anything relevant? (optional)", ["Studies/work", "Relationships", "Family", "Health", "Loneliness", "Money", "Major change", "Not sure"])
            st.caption("These fields are for reflection/history only. The classifier does not use them.")
            add_to_trend = st.checkbox("Add this result to Trend Tracker", value=True)
        submitted = st.form_submit_button("Analyze text", type="primary", use_container_width=True)
    if submitted:
        if len(user_text.strip()) < 10:
            st.warning("Please enter at least 10 characters so the model has some text to evaluate.")
        else:
            result = analyze(user_text, vectorizer, model)
            st.session_state.latest_result = result
            show_result(user_text, result)
            if add_to_trend:
                st.session_state.history.append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "type": text_type,
                    "category": result["label"], "risk_level": RISK_MAP[result["label"]]["level"],
                    "score": round(result["score"], 1), "self_reported_mood": mood,
                    "self_reported_stress": stress, "sleep": sleep,
                    "context_tags": ", ".join(context_tags),
                })
                st.success("Added to Trend Tracker for this session.")

elif page == "Reflection Prompts":
    st.markdown("<div class='hero'><h1>Reflection space</h1><p>Optional writing prompts to help organize thoughts—without scoring or judging your answers.</p></div>", unsafe_allow_html=True)
    st.info("This page does not send your writing to the classifier. The text stays in the current page session unless you copy it yourself; hosting and browser behavior may still affect privacy.")
    prompt_groups = {
        "Daily check-in": ["What stood out about today?", "What took energy, and what gave me energy?", "What is one thing I handled, even if imperfectly?"],
        "When things feel busy": ["What is the main thing occupying my mind?", "Which part can I act on, and which part can wait?", "What would make the next hour a little easier?"],
        "Support and connection": ["Who is one person I can be honest with?", "What kind of support would feel useful?", "What would I say to someone I care about in the same situation?"],
    }
    for group, prompts in prompt_groups.items():
        with st.expander(group, expanded=(group == "Daily check-in")):
            for p in prompts: st.markdown(f"- {p}")
    st.subheader("Write a private reflection")
    reflection = st.text_area("Your notes (optional)", height=220, placeholder="Start wherever feels comfortable…")
    st.download_button("Download reflection as .txt", data=reflection, file_name="my_reflection.txt", mime="text/plain", disabled=not reflection.strip())
    st.caption("Writing here is optional. You can skip any prompt that does not feel right for you.")

elif page == "Trend Tracker":
    st.markdown("<div class='hero'><h1>Trend tracker</h1><p>Review only the entries you opted to add during this session.</p></div>", unsafe_allow_html=True)
    st.caption("These are model-generated scores, not proof of changes in mental health. Session history may be lost when the session ends; downloaded files are your responsibility.")
    history = st.session_state.history
    if not history:
        st.info("No entries yet. Visit Analyze Text and choose to add a result to the tracker.")
    else:
        hdf = pd.DataFrame(history)
        hdf["entry"] = range(1, len(hdf) + 1)
        a, b, c = st.columns(3)
        with a: metric_card("Entries", len(hdf), "Current session")
        with b: metric_card("Latest score", f"{hdf['score'].iloc[-1]:.0f}/100", "Prototype score")
        with c: metric_card("Average score", f"{hdf['score'].mean():.0f}/100", "Across listed entries")
        chart = alt.Chart(hdf).mark_line(point=alt.OverlayMarkDef(filled=True, size=70), strokeWidth=3).encode(
            x=alt.X("entry:O", title="Entry", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("score:Q", title="Prototype score (0–100)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["entry", "time", "category", "score", "type"],
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
        if len(hdf) >= 4:
            recent = hdf["score"].tail(3).mean(); earlier = hdf["score"].iloc[:-3].mean(); diff = recent - earlier
            if diff >= 10: st.warning(f"The recent model-score average is higher by {diff:.0f} points. This is not a clinical trend assessment.")
            elif diff <= -10: st.info(f"The recent model-score average is lower by {abs(diff):.0f} points. This is not a clinical trend assessment.")
            else: st.info("The recent and earlier model-score averages are within 10 points of each other. This is not a clinical trend assessment.")
        else: st.caption("Add at least 4 entries to show the simple recent-versus-earlier comparison.")
        st.subheader("Entry history")
        st.dataframe(hdf.drop(columns=["entry"]), use_container_width=True, hide_index=True)
        left, right = st.columns([1, 1])
        with left:
            st.download_button("⬇ Download CSV", hdf.drop(columns=["entry"]).to_csv(index=False), file_name="mindscope_session_history.csv", mime="text/csv", use_container_width=True)
        with right:
            if st.button("Clear session history", use_container_width=True):
                st.session_state.history = []; st.rerun()

elif page == "Support Library":
    st.markdown("<div class='hero'><h1>Support library</h1><p>General, non-clinical ideas for grounding, reflection, and reaching out.</p></div>", unsafe_allow_html=True)
    st.caption("These suggestions are not treatment and may not fit every person. Choose only what feels appropriate.")
    for label, info in RISK_MAP.items():
        with st.expander(f"{info['emoji']} {info['level']} · {label}"):
            for tip in SUPPORT_TIPS[label]: st.write(f"- {tip}")
    st.divider()
    st.subheader("Reach a person or service")
    st.write("If you feel unsafe or overwhelmed, consider contacting someone you trust or a qualified professional. If there is immediate danger, contact local emergency services.")
    for item in HELPLINES:
        st.markdown(f"**{item['name']}:** {item['number']}")
    st.caption("Please confirm that contact details are current and appropriate for your location.")

elif page == "Model Insights":
    st.markdown("<div class='hero'><h1>Model insights</h1><p>Explore evaluation results and how this prototype makes predictions.</p></div>", unsafe_allow_html=True)
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as file: metrics = json.load(file)
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Deployed model", metrics.get("deployed_model", "Not listed"), "Selected during training")
        with c2: metric_card("Training rows", f"{metrics.get('train_size', '—'):,}" if isinstance(metrics.get('train_size'), int) else metrics.get('train_size', '—'), "Used to fit the model")
        with c3: metric_card("Test rows", f"{metrics.get('test_size', '—'):,}" if isinstance(metrics.get('test_size'), int) else metrics.get('test_size', '—'), "Held-out evaluation split")
        st.subheader("Model comparison")
        if metrics.get("comparison"): st.dataframe(pd.DataFrame(metrics["comparison"]), use_container_width=True, hide_index=True)
        if metrics.get("report"):
            st.subheader("Per-class evaluation report")
            st.dataframe(pd.DataFrame(metrics["report"]).T.round(3), use_container_width=True)
        cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path): st.image(cm_path, caption="Confusion matrix from the held-out test split", width=650)
        st.caption("Metrics describe performance on the available test split, not guaranteed performance on new people or contexts.")
    else:
        st.info("No metrics.json was found. Run `python train.py` to train and evaluate the models.")
    st.subheader("Prediction pipeline")
    st.write("1. Normalize text with the shared cleaner.  2. Convert text to TF-IDF features.  3. Obtain class probabilities from the trained classifier.  4. Apply the configured probability threshold and safety phrase rules.  5. Display the category, prototype score and word-weight explanation.")
    st.subheader("Key configuration")
    st.dataframe(pd.DataFrame([
        {"Setting": "Classes", "Configured value": "Normal, Anxiety, Depression, Suicidal"},
        {"Setting": "Low-confidence warning", "Configured value": f"Top probability below {LOW_CONFIDENCE:.0%}"},
        {"Setting": "Suicidal class threshold", "Configured value": f"{SUICIDAL_THRESHOLD:.0%}"},
        {"Setting": "Keyword score floor", "Configured value": f"{KEYWORD_MIN_SCORE:.0f}/100 when a configured phrase matches"},
    ]), use_container_width=True, hide_index=True)

elif page == "About & Privacy":
    st.markdown("<div class='hero'><h1>About MindScope</h1><p>A student-friendly text classification prototype with transparent limitations.</p></div>", unsafe_allow_html=True)
    st.subheader("What it does")
    st.write("The application uses a trained text classifier to assign one of four text categories. It also displays class probabilities, a weighted prototype score, selected word weights, and optional session-based trend history.")
    st.subheader("What it does not do")
    st.markdown("- It does not diagnose depression, anxiety, or any other condition.\n- It cannot know a person's intentions or situation from text alone.\n- A low score does not prove that someone is safe, and a high score does not prove a crisis.\n- The safety phrase rules can miss context or produce false alarms.\n- Self-reported mood, stress, sleep and context fields do not affect the classifier prediction.")
    st.subheader("Privacy notes")
    st.markdown("- Trend history is held in Streamlit session state and is not written to a database by this code.\n- Session storage is not a guarantee of privacy; hosting, server logs, browser configuration and shared devices may matter.\n- Avoid entering names, contact details, or other identifying information.\n- Downloaded CSVs and reflection files are saved by you and should be handled carefully.")
    st.subheader("Technical stack")
    st.write("Python · Streamlit · scikit-learn · TF-IDF · Logistic Regression model artifacts · pandas · Altair")
    st.caption(DISCLAIMER)

st.divider()
st.caption("MindScope · Student project prototype · Please interpret outputs cautiously and prioritize human support over automated labels.")

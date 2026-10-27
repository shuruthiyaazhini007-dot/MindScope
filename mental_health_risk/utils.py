import re
import html
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DATA_PATH = "data/dataset.csv"
TEXT_COL = "text"
LABEL_COL = "status"
MODEL_DIR = "models"

LABELS = ["Normal", "Anxiety", "Depression", "Suicidal"]

RISK_MAP = {
    "Normal": {"level": "Low Risk", "emoji": "🟢", "color": "#2e7d32"},
    "Anxiety": {"level": "Moderate Risk", "emoji": "🟡", "color": "#f9a825"},
    "Depression": {"level": "High Risk", "emoji": "🟠", "color": "#ef6c00"},
    "Suicidal": {"level": "Critical Risk", "emoji": "🔴", "color": "#c62828"},
}

CLASS_WEIGHTS = {"Normal": 0, "Anxiety": 35, "Depression": 65, "Suicidal": 100}

SUICIDAL_THRESHOLD = 0.30
LOW_CONFIDENCE = 0.50
KEYWORD_MIN_SCORE = 65.0

SAFETY_PATTERNS = [
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bend it all\b",
    r"\bwant to die\b",
    r"\bwanna die\b",
    r"\bsuicide\b",
    r"\bsuicidal\b",
    r"\bno reason to live\b",
    r"\bbetter off dead\b",
    r"\bhurt myself\b",
    r"\bself harm\b",
    r"\bcut myself\b",
    r"\bdont want to live\b",
    r"\bdo not want to live\b",
]

HELPLINES = [
    {"name": "Tele-MANAS (India, 24x7, free)", "number": "14416"},
    {"name": "Emergency services (India)", "number": "112"},
    {"name": "Other countries", "number": "https://findahelpline.com"},
]

SUPPORT_TIPS = {
    "Normal": [
        "Keep up routines that help you feel steady, like sleep, movement and time with people you trust.",
        "Journaling regularly makes it easier to notice changes early.",
    ],
    "Anxiety": [
        "Try slow breathing: inhale for 4 seconds, hold for 4, exhale for 6, repeat for a few minutes.",
        "Try the 5-4-3-2-1 grounding exercise: name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
        "Write down what is worrying you, then circle the parts you can act on today.",
        "If worry keeps interfering with daily life, consider talking to a counsellor.",
    ],
    "Depression": [
        "Pick one small task today, such as a short walk, a meal or a shower, and treat finishing it as a win.",
        "Reach out to one person you trust and tell them how you have been feeling.",
        "Low mood that lasts more than two weeks is worth discussing with a doctor or counsellor.",
    ],
    "Suicidal": [
        "You do not have to handle this alone. Please contact someone right now, a friend, a family member or a helpline.",
        "If you are in immediate danger, call your local emergency number.",
        "Try to move away from anything you could use to hurt yourself and stay near other people.",
    ],
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def keyword_hit(text):
    cleaned = clean_text(text)
    return [p for p in SAFETY_PATTERNS if re.search(p, cleaned)]


def risk_score(probs):
    return float(sum(probs[c] * CLASS_WEIGHTS[c] for c in probs))


def decide_label(probs):
    if probs.get("Suicidal", 0) >= SUICIDAL_THRESHOLD:
        return "Suicidal"
    return max(probs, key=probs.get)


def word_weights(text, vectorizer, model, label):
    vocab = vectorizer.vocabulary_
    class_index = list(model.classes_).index(label)
    coefs = model.coef_[class_index]
    weights = {}
    for token in re.findall(r"\w+", text):
        word = clean_text(token)
        if word in vocab and word not in ENGLISH_STOP_WORDS:
            weights[token.lower()] = float(coefs[vocab[word]])
    return weights


def highlight_html(text, vectorizer, model, label):
    weights = word_weights(text, vectorizer, model, label)
    positive = [w for w in weights.values() if w > 0]
    max_w = max(positive) if positive else 1.0
    parts = re.findall(r"\w+|\W+", text)
    out = []
    for part in parts:
        w = weights.get(part.lower(), 0)
        if re.match(r"\w+", part) and w > 0:
            alpha = round(0.15 + 0.65 * min(w / max_w, 1.0), 2)
            out.append(
                f"<span style='background: rgba(239,83,80,{alpha}); padding:1px 3px; border-radius:3px'>{html.escape(part)}</span>"
            )
        else:
            out.append(html.escape(part))
    return "".join(out).replace("\n", "<br>")


def top_words(text, vectorizer, model, label, n=8):
    weights = word_weights(text, vectorizer, model, label)
    ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    return [(w, round(v, 3)) for w, v in ranked if v > 0][:n]

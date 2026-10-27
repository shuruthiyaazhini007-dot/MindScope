import json
import os
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from utils import clean_text, DATA_PATH, TEXT_COL, LABEL_COL, MODEL_DIR, LABELS

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df[[TEXT_COL, LABEL_COL]].dropna().drop_duplicates()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.capitalize()
df = df[df[LABEL_COL].isin(LABELS)].copy()

print("Class distribution:")
print(df[LABEL_COL].value_counts())

df["clean"] = df[TEXT_COL].apply(clean_text)
df = df[df["clean"].str.len() > 0]

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df[LABEL_COL], test_size=0.2, random_state=42, stratify=df[LABEL_COL]
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), max_features=60000, min_df=2, sublinear_tf=True
)
Xtr = vectorizer.fit_transform(X_train)
Xte = vectorizer.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=2.0, class_weight="balanced"
    ),
    "Linear SVM": LinearSVC(class_weight="balanced"),
    "Complement Naive Bayes": ComplementNB(),
}

comparison = []
trained = {}
for name, clf in models.items():
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    row = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "suicidal_recall": round(
            recall_score(y_test, pred, labels=["Suicidal"], average="macro"), 4
        ),
    }
    comparison.append(row)
    trained[name] = (clf, pred)
    print(row)

final_name = "Logistic Regression"
final_model, final_pred = trained[final_name]

report = classification_report(y_test, final_pred, output_dict=True, zero_division=0)
print(classification_report(y_test, final_pred, zero_division=0))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    final_pred,
    labels=[l for l in LABELS if l in final_model.classes_],
    cmap="Blues",
    ax=ax,
    xticks_rotation=30,
)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)

joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
joblib.dump(final_model, os.path.join(MODEL_DIR, "model.joblib"))

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(
        {
            "deployed_model": final_name,
            "comparison": comparison,
            "report": report,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
        f,
        indent=2,
    )

print("Saved model, vectorizer, metrics and confusion matrix to", MODEL_DIR)

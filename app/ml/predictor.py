"""
Loads the trained ensemble artifacts ONCE at import time (not per-request —
that was the bug in the original Flask app, which retrained on every server
start) and exposes a single predict() function for the API layer to call.
"""
from app.ml.heuristics import extract_features, extract_features_dict, normalize_text
import os
import joblib
import numpy as np

from app.ml.heuristics import extract_features, extract_features_dict

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def _load(name):
    path = os.path.join(ARTIFACTS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing model artifact: {name}. "
            f"Run `python -m app.ml.train` from the project root first."
        )
    return joblib.load(path)


# Loaded once, reused for every request.
word_vectorizer = _load("word_vectorizer.pkl")
char_vectorizer = _load("char_vectorizer.pkl")
nb_model = _load("nb_model.pkl")
svm_model = _load("svm_model.pkl")
rf_model = _load("rf_model.pkl")
char_svm_model = _load("char_svm_model.pkl")
meta_model = _load("meta_model.pkl")

MODEL_NAMES = ["naive_bayes", "linear_svm", "random_forest", "char_ngram_svm"]



def predict(message: str) -> dict:
    clean_message = normalize_text(message)

    word_vec = word_vectorizer.transform([clean_message])
    char_vec = char_vectorizer.transform([clean_message])

    votes = {
        "naive_bayes": int(nb_model.predict(word_vec)[0]),
        "linear_svm": int(svm_model.predict(word_vec)[0]),
        "random_forest": int(rf_model.predict(word_vec)[0]),
        "char_ngram_svm": int(char_svm_model.predict(char_vec)[0]),
    }

    heuristic_flags = extract_features_dict(clean_message)
    heuristic_values = extract_features(clean_message)
    ...

    meta_features = np.array([[votes[name] for name in MODEL_NAMES] + heuristic_values])
    final_label = int(meta_model.predict(meta_features)[0])

    # KNN can also report how many of its 5 nearest neighbors agreed —
    # a rough, honest confidence signal (not a calibrated probability).
    neighbor_labels = meta_model.predict(meta_features)  # final label already computed
    distances, indices = meta_model.kneighbors(meta_features)
    neighbor_votes = meta_model._y[indices[0]]
    confidence = float(np.mean(neighbor_votes == final_label))

    return {
        "message": message,
        "is_spam": bool(final_label),
        "label": "spam" if final_label == 1 else "ham",
        "confidence": round(confidence, 2),
        "model_votes": {k: ("spam" if v == 1 else "ham") for k, v in votes.items()},
        "heuristic_flags": {k: bool(v) for k, v in heuristic_flags.items()},
    }

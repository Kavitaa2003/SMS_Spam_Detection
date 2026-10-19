"""
Trains the full spam-detection ensemble and saves all artifacts needed for
inference. Run this once after cloning the repo (see README) — it does not
get committed to git; it's regenerated locally in under a minute.

Pipeline recap:
  - 3 base models on word-level TF-IDF: Naive Bayes, Linear SVM, Random Forest
  - 1 base model on character n-gram TF-IDF: Linear SVM (catches obfuscated
    spelling / modern link-based patterns word-level features miss)
  - 8 rule-based heuristic features (URLs, urgency language, etc.)
  - A KNN meta-model trained on [4 base-model votes + 8 heuristic flags],
    using cross-validated predictions for the base-model votes to avoid
    data leakage into the meta-model.

Run from the project root:
    python -m app.ml.train
"""
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.ml.heuristics import extract_features

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "spam.csv")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df[['label', 'message']]
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )

    print(f"Training on {len(X_train)} messages, testing on {len(X_test)}...")

    # --- Feature extraction ---
    word_vectorizer = TfidfVectorizer(stop_words='english')
    X_train_word = word_vectorizer.fit_transform(X_train)
    X_test_word = word_vectorizer.transform(X_test)

    char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    X_train_char = char_vectorizer.fit_transform(X_train)
    X_test_char = char_vectorizer.transform(X_test)

    heuristics_train = np.array([extract_features(msg) for msg in X_train])
    heuristics_test = np.array([extract_features(msg) for msg in X_test])

    # --- Honest, cross-validated meta-features (avoids leakage) ---
    print("Generating cross-validated meta-features...")
    nb_meta_train = cross_val_predict(MultinomialNB(), X_train_word, y_train, cv=5)
    svm_meta_train = cross_val_predict(LinearSVC(), X_train_word, y_train, cv=5)
    rf_meta_train = cross_val_predict(
        RandomForestClassifier(n_estimators=200, random_state=42), X_train_word, y_train, cv=5
    )
    char_svm_meta_train = cross_val_predict(LinearSVC(), X_train_char, y_train, cv=5)

    meta_X_train = np.column_stack([nb_meta_train, svm_meta_train, rf_meta_train, char_svm_meta_train])
    meta_X_train = np.hstack([meta_X_train, heuristics_train])

    # --- Final base models, trained on all available training data ---
    print("Training final base models...")
    nb_final = MultinomialNB().fit(X_train_word, y_train)
    svm_final = LinearSVC().fit(X_train_word, y_train)
    rf_final = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train_word, y_train)
    char_svm_final = LinearSVC().fit(X_train_char, y_train)

    nb_meta_test = nb_final.predict(X_test_word)
    svm_meta_test = svm_final.predict(X_test_word)
    rf_meta_test = rf_final.predict(X_test_word)
    char_svm_meta_test = char_svm_final.predict(X_test_char)

    meta_X_test = np.column_stack([nb_meta_test, svm_meta_test, rf_meta_test, char_svm_meta_test])
    meta_X_test = np.hstack([meta_X_test, heuristics_test])

    # --- Meta-model ---
    print("Training KNN meta-model...")
    meta_model = KNeighborsClassifier(n_neighbors=5)
    meta_model.fit(meta_X_train, y_train)

    # --- Evaluate final pipeline ---
    y_pred = meta_model.predict(meta_X_test)
    print("\n=== Final Ensemble Performance (held-out test set) ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 score:  {f1_score(y_test, y_pred):.4f}")

    # --- Save everything needed for inference ---
    joblib.dump(word_vectorizer, os.path.join(ARTIFACTS_DIR, "word_vectorizer.pkl"))
    joblib.dump(char_vectorizer, os.path.join(ARTIFACTS_DIR, "char_vectorizer.pkl"))
    joblib.dump(nb_final, os.path.join(ARTIFACTS_DIR, "nb_model.pkl"))
    joblib.dump(svm_final, os.path.join(ARTIFACTS_DIR, "svm_model.pkl"))
    joblib.dump(rf_final, os.path.join(ARTIFACTS_DIR, "rf_model.pkl"))
    joblib.dump(char_svm_final, os.path.join(ARTIFACTS_DIR, "char_svm_model.pkl"))
    joblib.dump(meta_model, os.path.join(ARTIFACTS_DIR, "meta_model.pkl"))

    print(f"\nAll artifacts saved to {os.path.abspath(ARTIFACTS_DIR)}")
    print("You can now start the API with: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()

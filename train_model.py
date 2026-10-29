"""
TruthLens AI Fake News Detector - Machine Learning Model Trainer
Trains a TF-IDF + Logistic Regression classification pipeline on labelled news examples.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "news.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"


def train_and_save():
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}")
        return False

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10]

    print(f"Loaded {len(df)} samples from {DATA_PATH.name}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Validation Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Save trained pipeline and metadata
    model_bundle = {
        "pipeline": pipeline,
        "classes": list(pipeline.classes_),
        "sample_count": len(df),
        "accuracy": float(accuracy),
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"Model successfully saved to: {MODEL_PATH}")
    return True


if __name__ == "__main__":
    train_and_save()


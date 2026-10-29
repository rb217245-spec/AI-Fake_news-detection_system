"""
TruthLens AI Fake News Detector - Machine Learning Classifier Interface
Provides inference and diagnostics using the trained TF-IDF + Logistic Regression model.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fake_news_model.joblib"

_CACHED_MODEL_BUNDLE: Optional[Dict[str, Any]] = None


def is_model_available() -> bool:
    """Check if the trained model file exists on disk."""
    return MODEL_PATH.exists()


def get_model_info() -> Dict[str, Any]:
    """Get metadata about the trained ML model."""
    if not is_model_available():
        return {
            "available": False,
            "status": "Not trained",
            "message": "Run 'python train_model.py' to train the local model on the dataset.",
            "path": str(MODEL_PATH)
        }
    return {
        "available": True,
        "status": "Ready",
        "path": str(MODEL_PATH),
        "file_size_kb": round(MODEL_PATH.stat().st_size / 1024, 1)
    }


def _load_model() -> Optional[Dict[str, Any]]:
    global _CACHED_MODEL_BUNDLE
    if _CACHED_MODEL_BUNDLE is not None:
        return _CACHED_MODEL_BUNDLE

    if not is_model_available():
        return None

    try:
        loaded = joblib.load(MODEL_PATH)
        # Support both raw pipeline or dictionary bundle
        if isinstance(loaded, dict) and "pipeline" in loaded:
            _CACHED_MODEL_BUNDLE = loaded
        else:
            _CACHED_MODEL_BUNDLE = {
                "pipeline": loaded,
                "classes": list(loaded.classes_) if hasattr(loaded, "classes_") else ["FAKE", "REAL"],
                "accuracy": None
            }
        return _CACHED_MODEL_BUNDLE
    except Exception as exc:
        return None


def predict_text(text: str) -> Dict[str, Any]:
    """
    Classify a piece of text using the local TF-IDF + Logistic Regression model.
    Returns prediction label, confidence score, and clear educational notice.
    """
    cleaned = (text or "").strip()
    if len(cleaned) < 10:
        return {
            "available": False,
            "error": "Text too short for ML analysis (minimum 10 characters required)."
        }

    bundle = _load_model()
    if not bundle:
        return {
            "available": False,
            "error": "Local ML model not found. Run 'python train_model.py' to generate models/fake_news_model.joblib."
        }

    try:
        pipeline = bundle["pipeline"]
        classes = bundle.get("classes", list(pipeline.classes_))
        prediction = pipeline.predict([cleaned])[0]
        
        # Calculate probability/confidence if available
        confidence = 50.0
        probabilities = {}
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba([cleaned])[0]
            for cls_name, prob in zip(classes, probs):
                probabilities[str(cls_name)] = round(float(prob) * 100, 1)
            pred_idx = list(classes).index(prediction)
            confidence = round(float(probs[pred_idx]) * 100, 1)

        is_fake = str(prediction).upper() == "FAKE"

        return {
            "available": True,
            "prediction": "Likely Misleading / Fake Pattern" if is_fake else "Likely Real / Standard Reporting Pattern",
            "raw_label": str(prediction),
            "confidence": confidence,
            "probabilities": probabilities,
            "method": "TF-IDF N-grams + Logistic Regression",
            "disclaimer": "Local ML classifies linguistic n-gram patterns against the sample training set. It does not check live external facts."
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"ML prediction failed ({type(exc).__name__}): {str(exc)}"
        }


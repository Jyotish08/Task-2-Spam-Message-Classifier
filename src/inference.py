"""Inference Service module for Spam Message Classifier.

Provides safe, production-ready single and batch prediction interfaces,
input validation, probability scoring, and descriptive error handling.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import joblib

from src.config import SAVED_MODEL_PATH
from src.preprocessing import validate_input_text


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model artifact cannot be found on disk."""
    pass


class ModelLoadError(RuntimeError):
    """Raised when the model artifact exists but cannot be deserialized or is invalid."""
    pass


@dataclass
class PredictionResult:
    """Structured container for prediction output."""

    label: str
    is_spam: bool
    spam_probability: float
    ham_probability: float
    confidence: float
    processing_time_ms: float
    message_length: int
    sanitized_text_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to a serializable dictionary."""
        return asdict(self)


class SpamClassifierService:
    """Inference service wrapping the trained scikit-learn pipeline."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initializes the service and loads the serialized pipeline.

        Parameters
        ----------
        model_path : Optional[Union[str, Path]]
            Path to the saved joblib model. Defaults to SAVED_MODEL_PATH.
        """
        self.model_path = Path(model_path) if model_path else SAVED_MODEL_PATH
        self.pipeline: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads and validates the trained pipeline from disk."""
        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Trained model artifact not found at: {self.model_path.resolve()}\n"
                "Please train the model first by running:\n"
                "    python train.py"
            )

        try:
            self.pipeline = joblib.load(self.model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model from {self.model_path.resolve()}. "
                f"The file may be corrupted. Error: {exc}\n"
                "Try retraining the model by running 'python train.py'."
            ) from exc

        if not hasattr(self.pipeline, "predict"):
            raise ModelLoadError(
                f"The object loaded from {self.model_path} does not implement the standard 'predict' method."
            )

    def is_ready(self) -> bool:
        """Checks if the inference service has a loaded model ready to predict."""
        return self.pipeline is not None

    def predict(self, text: str) -> PredictionResult:
        """Runs prediction on a single user message with rigorous validation.

        Parameters
        ----------
        text : str
            Raw input SMS message.

        Returns
        -------
        PredictionResult
            Detailed prediction result containing label, confidence, and probabilities.

        Raises
        ------
        TypeError
            If text is not a string.
        ValueError
            If text is empty, whitespace-only, or exceeds safety limits.
        """
        validated_text = validate_input_text(text)

        t0 = time.perf_counter()
        raw_pred = self.pipeline.predict([validated_text])[0]

        # Extract probabilities if the underlying pipeline supports predict_proba
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([validated_text])[0]
            ham_prob = float(probs[0])
            spam_prob = float(probs[1])
        else:
            # Fallback for models without calibrated predict_proba
            spam_prob = 1.0 if raw_pred == 1 else 0.0
            ham_prob = 1.0 - spam_prob

        t1 = time.perf_counter()
        elapsed_ms = round((t1 - t0) * 1000.0, 3)

        label = "spam" if raw_pred == 1 else "ham"
        confidence = spam_prob if label == "spam" else ham_prob

        snippet = (
            validated_text[:80] + "..."
            if len(validated_text) > 80
            else validated_text
        )

        return PredictionResult(
            label=label,
            is_spam=(label == "spam"),
            spam_probability=round(spam_prob, 4),
            ham_probability=round(ham_prob, 4),
            confidence=round(confidence, 4),
            processing_time_ms=elapsed_ms,
            message_length=len(validated_text),
            sanitized_text_snippet=snippet,
        )

    def predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        """Runs prediction on a list of input messages.

        Parameters
        ----------
        texts : List[str]
            List of raw input messages.

        Returns
        -------
        List[PredictionResult]
            List of prediction results.
        """
        validated_texts = [validate_input_text(t) for t in texts]

        t0 = time.perf_counter()
        raw_preds = self.pipeline.predict(validated_texts)

        if hasattr(self.pipeline, "predict_proba"):
            probs_matrix = self.pipeline.predict_proba(validated_texts)
        else:
            probs_matrix = [[1.0 - p, float(p)] for p in raw_preds]

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0
        avg_ms = round(total_ms / max(len(texts), 1), 3)

        results: List[PredictionResult] = []
        for text, pred, probs in zip(validated_texts, raw_preds, probs_matrix):
            ham_prob = float(probs[0])
            spam_prob = float(probs[1])
            label = "spam" if pred == 1 else "ham"
            confidence = spam_prob if label == "spam" else ham_prob
            snippet = text[:80] + "..." if len(text) > 80 else text

            results.append(
                PredictionResult(
                    label=label,
                    is_spam=(label == "spam"),
                    spam_probability=round(spam_prob, 4),
                    ham_probability=round(ham_prob, 4),
                    confidence=round(confidence, 4),
                    processing_time_ms=avg_ms,
                    message_length=len(text),
                    sanitized_text_snippet=snippet,
                )
            )

        return results

"""Unit tests for src.inference and SpamClassifierService."""

from pathlib import Path
import tempfile
import joblib
import numpy as np
import pytest

from src.config import SAVED_MODEL_PATH
from src.inference import (
    ModelLoadError,
    ModelNotFoundError,
    PredictionResult,
    SpamClassifierService,
)


@pytest.fixture(scope="module")
def inference_service() -> SpamClassifierService:
    """Provides a loaded SpamClassifierService using the trained model artifact."""
    if not SAVED_MODEL_PATH.exists():
        pytest.skip("Model artifact not yet trained. Run train.py first.")
    return SpamClassifierService(SAVED_MODEL_PATH)


class TestSpamClassifierService:
    """Test suite for SpamClassifierService."""

    def test_predict_single_spam_message(self, inference_service: SpamClassifierService):
        text = "URGENT! You have won a £1,000 cash prize! Claim by calling 0800123456 now."
        result = inference_service.predict(text)

        assert isinstance(result, PredictionResult)
        assert result.label in {"ham", "spam"}
        assert result.is_spam is True
        assert 0.0 <= result.spam_probability <= 1.0
        assert 0.0 <= result.ham_probability <= 1.0
        assert pytest.approx(result.spam_probability + result.ham_probability, abs=1e-3) == 1.0
        assert result.processing_time_ms > 0
        assert len(result.sanitized_text_snippet) > 0

    def test_predict_single_ham_message(self, inference_service: SpamClassifierService):
        text = "Hey mom, I'll be home late tonight around 8pm for dinner."
        result = inference_service.predict(text)

        assert isinstance(result, PredictionResult)
        assert result.label == "ham"
        assert result.is_spam is False
        assert result.ham_probability > 0.5

    def test_predict_batch_messages(self, inference_service: SpamClassifierService):
        texts = [
            "Hello, are we still meeting today?",
            "Win £10,000 cash or car! Text PRIZE to 88088 now!",
            "Can you please send me the notes from lecture?",
        ]
        results = inference_service.predict_batch(texts)

        assert len(results) == 3
        assert [r.label for r in results] == ["ham", "spam", "ham"]

    def test_rejects_empty_and_whitespace_input(self, inference_service: SpamClassifierService):
        with pytest.raises(ValueError, match="cannot be empty"):
            inference_service.predict("")

        with pytest.raises(ValueError, match="cannot be empty"):
            inference_service.predict("   \n\t  ")

    def test_rejects_non_string_input(self, inference_service: SpamClassifierService):
        with pytest.raises(TypeError, match="cannot be None"):
            inference_service.predict(None)

        with pytest.raises(TypeError, match="must be a string"):
            inference_service.predict(99999)

    def test_handles_unicode_emojis_and_special_chars(self, inference_service: SpamClassifierService):
        # Emojis, punctuation, URLs, numbers
        special_text = "🎉 Congratulations 🏆 http://prize.com £££ 100% FREE $$$ call 0800!!"
        result = inference_service.predict(special_text)
        assert result.label in {"ham", "spam"}

    def test_handles_unusually_long_input_safely(self, inference_service: SpamClassifierService):
        # 1000 characters (valid)
        long_text = "Hello " * 160
        result = inference_service.predict(long_text)
        assert result.label in {"ham", "spam"}

        # Exceeding 5000 characters (rejected)
        oversized = "A" * 5001
        with pytest.raises(ValueError, match="exceeds safety limit"):
            inference_service.predict(oversized)

    def test_model_not_found_error(self, tmp_path: Path):
        nonexistent = tmp_path / "missing_model.joblib"
        with pytest.raises(ModelNotFoundError, match="Trained model artifact not found"):
            SpamClassifierService(nonexistent)

    def test_model_corrupted_error(self, tmp_path: Path):
        corrupted = tmp_path / "corrupted_model.joblib"
        corrupted.write_text("This is not a valid serialized joblib file", encoding="utf-8")
        with pytest.raises(ModelLoadError, match="Failed to load model"):
            SpamClassifierService(corrupted)


def test_model_persistence_and_reload_identity():
    """Verifies that reloading the serialized pipeline produces identical predictions."""
    if not SAVED_MODEL_PATH.exists():
        pytest.skip("Model artifact not yet trained.")

    pipe1 = joblib.load(SAVED_MODEL_PATH)
    pipe2 = joblib.load(SAVED_MODEL_PATH)

    test_samples = [
        "Are you available for a phone call?",
        "Urgent: Text STOP to 12345 to opt out of £1.50/week charges.",
    ]

    preds1 = pipe1.predict(test_samples)
    preds2 = pipe2.predict(test_samples)
    np.testing.assert_array_equal(preds1, preds2)

    probs1 = pipe1.predict_proba(test_samples)
    probs2 = pipe2.predict_proba(test_samples)
    np.testing.assert_allclose(probs1, probs2, atol=1e-6)

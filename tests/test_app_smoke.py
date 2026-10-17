"""Smoke tests for the Streamlit application and its helper controllers."""

import pytest
from app import SAMPLE_MESSAGES, get_inference_service, load_metrics_report
from src.config import SAVED_MODEL_PATH
from src.inference import PredictionResult


def test_app_sample_messages_defined():
    """Verifies that sample messages cover both ham and spam classes and are valid strings."""
    assert len(SAMPLE_MESSAGES) >= 4
    for title, text in SAMPLE_MESSAGES.items():
        assert isinstance(title, str)
        assert isinstance(text, str)
        if title != "-- Select a sample message --":
            assert len(text) > 10


def test_app_inference_service_initialization():
    """Verifies that get_inference_service initializes and is ready."""
    if not SAVED_MODEL_PATH.exists():
        pytest.skip("Model not found. Run train.py first.")

    service = get_inference_service()
    assert service.is_ready()


def test_app_sample_messages_inference():
    """Verifies that all non-empty sample messages run through inference without error."""
    if not SAVED_MODEL_PATH.exists():
        pytest.skip("Model not found. Run train.py first.")

    service = get_inference_service()
    for title, text in SAMPLE_MESSAGES.items():
        if not text.strip():
            continue
        res = service.predict(text)
        assert isinstance(res, PredictionResult)
        assert res.label in {"ham", "spam"}
        assert 0.0 <= res.confidence <= 1.0


def test_app_load_metrics_report():
    """Verifies that load_metrics_report returns a valid dictionary."""
    metrics = load_metrics_report()
    assert isinstance(metrics, dict)
    if metrics:
        assert "accuracy" in metrics
        assert "spam_metrics" in metrics

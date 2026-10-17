"""Unit tests verifying that evaluation metrics reports contain all required fields."""

import json
from pathlib import Path
import pytest

from src.config import COMPARISON_CSV_PATH, COMPARISON_JSON_PATH, METRICS_JSON_PATH


def test_metrics_json_report_structure_and_bounds():
    """Verifies that the generated metrics JSON contains all required fields and valid ranges."""
    if not METRICS_JSON_PATH.exists():
        pytest.skip(f"Metrics file {METRICS_JSON_PATH} not found. Run train.py first.")

    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Core high-level keys
    required_keys = [
        "model_name",
        "sample_count",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "spam_metrics",
        "ham_metrics",
        "confusion_matrix",
        "training_time_sec",
        "inference_latency",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing required top-level metric: {key}"

    # Verify per-class sub-keys
    for class_key in ["spam_metrics", "ham_metrics"]:
        sub = metrics[class_key]
        for sub_metric in ["precision", "recall", "f1_score", "support"]:
            assert sub_metric in sub, f"Missing {sub_metric} in {class_key}"
            if sub_metric != "support":
                assert 0.0 <= sub[sub_metric] <= 1.0, f"Out of bounds: {class_key}.{sub_metric}"

    # Verify confusion matrix
    cm = metrics["confusion_matrix"]
    for cm_key in ["tn", "fp", "fn", "tp", "matrix"]:
        assert cm_key in cm, f"Missing {cm_key} in confusion_matrix"
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == metrics["sample_count"]

    # Verify aggregates
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["weighted_f1"] <= 1.0

    # Latency structure
    latency = metrics["inference_latency"]
    assert "mean_single_prediction_ms" in latency
    assert "throughput_samples_per_sec" in latency
    assert latency["mean_single_prediction_ms"] > 0


def test_model_comparison_report_exists_and_contains_both_models():
    """Verifies that the model comparison report exists and contains Naive Bayes and Logistic Regression."""
    if not COMPARISON_JSON_PATH.exists() or not COMPARISON_CSV_PATH.exists():
        pytest.skip("Comparison report not found. Run train.py first.")

    with open(COMPARISON_JSON_PATH, "r", encoding="utf-8") as f:
        comp = json.load(f)

    assert len(comp) >= 2, "Expected at least 2 comparison models"
    model_names = [m["Model"] for m in comp]

    # Must contain both Naive Bayes and Logistic Regression
    assert any("MultinomialNB" in name for name in model_names), "MultinomialNB missing from comparison"
    assert any("LogisticRegression" in name for name in model_names), "LogisticRegression missing from comparison"

    # Verify one model is selected
    selected_count = sum(1 for m in comp if m.get("Selected") is True)
    assert selected_count == 1, "Exactly one model must be marked Selected"

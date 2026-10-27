"""Unit tests verifying that evaluation metrics reports contain all required fields."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.config import COMPARISON_CSV_PATH, COMPARISON_JSON_PATH, METRICS_JSON_PATH
from src.evaluation import compute_comprehensive_metrics, plot_confusion_matrix


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


def test_metric_calculation_known_small_examples():
    """Verifies precision, recall, F1, and confusion matrix against manually calculated small fixtures."""
    # 4 samples:
    # Sample 0: true=0 (ham),  pred=0 (ham)  -> TN
    # Sample 1: true=0 (ham),  pred=1 (spam) -> FP
    # Sample 2: true=1 (spam), pred=0 (ham)  -> FN
    # Sample 3: true=1 (spam), pred=1 (spam) -> TP
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.3, 0.9])

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob, model_name="FixtureModel")

    # Overall counts
    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == 0.50

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    assert cm["tn"] == 1
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tp"] == 1
    assert cm["matrix"] == [[1, 1], [1, 1]]

    # Spam metrics (positive class = 1)
    # Precision: TP / (TP + FP) = 1 / 2 = 0.50
    # Recall: TP / (TP + FN) = 1 / 2 = 0.50
    # F1: 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.50
    spam_m = metrics["spam_metrics"]
    assert spam_m["precision"] == 0.50
    assert spam_m["recall"] == 0.50
    assert spam_m["f1_score"] == 0.50
    assert spam_m["support"] == 2

    # Ham metrics (negative class = 0)
    ham_m = metrics["ham_metrics"]
    assert ham_m["precision"] == 0.50
    assert ham_m["recall"] == 0.50
    assert ham_m["f1_score"] == 0.50
    assert ham_m["support"] == 2


def test_confusion_matrix_plot_generation(tmp_path: Path):
    """Verifies that plot_confusion_matrix creates a valid non-empty PNG file."""
    dummy_cm = np.array([[800, 5], [10, 120]])
    target_png = tmp_path / "test_confusion_matrix.png"

    result_path = plot_confusion_matrix(
        cm=dummy_cm,
        model_name="TestClassifier",
        output_path=target_png,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 1000  # Non-trivial image file
    # Check PNG magic header bytes
    with open(result_path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"

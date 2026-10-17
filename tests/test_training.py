"""Unit tests for src.training."""

import numpy as np
import pandas as pd
import pytest

from src.training import (
    get_candidate_pipelines,
    prepare_train_test_split,
    select_best_model,
)


def test_get_candidate_pipelines_structure():
    """Verifies candidate pipelines have the required step names and estimators."""
    pipelines = get_candidate_pipelines()
    assert "MultinomialNB_Count" in pipelines
    assert "MultinomialNB_TFIDF" in pipelines
    assert "LogisticRegression_TFIDF" in pipelines

    for name, pipe in pipelines.items():
        assert "preprocessor" in pipe.named_steps
        assert "vectorizer" in pipe.named_steps
        assert "classifier" in pipe.named_steps


def test_prepare_train_test_split_stratification_and_reproducibility():
    """Verifies that split is stratified and perfectly reproducible with fixed seed."""
    records = []
    # 800 ham, 200 spam (20% spam ratio)
    for i in range(800):
        records.append({"label": "ham", "text": f"Ham unique message {i}"})
    for i in range(200):
        records.append({"label": "spam", "text": f"Spam unique message {i}"})

    df = pd.DataFrame(records)

    train1, test1 = prepare_train_test_split(df, test_size=0.20, random_state=42)
    train2, test2 = prepare_train_test_split(df, test_size=0.20, random_state=42)

    # Determinism
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)

    # Size check
    assert len(train1) == 800
    assert len(test1) == 200

    # Stratification check (spam ratio should be preserved at ~20%)
    train_spam_ratio = train1["target"].mean()
    test_spam_ratio = test1["target"].mean()
    assert pytest.approx(train_spam_ratio, abs=0.01) == 0.20
    assert pytest.approx(test_spam_ratio, abs=0.01) == 0.20


def test_select_best_model_prefers_precision_and_macro_f1():
    """Verifies model selection logic balances Macro-F1 and Spam Precision."""
    cv_dummy = {
        "ModelA": {
            "cv_macro_f1_mean": 0.95,
            "cv_spam_precision_mean": 0.99,  # High precision, low false positive
        },
        "ModelB": {
            "cv_macro_f1_mean": 0.96,
            "cv_spam_precision_mean": 0.88,  # Lower precision, more false alarms
        },
    }
    # Score A: 0.5 * 0.95 + 0.5 * 0.99 = 0.970
    # Score B: 0.5 * 0.96 + 0.5 * 0.88 = 0.920
    selected = select_best_model(cv_dummy)
    assert selected == "ModelA"

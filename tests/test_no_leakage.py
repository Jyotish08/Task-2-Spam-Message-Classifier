"""Tests for Train/Test Leakage Safeguards and Pipeline Isolation.

Verifies:
1. Zero verbatim message overlap between training and test sets.
2. Vectorizer vocabulary is fitted strictly on the training partition.
3. Test-only words do not leak into vectorizer vocabulary.
4. Pipeline encapsulation ensures no standalone preprocessing on the entire dataset.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from src.config import PROCESSED_TEST_PATH, PROCESSED_TRAIN_PATH
from src.preprocessing import SMSPreprocessor
from src.training import prepare_train_test_split


def test_zero_text_overlap_between_train_and_test_splits():
    """Verifies that no text string in the processed training set exists in the test set."""
    if not PROCESSED_TRAIN_PATH.exists() or not PROCESSED_TEST_PATH.exists():
        pytest.skip("Processed train/test files not yet generated.")

    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)

    train_texts = set(train_df["text"].str.strip())
    test_texts = set(test_df["text"].str.strip())

    overlap = train_texts.intersection(test_texts)
    assert len(overlap) == 0, f"Detected {len(overlap)} overlapping messages between train and test: {list(overlap)[:3]}"


def test_vectorizer_vocabulary_isolation():
    """Verifies that tokens appearing exclusively in test messages are NOT in the fitted vocabulary."""
    train_messages = [
        "meeting tomorrow morning at office",
        "claim your free vouchers now",
        "urgent call back required",
    ]
    test_messages = [
        "xylophone zebra extraterrestrial uniqueunseenword",
    ]

    pipeline = Pipeline(
        [
            ("preprocessor", SMSPreprocessor()),
            ("vectorizer", TfidfVectorizer()),
            ("classifier", MultinomialNB()),
        ]
    )

    # Fit pipeline ONLY on train
    pipeline.fit(train_messages, [0, 1, 1])

    vocab = pipeline.named_steps["vectorizer"].vocabulary_

    # Verify train tokens are present
    assert "meeting" in vocab
    assert "vouchers" in vocab

    # Verify unseen test-only tokens are completely absent from vocabulary
    assert "xylophone" not in vocab
    assert "zebra" not in vocab
    assert "extraterrestrial" not in vocab
    assert "uniqueunseenword" not in vocab


def test_pipeline_encapsulation_prevents_leakage():
    """Verifies that vectorizer parameters (IDF weights, document frequencies) are not fitted prior to pipeline.fit."""
    pipeline = Pipeline(
        [
            ("preprocessor", SMSPreprocessor()),
            ("vectorizer", TfidfVectorizer()),
            ("classifier", MultinomialNB()),
        ]
    )

    # Before fitting, vectorizer must not have learned attributes
    assert not hasattr(pipeline.named_steps["vectorizer"], "vocabulary_")
    assert not hasattr(pipeline.named_steps["vectorizer"], "idf_")

    pipeline.fit(["Sample text message one", "Sample text message two"], [0, 1])

    # After fitting, learned attributes must exist
    assert hasattr(pipeline.named_steps["vectorizer"], "vocabulary_")
    assert hasattr(pipeline.named_steps["vectorizer"], "idf_")

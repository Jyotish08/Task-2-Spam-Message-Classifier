"""End-to-End smoke test exercising the complete train/evaluate/load/predict lifecycle."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluation import compute_comprehensive_metrics
from src.inference import SpamClassifierService
from src.preprocessing import SMSPreprocessor, get_custom_stop_words
from src.training import get_candidate_pipelines, prepare_train_test_split


def test_full_pipeline_train_evaluate_serialize_and_predict(tmp_path: Path):
    """Executes a self-contained, isolated end-to-end pipeline run from synthetic data to inference."""
    # 1. Generate clean synthetic dataset
    records = []
    for i in range(40):
        records.append({"label": "ham", "text": f"Casual reminder about our group meeting at 3pm on Friday {i}."})
    for i in range(20):
        records.append({"label": "spam", "text": f"WINNER! Free cash prize £{100 + i}! Text CLAIM to 88088 now."})

    df = pd.DataFrame(records)

    # 2. Split with zero leakage
    train_df, test_df = prepare_train_test_split(df, test_size=0.25, random_state=42)
    assert len(train_df) == 45
    assert len(test_df) == 15

    # 3. Build and train pipeline
    pipelines = get_candidate_pipelines()
    pipeline = pipelines["MultinomialNB_TFIDF"]
    pipeline.fit(train_df["text"].tolist(), train_df["target"].values)

    # 4. Evaluate on test partition
    y_test = test_df["target"].values
    y_pred = pipeline.predict(test_df["text"].tolist())
    y_prob = pipeline.predict_proba(test_df["text"].tolist())[:, 1]

    metrics = compute_comprehensive_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["spam_metrics"]["precision"] <= 1.0
    assert 0.0 <= metrics["spam_metrics"]["recall"] <= 1.0

    # 5. Serialize pipeline artifact to temp location
    artifact_path = tmp_path / "smoke_pipeline.joblib"
    joblib.dump(pipeline, artifact_path)
    assert artifact_path.exists()

    # 6. Instantiate inference service with serialized artifact
    service = SpamClassifierService(model_path=artifact_path)
    assert service.is_ready()

    # 7. Predict on new incoming messages
    ham_res = service.predict("Are we meeting for lunch at 12:30 today?")
    assert ham_res.label in {"ham", "spam"}
    assert 0.0 <= ham_res.confidence <= 1.0

    spam_res = service.predict("URGENT! Text CLAIM to 88088 to receive your £1000 cash bonus now!")
    assert spam_res.label == "spam"
    assert spam_res.is_spam is True
    assert spam_res.spam_probability > 0.50

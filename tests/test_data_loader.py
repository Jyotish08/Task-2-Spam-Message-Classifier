"""Unit tests for src.data_loader."""

import json
from pathlib import Path
import tempfile
import pytest
import pandas as pd

from src.data_loader import (
    audit_dataset,
    compute_file_sha256,
    parse_sms_spam_file,
    validate_labels,
)


def test_parse_sms_spam_file_valid_and_malformed(tmp_path: Path):
    """Verifies that parse_sms_spam_file correctly extracts valid lines and catches malformed ones."""
    sample_content = (
        "ham\tHello, this is a normal message.\n"
        "spam\tFree entry in a £1000 prize draw! Text WIN to 80088.\n"
        "invalid_label\tThis line has an invalid label.\n"
        "This line has no tab separator at all.\n"
        "ham\tAnother valid ham message with numbers 12345.\n"
        "\n"  # Empty line should be skipped silently
    )
    test_file = tmp_path / "test_spam.txt"
    test_file.write_text(sample_content, encoding="utf-8")

    df, malformed = parse_sms_spam_file(test_file)

    # Valid records: exactly 3
    assert len(df) == 3
    assert list(df.columns) == ["label", "text"]
    assert df["label"].tolist() == ["ham", "spam", "ham"]
    assert df["text"].iloc[0] == "Hello, this is a normal message."

    # Malformed records: exactly 2 (invalid label, missing tab)
    assert len(malformed) == 2
    reasons = [m["reason"] for m in malformed]
    assert any("Invalid label" in r for r in reasons)
    assert any("Missing tab separator" in r for r in reasons)


def test_parse_sms_spam_file_missing_file():
    """Verifies FileNotFoundError when pointing to a nonexistent file."""
    nonexistent = Path("nonexistent_dir/does_not_exist.tsv")
    with pytest.raises(FileNotFoundError):
        parse_sms_spam_file(nonexistent)


def test_audit_dataset_metrics():
    """Verifies that audit_dataset computes accurate summary statistics."""
    records = [
        {"label": "ham", "text": "Are you coming today?"},
        {"label": "ham", "text": "Yes, I will be there at 5."},
        {"label": "ham", "text": "Are you coming today?"},  # duplicate ham
        {"label": "spam", "text": "Claim £500 cash now!"},
    ]
    df = pd.DataFrame(records)
    audit = audit_dataset(df, malformed_records=[])

    assert audit["total_records"] == 4
    assert audit["class_distribution"]["ham_count"] == 3
    assert audit["class_distribution"]["spam_count"] == 1
    assert audit["class_distribution"]["spam_percentage"] == 25.0
    assert audit["data_integrity"]["unique_messages_count"] == 3
    assert audit["data_integrity"]["duplicate_messages_count"] == 1
    assert audit["data_integrity"]["conflicting_label_messages_count"] == 0
    assert "char_length" in audit["length_distributions"]
    assert "token_length" in audit["length_distributions"]


def test_compute_file_sha256(tmp_path: Path):
    """Verifies deterministic SHA256 calculation."""
    test_file = tmp_path / "dummy.txt"
    test_file.write_text("Deterministic test payload for SHA256", encoding="utf-8")

    hash1 = compute_file_sha256(test_file)
    hash2 = compute_file_sha256(test_file)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_validate_labels_valid_and_invalid():
    """Verifies that validate_labels accepts ham/spam and raises ValueError for invalid/missing labels."""
    # Valid series should pass without error
    valid_series = pd.Series(["ham", "spam", "ham", "spam"])
    validate_labels(valid_series)

    # Invalid label should raise ValueError
    invalid_series = pd.Series(["ham", "unsolicited_commercial", "spam"])
    with pytest.raises(ValueError, match="invalid target labels"):
        validate_labels(invalid_series)

    # Missing label (NaN) should raise ValueError
    missing_series = pd.Series(["ham", None, "spam"])
    with pytest.raises(ValueError, match="missing .* label values"):
        validate_labels(missing_series)

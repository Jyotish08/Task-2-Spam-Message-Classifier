"""Data Loader module for Spam Message Classifier.

Handles deterministic downloading, extraction, verification, parsing,
and auditing of the UCI SMS Spam Collection dataset.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import zipfile

import pandas as pd
import requests

from src.config import (
    DATA_AUDIT_JSON_PATH,
    DATASET_CITATION,
    DATASET_DOI,
    DATASET_LICENSE,
    RAW_DATA_PATH,
    RAW_METADATA_PATH,
    RAW_ZIP_PATH,
    UCI_DATASET_URL,
    VALID_LABELS,
)


def compute_file_sha256(filepath: Path) -> str:
    """Computes the SHA256 hexadecimal hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_uci_dataset(
    url: str = UCI_DATASET_URL,
    zip_target: Path = RAW_ZIP_PATH,
    extract_dir: Path = RAW_DATA_PATH.parent,
    force_download: bool = False,
) -> Path:
    """Downloads the official UCI SMS Spam Collection archive and extracts it.

    Parameters
    ----------
    url : str
        Official UCI archive URL.
    zip_target : Path
        Destination path for the zip archive.
    extract_dir : Path
        Directory where archive contents will be extracted.
    force_download : bool
        If True, re-downloads even if the raw data file already exists.

    Returns
    -------
    Path
        Path to the extracted SMSSpamCollection file.

    Raises
    ------
    RuntimeError
        If download or extraction fails.
    """
    if RAW_DATA_PATH.exists() and not force_download:
        return RAW_DATA_PATH

    extract_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download zip archive
    if not zip_target.exists() or force_download:
        try:
            response = requests.get(
                url,
                stream=True,
                timeout=30,
                headers={"User-Agent": "CodeNova-SpamClassifier/1.0"},
            )
            response.raise_for_status()
            with open(zip_target, "wb") as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to download UCI SMS Spam Collection dataset from {url}.\n"
                f"Error: {exc}\n"
                f"Manual Resolution Procedure:\n"
                f"1. Download '{url}' manually in your browser.\n"
                f"2. Save the zip file to: {zip_target.resolve()}\n"
                f"3. Extract 'SMSSpamCollection' to: {RAW_DATA_PATH.resolve()}"
            ) from exc

    # Step 2: Extract archive
    try:
        with zipfile.ZipFile(zip_target, "r") as archive:
            archive.extractall(extract_dir)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"The downloaded file '{zip_target}' is not a valid zip archive. "
            "Please delete it and rerun to re-download."
        ) from exc

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected raw file '{RAW_DATA_PATH.name}' not found after extracting '{zip_target}'."
        )

    # Step 3: Record provenance metadata
    sha256_hash = compute_file_sha256(RAW_DATA_PATH)
    metadata: Dict[str, Any] = {
        "dataset_name": "SMS Spam Collection",
        "source_url": url,
        "citation": DATASET_CITATION,
        "doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file_name": RAW_DATA_PATH.name,
        "raw_file_sha256": sha256_hash,
        "raw_file_size_bytes": RAW_DATA_PATH.stat().st_size,
    }
    with open(RAW_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return RAW_DATA_PATH


def parse_sms_spam_file(
    file_path: Path = RAW_DATA_PATH,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Parses the tab-delimited SMSSpamCollection file with rigorous validation.

    Each record is expected to follow the format:
        <label>\\t<message text>

    Parameters
    ----------
    file_path : Path
        Path to the raw SMSSpamCollection file.

    Returns
    -------
    Tuple[pd.DataFrame, List[Dict[str, Any]]]
        - DataFrame containing valid records with columns: 'label', 'text'
        - List of malformed records detected during parsing.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at {file_path}. Run download_uci_dataset() first."
        )

    valid_records: List[Dict[str, str]] = []
    malformed_records: List[Dict[str, Any]] = []

    # The official dataset is encoded in UTF-8 or latin-1
    # We attempt utf-8 with fallback to latin-1
    content: str
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            lines = f.readlines()

    for line_idx, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if not line:
            # Empty trailing line
            continue

        parts = line.split("\t", 1)
        if len(parts) != 2:
            malformed_records.append(
                {
                    "line_number": line_idx,
                    "reason": "Missing tab separator",
                    "preview": line[:80],
                }
            )
            continue

        raw_label, raw_text = parts[0].strip().lower(), parts[1].strip()

        if raw_label not in VALID_LABELS:
            malformed_records.append(
                {
                    "line_number": line_idx,
                    "reason": f"Invalid label '{raw_label}'",
                    "preview": line[:80],
                }
            )
            continue

        valid_records.append({"label": raw_label, "text": raw_text})

    df = pd.DataFrame(valid_records)
    return df, malformed_records


def validate_labels(labels: pd.Series) -> None:
    """Validates that all target labels are strictly known and valid ('ham' or 'spam').

    Parameters
    ----------
    labels : pd.Series
        Series of raw or normalized label strings.

    Raises
    ------
    ValueError
        If any invalid, missing, or unrecognized labels are detected.
    """
    if labels.isna().any():
        raise ValueError("Dataset contains missing (NaN) label values.")

    invalid = set(labels.dropna().unique()) - VALID_LABELS
    if invalid:
        raise ValueError(
            f"Dataset contains invalid target labels: {invalid}. "
            f"Expected only valid labels: {sorted(list(VALID_LABELS))}."
        )


def audit_dataset(
    df: pd.DataFrame,
    malformed_records: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Performs a comprehensive audit of the SMS Spam dataset.

    Computes:
    - Shape and column names
    - Class balance and imbalance ratio
    - Empty and missing messages
    - Exact duplicate analysis (including label conflicting duplicates)
    - Character and token length distributions

    Parameters
    ----------
    df : pd.DataFrame
        Parsed dataset with columns 'label' and 'text'.
    malformed_records : Optional[List[Dict[str, Any]]]
        List of malformed lines detected during raw parsing.
    save_path : Optional[Path]
        If provided, persists the audit report as JSON to this path.

    Returns
    -------
    Dict[str, Any]
        Audit report dictionary.
    """
    total_records = len(df)
    class_counts = df["label"].value_counts().to_dict()
    ham_count = class_counts.get("ham", 0)
    spam_count = class_counts.get("spam", 0)
    spam_percentage = (spam_count / total_records * 100) if total_records > 0 else 0.0

    # Missing & empty values
    missing_text = int(df["text"].isna().sum())
    empty_text = int((df["text"].str.strip() == "").sum())

    # Duplicate analysis
    total_exact_duplicates = int(df.duplicated(subset=["text"]).sum())
    unique_texts = int(df["text"].nunique())

    # Conflicting duplicates (same text, different label)
    text_label_nunique = df.groupby("text")["label"].nunique()
    conflicting_texts = text_label_nunique[text_label_nunique > 1].index.tolist()
    conflicting_count = len(conflicting_texts)

    # Length statistics
    df_with_len = df.copy()
    df_with_len["char_length"] = df_with_len["text"].str.len()
    df_with_len["token_length"] = df_with_len["text"].apply(lambda t: len(str(t).split()))

    def get_distribution_stats(series: pd.Series) -> Dict[str, float]:
        desc = series.describe(percentiles=[0.25, 0.50, 0.75, 0.95]).to_dict()
        return {
            "mean": round(float(desc.get("mean", 0.0)), 2),
            "std": round(float(desc.get("std", 0.0)), 2),
            "min": int(desc.get("min", 0)),
            "p25": int(desc.get("25%", 0)),
            "median": int(desc.get("50%", 0)),
            "p75": int(desc.get("75%", 0)),
            "p95": int(desc.get("95%", 0)),
            "max": int(desc.get("max", 0)),
        }

    ham_subset = df_with_len[df_with_len["label"] == "ham"]
    spam_subset = df_with_len[df_with_len["label"] == "spam"]

    audit_report: Dict[str, Any] = {
        "total_records": total_records,
        "columns": list(df.columns),
        "class_distribution": {
            "ham_count": ham_count,
            "spam_count": spam_count,
            "spam_percentage": round(spam_percentage, 2),
            "imbalance_ratio": f"1 spam : {round(ham_count / max(spam_count, 1), 2)} ham",
        },
        "data_integrity": {
            "missing_text_count": missing_text,
            "empty_text_count": empty_text,
            "unique_messages_count": unique_texts,
            "duplicate_messages_count": total_exact_duplicates,
            "conflicting_label_messages_count": conflicting_count,
            "conflicting_samples_preview": conflicting_texts[:5],
            "malformed_lines_count": len(malformed_records) if malformed_records else 0,
            "malformed_lines_preview": malformed_records[:3] if malformed_records else [],
        },
        "length_distributions": {
            "char_length": {
                "overall": get_distribution_stats(df_with_len["char_length"]),
                "ham": get_distribution_stats(ham_subset["char_length"]),
                "spam": get_distribution_stats(spam_subset["char_length"]),
            },
            "token_length": {
                "overall": get_distribution_stats(df_with_len["token_length"]),
                "ham": get_distribution_stats(ham_subset["token_length"]),
                "spam": get_distribution_stats(spam_subset["token_length"]),
            },
        },
    }

    # Save to metrics directory only when save_path is explicitly provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)

    return audit_report


def load_raw_dataset(
    force_download: bool = False,
    save_audit: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience pipeline to download (if needed), parse, and audit the dataset.

    Parameters
    ----------
    force_download : bool, default=False
        Whether to re-download the archive even if SMSSpamCollection exists.
    save_audit : bool, default=True
        Whether to persist the resulting audit JSON report to DATA_AUDIT_JSON_PATH.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - Valid DataFrame
        - Audit report dictionary
    """
    raw_path = download_uci_dataset(force_download=force_download)
    df, malformed = parse_sms_spam_file(raw_path)
    validate_labels(df["label"])
    save_path = DATA_AUDIT_JSON_PATH if save_audit else None
    audit = audit_dataset(df, malformed, save_path=save_path)
    return df, audit

"""Configuration module for Spam Message Classifier.

Defines deterministic paths, random seeds, model hyperparameter grids,
and dataset source metadata.
"""

from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Ensure all primary directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Provenance & Metadata
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_ZIP_NAME = "sms+spam+collection.zip"
DATASET_RAW_FILENAME = "SMSSpamCollection"
DATASET_README_FILENAME = "readme"
DATASET_METADATA_FILENAME = "metadata.json"

DATASET_CITATION = (
    "Almeida, T. and Hidalgo, J. (2011). SMS Spam Collection. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84"
)
DATASET_DOI = "https://doi.org/10.24432/C5CC84"
DATASET_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"

# File Paths
RAW_ZIP_PATH = RAW_DATA_DIR / DATASET_ZIP_NAME
RAW_DATA_PATH = RAW_DATA_DIR / DATASET_RAW_FILENAME
RAW_METADATA_PATH = RAW_DATA_DIR / DATASET_METADATA_FILENAME

PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

SAVED_MODEL_PATH = MODELS_DIR / "spam_classifier_pipeline.joblib"
METRICS_JSON_PATH = METRICS_DIR / "evaluation_metrics.json"
COMPARISON_JSON_PATH = METRICS_DIR / "model_comparison.json"
COMPARISON_CSV_PATH = METRICS_DIR / "model_comparison.csv"
CONFUSION_MATRIX_PNG = FIGURES_DIR / "confusion_matrix.png"
ROC_PR_CURVES_PNG = FIGURES_DIR / "roc_pr_curves.png"
CLASS_DISTRIBUTION_PNG = FIGURES_DIR / "class_distribution.png"
LENGTH_DISTRIBUTION_PNG = FIGURES_DIR / "message_length_distribution.png"
DATA_AUDIT_JSON_PATH = METRICS_DIR / "data_audit.json"
THRESHOLD_ANALYSIS_JSON_PATH = METRICS_DIR / "threshold_analysis.json"

# Reproducibility & Splitting Configuration
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Label Definitions
LABEL_HAM = "ham"
LABEL_SPAM = "spam"
VALID_LABELS = {LABEL_HAM, LABEL_SPAM}

LABEL_ENCODING = {
    LABEL_HAM: 0,
    LABEL_SPAM: 1,
}
LABEL_DECODING = {
    0: LABEL_HAM,
    1: LABEL_SPAM,
}

# Max allowed input length for safety (in characters)
MAX_INPUT_LENGTH = 5000

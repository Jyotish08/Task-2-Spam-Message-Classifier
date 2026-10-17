# SMS Spam Message Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![App: Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Tests: Pytest](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)](tests/)

A reproducible, production-grade Natural Language Processing (NLP) system and interactive Streamlit web application for detecting SMS spam messages. Developed for the **CodeNova AI/ML Internship Program, Batch 01 — Task 2**.

---

## Table of Contents
- [1. Problem Statement & Objectives](#1-problem-statement--objectives)
- [2. Dataset Attribution & Audit](#2-dataset-attribution--audit)
- [3. Project Architecture](#3-project-architecture)
- [4. Installation & Environment Setup](#4-installation--environment-setup)
- [5. Reproducible Training Pipeline](#5-reproducible-training-pipeline)
- [6. Evaluation Methodology & Model Selection](#6-evaluation-methodology--model-selection)
- [7. Actual Test Set Benchmark Results](#7-actual-test-set-benchmark-results)
- [8. Error Analysis: Plain-Language Inspection](#8-error-analysis-plain-language-inspection)
- [9. Inference & Interactive Prediction](#9-inference--interactive-prediction)
- [10. Interactive Streamlit Web Application](#10-interactive-streamlit-web-application)
- [11. Testing & Quality Assurance](#11-testing--quality-assurance)
- [12. Limitations & Security Considerations](#12-limitations--security-considerations)
- [13. CodeNova Task 2 Compliance Checklist](#13-codenova-task-2-compliance-checklist)

---

## 1. Problem Statement & Objectives

Mobile spam and unsolicited SMS messages expose smartphone users to financial fraud, malware phishing links, and unwanted promotional bombardment. The goal of this project is to develop an end-to-end, leak-free text classification system that accurately flags spam messages while strictly minimizing False Positives (preventing legitimate messages such as banking OTPs, personal updates, or medical notifications from being wrongly classified as spam).

### Core Task 2 Requirements
1. **Text Preprocessing**: Robust Unicode normalization (NFKC), currency/financial token preservation (`£`, `$`, `€`), punctuation separation, whitespace normalization, and boundary validation.
2. **Feature Extraction**: TF-IDF (unigram + bigram) and Count Vectorization with sublinear term-frequency scaling.
3. **Algorithm Comparison**: Multi-model benchmarking comparing **Multinomial Naive Bayes** (Count and TF-IDF) against **Logistic Regression** (TF-IDF).
4. **Comprehensive Evaluation**: Metrics reporting for Precision, Recall, F1-Score per class, Macro-F1, Weighted-F1, ROC-AUC, PR-AUC, and full Confusion Matrix.
5. **Interactive Prediction**: Streamlit web interface and versatile command-line interface (CLI) with confidence and probability scoring.
6. **Artifact Persistence**: Fully serialized, reproducible Scikit-Learn `Pipeline` saved via Joblib.

---

## 2. Dataset Attribution & Audit

### Provenance & Citation
The dataset used is the official **UCI Machine Learning Repository SMS Spam Collection**:

> **Citation:**  
> Almeida, T. and Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository.  
> [https://doi.org/10.24432/C5CC84](https://doi.org/10.24432/C5CC84)

- **Source Archive**: `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`
- **Format**: Tab-separated records (`label\tmessage text`)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

### Dataset Audit Findings
Automated dataset profiling (`reports/metrics/data_audit.json`) revealed:
- **Total Records**: 5,574
- **Raw Class Breakdown**:
  - `ham` (legitimate): 4,827 (86.60%)
  - `spam`: 747 (13.40%)
  - **Imbalance Ratio**: 1 spam for every 6.46 ham messages
- **Data Integrity**:
  - Missing text values: 0
  - Empty text values: 0
  - Conflicting label duplicates: 0
  - Malformed records: 0
- **Duplicate & Leakage Analysis**:
  - The raw dataset contains 414 duplicate messages (94 repeated spams, 320 repeated hams).
  - 11 messages differed solely by leading/trailing whitespace.
  - **Zero-Leakage Safeguard**: Our training pipeline strips whitespace and deduplicates identical texts before splitting, ensuring **0 verbatim message overlap** between the training partition and the held-out test partition.
- **Length Distributions**:
  - **Ham**: Mean 71.4 characters (median 52 chars, max 910 chars), mean 14.3 tokens.
  - **Spam**: Mean 138.7 characters (median 149 chars, tightly bounded with max 223 chars), mean 23.9 tokens. Spam messages are concentrated near the traditional 160-character SMS limit.

---

## 3. Project Architecture

The repository enforces modular separation of concerns:

```
Task 2 Spam Message Classifier/
├── app.py                     # Interactive Streamlit Web Application
├── train.py                   # End-to-end training, CV, evaluation & serialization CLI
├── predict.py                 # Single, interactive, and batch CLI inference tool
├── requirements.txt           # Pinned & bounded dependency declarations
├── pyproject.toml             # Packaging metadata and pytest configuration
├── README.md                  # Complete technical documentation
├── LICENSE                    # MIT Open Source License
├── .gitignore                 # Excludes caches, venvs, and generated binaries
├── .env.example               # Template environment configuration
│
├── data/
│   ├── raw/                   # Downloaded raw zip archive and metadata
│   │   ├── metadata.json      # Provenance, SHA256 checksum, download timestamp
│   │   └── SMSSpamCollection  # Raw extracted TSV dataset
│   └── processed/             # Isolated stratified train/test CSV partitions
│       ├── train.csv          # 4,128 samples (80%)
│       └── test.csv           # 1,032 samples (20%)
│
├── models/
│   ├── .gitkeep
│   └── spam_classifier_pipeline.joblib  # Serialized fitted Pipeline artifact
│
├── reports/
│   ├── figures/
│   │   ├── class_distribution.png        # Bar chart of dataset class balance
│   │   ├── confusion_matrix.png          # Annotated heatmap of test confusion matrix
│   │   ├── message_length_distribution.png# Ham vs. Spam length density KDE
│   │   └── roc_pr_curves.png             # ROC and Precision-Recall curves
│   └── metrics/
│       ├── data_audit.json               # Full dataset profiling statistics
│       ├── model_comparison.csv          # Side-by-side CV and test metrics table
│       ├── model_comparison.json         # JSON structured benchmark comparisons
│       ├── evaluation_metrics.json       # Detailed test set metrics for chosen model
│       └── error_analysis.json           # Plain-language FP and FN case studies
│
├── src/
│   ├── __init__.py            # Package root
│   ├── config.py              # Central deterministic paths and hyperparameter constants
│   ├── data_loader.py         # Automated UCI downloader, parser, and auditor
│   ├── preprocessing.py       # Scikit-learn SMSPreprocessor & input validator
│   ├── training.py            # Pipeline factory, 5-fold Stratified CV, model selection
│   ├── evaluation.py          # Metric calculations, plotting routines, latency benchmarks
│   └── inference.py           # Production SpamClassifierService & PredictionResult
│
└── tests/
    ├── conftest.py            # Pytest path discovery setup
    ├── test_app_smoke.py      # Streamlit smoke tests and sample verification
    ├── test_data_loader.py    # TSV parser and audit unit tests
    ├── test_inference.py      # Inference service, edge cases, and corruption handling
    ├── test_metrics_report.py # Metrics JSON schema and numeric bounds validation
    ├── test_no_leakage.py     # Leakage safeguards & vocabulary isolation tests
    ├── test_preprocessing.py  # Input sanitization, unicode & transformer unit tests
    └── test_training.py       # Pipeline creation, stratification, and model selection tests
```

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 (CPython x86_64)
- Git

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Task 2 Spam Message Classifier"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # On Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install pinned dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify environment setup:**
   ```bash
   python -c "import pandas, sklearn, streamlit, pytest; print('Environment verified!')"
   ```

---

## 5. Reproducible Training Pipeline

To execute the end-to-end training pipeline, run:

```bash
python train.py
```

### What `train.py` executes:
1. **Automated Ingestion**: Checks if `data/raw/SMSSpamCollection` exists; if missing, automatically downloads `sms+spam+collection.zip` directly from UCI, verifies the SHA256 checksum, and extracts it.
2. **Data Audit**: Audits class imbalance, duplicate counts, and token length distributions, saving `reports/metrics/data_audit.json`.
3. **Stratified Split**: Performs an 80/20 train/test stratified split using `random_state=42`. Verifies 0 text overlap.
4. **Stratified 5-Fold Cross-Validation**: Evaluates `MultinomialNB_Count`, `MultinomialNB_TFIDF`, and `LogisticRegression_TFIDF` across all folds.
5. **Model Selection**: Selects the best pipeline based on the error-cost formula prioritizing Spam Precision while maintaining high Macro-F1.
6. **Held-Out Test Set Evaluation**: Evaluates the selected model on the isolated test set (1,032 samples).
7. **Artifact Generation**: Saves the fitted pipeline to `models/spam_classifier_pipeline.joblib`, generates all metric JSON/CSV reports, and renders high-resolution visualization figures.

---

## 6. Evaluation Methodology & Model Selection

### Asymmetric Error Costs in SMS Spam Filtering
In SMS spam classification, prediction errors have profoundly asymmetric real-world consequences:
- **False Positive (FP - High Cost)**: A legitimate message (e.g. medical appointment reminder, one-time banking passcode [OTP], flight update, or personal message from family) is misclassified as Spam. The user misses critical communication.
- **False Negative (FN - Low Cost)**: A spam message arrives in the user's inbox. The user experiences minor annoyance and manually discards the text.

**Selection Rule**: Because False Positives severely harm user trust, our model selection criteria explicitly prioritizes **Spam Precision** (minimizing False Positives) alongside **Macro-F1**:

$$\text{Composite Score} = 0.5 \times \text{CV Macro-F1} + 0.5 \times \text{CV Spam Precision}$$

### Cross-Validation Comparison on Training Split (N=4,128)

| Pipeline Model | Feature Vectorization | 5-Fold CV Macro-F1 | 5-Fold CV Spam Precision | 5-Fold CV Spam Recall | Selected |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MultinomialNB_Count** | CountVectorizer (1,2 n-grams) | 0.9677 (±0.007) | 0.9635 (±0.016) | 0.9160 (±0.021) | No |
| **MultinomialNB_TFIDF** | TfidfVectorizer (sublinear TF) | **0.9676** (±0.006) | **0.9872** (±0.014) | 0.8809 (±0.019) | **YES** |
| **LogisticRegression_TFIDF** | TfidfVectorizer (balanced) | 0.9670 (±0.008) | 0.9470 (±0.018) | 0.9238 (±0.020) | No |

**Rationale**: `MultinomialNB_TFIDF` achieved an outstanding **98.72% CV Spam Precision**, producing virtually zero false alarms on legitimate messages while achieving competitive Macro-F1.

---

## 7. Actual Test Set Benchmark Results

Evaluation was performed on the **untouched held-out test partition** ($N = 1,032$ samples: 904 Ham, 128 Spam).

### Side-by-Side Model Comparison Table (Held-Out Test Set)

| Model Name | Accuracy | Macro-F1 | Spam Precision | Spam Recall | Spam F1 | Ham F1 | ROC-AUC | PR-AUC | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MultinomialNB_Count** | 98.74% | 0.9709 | 95.28% | 94.53% | 0.9490 | 0.9928 | 0.9799 | 0.9651 | 6 | 7 |
| **MultinomialNB_TFIDF** *(Selected)* | **98.64%** | **0.9679** | **97.50%** | **91.41%** | **0.9435** | **0.9923** | **0.9884** | **0.9720** | **3** | **11** |
| **LogisticRegression_TFIDF** | 98.55% | 0.9669 | 93.13% | 95.31% | 0.9421 | 0.9917 | 0.9974 | 0.9861 | 9 | 6 |

### Selected Model Detailed Performance: `MultinomialNB_TFIDF`

- **Overall Accuracy**: **98.64%** (1,018 / 1,032 correct)
- **Macro-Averaged F1**: **0.9679**
- **Weighted-Averaged F1**: **0.9862**
- **ROC-AUC**: **0.9884** | **PR-AUC**: **0.9720**
- **Training Time**: 0.082 seconds
- **Single Prediction Latency**: 1.70 ms (P95: 1.99 ms)
- **Batch Throughput**: 16,702 samples / second

### Confusion Matrix Breakdown

```
                    PREDICTED HAM    PREDICTED SPAM
ACTUAL HAM (904)         901 (TN)           3 (FP)   <-- Only 3 legitimate messages flagged
ACTUAL SPAM (128)         11 (FN)         117 (TP)   <-- 117 spam messages caught
```

- **True Negatives (TN)**: 901 legitimate messages correctly delivered.
- **False Positives (FP)**: **3** (Spam False Alarm Rate: $3 / 904 = 0.33\%$).
- **False Negatives (FN)**: **11** (Missed spam rate: $11 / 128 = 8.59\%$).
- **True Positives (TP)**: **117** spam messages caught.

### Visualizations

The training pipeline generates visual artifacts in `reports/figures/`:

1. **Confusion Matrix**: `reports/figures/confusion_matrix.png`  
   Annotated heatmap displaying both raw counts and percentage proportions.
2. **ROC & PR Curves**: `reports/figures/roc_pr_curves.png`  
   Receiver Operating Characteristic (AUC = 0.9884) and Precision-Recall curve (AUC = 0.9720) demonstrating superior discrimination above the baseline.
3. **Class Distribution**: `reports/figures/class_distribution.png`  
   Visualizes the 6.46:1 imbalance between Ham (86.6%) and Spam (13.4%).
4. **Length Distribution**: `reports/figures/message_length_distribution.png`  
   Kernel Density Estimation comparing message lengths (ham median 52 chars vs. spam median 149 chars).

---

## 8. Error Analysis: Plain-Language Inspection

A rigorous ML system must examine and explain its failure modes (`reports/metrics/error_analysis.json`).

### 1. False Positives (Legitimate Ham misclassified as Spam: 3 cases)
- **Case 1**: `"Nokia phone is lovly....."` ($P(\text{Spam}) = 0.9293$)  
  *Root Cause*: The token `"nokia"` and `"phone"` in the 2011 UCI dataset appear with overwhelming frequency in promotional handset offers (`"Claim your free Nokia mobile"`). Without sufficient conversational context words, the model associates the brand name with spam.
- **Case 2**: `"Can U get 2 phone hack op...?"` ($P(\text{Spam}) = 0.5847$)  
  *Root Cause*: Contains `"hack"`, `"phone"`, and shorthand `"u"`, mimicking suspicious or unauthorized activity patterns.
- **Case 3**: `"Unlimited text message for only £5.00"` ($P(\text{Spam}) = 0.9120$)  
  *Root Cause*: Legitimate peer-to-peer discussion about a mobile plan, but contains the currency symbol `£` and `"unlimited"`, which are strong spam feature triggers.

### 2. False Negatives (Spam missed as Ham: 11 cases)
- **Case 1**: `"Sorry I missed your call let's talk when you have the time. I'm on 070..."` ($P(\text{Spam}) = 0.0013$)  
  *Root Cause*: Deceptive conversational disguise. The spammer opens with polite conversational phrasing (`"Sorry I missed your call"`) before appending a premium-rate callback number (`070...`). The heavy presence of conversational ham words lowers the spam probability.
- **Case 2**: `"Check Out Choose Your Babe Videos @ sms.shsex.net UN fgkslpo PW fgkslpo"` ($P(\text{Spam}) = 0.2019$)  
  *Root Cause*: Contains obscure random URL parameters and unusual tokens that were rare or unseen in the training vocabulary, failing to trigger sufficient TF-IDF feature weight.

---

## 9. Inference & Interactive Prediction

The project provides an inference CLI (`predict.py`) backed by the production `SpamClassifierService`.

### 1. Single Message Prediction
```bash
python predict.py "URGENT! You have won a £1,000 cash prize. Call 0800 now to claim."
```
**Output:**
```
[ SPAM ]  URGENT! You have won a £1,000 cash prize. Call 0800 now to claim.
        Label: SPAM | Confidence: 100.00% (Spam: 100.00%, Ham: 0.00%)
        Latency: 1.84 ms | Length: 65 chars
```

```bash
python predict.py "Hey, are you free to grab lunch together at 1 pm?"
```
**Output:**
```
[  HAM ]  Hey, are you free to grab lunch together at 1 pm?
        Label: HAM | Confidence: 98.13% (Spam: 1.87%, Ham: 98.13%)
        Latency: 1.72 ms | Length: 49 chars
```

### 2. Interactive Terminal Shell
```bash
python predict.py --interactive
```
Launches an interactive REPL where you can enter multiple messages consecutively.

### 3. Batch File Classification & JSON Output
```bash
# JSON output for API or pipeline integration
python predict.py "Free entry to win cash! Text WIN to 80088." --json
```

---

## 10. Interactive Streamlit Web Application

To launch the web user interface:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Streamlit App Features:
- **Custom Message Text Area**: Live character and token counter with safety truncation limits.
- **Sample Message Selector**: Pre-loaded with realistic spam and ham scenarios (including edge-case traps like casual messages containing `"free"`).
- **Classification Verdict**: Clean visual status badge (Spam: Warning/Red, Ham: Verified/Green).
- **Confidence & Probability Meter**: Visual progress bar indicating estimated spam probability.
- **Performance Metrics Tab**: Displays real test set metrics, confusion matrix breakdown, and ROC/PR curve plots.
- **Limitations & Ethics Advisory**: Plain-language disclosure of model error trade-offs and domain bounds.
- **Fail-Safe Error Handling**: If the model has not been trained, displays actionable instructions to run `python train.py` without leaking server traces.

---

## 11. Testing & Quality Assurance

The project includes an automated test suite executed via `pytest`:

```bash
pytest -v
```

### Test Suite Summary (38 Tests Passing)
- **`tests/test_data_loader.py`**: Verifies TSV parsing, malformed record detection, dataset audit metrics, and deterministic SHA256 checksums.
- **`tests/test_preprocessing.py`**: Tests input validation (rejecting `None`, numbers, empty/whitespace strings, and oversized texts), currency normalization, punctuation separation, and Scikit-Learn transformer compatibility.
- **`tests/test_training.py`**: Verifies candidate pipeline step naming, stratified splitting determinism, and model selection scoring logic.
- **`tests/test_no_leakage.py`**: Asserts **zero text overlap** between training and test sets, verifies that test-only tokens never appear in the fitted vectorizer vocabulary, and proves vectorizer encapsulation within the pipeline.
- **`tests/test_inference.py`**: Validates `SpamClassifierService` on single and batch predictions, checks probability normalization ($P(\text{ham}) + P(\text{spam}) = 1.0$), handles corrupted model files (`ModelLoadError`), and checks missing files (`ModelNotFoundError`).
- **`tests/test_metrics_report.py`**: Verifies that `evaluation_metrics.json` and `model_comparison.json` contain all required keys, valid ranges ($[0.0, 1.0]$), and comparison models.
- **`tests/test_app_smoke.py`**: Smoke tests for Streamlit application helper functions and verifies that all sample messages classify without error.

---

## 12. Limitations & Security Considerations

1. **Adversarial Obfuscation (Leetspeak / Homoglyphs)**: Modern spammers frequently use character substitutions (`FR33`, `C@LL`, `W!N`). Classical n-gram TF-IDF models may miss heavily obfuscated text unless character-level embeddings or subword tokenizers are introduced.
2. **Temporal & Vocabulary Drift**: The UCI SMS Spam Collection was gathered in 2011. Emerging modern spam vectors (e.g., cryptocurrency giveaways, WhatsApp verification codes, multi-factor authentication phishing) have vocabulary not present in 2011 training data.
3. **Conversational Disguise**: As shown in our error analysis, spammers who disguise their message with colloquial openings (`"Sorry I missed your call"`) can reduce their TF-IDF spam score.
4. **Context-Free Short Texts**: Single-word messages (e.g. `"Free"`, `"Win"`) lack grammatical context and may yield ambiguous probability estimates.
5. **Language Specificity**: The model is trained exclusively on English SMS messages and should not be used on multilingual text.

---

## 13. CodeNova Task 2 Compliance Checklist

| CodeNova Task 2 Requirement | Implementation Location | Evidence & Verification Status |
| :--- | :--- | :--- |
| **Official Dataset Source** | `src/data_loader.py` | UCI SMS Spam Collection (Almeida & Hidalgo, 2011), SHA256 verified, metadata recorded in `data/raw/metadata.json` |
| **Text Preprocessing** | `src/preprocessing.py` | Unicode NFKC normalization, currency indicators (`$`, `£`, `€`), punctuation separation, empty/whitespace validation |
| **Vectorization** | `src/training.py` | `TfidfVectorizer` (unigrams + bigrams, sublinear TF) and `CountVectorizer` |
| **Two+ Classification Algorithms** | `src/training.py` | Multinomial Naive Bayes (Count & TF-IDF) and Logistic Regression (TF-IDF) |
| **Stratified CV & Train/Test Split** | `src/training.py` | 5-Fold Stratified CV on train (4,128), held-out test (1,032), random seed 42 |
| **Zero Data Leakage** | `tests/test_no_leakage.py` | Pipeline encapsulation; 0 text overlap between train and test splits verified by automated unit tests |
| **Precision, Recall, F1, Confusion Matrix** | `src/evaluation.py` | Test Precision: 97.50%, Recall: 91.41%, Macro-F1: 0.9679, full confusion matrix generated in `reports/` |
| **Plain-Language Error Analysis** | `reports/metrics/error_analysis.json` | Detailed inspection and linguistic root-cause explanation of False Positives and False Negatives |
| **Interactive Custom Prediction** | `predict.py` | Single message CLI, interactive REPL shell, batch mode, and JSON output |
| **Streamlit Web Application** | `app.py` | Interactive web app with custom input, sample selector, risk warnings, metrics panel, and HTTP 200 verification |
| **Saved Model Pipeline** | `models/spam_classifier_pipeline.joblib` | Fully fitted Scikit-Learn `Pipeline` serialized with Joblib |
| **Pytest Test Suite** | `tests/` | 38 unit and smoke tests passing in < 4 seconds |
| **Complete Documentation & Metrics** | `README.md` | Full architecture documentation, exact benchmark tables, error cost trade-offs, and sample predictions |

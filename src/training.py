"""Training and Model Selection module for Spam Message Classifier.

Builds scikit-learn pipelines, executes stratified cross-validation,
compares multiple algorithms, evaluates on an isolated test set,
and serializes artifacts.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import (
    COMPARISON_CSV_PATH,
    COMPARISON_JSON_PATH,
    CONFUSION_MATRIX_PNG,
    CV_FOLDS,
    METRICS_JSON_PATH,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    RANDOM_SEED,
    REPORTS_DIR,
    ROC_PR_CURVES_PNG,
    SAVED_MODEL_PATH,
    TEST_SIZE,
)
from src.evaluation import (
    analyze_errors,
    benchmark_inference_latency,
    compute_comprehensive_metrics,
    plot_confusion_matrix,
    plot_dataset_distributions,
    plot_roc_and_pr_curves,
)
from src.preprocessing import SMSPreprocessor


def prepare_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    drop_duplicates: bool = True,
    save_csv: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset into stratified training and testing partitions.

    Ensures zero data leakage between train and test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with columns 'label' and 'text'.
    test_size : float
        Proportion of dataset to reserve for held-out testing (e.g. 0.20).
    random_state : int
        Deterministic random seed.
    drop_duplicates : bool
        Whether to deduplicate verbatim messages before splitting to eliminate
        exact text overlap between train and test.
    save_csv : bool
        Whether to persist train.csv and test.csv to disk.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    clean_df = df.copy()
    clean_df["text"] = clean_df["text"].astype(str).str.strip()

    if drop_duplicates:
        clean_df = clean_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Encode label: 'ham' -> 0, 'spam' -> 1
    clean_df["target"] = (clean_df["label"] == "spam").astype(int)

    train_df, test_df = train_test_split(
        clean_df,
        test_size=test_size,
        random_state=random_state,
        stratify=clean_df["target"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if save_csv:
        PROCESSED_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
        test_df.to_csv(PROCESSED_TEST_PATH, index=False)

    return train_df, test_df


def get_candidate_pipelines() -> Dict[str, Pipeline]:
    """Defines candidate pipelines comparing vectorizers and classification models.

    Models:
    1. MultinomialNB + CountVectorizer (Baseline bag-of-words)
    2. MultinomialNB + TfidfVectorizer (Naive Bayes with TF-IDF weighting)
    3. LogisticRegression + TfidfVectorizer (Linear classifier with calibrated probabilities)

    Returns
    -------
    Dict[str, Pipeline]
        Candidate scikit-learn pipelines.
    """
    pipelines = {
        "MultinomialNB_Count": Pipeline(
            [
                ("preprocessor", SMSPreprocessor(lowercase=True)),
                (
                    "vectorizer",
                    CountVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                ("classifier", MultinomialNB(alpha=0.1)),
            ]
        ),
        "MultinomialNB_TFIDF": Pipeline(
            [
                ("preprocessor", SMSPreprocessor(lowercase=True)),
                (
                    "vectorizer",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", MultinomialNB(alpha=0.1)),
            ]
        ),
        "LogisticRegression_TFIDF": Pipeline(
            [
                ("preprocessor", SMSPreprocessor(lowercase=True)),
                (
                    "vectorizer",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=5.0,
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }
    return pipelines


def compare_models_cross_validation(
    pipelines: Dict[str, Pipeline],
    X_train: List[str],
    y_train: np.ndarray,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """Evaluates candidate pipelines using Stratified K-Fold Cross-Validation.

    Parameters
    ----------
    pipelines : Dict[str, Pipeline]
        Named candidate pipelines.
    X_train : List[str]
        Training text messages.
    y_train : np.ndarray
        Training binary labels (0=ham, 1=spam).
    cv_folds : int
        Number of cross-validation folds.
    random_state : int
        Seed for CV split shuffling.

    Returns
    -------
    Dict[str, Dict[str, float]]
        CV metrics for each model.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "spam_precision": "precision",
        "spam_recall": "recall",
        "spam_f1": "f1",
    }

    results: Dict[str, Dict[str, float]] = {}

    for name, pipe in pipelines.items():
        print(f"    -> Evaluating {name} ({cv_folds}-fold CV)...", flush=True)
        t0 = time.perf_counter()
        cv_res = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )
        t1 = time.perf_counter()
        print(f"       done in {t1-t0:.2f}s | Macro-F1: {np.mean(cv_res['test_macro_f1']):.4f}", flush=True)

        results[name] = {
            "cv_accuracy_mean": round(float(np.mean(cv_res["test_accuracy"])), 4),
            "cv_accuracy_std": round(float(np.std(cv_res["test_accuracy"])), 4),
            "cv_macro_f1_mean": round(float(np.mean(cv_res["test_macro_f1"])), 4),
            "cv_macro_f1_std": round(float(np.std(cv_res["test_macro_f1"])), 4),
            "cv_spam_precision_mean": round(float(np.mean(cv_res["test_spam_precision"])), 4),
            "cv_spam_precision_std": round(float(np.std(cv_res["test_spam_precision"])), 4),
            "cv_spam_recall_mean": round(float(np.mean(cv_res["test_spam_recall"])), 4),
            "cv_spam_recall_std": round(float(np.std(cv_res["test_spam_recall"])), 4),
            "cv_spam_f1_mean": round(float(np.mean(cv_res["test_spam_f1"])), 4),
            "cv_spam_f1_std": round(float(np.std(cv_res["test_spam_f1"])), 4),
            "cv_fit_time_sec": round(float(t1 - t0), 3),
        }

    return results


def select_best_model(cv_results: Dict[str, Dict[str, float]]) -> str:
    """Selects the best performing model using documented error-cost trade-offs.

    Selection Criteria:
    - Primary objective: Maximize Macro-F1 and Spam Precision.
    - False Positive Cost: Flagging legitimate messages as spam harms user trust.
      High Spam Precision (ideally > 0.95) is required.
    - False Negative Cost: Letting spam slip through is annoying but less critical.
    - Formula: Combined score = 0.5 * cv_macro_f1 + 0.5 * cv_spam_precision.

    Parameters
    ----------
    cv_results : Dict[str, Dict[str, float]]
        Cross-validation results.

    Returns
    -------
    str
        Name of the chosen pipeline.
    """
    best_name = ""
    best_score = -1.0

    for name, metrics in cv_results.items():
        macro_f1 = metrics["cv_macro_f1_mean"]
        precision = metrics["cv_spam_precision_mean"]
        # Balanced score rewarding both overall discrimination and low false positives
        composite_score = 0.5 * macro_f1 + 0.5 * precision

        if composite_score > best_score:
            best_score = composite_score
            best_name = name

    return best_name


def train_and_evaluate_pipeline(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Full end-to-end training, cross-validation, test evaluation, and artifact generation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or pre-parsed dataset.
    drop_duplicates : bool
        Whether to deduplicate messages to ensure zero text leakage.
    save_artifacts : bool
        Whether to save model files, reports, and figures to disk.

    Returns
    -------
    Dict[str, Any]
        Complete results dictionary with CV comparison, final test metrics,
        and latency benchmarks.
    """
    # Step 1: Generate dataset audit figures
    plot_dataset_distributions(df)

    # Step 2: Stratified train/test split
    train_df, test_df = prepare_train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        drop_duplicates=drop_duplicates,
        save_csv=save_artifacts,
    )

    X_train = train_df["text"].tolist()
    y_train = train_df["target"].values
    X_test = test_df["text"].tolist()
    y_test = test_df["target"].values

    # Step 3: Cross-validation comparison of candidate models
    pipelines = get_candidate_pipelines()
    cv_results = compare_models_cross_validation(pipelines, X_train, y_train)

    # Step 4: Model selection based on documented criteria
    selected_model_name = select_best_model(cv_results)
    best_pipeline = pipelines[selected_model_name]

    # Step 5: Fit chosen model on entire training partition
    t_train_start = time.perf_counter()
    best_pipeline.fit(X_train, y_train)
    t_train_end = time.perf_counter()
    training_time_sec = round(t_train_end - t_train_start, 4)

    # Step 6: Evaluate on the isolated held-out test partition
    y_test_pred = best_pipeline.predict(X_test)

    # Probabilities for spam class
    if hasattr(best_pipeline, "predict_proba"):
        y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    else:
        y_test_prob = None

    test_metrics = compute_comprehensive_metrics(
        y_true=y_test,
        y_pred=y_test_pred,
        y_prob=y_test_prob,
        model_name=selected_model_name,
    )
    test_metrics["training_time_sec"] = training_time_sec

    # Step 7: Benchmark inference latency
    latency_bench = benchmark_inference_latency(best_pipeline, X_test)
    test_metrics["inference_latency"] = latency_bench

    # Step 8: Error analysis (plain language)
    error_report = analyze_errors(
        texts=X_test,
        y_true=y_test,
        y_pred=y_test_pred,
        y_prob=y_test_prob,
    )

    # Step 9: Also evaluate all candidate models on test set for side-by-side comparison table
    comparison_records: List[Dict[str, Any]] = []
    for name, pipe in pipelines.items():
        if name != selected_model_name:
            pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
        m = compute_comprehensive_metrics(y_test, pred, prob, model_name=name)
        comparison_records.append(
            {
                "Model": name,
                "CV Macro-F1": cv_results[name]["cv_macro_f1_mean"],
                "CV Spam Precision": cv_results[name]["cv_spam_precision_mean"],
                "Test Accuracy": m["accuracy"],
                "Test Macro-F1": m["macro_f1"],
                "Test Spam Precision": m["spam_metrics"]["precision"],
                "Test Spam Recall": m["spam_metrics"]["recall"],
                "Test Spam F1": m["spam_metrics"]["f1_score"],
                "Test ROC-AUC": m["roc_auc"],
                "Test PR-AUC": m["pr_auc"],
                "False Positives": m["confusion_matrix"]["fp"],
                "False Negatives": m["confusion_matrix"]["fn"],
                "Selected": (name == selected_model_name),
            }
        )

    comparison_df = pd.DataFrame(comparison_records)

    # Step 10: Save artifacts
    if save_artifacts:
        SAVED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, SAVED_MODEL_PATH)

        # Save metrics JSON & CSV
        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        with open(COMPARISON_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(comparison_records, f, indent=2)

        comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)

        error_analysis_path = REPORTS_DIR / "metrics" / "error_analysis.json"
        with open(error_analysis_path, "w", encoding="utf-8") as f:
            json.dump(error_report, f, indent=2)

        # Generate plots
        plot_confusion_matrix(
            np.array(test_metrics["confusion_matrix"]["matrix"]),
            model_name=selected_model_name,
            output_path=CONFUSION_MATRIX_PNG,
        )

        if y_test_prob is not None:
            plot_roc_and_pr_curves(
                y_true=y_test,
                y_prob=y_test_prob,
                model_name=selected_model_name,
                output_path=ROC_PR_CURVES_PNG,
            )

    return {
        "selected_model": selected_model_name,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "comparison_table": comparison_df,
        "error_report": error_report,
    }

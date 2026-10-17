"""Evaluation module for Spam Message Classifier.

Computes comprehensive classification metrics, confusion matrices, ROC/PR curves,
inference latency benchmarks, and plain-language error analyses.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    CLASS_DISTRIBUTION_PNG,
    CONFUSION_MATRIX_PNG,
    FIGURES_DIR,
    LENGTH_DISTRIBUTION_PNG,
    METRICS_DIR,
    ROC_PR_CURVES_PNG,
)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Classifier",
) -> Dict[str, Any]:
    """Computes all standard and granular NLP classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (0 for ham, 1 for spam).
    y_pred : np.ndarray
        Predicted class labels (0 or 1).
    y_prob : Optional[np.ndarray]
        Predicted probability of the positive class (spam, label 1).
    model_name : str
        Human-readable name of the evaluated model.

    Returns
    -------
    Dict[str, Any]
        Dictionary with all scalar metrics and confusion matrix counts.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Per-class metrics
    spam_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    spam_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spam_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    ham_precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    ham_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    ham_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    # Aggregates
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Probability-based metrics (ROC-AUC & PR-AUC)
    roc_auc = None
    pr_auc = None
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recalls, precisions))
        except ValueError:
            pass

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(y_true),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "spam_metrics": {
            "precision": round(float(spam_precision), 4),
            "recall": round(float(spam_recall), 4),
            "f1_score": round(float(spam_f1), 4),
            "support": int(tp + fn),
        },
        "ham_metrics": {
            "precision": round(float(ham_precision), 4),
            "recall": round(float(ham_recall), 4),
            "f1_score": round(float(ham_f1), 4),
            "support": int(tn + fp),
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
    }

    return metrics


def benchmark_inference_latency(
    pipeline: Any,
    sample_texts: List[str],
    num_iterations: int = 5,
) -> Dict[str, float]:
    """Measures single-message and batch inference latency.

    Parameters
    ----------
    pipeline : Any
        Fitted Scikit-Learn Pipeline.
    sample_texts : List[str]
        List of texts for benchmarking.
    num_iterations : int
        Number of warm-up and measurement passes.

    Returns
    -------
    Dict[str, float]
        Latency benchmarks (mean ms per sample, batch latency).
    """
    if not sample_texts:
        return {"mean_ms_per_sample": 0.0, "batch_total_ms": 0.0}

    # Warm-up pass
    _ = pipeline.predict(sample_texts[:5])

    # Single-sample latency timing
    single_times = []
    bench_subset = sample_texts[: min(100, len(sample_texts))]
    for text in bench_subset:
        t0 = time.perf_counter()
        _ = pipeline.predict([text])
        t1 = time.perf_counter()
        single_times.append((t1 - t0) * 1000.0)

    # Batch latency timing
    batch_times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = pipeline.predict(sample_texts)
        t1 = time.perf_counter()
        batch_times.append((t1 - t0) * 1000.0)

    return {
        "mean_single_prediction_ms": round(float(np.mean(single_times)), 3),
        "p95_single_prediction_ms": round(float(np.percentile(single_times, 95)), 3),
        "mean_batch_prediction_ms": round(float(np.mean(batch_times)), 3),
        "throughput_samples_per_sec": round(
            float(len(sample_texts) / (np.mean(batch_times) / 1000.0)), 1
        ),
    }


def analyze_errors(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extracts and analyzes False Positives and False Negatives with contextual explanations.

    False Positive (FP): Legitimate ham misclassified as spam.
    False Negative (FN): Spam message missed and misclassified as ham.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Dictionary with keys 'false_positives' and 'false_negatives'.
    """
    fps = []
    fns = []

    for idx, (text, true_lbl, pred_lbl) in enumerate(zip(texts, y_true, y_pred)):
        prob = float(y_prob[idx]) if y_prob is not None else None
        if true_lbl == 0 and pred_lbl == 1:
            # False Positive
            fps.append(
                {
                    "index": idx,
                    "text": text,
                    "true_label": "ham",
                    "predicted_label": "spam",
                    "spam_probability": round(prob, 4) if prob is not None else None,
                    "explanation": (
                        "Legitimate message mistakenly classified as spam. "
                        "Likely contains keywords typically correlated with spam "
                        "(e.g., 'free', numbers, web links, or prize phrasing)."
                    ),
                }
            )
        elif true_lbl == 1 and pred_lbl == 0:
            # False Negative
            fns.append(
                {
                    "index": idx,
                    "text": text,
                    "true_label": "spam",
                    "predicted_label": "ham",
                    "spam_probability": round(prob, 4) if prob is not None else None,
                    "explanation": (
                        "Spam message missed and classified as ham. "
                        "Likely written in colloquial or subtle conversational style "
                        "lacking overt promotional buzzwords."
                    ),
                }
            )

    return {"false_positives": fps, "false_negatives": fns}


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: Path = CONFUSION_MATRIX_PNG,
) -> Path:
    """Plots and saves an annotated confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        2x2 confusion matrix [[TN, FP], [FN, TP]].
    model_name : str
        Name of the model for the plot title.
    output_path : Path
        File path to save the generated image.

    Returns
    -------
    Path
        Path to saved figure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))

    tn, fp, fn, tp = cm.ravel()
    total = np.sum(cm)

    labels = [
        [f"True Ham (TN)\n{tn}\n({tn/total*100:.1f}%)", f"False Spam (FP)\n{fp}\n({fp/total*100:.1f}%)"],
        [f"Missed Spam (FN)\n{fn}\n({fn/total*100:.1f}%)", f"True Spam (TP)\n{tp}\n({tp/total*100:.1f}%)"],
    ]

    sns.heatmap(
        cm,
        annot=np.array(labels),
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted Ham", "Predicted Spam"],
        yticklabels=["Actual Ham", "Actual Spam"],
        annot_kws={"size": 11, "fontweight": "bold"},
    )

    plt.title(f"Confusion Matrix: {model_name}\n(Test Set, N={total})", fontsize=13, pad=12)
    plt.ylabel("Actual True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_path: Path = ROC_PR_CURVES_PNG,
) -> Path:
    """Plots and saves ROC and Precision-Recall curves.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (0 for ham, 1 for spam).
    y_prob : np.ndarray
        Predicted probabilities for spam class.
    model_name : str
        Model title.
    output_path : Path
        Destination path.

    Returns
    -------
    Path
        Path to saved figure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Chance")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10)
    axes[0].set_title(f"Receiver Operating Characteristic (ROC)\n{model_name}", fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    baseline_pr = float(np.mean(y_true))
    axes[1].plot(recall, precision, color="#d62728", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    axes[1].axhline(y=baseline_pr, color="gray", lw=1, linestyle="--", label=f"Baseline ({baseline_pr:.2%})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Spam Recall", fontsize=10)
    axes[1].set_ylabel("Spam Precision", fontsize=10)
    axes[1].set_title(f"Precision-Recall Curve\n{model_name}", fontsize=11)
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_dataset_distributions(
    df: pd.DataFrame,
    class_fig_path: Path = CLASS_DISTRIBUTION_PNG,
    len_fig_path: Path = LENGTH_DISTRIBUTION_PNG,
) -> Tuple[Path, Path]:
    """Generates distribution plots for dataset audit and visual verification.

    1. Class distribution bar plot.
    2. Character length distribution by class.
    """
    class_fig_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Class Distribution
    plt.figure(figsize=(6, 5))
    counts = df["label"].value_counts()
    colors = ["#2ca02c", "#d62728"]
    bars = plt.bar(counts.index.str.upper(), counts.values, color=colors, width=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 30,
            f"{h:,}\n({h/len(df)*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.title("Class Distribution: UCI SMS Spam Collection", fontsize=12, pad=12)
    plt.ylabel("Number of Messages", fontsize=10)
    plt.xlabel("Category", fontsize=10)
    plt.ylim(0, len(df) * 1.08)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(class_fig_path, dpi=300)
    plt.close()

    # 2. Length Distribution
    df_copy = df.copy()
    df_copy["char_length"] = df_copy["text"].str.len()
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df_copy[df_copy["char_length"] <= 300],
        x="char_length",
        hue="label",
        bins=50,
        kde=True,
        palette={"ham": "#2ca02c", "spam": "#d62728"},
        stat="density",
        common_norm=False,
    )
    plt.title("Message Character Length Distribution (Ham vs. Spam, ≤ 300 chars)", fontsize=12)
    plt.xlabel("Character Length", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(len_fig_path, dpi=300)
    plt.close()

    return class_fig_path, len_fig_path

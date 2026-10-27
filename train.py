"""CLI Training and Evaluation Script for Spam Message Classifier.

Executes the full pipeline:
1. Data loading and validation (UCI SMS Spam Collection).
2. Data audit and length distribution analysis.
3. Stratified Train/Test split with zero-leakage safeguards.
4. Stratified 5-Fold Cross-Validation model comparison.
5. Model selection based on Macro-F1 and Spam Precision trade-offs.
6. Test set evaluation, metrics reporting, and artifact serialization.
"""

import argparse
import sys
import time

# Ensure immediate console output flushing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from src.config import (
    COMPARISON_CSV_PATH,
    DATA_AUDIT_JSON_PATH,
    METRICS_JSON_PATH,
    SAVED_MODEL_PATH,
    THRESHOLD_ANALYSIS_JSON_PATH,
)
from src.data_loader import load_raw_dataset
from src.training import train_and_evaluate_pipeline


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for training script."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the CodeNova Spam Message Classifier pipeline."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the raw dataset from UCI even if already present.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep verbatim duplicate messages instead of deduplicating before split.",
    )
    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_arguments()
    print("=" * 70)
    print("   CodeNova AI/ML Internship - Task 2: Spam Message Classifier   ")
    print("=" * 70)

    # Step 1: Data Ingestion & Audit
    print("\n[Step 1/5] Ingesting and Auditing Dataset from UCI...")
    t0 = time.time()
    try:
        df, audit = load_raw_dataset(force_download=args.force_download)
    except Exception as exc:
        print(f"\n[ERROR] Failed to load dataset: {exc}", file=sys.stderr)
        return 1

    dist = audit["class_distribution"]
    print(f" -> Total Raw Records: {audit['total_records']:,}")
    print(f" -> Class Distribution: {dist['ham_count']:,} Ham ({100-dist['spam_percentage']}%), "
          f"{dist['spam_count']:,} Spam ({dist['spam_percentage']}%)")
    print(f" -> Class Imbalance: {dist['imbalance_ratio']}")
    print(f" -> Unique Messages: {audit['data_integrity']['unique_messages_count']:,} "
          f"({audit['data_integrity']['duplicate_messages_count']:,} exact duplicates)")
    print(f" -> Conflicting Labels: {audit['data_integrity']['conflicting_label_messages_count']}")
    print(f" -> Audit report saved: {DATA_AUDIT_JSON_PATH}")

    # Step 2: Training & Model Comparison
    drop_dups = not args.keep_duplicates
    if drop_dups:
        print(f"\n[Step 2/5] Splitting Data (Deduplicated: {audit['data_integrity']['unique_messages_count']:,} samples)...")
    else:
        print(f"\n[Step 2/5] Splitting Data (All {audit['total_records']:,} samples)...")

    print(" -> 80% Train / 20% Held-out Test (Stratified split, Random Seed = 42)")
    print(" -> Running 5-Fold Stratified Cross-Validation across candidate models...")

    try:
        results = train_and_evaluate_pipeline(
            df=df,
            drop_duplicates=drop_dups,
            save_artifacts=True,
        )
    except Exception as exc:
        print(f"\n[ERROR] Pipeline training failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Model Comparison Table
    print("\n[Step 3/5] Cross-Validation & Test Set Comparison:")
    comp_df = results["comparison_table"]
    print("-" * 70)
    print(comp_df.to_string(index=False))
    print("-" * 70)
    print(f" -> Model comparison saved: {COMPARISON_CSV_PATH}")

    # Step 4: Selected Model Test Performance
    selected = results["selected_model"]
    test_m = results["test_metrics"]
    cm = test_m["confusion_matrix"]

    print(f"\n[Step 4/5] Selected Model: {selected}")
    print(f" -> Test Set Accuracy:        {test_m['accuracy']:.4f}")
    print(f" -> Test Macro-F1:            {test_m['macro_f1']:.4f}")
    print(f" -> Test Spam Precision:      {test_m['spam_metrics']['precision']:.4f}")
    print(f" -> Test Spam Recall:         {test_m['spam_metrics']['recall']:.4f}")
    print(f" -> Test Spam F1-Score:       {test_m['spam_metrics']['f1_score']:.4f}")
    print(f" -> Test Ham Precision:       {test_m['ham_metrics']['precision']:.4f}")
    print(f" -> Test Ham Recall:          {test_m['ham_metrics']['recall']:.4f}")
    print(f" -> Test Ham F1-Score:        {test_m['ham_metrics']['f1_score']:.4f}")
    if test_m.get("roc_auc"):
        print(f" -> Test ROC-AUC:             {test_m['roc_auc']:.4f}")
    if test_m.get("pr_auc"):
        print(f" -> Test PR-AUC:              {test_m['pr_auc']:.4f}")

    print("\nConfusion Matrix Breakdown (Held-out Test Set):")
    print(f"    True Negative  (Ham correctly identified):  {cm['tn']:,}")
    print(f"    False Positive (Ham misclassified as Spam): {cm['fp']:,}  <-- Cost: Legitimate message flagged")
    print(f"    False Negative (Spam missed as Ham):        {cm['fn']:,}  <-- Cost: Spam reaches inbox")
    print(f"    True Positive  (Spam correctly caught):     {cm['tp']:,}")

    # Latency
    latency = test_m["inference_latency"]
    print(f"\nLatency Benchmark:")
    print(f" -> Mean single message prediction: {latency['mean_single_prediction_ms']} ms")
    print(f" -> P95 single message prediction:  {latency['p95_single_prediction_ms']} ms")
    print(f" -> Batch throughput:               {latency['throughput_samples_per_sec']} samples/sec")

    # Threshold analysis summary
    th_eval = results.get("threshold_eval", {})
    if th_eval:
        print(f"\nDecision Threshold Analysis (Cross-Validation on Training Set):")
        print(f" -> Recommended Operating Threshold: {th_eval.get('recommended_threshold')}")
        print("    Threshold | CV Spam Prec | CV Spam Rec | CV Spam F1 | CV False Positives")
        print("    " + "-" * 62)
        for rec in th_eval.get("evaluated_thresholds", []):
            print(f"    {rec['threshold']:<9.2f} | {rec['cv_spam_precision']:<12.4f} | {rec['cv_spam_recall']:<11.4f} | {rec['cv_spam_f1']:<10.4f} | {rec['cv_false_positives']}")

    # Step 5: Artifacts
    print(f"\n[Step 5/5] Artifacts Successfully Saved:")
    print(f" -> Pipeline:           {SAVED_MODEL_PATH}")
    print(f" -> Metrics JSON:       {METRICS_JSON_PATH}")
    print(f" -> Threshold JSON:     {THRESHOLD_ANALYSIS_JSON_PATH}")

    # Error analysis summary
    error_rep = results["error_report"]
    fps = error_rep["false_positives"]
    fns = error_rep["false_negatives"]
    print(f"\nError Analysis Summary:")
    print(f" -> False Positives: {len(fps)}")
    if fps:
        print(f"    Example FP: \"{fps[0]['text'][:70]}...\" (P(Spam)={fps[0]['spam_probability']})")
    print(f" -> False Negatives: {len(fns)}")
    if fns:
        print(f"    Example FN: \"{fns[0]['text'][:70]}...\" (P(Spam)={fns[0]['spam_probability']})")

    elapsed = round(time.time() - t0, 2)
    print(f"\nPipeline completed successfully in {elapsed}s.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""CLI Inference Interface for Spam Message Classifier.

Supports:
- Single message classification: python predict.py "Your message here"
- Interactive mode: python predict.py --interactive
- Batch file classification: python predict.py --file messages.txt
"""

import argparse
import json
import sys
from typing import List

# Ensure robust console UTF-8 handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.inference import (
    ModelLoadError,
    ModelNotFoundError,
    PredictionResult,
    SpamClassifierService,
)


def format_result(result: PredictionResult) -> str:
    """Formats a PredictionResult for human-readable console display."""
    badge = "[ SPAM ]" if result.is_spam else "[  HAM ]"
    conf_pct = result.confidence * 100
    spam_pct = result.spam_probability * 100
    ham_pct = result.ham_probability * 100

    output = [
        f"{badge}  {result.sanitized_text_snippet}",
        f"        Label: {result.label.upper()} | Confidence: {conf_pct:.2f}% "
        f"(Spam: {spam_pct:.2f}%, Ham: {ham_pct:.2f}%)",
        f"        Latency: {result.processing_time_ms} ms | Length: {result.message_length} chars",
    ]
    return "\n".join(output)


def run_interactive(service: SpamClassifierService) -> None:
    """Runs an interactive REPL session for testing messages."""
    print("=" * 65)
    print(" Spam Message Classifier - Interactive Testing Shell")
    print(" Type a message and press Enter to classify.")
    print(" Type 'quit', 'exit', or press Ctrl+C to terminate.")
    print("=" * 65)

    while True:
        try:
            user_input = input("\n[Enter Message] > ").strip()
            if not user_input:
                print("  [Warning] Empty message entered. Please provide text.")
                continue

            if user_input.lower() in {"quit", "exit", "q"}:
                print("\nExiting interactive session.")
                break

            result = service.predict(user_input)
            print("\n" + format_result(result))

            if result.is_spam:
                print("  [!] WARNING: This message displays strong promotional/phishing cues.")
            else:
                print("  [OK] INFO: This message appears to be standard conversational text.")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break
        except Exception as exc:
            print(f"  [Error] {exc}")


def run_smoke_test(service: SpamClassifierService) -> int:
    """Executes a comprehensive non-interactive smoke test on representative inputs.

    Tests:
    1. Ordinary legitimate message (ham)
    2. Obvious promotional/financial spam (spam)
    3. Borderline message containing 'free' in benign context (ham)
    4. Legitimate message with currency symbol and numbers (ham)
    5. Obfuscated / high-punctuation spam with URL (spam)
    6. Unicode emoji text (ham)
    7. Validation error handling on empty, whitespace, and non-string inputs.

    Returns
    -------
    int
        0 on complete success, 1 on failure.
    """
    print("=" * 75)
    print(" Spam Message Classifier - Model Inference Smoke Test")
    print("=" * 75)

    test_cases = [
        (
            "Ordinary Ham",
            "Hey mom, I will be home late tonight around 8pm for dinner.",
            "ham",
        ),
        (
            "Obvious Spam",
            "URGENT! You have won a £1,000 cash prize! Claim by texting PRIZE to 88088 now.",
            "spam",
        ),
        (
            "Borderline 'Free' Ham",
            "Are you free tonight? Let's grab dinner and discuss the machine learning assignment.",
            "ham",
        ),
        (
            "Ham with Currency/Numbers",
            "Thanks for dinner last night, I just transferred £25.50 to your account.",
            "ham",
        ),
        (
            "Phishing Spam with URL",
            "ALERT: Bank account suspended due to suspicious activity. Verify immediately at http://secure-login.co.uk",
            "spam",
        ),
        (
            "Unicode Emoji Ham",
            "🎉 Happy Birthday Sarah! Wishing you a fantastic year ahead! 🎂🎈",
            "ham",
        ),
    ]

    passed = 0
    total = len(test_cases) + 3  # 6 inputs + 3 validation checks

    print(f"\n[Part 1/2] Evaluating {len(test_cases)} Representative Text Messages:\n")
    print(f"{'Case':<22} | {'Snippet':<30} | {'Pred':<5} | {'P(Spam)':<7} | {'Latency':<8} | {'Status'}")
    print("-" * 88)

    for case_name, text, expected_label in test_cases:
        try:
            res = service.predict(text)
            snippet = (text[:27] + "...") if len(text) > 27 else text
            status = "PASS" if res.label == expected_label else f"MISMATCH(got {res.label})"
            print(
                f"{case_name:<22} | {snippet:<30} | {res.label.upper():<5} | "
                f"{res.spam_probability:<7.4f} | {res.processing_time_ms:<5.2f} ms | {status}"
            )
            if res.label == expected_label:
                passed += 1
            else:
                print(f"    [Notice] Case '{case_name}' predicted as '{res.label}', expected '{expected_label}'")
        except Exception as exc:
            print(f"{case_name:<22} | ERROR: {exc} | FAIL")

    print("\n[Part 2/2] Validating Robust Input Error Handling:")
    # Check 1: Empty string raises ValueError
    try:
        service.predict("")
        print("  [FAIL] Empty string check FAILED (no error raised)")
    except ValueError:
        print("  [OK] Empty string check PASSED (raised ValueError)")
        passed += 1

    # Check 2: Whitespace only raises ValueError
    try:
        service.predict("   \t\n  ")
        print("  [FAIL] Whitespace-only check FAILED (no error raised)")
    except ValueError:
        print("  [OK] Whitespace-only check PASSED (raised ValueError)")
        passed += 1

    # Check 3: Non-string raises TypeError
    try:
        service.predict(None)  # type: ignore
        print("  [FAIL] Non-string input check FAILED (no error raised)")
    except TypeError:
        print("  [OK] Non-string input check PASSED (raised TypeError)")
        passed += 1

    print("\n" + "=" * 75)
    print(f" Smoke Test Summary: {passed}/{total} checks passed.")
    print("=" * 75)

    if passed == total:
        print("\n[SMOKE TEST PASSED] Saved model pipeline is verified, reproducible, and ready for production.\n")
        return 0
    else:
        print(f"\n[SMOKE TEST INCOMPLETE] {total - passed} checks had unexpected outcomes.\n")
        return 1


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Spam Message Classifier CLI Inference Tool"
    )
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Custom text message to classify.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch interactive terminal prediction mode.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute non-interactive model inference smoke test and exit.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Path to a text file containing one message per line.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output prediction results in JSON format.",
    )

    args = parser.parse_args()

    # Load Inference Service
    try:
        service = SpamClassifierService()
    except (ModelNotFoundError, ModelLoadError) as err:
        print(f"\n[Error] {err}", file=sys.stderr)
        return 1

    # Case 0: Smoke Test Mode
    if args.smoke_test:
        return run_smoke_test(service)

    # Case 1: Interactive Mode
    if args.interactive:
        run_interactive(service)
        return 0

    # Case 2: Batch File Mode
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as exc:
            print(f"[Error] Failed to read input file '{args.file}': {exc}", file=sys.stderr)
            return 1

        if not lines:
            print(f"[Warning] File '{args.file}' contained no non-empty lines.")
            return 0

        results = service.predict_batch(lines)

        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            print(f"\nBatch Evaluation ({len(results)} messages):")
            print("-" * 65)
            for res in results:
                print(format_result(res))
                print("-" * 65)
        return 0

    # Case 3: Single Message from argument
    if args.message:
        try:
            result = service.predict(args.message)
        except Exception as exc:
            print(f"[Error] {exc}", file=sys.stderr)
            return 1

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("\n" + format_result(result))
        return 0

    # No arguments provided: print help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

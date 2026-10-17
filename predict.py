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
                print("  [✓] INFO: This message appears to be standard conversational text.")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break
        except Exception as exc:
            print(f"  [Error] {exc}")


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

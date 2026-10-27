"""Unit and integration tests for predict.py CLI interface."""

import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from src.config import SAVED_MODEL_PATH


@pytest.fixture(scope="module")
def python_executable() -> str:
    """Returns the current Python executable path."""
    return sys.executable


@pytest.fixture(scope="module")
def utf8_env() -> dict:
    """Provides environment variables with UTF-8 encoding enabled for subprocesses."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_cli_help(python_executable: str, utf8_env: dict):
    """Verifies that predict.py --help returns exit code 0 and usage information."""
    result = subprocess.run(
        [python_executable, "predict.py", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        env=utf8_env,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "--smoke-test" in result.stdout
    assert "--interactive" in result.stdout


def test_cli_smoke_test(python_executable: str, utf8_env: dict):
    """Verifies that predict.py --smoke-test executes cleanly with exit code 0."""
    if not SAVED_MODEL_PATH.exists():
        pytest.skip("Model artifact not yet trained. Run train.py first.")

    result = subprocess.run(
        [python_executable, "predict.py", "--smoke-test"],
        capture_output=True,
        text=True,
        timeout=30,
        env=utf8_env,
    )
    assert result.returncode == 0, f"Smoke test failed with stderr: {result.stderr}"
    assert "[SMOKE TEST PASSED]" in result.stdout
    assert "Part 1/2" in result.stdout
    assert "Part 2/2" in result.stdout


def test_cli_single_message_ham(python_executable: str, utf8_env: dict):
    """Verifies CLI single message classification for an ordinary ham message."""
    if not SAVED_MODEL_PATH.exists():
        pytest.skip("Model artifact not yet trained.")

    msg = "Are you available to join us for coffee later this afternoon?"
    result = subprocess.run(
        [python_executable, "predict.py", msg],
        capture_output=True,
        text=True,
        timeout=15,
        env=utf8_env,
    )
    assert result.returncode == 0
    assert "HAM" in result.stdout


def test_cli_single_message_json(python_executable: str, utf8_env: dict):
    """Verifies that --json outputs valid, parseable JSON with all schema keys."""
    if not SAVED_MODEL_PATH.exists():
        pytest.skip("Model artifact not yet trained.")

    msg = "URGENT: Claim £1000 prize now! Call 0800123456."
    result = subprocess.run(
        [python_executable, "predict.py", msg, "--json"],
        capture_output=True,
        text=True,
        timeout=15,
        env=utf8_env,
    )
    assert result.returncode == 0
    parsed = json.loads(result.stdout.strip())
    assert "label" in parsed
    assert "is_spam" in parsed
    assert "spam_probability" in parsed
    assert "ham_probability" in parsed
    assert "confidence" in parsed
    assert parsed["label"] == "spam"
    assert parsed["is_spam"] is True


def test_cli_empty_message_validation_error(python_executable: str, utf8_env: dict):
    """Verifies that passing an empty or whitespace message returns an error exit code."""
    if not SAVED_MODEL_PATH.exists():
        pytest.skip("Model artifact not yet trained.")

    result = subprocess.run(
        [python_executable, "predict.py", "   "],
        capture_output=True,
        text=True,
        timeout=15,
        env=utf8_env,
    )
    assert result.returncode != 0
    assert "cannot be empty" in result.stderr.lower() or "cannot be empty" in result.stdout.lower()


def test_cli_interactive_exit_commands(python_executable: str, utf8_env: dict):
    """Verifies that the interactive REPL terminates cleanly on 'quit' or 'exit'."""
    if not SAVED_MODEL_PATH.exists():
        pytest.skip("Model artifact not yet trained.")

    for exit_cmd in ["quit\n", "exit\n", "q\n"]:
        proc = subprocess.run(
            [python_executable, "predict.py", "-i"],
            input=exit_cmd,
            capture_output=True,
            text=True,
            timeout=15,
            env=utf8_env,
        )
        assert proc.returncode == 0
        assert "Exiting interactive session" in proc.stdout

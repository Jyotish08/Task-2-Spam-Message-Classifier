"""Text Preprocessing and Input Validation module for Spam Message Classifier.

Provides deterministic cleaning, normalization, tokenization, negation-preserving
stop-word filtering, and robust input validation for SMS messages.
"""

import re
from typing import Any, FrozenSet, List, Optional
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import MAX_INPUT_LENGTH

# Negation words that carry critical semantic signal in spam detection and must NOT
# be stripped as ordinary stop words (e.g. "not spam", "no fee", "never expires").
NEGATION_WORDS: FrozenSet[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "nor",
        "neither",
        "none",
        "nobody",
        "nowhere",
        "nothing",
        "hardly",
        "scarcely",
        "barely",
        "cannot",
        "without",
    }
)


def get_custom_stop_words(preserve_negations: bool = True) -> List[str]:
    """Returns a deterministic, sorted stop-word list with optional negation preservation.

    Parameters
    ----------
    preserve_negations : bool, default=True
        If True, excludes critical negation words (e.g. 'not', 'no', 'never')
        from the stop-word removal list so semantic negation is preserved.

    Returns
    -------
    List[str]
        Sorted list of stop words suitable for CountVectorizer and TfidfVectorizer.
    """
    if preserve_negations:
        filtered = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS
    else:
        filtered = set(ENGLISH_STOP_WORDS)
    return sorted(list(filtered))


def tokenize_text(text: str) -> List[str]:
    """Tokenizes normalized text into word tokens while preserving alphanumeric cues and currency tokens.

    Parameters
    ----------
    text : str
        Normalized input text.

    Returns
    -------
    List[str]
        List of lowercase string tokens.
    """
    if not isinstance(text, str):
        return []
    # Match alphanumeric words, numbers, and currency words
    tokens = re.findall(r"(?u)\b\w+\b", text.lower())
    return tokens


def validate_input_text(
    text: Any,
    max_length: int = MAX_INPUT_LENGTH,
    allow_empty: bool = False,
) -> str:
    """Validates raw input message text for inference and pipeline ingestion.

    Parameters
    ----------
    text : Any
        Input text to validate.
    max_length : int
        Maximum allowed character length.
    allow_empty : bool
        Whether to allow empty strings (default False).

    Returns
    -------
    str
        Validated, stripped string.

    Raises
    ------
    TypeError
        If text is not a string.
    ValueError
        If text is empty or whitespace-only (when allow_empty=False),
        or exceeds max_length.
    """
    if text is None:
        raise TypeError("Input message cannot be None. Expected a string.")

    if not isinstance(text, str):
        raise TypeError(
            f"Input message must be a string, got {type(text).__name__}."
        )

    stripped = text.strip()

    if not allow_empty and not stripped:
        raise ValueError(
            "Input message cannot be empty or contain only whitespace."
        )

    if len(text) > max_length:
        raise ValueError(
            f"Input message length ({len(text)} characters) exceeds safety limit "
            f"of {max_length} characters."
        )

    return stripped


def clean_text(text: str) -> str:
    """Cleans and normalizes an SMS message for feature extraction.

    Transformations applied:
    1. Unicode normalization (NFKC) to unify accents, ligatures, and full-width chars.
    2. Normalize redundant whitespace (multiple spaces, tabs, newlines -> single space).
    3. Normalization of currency symbols ($ -> ' dollar ', £ -> ' pound ', € -> ' euro ')
       so financial prize cues are preserved as distinct informative tokens.
    4. Normalization of repeated punctuation (e.g. '!!!!' -> ' ! ') to preserve emphasis.
    5. Stripping boundary whitespace.

    Parameters
    ----------
    text : str
        Raw input SMS text.

    Returns
    -------
    str
        Cleaned and normalized text.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1. Unicode normalization
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Preserve financial & currency indicators common in spam
    normalized = normalized.replace("£", " pound ")
    normalized = normalized.replace("$", " dollar ")
    normalized = normalized.replace("€", " euro ")

    # 3. Separate key punctuation marks that often signify urgent calls-to-action
    # e.g., exclamation marks, question marks, percentage
    normalized = re.sub(r"([!?,%])+", r" \1 ", normalized)

    # 4. Collapse multiple whitespace characters into a single space
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


class SMSPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible transformer for SMS text preprocessing.

    Encapsulates text normalization directly inside a scikit-learn Pipeline
    to prevent data leakage and guarantee that identical preprocessing is
    applied during training, cross-validation, testing, and production inference.
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def fit(self, X: Any, y: Optional[Any] = None) -> "SMSPreprocessor":
        """Fit method (stateless transformer)."""
        return self

    def transform(self, X: Any) -> List[str]:
        """Transforms a collection of raw text messages.

        Parameters
        ----------
        X : iterable of strings
            Raw SMS texts.

        Returns
        -------
        List[str]
            Cleaned and normalized texts.
        """
        cleaned_list: List[str] = []
        for item in X:
            if not isinstance(item, str):
                item = str(item) if item is not None else ""
            cleaned = clean_text(item)
            if self.lowercase:
                cleaned = cleaned.lower()
            cleaned_list.append(cleaned)
        return cleaned_list

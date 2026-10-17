"""Unit tests for src.preprocessing."""

import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from src.preprocessing import SMSPreprocessor, clean_text, validate_input_text


class TestValidateInputText:
    """Test suite for validate_input_text."""

    def test_valid_input_string(self):
        result = validate_input_text("  Valid SMS Message  ")
        assert result == "Valid SMS Message"

    def test_none_input_raises_type_error(self):
        with pytest.raises(TypeError, match="cannot be None"):
            validate_input_text(None)

    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a string"):
            validate_input_text(12345)
        with pytest.raises(TypeError, match="must be a string"):
            validate_input_text(["not", "a", "string"])

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_input_text("")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_input_text("     \t\n  ")

    def test_excessive_length_raises_value_error(self):
        oversized_text = "A" * 5001
        with pytest.raises(ValueError, match="exceeds safety limit"):
            validate_input_text(oversized_text, max_length=5000)

    def test_allow_empty_flag(self):
        assert validate_input_text("", allow_empty=True) == ""


class TestCleanText:
    """Test suite for clean_text."""

    def test_currency_symbol_normalization(self):
        text = "Win £500 or $1000 or €200 now!"
        cleaned = clean_text(text)
        assert "pound" in cleaned
        assert "dollar" in cleaned
        assert "euro" in cleaned
        assert "£" not in cleaned
        assert "$" not in cleaned
        assert "€" not in cleaned

    def test_repeated_punctuation_normalized(self):
        text = "URGENT!!!!! Call now????"
        cleaned = clean_text(text)
        assert " ! " in cleaned
        assert cleaned.endswith(" ?")

    def test_unicode_normalization(self):
        # Fullwidth latin characters or ligatures
        fullwidth_text = "ＦＲＥＥ ｓｐａｍ"
        cleaned = clean_text(fullwidth_text)
        assert "FREE spam" in cleaned

    def test_redundant_whitespace_collapsed(self):
        text = "Hello   \t\n   world   from   SMS  "
        cleaned = clean_text(text)
        assert cleaned == "Hello world from SMS"


class TestSMSPreprocessorTransformer:
    """Test suite for Scikit-Learn transformer SMSPreprocessor."""

    def test_fit_transform_pipeline(self):
        pipe = Pipeline(
            [
                ("preprocessor", SMSPreprocessor(lowercase=True)),
                ("vectorizer", CountVectorizer()),
            ]
        )
        sample_texts = [
            "Win £500 cash now!",
            "Hey, are you free for lunch?",
            "ALERT: Call 0800 now!",
        ]
        # Transform should lowercase and normalize
        matrix = pipe.fit_transform(sample_texts)
        assert matrix.shape[0] == 3
        vocab = pipe.named_steps["vectorizer"].vocabulary_
        assert "pound" in vocab  # from £ normalization
        assert "win" in vocab

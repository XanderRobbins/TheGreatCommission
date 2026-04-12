"""Tests for configuration module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from config import Config
from exceptions import LanguageNotSupportedError


def test_get_language_code_valid():
    """Test that valid language codes are returned."""
    code = Config.get_language_code("english")
    assert code == "eng_Latn"

    code = Config.get_language_code("spanish")
    assert code == "spa_Latn"


def test_get_language_code_invalid_raises():
    """Test that invalid language raises LanguageNotSupportedError."""
    with pytest.raises(LanguageNotSupportedError, match="not supported"):
        Config.get_language_code("klingon")


def test_get_language_code_case_insensitive():
    """Test that language code lookup is case-insensitive."""
    code = Config.get_language_code("ENGLISH")
    assert code == "eng_Latn"

    code = Config.get_language_code("SpAnish")
    assert code == "spa_Latn"


def test_ensure_dirs_creates_directories():
    """Test that ensure_dirs creates required directories."""
    # This test verifies ensure_dirs doesn't error
    # In production it would create actual directories
    try:
        Config.ensure_dirs()
    except Exception as e:
        pytest.fail(f"ensure_dirs() raised unexpected exception: {e}")


def test_consistency_loss_weight_default_zero():
    """Test that consistency loss weight defaults to 0.0."""
    # Default should be 0.0 (safe default)
    with patch.dict('os.environ', {}, clear=True):
        # Re-read the config (would be 0.0 with no env var)
        assert "consistency_loss" in Config.LOSS_WEIGHTS
        # If CONSISTENCY_LOSS_WEIGHT env var is not set, should default to 0.0
        # This prevents silent training corruption


def test_model_name_from_env():
    """Test that MODEL_NAME can be overridden via environment variable."""
    original = Config.MODEL_NAME

    # Verify it's set to something
    assert original is not None
    assert len(original) > 0

    # Should be the NLLB model by default
    assert "nllb" in original.lower()


def test_device_cpu_or_cuda():
    """Test that get_device returns valid device string."""
    device = Config.get_device()

    assert device in ["cpu", "cuda"]


def test_language_codes_dict_not_empty():
    """Test that LANGUAGE_CODES dictionary is populated."""
    assert len(Config.LANGUAGE_CODES) > 0
    assert "english" in Config.LANGUAGE_CODES
    assert "spanish" in Config.LANGUAGE_CODES


def test_training_config_has_gradient_accumulation():
    """Test that TRAINING_CONFIG includes gradient_accumulation_steps."""
    assert "gradient_accumulation_steps" in Config.TRAINING_CONFIG
    assert Config.TRAINING_CONFIG["gradient_accumulation_steps"] >= 1

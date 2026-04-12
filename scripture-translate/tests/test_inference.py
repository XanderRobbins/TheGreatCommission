"""Tests for inference engine (Bug 2, Bug 3 fixes)."""

import pytest
import torch
from unittest.mock import MagicMock, patch
from inference import ScriptureTranslator, TranslationResult


def test_confidence_not_always_one():
    """Test that confidence score is NOT always 1.0 (Bug 3 fix).

    Bug 3: softmax([x]) is always 1.0, but sequences_scores is log-prob.
    Fixed by: confidence = exp(log_prob).clamp(0, 1)
    """
    # Mock model and tokenizer
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids = MagicMock(return_value=1)
    tokenizer.decode = MagicMock(return_value="test translation")

    translator = ScriptureTranslator(model, tokenizer)

    # Mock generate output with negative log-prob (should result in < 1.0 confidence)
    mock_outputs = MagicMock()
    mock_outputs.sequences = [torch.tensor([1, 2, 3])]
    mock_outputs.sequences_scores = torch.tensor([-2.5])  # log-prob

    model.generate = MagicMock(return_value=mock_outputs)

    # Call translate_verse
    result = translator.translate_verse(
        "test",
        source_lang="eng_Latn",
        target_lang="spa_Latn",
    )

    # Confidence should be approximately exp(-2.5) ≈ 0.082, NOT 1.0
    assert result.confidence < 1.0
    assert result.confidence > 0.0
    assert abs(result.confidence - torch.exp(torch.tensor(-2.5)).item()) < 0.01


def test_translate_batch_real_batching():
    """Test that translate_batch uses batching, not single-item loops (Bug 2).

    Bug 2: batch_size parameter was accepted but ignored.
    Fixed by: Grouping verses into chunks, one model.generate() call per chunk.
    """
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids = MagicMock(return_value=1)
    tokenizer.decode = MagicMock(return_value="translation")

    translator = ScriptureTranslator(model, tokenizer)

    # Mock model output: 5 sequences per batch call (matching batch_size=5)
    # Note: Each batch call should return sequences matching that batch's size
    def create_mock_outputs(batch_size=5):
        outputs = MagicMock()
        # Create tensor with shape (batch_size, 2) to match batch processing
        outputs.sequences = torch.zeros((batch_size, 2), dtype=torch.long)
        outputs.sequences_scores = torch.tensor([-1.0] * batch_size)
        return outputs

    # Use side_effect to return properly-sized outputs for each batch
    model.generate = MagicMock(side_effect=[
        create_mock_outputs(5),  # First batch of 5 verses
        create_mock_outputs(5),  # Second batch of 5 verses
    ])

    # Create 10 verses
    verses = [{'text': f'verse {i}'} for i in range(10)]

    # Translate with batch_size=5 (should make 2 model calls, not 10)
    results = translator.translate_batch(
        verses,
        source_lang="eng_Latn",
        target_lang="spa_Latn",
        batch_size=5,
        show_progress=False,
    )

    assert len(results) == 10, f"Expected 10 results, got {len(results)}"
    # Should be called twice (batches of 5 and 5), not 10 times
    assert model.generate.call_count == 2


def test_translation_result_json_serializable():
    """Test that TranslationResult can be serialized to JSON."""
    result = TranslationResult(
        primary="translation",
        confidence=0.95,
        alternatives=["alt1", "alt2"],
        theological_terms={"salvation": "salvación"},
        consistency_enforced=False,
        source_text="source",
        target_language="spa_Latn",
    )

    # Should be serializable without error
    result_dict = result.to_dict()

    assert result_dict["primary"] == "translation"
    assert result_dict["confidence"] == 0.95
    assert isinstance(result_dict["alternatives"], list)
    assert isinstance(result_dict["theological_terms"], dict)


def test_translation_result_none_defaults():
    """Test that TranslationResult handles None fields correctly."""
    result = TranslationResult(
        primary="test",
        confidence=0.8,
    )

    result_dict = result.to_dict()
    assert result_dict["alternatives"] == []
    assert result_dict["theological_terms"] == {}
    assert result_dict["consistency_enforced"] is False

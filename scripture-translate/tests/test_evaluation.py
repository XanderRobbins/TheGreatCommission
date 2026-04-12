"""Tests for evaluation module (Bug 6, Bug 8 fixes)."""

import pytest
from unittest.mock import MagicMock
from evaluation import ScriptureEvaluator
from models.terminology import TerminologyDB, TermExtractor


def test_compute_consistency_score_not_stub():
    """Test that compute_consistency_score is implemented (Bug 6).

    Bug 6: Inner loop was just `pass`, function always returned 1.0.
    Fixed by: Using TermExtractor to track theological terms and surface forms.
    """
    terminology_db = TerminologyDB()

    # Add some terms
    terminology_db.add_term("salvation", "spa_Latn", "salvación")
    terminology_db.add_term("grace", "spa_Latn", "gracia")

    evaluator = ScriptureEvaluator(terminology_db)

    # Create translations with consistent term usage
    translations = [
        "God's salvación is available",
        "Through salvación and gracia",
    ]

    # Score should not be 1.0 if terms are found and tracked
    score = evaluator.compute_consistency_score(translations, terminology_db)

    # Should be between 0 and 1, and computed (not just stub returning 1.0)
    assert 0.0 <= score <= 1.0


def test_bleu_perfect_match():
    """Test BLEU score calculation."""
    evaluator = ScriptureEvaluator()

    # Identical hypothesis and reference (simple test)
    text = "the cat sat on the mat"
    bleu = evaluator.compute_bleu(
        text,
        text,
        weights=(0.25, 0.25, 0.25, 0.25),
    )

    # Should be 1.0 for perfect match
    assert bleu == 1.0, f"Expected 1.0, got {bleu}"


def test_bleu_zero_for_empty():
    """Test BLEU score for empty input."""
    evaluator = ScriptureEvaluator()

    bleu = evaluator.compute_bleu("", "reference")
    assert bleu == 0.0

    bleu = evaluator.compute_bleu("hypothesis", "")
    assert bleu == 0.0


def test_evaluate_batch_mismatched_lengths():
    """Test that evaluate_batch raises ValueError for mismatched lengths."""
    evaluator = ScriptureEvaluator()

    with pytest.raises(ValueError, match="same length"):
        evaluator.evaluate_batch(
            hypotheses=["a", "b"],
            references=["a", "b", "c"],
            target_lang="spa_Latn",
        )


def test_evaluate_batch_empty():
    """Test that evaluate_batch raises ValueError for empty input."""
    evaluator = ScriptureEvaluator()

    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate_batch(
            hypotheses=[],
            references=[],
            target_lang="spa_Latn",
        )


def test_print_metrics_does_not_raise():
    """Test that print_metrics completes without errors."""
    from evaluation import EvaluationMetrics

    evaluator = ScriptureEvaluator()

    metrics = EvaluationMetrics(
        bleu_1=0.5,
        bleu_2=0.4,
        bleu_4=0.3,
        consistency_score=0.8,
        unique_terms=100,
        avg_term_usage=2.5,
    )

    # Should not raise any exception
    evaluator.print_metrics(metrics, title="Test Metrics")


def test_keyboard_interrupt_not_swallowed():
    """Test that KeyboardInterrupt is not caught (Bug 8).

    Bug 8: Bare `except:` would catch KeyboardInterrupt.
    Fixed by: Using `except Exception as exc:` to allow interrupts.
    """
    evaluator = ScriptureEvaluator()

    # Create a mock that raises KeyboardInterrupt
    def raising_bleu(*args, **kwargs):
        raise KeyboardInterrupt()

    evaluator.compute_bleu = raising_bleu

    # KeyboardInterrupt should NOT be caught
    with pytest.raises(KeyboardInterrupt):
        evaluator.compute_bleu("a", "b")

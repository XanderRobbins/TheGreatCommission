"""Tests for data loaders module."""

import tempfile
import json
from pathlib import Path
import pytest
from data.loaders import create_data_splits, BibleDataLoader


def test_create_data_splits_reproducible():
    """Test that create_data_splits with same seed produces identical splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        test_file = Path(tmpdir) / "test_verses.jsonl"
        verses = [
            {"source": f"verse {i}", "target": f"verso {i}"}
            for i in range(100)
        ]

        with open(test_file, 'w', encoding='utf-8') as f:
            for verse in verses:
                f.write(json.dumps(verse) + "\n")

        # Split with seed 42
        train1, val1, test1 = create_data_splits(test_file, seed=42)

        # Read first split
        with open(train1, 'r', encoding='utf-8') as f:
            train1_lines = [line.strip() for line in f if line.strip()]

        # Split again with same seed
        train2, val2, test2 = create_data_splits(test_file, seed=42)

        # Read second split
        with open(train2, 'r', encoding='utf-8') as f:
            train2_lines = [line.strip() for line in f if line.strip()]

        # Should be identical
        assert train1_lines == train2_lines


def test_create_data_splits_different_seeds():
    """Test that different seeds produce different splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_verses.jsonl"
        verses = [
            {"source": f"verse {i}", "target": f"verso {i}"}
            for i in range(100)
        ]

        with open(test_file, 'w', encoding='utf-8') as f:
            for verse in verses:
                f.write(json.dumps(verse) + "\n")

        # Split with seed 42
        train1, _, _ = create_data_splits(test_file, seed=42)

        with open(train1, 'r', encoding='utf-8') as f:
            train1_lines = [line.strip() for line in f if line.strip()]

        # Split with seed 99
        train2, _, _ = create_data_splits(test_file, seed=99)

        with open(train2, 'r', encoding='utf-8') as f:
            train2_lines = [line.strip() for line in f if line.strip()]

        # Should be different (extremely unlikely to be same with different seeds)
        assert len(train1_lines) == len(train2_lines)
        assert train1_lines != train2_lines


def test_create_data_splits_respects_ratios():
    """Test that data splits respect train/val/test ratios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_verses.jsonl"
        verses = [
            {"source": f"verse {i}", "target": f"verso {i}"}
            for i in range(100)
        ]

        with open(test_file, 'w', encoding='utf-8') as f:
            for verse in verses:
                f.write(json.dumps(verse) + "\n")

        train_path, val_path, test_path = create_data_splits(
            test_file,
            train_ratio=0.8,
            val_ratio=0.1,
        )

        with open(train_path, 'r', encoding='utf-8') as f:
            train_count = sum(1 for line in f if line.strip())

        with open(val_path, 'r', encoding='utf-8') as f:
            val_count = sum(1 for line in f if line.strip())

        with open(test_path, 'r', encoding='utf-8') as f:
            test_count = sum(1 for line in f if line.strip())

        total = train_count + val_count + test_count
        assert total == 100

        # Approximate ratio checks
        assert 75 <= train_count <= 85  # ~80%
        assert 5 <= val_count <= 15      # ~10%
        assert 5 <= test_count <= 15     # ~10%


def test_bible_data_loader_initialization():
    """Test that BibleDataLoader initializes without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = BibleDataLoader(Path(tmpdir))

        assert loader.data_dir == Path(tmpdir)
        assert len(loader.ALL_BOOKS) > 0
        assert "Genesis" in loader.ALL_BOOKS


def test_bible_verse_reference():
    """Test BibleVerse.reference() formatting."""
    from data.loaders import BibleVerse

    verse = BibleVerse(
        book="Genesis",
        chapter=1,
        verse=1,
        text="In the beginning...",
        language="en"
    )

    assert verse.reference() == "Genesis 1:1"


def test_save_parallel_corpus_parameter_name():
    """Test that save_parallel_corpus uses 'output_format' not 'format'.

    This tests the fix for shadowing the builtin 'format' function.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = BibleDataLoader(Path(tmpdir))

        # Add some test verses
        from data.loaders import BibleVerse
        loader.verses["english"] = [
            BibleVerse("Genesis", 1, 1, "In the beginning", "en"),
        ]
        loader.verses["spanish"] = [
            BibleVerse("Genesis", 1, 1, "En el principio", "es"),
        ]

        # Should accept output_format parameter
        output_file = Path(tmpdir) / "corpus.jsonl"
        loader.save_parallel_corpus(
            "english",
            "spanish",
            output_file,
            output_format="jsonl",
        )

        assert output_file.exists()

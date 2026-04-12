"""Tests for terminology database (Bug 1 fix)."""

import tempfile
from pathlib import Path
import pytest
from models.terminology import TerminologyDB


def test_load_preserves_defaultdict_semantics():
    """Test that load() reconstructs nested defaultdicts.

    Bug 1: After save()->load(), record_usage() would raise KeyError
    because defaultdicts were replaced with plain dicts.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "terminology.json"

        # Create and populate database
        db1 = TerminologyDB(db_path)
        db1.add_term("salvation", "spa_Latn", "salvación", confidence=0.98)
        db1.add_term("grace", "spa_Latn", "gracia", confidence=0.97)
        db1.save()

        # Load from saved file
        db2 = TerminologyDB(db_path)

        # This should NOT raise KeyError (Bug 1 would fail here)
        db2.record_usage("salvation", "spa_Latn")
        db2.record_usage("grace", "spa_Latn")

        # Verify usage was recorded
        assert db2.get_usage_count("salvation", "spa_Latn") == 1
        assert db2.get_usage_count("grace", "spa_Latn") == 1


def test_add_term_conflict_no_override():
    """Test that term conflicts are preserved without override."""
    db = TerminologyDB()

    # Add initial term
    success1 = db.add_term("salvation", "spa_Latn", "salvación")
    assert success1 is True

    # Try to add conflicting term without override
    success2 = db.add_term("salvation", "spa_Latn", "redención", override=False)
    assert success2 is False  # Should fail

    # Original term should be preserved
    result = db.lookup("salvation", "spa_Latn")
    assert result == "salvación"


def test_theological_terms_is_frozenset():
    """Test that THEOLOGICAL_TERMS is immutable."""
    db = TerminologyDB()

    # Should be frozenset (immutable)
    assert isinstance(db.THEOLOGICAL_TERMS, frozenset)

    # Attempting to add should raise AttributeError
    with pytest.raises(AttributeError):
        db.THEOLOGICAL_TERMS.add("test")


def test_get_usage_count_after_load():
    """Test usage tracking survives save/load cycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "terminology.json"

        # Create, populate, record usage
        db1 = TerminologyDB(db_path)
        db1.add_term("faith", "spa_Latn", "fe")
        db1.record_usage("faith", "spa_Latn")
        db1.record_usage("faith", "spa_Latn")
        db1.save()

        # Load and verify usage counts survived
        db2 = TerminologyDB(db_path)
        assert db2.get_usage_count("faith", "spa_Latn") == 2

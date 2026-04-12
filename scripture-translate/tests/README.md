# Tests

Test suite for Scripture Translation System covering all Phase 1 bug fixes and critical functionality.

## Running Tests

```bash
# Install pytest if not already installed
pip install pytest pytest-mock

# Run all tests
pytest

# Run specific test file
pytest tests/test_terminology_db.py -v

# Run specific test
pytest tests/test_terminology_db.py::test_load_preserves_defaultdict_semantics -v

# Run with coverage
pytest --cov=. --cov-report=term-report
```

## Test Coverage

### test_terminology_db.py
- **Bug 1 Fix**: Verify `load()` reconstructs nested defaultdicts (no KeyError on `record_usage()`)
- Verify term conflicts are preserved
- Verify THEOLOGICAL_TERMS is immutable (frozenset)
- Verify usage counts survive save/load cycle

### test_inference.py
- **Bug 3 Fix**: Confidence score is NOT always 1.0 (properly calculates from log-prob)
- **Bug 2 Fix**: Batching is real (not single-item loops)
- Verify TranslationResult is JSON serializable
- Verify None fields have proper defaults

### test_evaluation.py
- **Bug 6 Fix**: `compute_consistency_score()` is fully implemented (not stub)
- **Bug 8 Fix**: KeyboardInterrupt is not swallowed
- BLEU score calculation accuracy
- Proper error handling for empty/mismatched batches
- `print_metrics()` completes without errors

### test_config.py
- Language code validation raises `LanguageNotSupportedError` for unsupported languages
- Language codes are case-insensitive
- Directories are created properly
- Consistency loss weight defaults to 0.0 (safe)
- Device detection works (CPU or CUDA)

### test_data_loaders.py
- **Seed Reproducibility**: Same seed produces identical splits
- **Different Seeds**: Different seeds produce different splits
- Ratios are respected (train/val/test)
- BibleDataLoader initializes properly
- BibleVerse reference formatting is correct
- `save_parallel_corpus()` uses `output_format` parameter (not `format` builtin)

## Key Testing Principles

1. **Bug Verification**: Each test directly verifies a Phase 1 bug was fixed
2. **Isolation**: Tests use temporary directories and mocks to avoid side effects
3. **Clarity**: Test names clearly describe what is being verified
4. **Completeness**: Cover both happy path and error cases

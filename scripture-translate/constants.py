"""Global constants for the Scripture Translation system.

Magic numbers and string literals are centralized here to:
1. Enable single-point configuration changes
2. Document the relationship between constants used in different files
3. Prevent typos and inconsistencies
"""

# ============================================================================
# Tokenization & Sequence Lengths
# ============================================================================

MAX_SOURCE_LENGTH: int = 512
"""Maximum sequence length for encoder input (English verses)."""

MAX_TARGET_LENGTH: int = 256
"""Maximum sequence length for decoder output (target language verses)."""

NLLB_VOCAB_SIZE: int = 256206
"""Vocabulary size for NLLB tokenizer."""

# ============================================================================
# Model Generation & Inference
# ============================================================================

DEFAULT_NUM_BEAMS: int = 5
"""Default beam width for beam search decoding."""

MIN_CONFIDENCE_THRESHOLD: float = 0.0
"""Minimum confidence for translation output."""

CONFIDENCE_CLIP_MIN: float = 0.0
CONFIDENCE_CLIP_MAX: float = 1.0
"""Bounds for confidence score clamping."""

# ============================================================================
# Training
# ============================================================================

GRAD_CLIP_NORM: float = 1.0
"""Maximum gradient norm for clipping during training."""

DEFAULT_LOG_EVERY_N_STEPS: int = 100
"""Default frequency of training log output (baseline training)."""

DEFAULT_LORA_LOG_EVERY_N_STEPS: int = 50
"""Default frequency of training log output (LoRA fine-tuning)."""

# ============================================================================
# Evaluation
# ============================================================================

MIN_BLEU_SCORE: float = 20.0
"""Minimum acceptable BLEU-4 score (quality threshold)."""

MIN_CONSISTENCY_SCORE: float = 0.85
"""Minimum acceptable terminology consistency score (0-1)."""

MIN_HUMAN_RATING: float = 3.5
"""Minimum acceptable human rating (on 5-point scale)."""

# ============================================================================
# Terminology Database
# ============================================================================

TERM_DEFAULT_CONFIDENCE: float = 0.9
"""Default confidence for newly added terms."""

TERM_REVIEWED_CONFIDENCE: float = 0.95
"""Confidence for terms that have been reviewed by a human."""

TERMINOLOGY_DB_VERSION: str = "1.0"
"""Schema version for terminology database."""

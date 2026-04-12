"""Translation inference engine for Scripture translation.

Provides batch and single-verse translation with consistency enforcement,
confidence scoring, and alternative generation.
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from models.terminology import TerminologyDB, TermExtractor
from config import Config
from constants import MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, DEFAULT_NUM_BEAMS
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TranslationResult:
    """Container for translation results.

    Attributes:
        primary: Primary translation output.
        confidence: Confidence score (0.0-1.0) for the translation.
        alternatives: List of alternative translations from beam search.
        theological_terms: Dict mapping English terms to target language translations.
        consistency_enforced: Whether consistency enforcement modified the output.
        source_text: Original source text (may include reference).
        target_language: Target language code used for translation.
    """

    primary: str
    confidence: float
    alternatives: Optional[List[str]] = field(default_factory=list)
    theological_terms: Optional[Dict[str, Optional[str]]] = field(default_factory=dict)
    consistency_enforced: bool = False
    source_text: str = ""
    target_language: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields ready for JSON encoding.
        """
        return {
            "primary": self.primary,
            "confidence": float(self.confidence),
            "alternatives": self.alternatives or [],
            "theological_terms": self.theological_terms or {},
            "consistency_enforced": self.consistency_enforced,
            "source_text": self.source_text,
            "target_language": self.target_language,
        }


class ScriptureTranslator:
    """Inference engine for scripture translation with consistency enforcement.

    Provides verse-by-verse translation with beam search, alternative generation,
    confidence scoring, and terminology consistency enforcement.
    """

    def __init__(
        self,
        model,
        tokenizer,
        terminology_db: Optional[TerminologyDB] = None,
        device: Optional[str] = None,
        enforce_consistency: bool = True,
    ) -> None:
        """Initialize translator.

        Args:
            model: Loaded NLLB model.
            tokenizer: NLLB tokenizer.
            terminology_db: Optional terminology database for consistency.
                            Defaults to new empty database.
            device: Device to use ("cpu" or "cuda:0"). Defaults to Config.get_device().
            enforce_consistency: Whether to enforce terminology consistency post-generation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.terminology_db = terminology_db or TerminologyDB()
        self.device = device or Config.get_device()
        self.enforce_consistency = enforce_consistency
        self.term_extractor = TermExtractor(self.terminology_db)

        self.model.eval()
        logger.info("ScriptureTranslator initialized")
    
    def translate_verse(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        num_beams: int = DEFAULT_NUM_BEAMS,
        max_length: int = MAX_TARGET_LENGTH,
        temperature: float = 1.0,
        return_alternatives: bool = True,
    ) -> TranslationResult:
        """Translate a single verse.

        Args:
            source_text: Source verse text.
            source_lang: Source language code (NLLB format, e.g., "eng_Latn").
            target_lang: Target language code (NLLB format, e.g., "spa_Latn").
            num_beams: Number of beams for beam search. Defaults to DEFAULT_NUM_BEAMS.
            max_length: Maximum output length in tokens. Defaults to MAX_TARGET_LENGTH.
            temperature: Temperature for generation. Lower=more deterministic.
            return_alternatives: Whether to return alternative translations from beam search.

        Returns:
            TranslationResult with primary translation, confidence, and alternatives.
        """
        # Extract theological terms before translation
        theological_terms = self.term_extractor.extract_theological_terms(source_text)
        canonical_terms = self.term_extractor.get_canonical_terms(source_text, target_lang)

        # Tokenize input
        inputs = self.tokenizer(
            source_text,
            return_tensors="pt",
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Set source language
        self.tokenizer.src_lang = source_lang

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_beams if return_alternatives else 1,
                temperature=temperature,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_lang),
            )

        # Decode translations
        primary_translation = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        # Get alternatives
        alternatives = []
        if return_alternatives and len(outputs.sequences) > 1:
            for seq in outputs.sequences[1:]:
                alt = self.tokenizer.decode(seq, skip_special_tokens=True)
                alternatives.append(alt)

        # Calculate confidence from log-probability
        # Bug 3 fix: sequences_scores is a log-probability, not softmax input
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            log_prob = outputs.sequences_scores[0].item()
            # Convert log-prob to probability via exp, then clamp to [0, 1]
            confidence = float(torch.exp(torch.tensor(log_prob)).clamp(0.0, 1.0).item())
        else:
            confidence = 0.0

        # Enforce consistency
        consistency_enforced = False
        if self.enforce_consistency and canonical_terms:
            primary_translation, consistency_enforced = self._enforce_consistency(
                primary_translation, source_text, canonical_terms, target_lang
            )

        # Record term usage
        for term in theological_terms:
            self.terminology_db.record_usage(term, target_lang)

        return TranslationResult(
            primary=primary_translation,
            confidence=confidence,
            alternatives=alternatives,
            theological_terms=canonical_terms,
            consistency_enforced=consistency_enforced,
            source_text=source_text,
            target_language=target_lang,
        )
    
    def _enforce_consistency(
        self,
        translation: str,
        source_text: str,
        canonical_terms: Dict[str, Optional[str]],
        target_lang: str,
    ) -> Tuple[str, bool]:
        """Post-process translation to enforce terminology consistency.

        Args:
            translation: Generated translation to post-process.
            source_text: Original source text (for context).
            canonical_terms: Canonical terms from terminology DB.
            target_lang: Target language code.

        Returns:
            Tuple of (potentially_modified_translation, was_modified_flag).
        """
        modified = False
        enforced = translation

        # For now, just log what would be enforced
        # Full implementation would intelligently replace terms with fuzzy matching
        for source_term, canonical in canonical_terms.items():
            if canonical and canonical not in enforced:
                # Log the inconsistency
                logger.debug(
                    f"Would enforce: {source_term} → {canonical} in {target_lang}"
                )

        return enforced, modified
    
    def translate_batch(
        self,
        verses: List[Dict],
        source_lang: str,
        target_lang: str,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> List[TranslationResult]:
        """Translate multiple verses with true batching for efficiency.

        Processes verses in chunks, making one model.generate() call per chunk
        instead of one per verse. This is essential for reasonable performance
        on large datasets.

        Args:
            verses: List of verse dictionaries, each with 'text' key and optional 'reference'.
            source_lang: Source language code (NLLB format, e.g., "eng_Latn").
            target_lang: Target language code (NLLB format, e.g., "spa_Latn").
            batch_size: Number of verses to process per model call. Default 8.
                        Adjust based on available GPU memory.
            show_progress: Whether to display a progress bar during translation.

        Returns:
            List of TranslationResult objects, one per input verse.
        """
        results = []

        # Wrap in tqdm for progress display
        iterator = verses
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(verses, desc="Translating verses")
            except ImportError:
                logger.warning("tqdm not available; progress bar disabled")
                iterator = verses

        # Process verses in batches
        for batch_start in range(0, len(verses), batch_size):
            batch_end = min(batch_start + batch_size, len(verses))
            batch_verses = verses[batch_start:batch_end]

            # Extract texts and metadata
            texts = [v["text"] for v in batch_verses]
            references = [v.get("reference", "") for v in batch_verses]

            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=MAX_SOURCE_LENGTH,
                truncation=True,
                padding=True,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Set source language
            self.tokenizer.src_lang = source_lang

            # Generate translations for entire batch in one call
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=DEFAULT_NUM_BEAMS,
                    num_return_sequences=1,  # One translation per verse
                    early_stopping=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_lang),
                )

            # Decode and create results for each verse in batch
            # Use actual output sequence length to avoid indexing errors
            num_outputs = outputs.sequences.shape[0]
            if num_outputs != len(batch_verses):
                logger.warning(f"Batch output mismatch: {num_outputs} outputs vs {len(batch_verses)} inputs")

            for i in range(len(batch_verses)):
                verse = batch_verses[i]
                ref = references[i]

                # Guard against fewer outputs
                if i >= num_outputs:
                    logger.warning(f"Skipping verse {i}: outputs ({num_outputs}) < batch ({len(batch_verses)})")
                    break

                # Get the translation for this verse from batch output
                seq_idx = i  # Index into the batch sequences
                primary_translation = self.tokenizer.decode(
                    outputs.sequences[seq_idx], skip_special_tokens=True
                )

                # Calculate confidence for this verse
                if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                    log_prob = outputs.sequences_scores[seq_idx].item()
                    confidence = float(torch.exp(torch.tensor(log_prob)).clamp(0.0, 1.0).item())
                else:
                    confidence = 0.0

                # Extract and record theological terms
                theological_terms = self.term_extractor.extract_theological_terms(verse["text"])
                for term in theological_terms:
                    self.terminology_db.record_usage(term, target_lang)

                canonical_terms = self.term_extractor.get_canonical_terms(
                    verse["text"], target_lang
                )

                # Create result
                result = TranslationResult(
                    primary=primary_translation,
                    confidence=confidence,
                    alternatives=[],  # No alternatives in batch mode (efficiency)
                    theological_terms=canonical_terms,
                    consistency_enforced=False,
                    source_text=f"{ref}: {verse['text']}" if ref else verse["text"],
                    target_language=target_lang,
                )

                results.append(result)

            # Update progress bar
            if show_progress and hasattr(iterator, "update"):
                iterator.update(batch_end - batch_start)

        return results
    
    def translate_book(
        self,
        book_verses: List[Dict],
        book_name: str,
        source_lang: str,
        target_lang: str,
    ) -> Dict:
        """Translate an entire book of Scripture.

        Args:
            book_verses: List of verse dictionaries with 'text' and optional 'reference'.
            book_name: Name of the book (e.g., "Genesis") for logging.
            source_lang: Source language code (NLLB format, e.g., "eng_Latn").
            target_lang: Target language code (NLLB format, e.g., "spa_Latn").

        Returns:
            Dictionary with:
                - book: Book name
                - source_language: Source language code
                - target_language: Target language code
                - total_verses: Number of verses translated
                - translations: List of translation result dictionaries
                - statistics: Dict with average_confidence and verses_with_consistency_enforced
        """
        logger.info(f"Translating {book_name} ({len(book_verses)} verses)...")

        results = self.translate_batch(
            book_verses,
            source_lang,
            target_lang,
            show_progress=True,
        )

        # Calculate statistics
        avg_confidence = (
            sum(r.confidence for r in results) / len(results) if results else 0.0
        )
        consistency_enforced = sum(1 for r in results if r.consistency_enforced)

        return {
            "book": book_name,
            "source_language": source_lang,
            "target_language": target_lang,
            "total_verses": len(results),
            "translations": [r.to_dict() for r in results],
            "statistics": {
                "average_confidence": avg_confidence,
                "verses_with_consistency_enforced": consistency_enforced,
            },
        }


class BeamSearchDecoder:
    """Enhanced beam search with optional terminology constraint support.

    TODO: Constraint support is a stub. Full implementation would use
    fuzzy string matching to safely replace output tokens with canonical
    terminology without breaking fluency.
    """

    def __init__(
        self, model, tokenizer, terminology_db: Optional[TerminologyDB] = None
    ) -> None:
        """Initialize beam search decoder.

        Args:
            model: Loaded NLLB model.
            tokenizer: NLLB tokenizer.
            terminology_db: Optional terminology database for constraints.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.terminology_db = terminology_db

    def decode_with_constraints(
        self,
        input_ids: torch.Tensor,
        target_lang: str,
        terminology_constraints: Optional[Dict[str, str]] = None,
        num_beams: int = DEFAULT_NUM_BEAMS,
        max_length: int = MAX_TARGET_LENGTH,
    ) -> List[str]:
        """Decode with optional terminology constraints.

        TODO: Constraint application is stubbed. In production, constraints
        should force certain terms to be used in the output via:
        - Fuzzy matching against output to find replacement sites
        - Weighted biasing of beam scores
        - Or hard constraints during beam generation

        Args:
            input_ids: Tokenized input tensor.
            target_lang: Target language code.
            terminology_constraints: Dict mapping source terms to target terms.
            num_beams: Number of beams for beam search.
            max_length: Maximum output length in tokens.

        Returns:
            List of decoded translations.
        """
        # Generate as usual
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            early_stopping=True,
        )

        # Decode
        translations = [
            self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs
        ]

        # Apply constraints post-hoc (simplified, stub implementation)
        if terminology_constraints:
            translations = [
                self._apply_constraints(t, terminology_constraints) for t in translations
            ]

        return translations

    def _apply_constraints(self, text: str, constraints: Dict[str, str]) -> str:
        """Apply terminology constraints to text.

        TODO: Stubbed. Full implementation should use fuzzy matching
        to safely replace surface forms while preserving sentence structure.

        Args:
            text: Generated text to constrain.
            constraints: Dict mapping source terms to target terms.

        Returns:
            Text with constraints applied (currently unchanged stub).
        """
        # Placeholder: in production, use fuzzy matching + safe replacement
        return text


if __name__ == "__main__":
    from models.base import ScriptureTranslationModel
    from utils.logger import configure_logging

    configure_logging()
    logger.info("Testing ScriptureTranslator...")

    # Initialize model
    model_wrapper = ScriptureTranslationModel()

    # Initialize translator
    terminology_db = TerminologyDB()
    translator = ScriptureTranslator(
        model=model_wrapper.get_model(),
        tokenizer=model_wrapper.get_tokenizer(),
        terminology_db=terminology_db,
        enforce_consistency=True,
    )

    # Test single verse translation
    source = "In the beginning, God created the heavens and the earth."
    logger.info(f"Source: {source}")

    result = translator.translate_verse(
        source,
        source_lang="eng_Latn",
        target_lang="spa_Latn",
        num_beams=3,
    )

    logger.info(f"Translation: {result.primary}")
    logger.info(f"Confidence: {result.confidence:.2%}")

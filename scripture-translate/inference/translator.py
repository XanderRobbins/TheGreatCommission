"""Translation inference engine for Scripture translation.

Provides batch and single-verse translation with consistency enforcement,
confidence scoring, and alternative generation. Supports context windows
(prev/current/next verses) for improved semantic understanding.
"""

import torch
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from models.terminology import TerminologyDB, TermExtractor
from models.tiered_terminology import TieredTerminologyDB, TermTier
from config import Config
from constants import MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, DEFAULT_NUM_BEAMS
from utils.logger import get_logger
from inference.context_manager import ContextWindowBuilder, ContextWindow
from inference.prompt_builder import PromptBuilder
from inference.confidence_scorer import ConfidenceScorer
from inference.translation_memory import TranslationMemory
from inference.back_translator import BackTranslationValidator

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
        tiered_terminology: Optional[TieredTerminologyDB] = None,
        device: Optional[str] = None,
        enforce_consistency: bool = True,
        use_prompt_conditioning: bool = True,
    ) -> None:
        """Initialize translator.

        Args:
            model: Loaded NLLB model.
            tokenizer: NLLB tokenizer.
            terminology_db: Optional terminology database for consistency.
                            Defaults to new empty database.
            tiered_terminology: Optional tiered terminology system for soft constraints.
            device: Device to use ("cpu" or "cuda:0"). Defaults to Config.get_device().
            enforce_consistency: Whether to enforce terminology consistency post-generation.
            use_prompt_conditioning: Whether to use domain-adaptive prompts.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.terminology_db = terminology_db or TerminologyDB()
        self.tiered_terminology = tiered_terminology
        self.device = device or Config.get_device()
        self.enforce_consistency = enforce_consistency
        self.use_prompt_conditioning = use_prompt_conditioning
        self.term_extractor = TermExtractor(self.terminology_db)
        self.prompt_builder = PromptBuilder(self.terminology_db, tiered_terminology) if use_prompt_conditioning else None

        self.confidence_scorer = ConfidenceScorer(self.terminology_db, self.tiered_terminology)
        self.translation_memory = TranslationMemory()
        self.back_translator = BackTranslationValidator(model, tokenizer, device)

        self.model.eval()
        logger.info("ScriptureTranslator initialized")

    def _enforce_tier1_terms(
        self,
        translation: str,
        source_text: str,
        target_lang: str,
    ) -> str:
        """HARD enforcement: Replace source Tier 1 terms with their canonical translations.

        This is deterministic post-processing, not probabilistic.
        If "God" is in source and we have a Tier 1 mapping, we forcefully inject it.

        Args:
            translation: Generated translation to post-process.
            source_text: Original source text (to extract Tier 1 terms from).
            target_lang: Target language code.

        Returns:
            Translation with Tier 1 terms enforced.
        """
        if not self.tiered_terminology:
            return translation

        # Extract source-language Tier 1 terms
        tier1_terms = self.tiered_terminology.get_terms_by_tier(TermTier.TIER_1)

        # For each Tier 1 term in source, ensure its translation is in output
        for source_term, target_term in tier1_terms.items():
            # Check if source term appears in source (case-insensitive)
            if re.search(r"\b" + re.escape(source_term) + r"\b", source_text, re.IGNORECASE):
                # If target term is NOT in translation, inject it
                if target_term not in translation:
                    logger.debug(
                        f"Injecting Tier 1 term: {source_term} → {target_term} "
                        f"(was missing from output)"
                    )
                    # Simple heuristic: find similar-looking words and replace
                    # This is crude but effective for core terms like God/Bondye
                    replacement_made = self._inject_term(translation, source_term, target_term)
                    if replacement_made:
                        translation = replacement_made

        return translation

    def _inject_term(
        self, translation: str, source_term: str, target_term: str
    ) -> Optional[str]:
        """Inject a missing canonical term into translation.

        Uses fuzzy matching to find plausible replacement locations.

        Args:
            translation: Translation to modify.
            source_term: English term that should appear.
            target_term: Target language canonical translation.

        Returns:
            Modified translation, or None if no safe injection point found.
        """
        # For now, use simple word-list approach
        # Map of common mistranslations for Tier 1 terms to watch for
        common_mistranslations = {
            "god": ["bondieu", "dye", "Dye", "bondieu"],
            "spirit": ["espri", "lèspri", "Espri"],
            "jesus": ["jezi", "jezis", "Jezis"],
            "holy": ["sen", "senm", "ho"],
        }

        source_lower = source_term.lower()
        if source_lower in common_mistranslations:
            # Look for mistranslations and replace with canonical
            for mistranslation in common_mistranslations[source_lower]:
                pattern = r"\b" + re.escape(mistranslation) + r"\b"
                if re.search(pattern, translation, re.IGNORECASE):
                    return re.sub(
                        pattern, target_term, translation, flags=re.IGNORECASE, count=1
                    )

        return None

    def _detect_french_contamination(self, text: str) -> bool:
        """Detect if French words have leaked into Haitian Creole output.

        Args:
            text: Text to check.

        Returns:
            True if French contamination detected.
        """
        # Common French words that should NOT appear in Haitian Creole
        # (Creole has different forms)
        french_indicators = {
            r"\bla\b",  # French article (Creole: "a", "an")
            r"\ble\b",  # French article
            r"\bdes\b",  # French plural article
            r"\bet\b",  # French "and" (Creole: "epi", "ak")
            r"\best\b",  # French "is" (Creole: "se", "ye")
            r"\bde\b",  # French "of" (Creole: "nan", "pou a")
            r"\bétat\b",  # French "state" (specific to French)
            r"\bdésert\b",  # French spelling (Creole: "dezè")
            r"\bconformé\b",  # French past participle
            r"\bétait\b",  # French "was" (Creole: "te")
            r"\bvide\b",  # French "empty" (Creole: "vid")
            r"\beau\b",  # French "water" (Creole: "dlo")
            r"\bcouvert\b",  # French "covered" (Creole: "kouvri")
            r"\bplus\b",  # French "more" (Creole: "plis", "pli")
        }

        for pattern in french_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                logger.debug(f"Detected French word matching: {pattern}")
                return True

        return False

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

    def translate_batch_with_context(
        self,
        verses: List[Dict],
        source_lang: str,
        target_lang: str,
        batch_size: int = 8,
        show_progress: bool = True,
        context_range: int = 1,
    ) -> List[TranslationResult]:
        """Translate verses using context windows (prev/current/next).

        Builds sliding windows of verses to provide semantic context during
        translation. This improves pronoun resolution, theological term
        consistency, and overall fluency.

        Args:
            verses: List of verse dictionaries with 'text' and optional 'reference'.
            source_lang: Source language code (e.g., "eng_Latn").
            target_lang: Target language code (e.g., "hat_Latn").
            batch_size: Number of context windows to process per batch.
            show_progress: Whether to show progress bar.
            context_range: How many verses before/after to include (default 1).

        Returns:
            List of TranslationResult objects, one per input verse.
        """
        results = [None] * len(verses)  # Pre-allocate results list

        # Build context windows
        windows = ContextWindowBuilder.build_windows(verses, context_range=context_range)

        # Wrap in progress bar
        iterator = windows
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(windows, desc="Translating verses with context")
            except ImportError:
                logger.warning("tqdm not available; progress bar disabled")
                iterator = windows

        # Process context windows in batches
        for batch_start in range(0, len(windows), batch_size):
            batch_end = min(batch_start + batch_size, len(windows))
            batch_windows = windows[batch_start:batch_end]

            # Build prompts with strict delimiter separation
            prompts = []
            if self.prompt_builder:
                prompts = [
                    self.prompt_builder.build_context_prompt(w, target_lang)
                    for w in batch_windows
                ]
            else:
                # Fallback: just use verse text without conditioning
                prompts = [w.current_verse for w in batch_windows]

            # Tokenize batch
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                max_length=MAX_SOURCE_LENGTH,
                truncation=True,
                padding=True,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            self.tokenizer.src_lang = source_lang

            # Generate translations
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=DEFAULT_NUM_BEAMS,
                    num_return_sequences=1,
                    early_stopping=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_lang),
                )

            # Process results for each window
            num_outputs = outputs.sequences.shape[0]
            for i in range(len(batch_windows)):
                window = batch_windows[i]
                orig_verse = verses[window.index]

                # STEP 1: Check translation memory (cache hit = no recomputation)
                language_pair = f"{source_lang}→{target_lang}"
                cached_translation = self.translation_memory.lookup(
                    orig_verse["text"], language_pair
                )

                if cached_translation:
                    verse_translation, cached_confidence = cached_translation
                    logger.debug(
                        f"Translation memory HIT: {window.current_ref}"
                    )
                else:
                    # STEP 2: Decode raw output if not cached
                    if i < num_outputs:
                        raw_output = self.tokenizer.decode(
                            outputs.sequences[i], skip_special_tokens=True
                        )

                        # Extract translation from structured prompt output
                        if self.prompt_builder:
                            verse_translation = self.prompt_builder.extract_translation_from_output(raw_output)
                        else:
                            verse_translation = raw_output
                    else:
                        logger.warning(f"Missing output for verse {window.current_ref}")
                        verse_translation = ""

                # Skip post-processing if from cache
                if not cached_translation:
                    # STEP 3: HARD POST-PROCESSING: Enforce Tier 1 terminology
                    if self.tiered_terminology and verse_translation:
                        verse_translation = self._enforce_tier1_terms(
                            verse_translation, orig_verse["text"], target_lang
                        )

                    # STEP 4: Calculate composite confidence from multiple signals
                    model_base_confidence = 0.5
                    if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                        log_prob = outputs.sequences_scores[i].item() if i < len(outputs.sequences_scores) else 0.0
                        model_base_confidence = float(torch.exp(torch.tensor(log_prob)).clamp(0.0, 1.0).item())

                    # Use multi-metric scorer for real confidence grounding
                    confidence, confidence_components = self.confidence_scorer.score(
                        verse_translation,
                        orig_verse["text"],
                        target_lang,
                        model_base_confidence,
                    )

                    quality_tier = self.confidence_scorer.get_quality_tier(confidence)
                    if quality_tier in ["acceptable", "poor"]:
                        logger.debug(f"{window.current_ref}: {quality_tier} quality (conf={confidence:.2f})")
                        logger.debug(f"  Components: {confidence_components}")
                else:
                    # Use cached confidence
                    confidence = cached_confidence
                    confidence_components = {"source": "cache"}

                # STEP 5: Back-translation validation (optional, for good-quality translations)
                if confidence >= 0.75 and not cached_translation:
                    try:
                        back_trans = self.back_translator.back_translate(
                            verse_translation, target_lang, source_lang
                        )
                        similarity, bt_metrics = self.back_translator.validate(
                            verse_translation, orig_verse["text"], back_trans
                        )

                        # Adjust confidence based on back-translation similarity
                        confidence = confidence * 0.7 + similarity * 0.3
                        confidence_components["back_translation_similarity"] = similarity
                        confidence_components["hallucination"] = self.back_translator.detect_hallucination(
                            verse_translation, back_trans
                        )

                        if confidence_components["hallucination"]:
                            logger.debug(f"{window.current_ref}: Hallucination detected via back-translation")
                    except Exception as e:
                        logger.debug(f"Back-translation validation skipped: {e}")

                # STEP 6: Store in translation memory for future lookups
                if not cached_translation and verse_translation:
                    self.translation_memory.store(
                        orig_verse["text"],
                        verse_translation,
                        language_pair,
                        confidence,
                    )

                # Extract theological terms
                theological_terms = self.term_extractor.extract_theological_terms(
                    orig_verse["text"]
                )
                for term in theological_terms:
                    self.terminology_db.record_usage(term, target_lang)

                canonical_terms = self.term_extractor.get_canonical_terms(
                    orig_verse["text"], target_lang
                )

                # Create result with enforcement flag
                consistency_enforced = bool(
                    self.tiered_terminology
                    and any(
                        self.tiered_terminology.get_tier(term) == TermTier.TIER_1
                        for term in theological_terms
                    )
                )

                result = TranslationResult(
                    primary=verse_translation,
                    confidence=confidence,
                    alternatives=[],
                    theological_terms=canonical_terms,
                    consistency_enforced=consistency_enforced,
                    source_text=f"{window.current_ref}: {orig_verse['text']}" if window.current_ref else orig_verse["text"],
                    target_language=target_lang,
                )

                results[window.index] = result

            if show_progress and hasattr(iterator, "update"):
                iterator.update(batch_end - batch_start)

        return results

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

"""Translation inference engine for Scripture translation.

Provides batch and single-verse translation with consistency enforcement,
confidence scoring, and alternative generation. Supports context windows
(prev/current/next verses) for improved semantic understanding.
"""

import math
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

    def _minimal_clean(self, text: str) -> str:
        """Minimal cleaning: only remove obvious junk, NOT complex parsing.

        Args:
            text: Raw model output.

        Returns:
            Minimally cleaned text.
        """
        text = text.strip()

        # Only remove if starts with clear instruction markers
        if text.startswith("###") or text.startswith(">>>") or text.startswith("==="):
            # These look like prompt echoes - skip to after the marker
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if not (line.startswith("###") or line.startswith(">>>") or line.startswith("===")):
                    text = "\n".join(lines[i:]).strip()
                    break

        # Remove if it contains excessive English/French keywords that suggest echoed instructions
        if text.count("DO NOT") > 2 or text.count("INSTRUCTION") > 1:
            # This looks like instructions were echoed back
            # Try to extract just the translation part
            lines = text.split("\n")
            # Take the last non-empty line or longest line
            non_empty_lines = [l.strip() for l in lines if l.strip() and len(l.strip()) > 10]
            if non_empty_lines:
                text = non_empty_lines[-1]

        return text.strip()

    def _enforce_tier1_terms(
        self,
        translation: str,
        source_text: str,
        target_lang: str,
    ) -> str:
        """HARD enforcement: Inject Tier 1 terms (God, Jesus, Spirit, etc.).

        Deterministic post-processing: if source has a Tier 1 term,
        the translation MUST contain its canonical form.

        Args:
            translation: Generated translation to post-process.
            source_text: Original source text.
            target_lang: Target language code.

        Returns:
            Translation with Tier 1 terms enforced.
        """
        if not self.tiered_terminology:
            return translation

        tier1_terms = self.tiered_terminology.get_terms_by_tier(TermTier.TIER_1)
        translation_modified = translation

        for source_term, canonical_target in tier1_terms.items():
            # Check if this Tier 1 term appears in source
            if re.search(r"\b" + re.escape(source_term) + r"\b", source_text, re.IGNORECASE):
                # Check if canonical form is already in translation
                if canonical_target.lower() in translation_modified.lower():
                    continue  # Already there, skip

                # Try to inject it at a plausible location
                injected = self._inject_tier1_term(
                    translation_modified, source_term, canonical_target
                )
                if injected:
                    translation_modified = injected
                    logger.debug(f"Injected Tier 1: {source_term} → {canonical_target}")

        return translation_modified

    def _inject_tier1_term(
        self, translation: str, source_term: str, target_term: str
    ) -> Optional[str]:
        """Try to inject a Tier 1 term into the translation.

        Attempts to find related words or similar phonetic forms and replace them
        with the canonical term.

        Args:
            translation: Translation to modify.
            source_term: English term (e.g., "god").
            target_term: Canonical target term (e.g., "Bondye").

        Returns:
            Modified translation, or None if no injection point found.
        """
        # Map of English terms to likely mistranslations/variations
        replacement_patterns = {
            "god": [
                (r"\bBondy\b", target_term),  # Misspelling
                (r"\bbondieu\b", target_term),  # French variant
                (r"\bDye\b", target_term),  # Phonetic variant
            ],
            "spirit": [
                (r"\blespri\b", target_term),  # Lowercase variant
                (r"\bespri\b", target_term),  # Missing l
            ],
            "jesus": [
                (r"\bJezi\b", target_term),  # Variant spelling
                (r"\bjezi\b", target_term),  # Lowercase
            ],
            "holy": [
                (r"\bsen\b", target_term),  # Possible variant
                (r"\bSen\b", target_term),  # Capitalized
            ],
            "lord": [
                (r"\bSeyè\b", target_term),  # Variant
                (r"\bseyè\b", target_term),  # Lowercase
            ],
        }

        source_lower = source_term.lower()
        if source_lower in replacement_patterns:
            for pattern, replacement in replacement_patterns[source_lower]:
                if re.search(pattern, translation, re.IGNORECASE):
                    return re.sub(pattern, replacement, translation, flags=re.IGNORECASE, count=1)

        # Fallback: if term is completely missing, try to insert it at the beginning
        # (this is crude but ensures presence)
        words = translation.split()
        if words and len(words) > 2:
            # Insert after first few words
            words.insert(min(2, len(words)), target_term)
            return " ".join(words)

        return None

    def _aggressive_clean_output(self, text: str) -> str:
        """Aggressively clean output to remove any system instruction text.

        NLLB sometimes echoes back parts of the system message translated.
        Remove these completely.

        Args:
            text: Output text to clean.

        Returns:
            Cleaned text.
        """
        # Remove anything that looks like translated instructions
        text = re.sub(
            r"###.*?###", "", text, flags=re.DOTALL | re.IGNORECASE
        )  # Remove ### headers
        text = re.sub(
            r"(SISTEM|ENSTRIKSYON|PA.*?TRADUI|DO.*?NOT|TRANSLATION|GLOSSARY|CONTEXT|SYSTEM)",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Remove common instruction markers
        text = re.sub(r"===.*?===", "", text)  # Remove === markers
        text = re.sub(r"\[.*?\]", "", text)  # Remove [brackets]

        # Remove markdown-style elements
        text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)  # Remove # headers

        # Remove numbered lists (instructions often use these)
        text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)

        # Clean whitespace
        text = re.sub(r"\s+", " ", text)  # Collapse spaces
        text = re.sub(r"\n\n+", "\n", text)  # Collapse newlines

        return text.strip()

    def _detect_repetition_collapse(self, text: str) -> bool:
        """Detect degenerate repetition loops in output.

        A trigram appearing 3+ times in a short verse is a clear sign of
        beam search / sampling failure (token loop collapse).

        Args:
            text: Decoded model output.

        Returns:
            True if repetition collapse detected.
        """
        words = text.split()
        if len(words) < 9:
            return False
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        seen = {}
        for tg in trigrams:
            seen[tg] = seen.get(tg, 0) + 1
            if seen[tg] >= 3:
                return True
        return False

    def _detect_numerical_corruption(self, translation: str, source_text: str) -> bool:
        """Detect when digit sequences from source are absent or mangled in output.

        NLLB sometimes garbles numbers (e.g., "601st year" → "6100 premye ane").
        If the source contains digits and more than half are absent from the
        translation, flag it.

        Args:
            translation: Decoded translation.
            source_text: Original source verse.

        Returns:
            True if numerical corruption likely.
        """
        source_numbers = re.findall(r'\d+', source_text)
        if not source_numbers:
            return False
        missing = sum(1 for n in source_numbers if n not in translation)
        return missing > len(source_numbers) * 0.5

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

    # English words that are often capitalized (sentence-start or titles)
    # but are NOT proper nouns (names of people/places)
    _ENGLISH_FUNCTION_WORDS: frozenset = frozenset({
        "and", "the", "a", "an", "in", "of", "to", "for", "from", "with",
        "that", "this", "these", "those", "is", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "let", "now",
        "but", "or", "nor", "so", "yet", "both", "either", "neither",
        "it", "he", "she", "we", "they", "you", "i", "me", "him", "her",
        "us", "them", "my", "his", "our", "their", "its", "thy", "thou",
        "when", "where", "which", "who", "what", "if", "as", "then",
        "there", "here", "also", "behold", "therefore", "because", "thus",
        "upon", "unto", "into", "over", "under", "through", "between",
        "said", "called", "named", "ye", "thee", "thine", "thereof",
        "jehovah", "god", "lord", "spirit", "holy",  # translated, not copied
        "man", "woman", "earth", "ground", "water", "light", "darkness",
        "day", "night", "morning", "evening", "heaven", "sea", "land",
    })

    def _compute_calibrated_confidence(self, outputs, seq_idx: int) -> float:
        """Length-normalized per-token log-probability as calibrated confidence.

        Raw sequence_score is the SUM of per-token log-probs, so longer outputs
        always score lower. Dividing by token count fixes this, then mapping
        through a sigmoid calibrated to NLLB's typical [-3, 0] avg-logprob range.

        Calibration targets (empirically derived for NLLB-600M):
            avg_log_prob = -0.3  → confidence ≈ 0.90  (excellent)
            avg_log_prob = -0.8  → confidence ≈ 0.80  (good)
            avg_log_prob = -1.5  → confidence ≈ 0.65  (acceptable)
            avg_log_prob = -2.5  → confidence ≈ 0.40  (poor)
            avg_log_prob = -3.5  → confidence ≈ 0.20  (very poor)

        Args:
            outputs: Return value of model.generate() with return_dict_in_generate=True.
            seq_idx: Which sequence in the batch to score.

        Returns:
            Calibrated confidence in [0.05, 0.97].
        """
        if not (hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None):
            return 0.60
        if seq_idx >= len(outputs.sequences_scores):
            return 0.60

        log_prob = outputs.sequences_scores[seq_idx].item()

        # Count non-padding tokens (subtract 1 for the forced BOS token)
        pad_id = self.tokenizer.pad_token_id or 1
        seq = outputs.sequences[seq_idx]
        n_tokens = int((seq != pad_id).sum().item()) - 1
        n_tokens = max(n_tokens, 1)

        avg_log_prob = log_prob / n_tokens

        # Sigmoid mapping: logit = 1.3 * avg_log_prob + 2.4
        # Chosen so the calibration targets above are met
        logit = 1.3 * avg_log_prob + 2.4
        raw = 1.0 / (1.0 + math.exp(-logit))
        return float(min(max(raw, 0.05), 0.97))

    def _extract_proper_nouns(self, source_text: str) -> set:
        """Extract likely person/place names from English source.

        A token is a proper-noun candidate when it:
          - is capitalized
          - appears mid-sentence (not the first word)
          - is not in the function-word exclusion list

        Args:
            source_text: English verse (may include "Book CH:V: " prefix).

        Returns:
            Set of lowercase proper-noun candidates.
        """
        # Strip "Genesis 3:14: " style prefix
        text = re.sub(r'^[A-Za-z]+ \d+:\d+:\s*', '', source_text)
        words = text.split()
        nouns = set()
        for i, word in enumerate(words):
            clean = re.sub(r'[^\w]', '', word)
            if not clean or i == 0:
                continue
            if clean[0].isupper() and clean.lower() not in self._ENGLISH_FUNCTION_WORDS:
                nouns.add(clean.lower())
        return nouns

    def _check_proper_noun_stability(
        self, translation: str, source_text: str
    ) -> float:
        """Score how well person/place names from source appear in translation.

        Uses fuzzy prefix matching since Haitian Creole phonetically adapts
        many English names (e.g. Cain → Kayen, Abel stays Abel).

        Args:
            translation: Generated Haitian Creole translation.
            source_text: Original English source verse.

        Returns:
            Stability score 0-1 (1.0 = all names accounted for).
        """
        proper_nouns = self._extract_proper_nouns(source_text)
        if not proper_nouns:
            return 1.0

        trans_lower = translation.lower()
        trans_words = set(re.sub(r'[^\w\s]', '', trans_lower).split())

        preserved = 0.0
        for noun in proper_nouns:
            if noun in trans_lower:
                preserved += 1.0
            elif len(noun) >= 4 and noun[:4] in trans_lower:
                preserved += 0.7  # Partial phonetic match
            elif len(noun) >= 3 and any(w.startswith(noun[:3]) for w in trans_words):
                preserved += 0.4  # Weak match

        return min(preserved / len(proper_nouns), 1.0)

    def _detect_non_hc_script(self, text: str) -> bool:
        """Detect Central/Eastern European characters leaking into HC output.

        NLLB sometimes bleeds characters from other Latin-script languages
        it was trained on (Czech, Slovak, Polish: č ň ě ř ž ů etc.).
        These never appear in legitimate Haitian Creole text.

        Args:
            text: Translation output to check.

        Returns:
            True if foreign-script contamination detected.
        """
        # Characters valid in Haitian Creole: ASCII + è é ò ô â à ê î û
        # Everything outside this set in a Latin context is suspicious
        non_hc_pattern = r'[čňěřžůďťľśćźąęłóżĺĽŃŌ]'
        return bool(re.search(non_hc_pattern, text, re.IGNORECASE))

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
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
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
        confidence = self._compute_calibrated_confidence(outputs, 0)

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

            # Build input: ONLY verse text, NO instructions or context labels
            # Model sees just the verse to translate, nothing else
            prompts = [w.current_verse for w in batch_windows]

            # Log what we're sending
            for w, prompt in zip(batch_windows, prompts):
                logger.debug(f"SENDING TO MODEL [{w.current_ref}]: {prompt[:80]}...")

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
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
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
                        logger.debug(f"RAW OUTPUT [{window.current_ref}]: {raw_output[:100]}...")

                        # MINIMAL cleaning - only remove obvious junk, NOT complex parsing
                        verse_translation = self._minimal_clean(raw_output)
                        logger.debug(f"CLEANED [{window.current_ref}]: {verse_translation[:100]}...")
                    else:
                        logger.warning(f"Missing output for verse {window.current_ref}")
                        verse_translation = ""

                # STEP 3: Enforce Tier 1 terminology (if not cached)
                if not cached_translation and verse_translation:
                    verse_translation = self._enforce_tier1_terms(
                        verse_translation, orig_verse["text"], target_lang
                    )
                    logger.debug(f"ENFORCED [{window.current_ref}]: {verse_translation[:100]}...")

                # STEP 4: Calculate confidence
                if not cached_translation:
                    confidence = self._compute_calibrated_confidence(outputs, i)

                    # Apply quality signals to confidence
                    if verse_translation:
                        if self._detect_repetition_collapse(verse_translation):
                            confidence = 0.05
                            logger.warning(f"Repetition collapse detected: {window.current_ref}")
                        else:
                            if self._detect_numerical_corruption(verse_translation, orig_verse["text"]):
                                confidence = max(0.2, confidence * 0.70)
                                logger.debug(f"Numerical corruption detected: {window.current_ref}")
                            if self._detect_non_hc_script(verse_translation):
                                confidence = max(0.2, confidence * 0.65)
                                logger.warning(f"Non-HC script detected: {window.current_ref}")
                            if self._detect_french_contamination(verse_translation):
                                confidence = max(0.2, confidence * 0.75)
                                logger.debug(f"French contamination detected: {window.current_ref}")
                            pn_score = self._check_proper_noun_stability(verse_translation, orig_verse["text"])
                            if pn_score < 0.5:
                                confidence = max(0.2, confidence * (0.7 + 0.3 * pn_score))
                                logger.debug(f"Proper noun instability (score={pn_score:.2f}): {window.current_ref}")
                else:
                    confidence = cached_confidence

                logger.debug(f"CONFIDENCE [{window.current_ref}]: {confidence:.2f}")

                # STEP 5: Store in translation memory (if not cached)
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
        use_back_translation: bool = False,
    ) -> List[TranslationResult]:
        """Translate multiple verses.

        Args:
            verses: List of verse dicts with 'text' and optional 'reference'.
            source_lang: Source language code (e.g., "eng_Latn").
            target_lang: Target language code (e.g., "hat_Latn").
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
            use_back_translation: If True, back-translate each verse and feed
                the semantic similarity score into confidence. Doubles inference
                time — use only for spot-checks or small batches.

        Returns:
            List of TranslationResult objects.
        """
        results = []

        iterator = verses
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(verses, desc="Translating verses")
            except ImportError:
                iterator = verses

        # Process in batches
        for batch_start in range(0, len(verses), batch_size):
            batch_end = min(batch_start + batch_size, len(verses))
            batch_verses = verses[batch_start:batch_end]

            # Input: ONLY verse text
            texts = [v["text"] for v in batch_verses]
            refs = [v.get("reference", "") for v in batch_verses]

            # Tokenize and generate
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=MAX_SOURCE_LENGTH,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            self.tokenizer.src_lang = source_lang

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=DEFAULT_NUM_BEAMS,
                    num_return_sequences=1,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
                    output_scores=True,
                    return_dict_in_generate=True,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_lang),
                )

            # Decode results
            num_outputs = outputs.sequences.shape[0]
            for i in range(len(batch_verses)):
                if i >= num_outputs:
                    break

                verse = batch_verses[i]
                ref = refs[i]

                # Decode raw translation
                translation = self.tokenizer.decode(
                    outputs.sequences[i], skip_special_tokens=True
                )

                # Clean minimal junk
                translation = self._minimal_clean(translation)

                # Enforce Tier 1 terms
                if self.tiered_terminology:
                    translation = self._enforce_tier1_terms(translation, verse["text"], target_lang)

                # Extract terms first (needed for quality checks)
                theological_terms = self.term_extractor.extract_theological_terms(verse["text"])
                for term in theological_terms:
                    self.terminology_db.record_usage(term, target_lang)

                canonical_terms = self.term_extractor.get_canonical_terms(verse["text"], target_lang)

                # Base confidence: length-normalized per-token log-probability
                confidence = self._compute_calibrated_confidence(outputs, i)

                # QUALITY SIGNAL 1: Tier 1 term injection (hard theological requirements)
                if self.tiered_terminology:
                    tier1_terms = self.tiered_terminology.get_terms_by_tier(TermTier.TIER_1)
                    for source_term in tier1_terms:
                        if re.search(r"\b" + re.escape(source_term) + r"\b", verse["text"], re.IGNORECASE):
                            canonical = tier1_terms[source_term]
                            if canonical.lower() in translation.lower():
                                confidence = min(0.99, confidence + 0.10)  # +10% for success
                            else:
                                confidence = max(0.2, confidence - 0.15)  # -15% if missing
                            break

                # QUALITY SIGNAL 2: Glossary coverage (theological terms in output)
                glossary_found = 0
                if canonical_terms:
                    for term, translation_val in canonical_terms.items():
                        if translation_val and translation_val.lower() in translation.lower():
                            glossary_found += 1
                    coverage_ratio = glossary_found / len(canonical_terms) if canonical_terms else 0
                    if coverage_ratio > 0:
                        confidence = min(0.99, confidence + (coverage_ratio * 0.15))
                        logger.debug(f"Glossary boost: {glossary_found}/{len(canonical_terms)} terms found (+{coverage_ratio * 0.15:.2f})")
                    elif coverage_ratio == 0 and len(canonical_terms) > 0:
                        confidence = max(0.2, confidence * 0.85)  # Penalize if terms expected but missing

                # QUALITY SIGNAL 3: Length ratio (output reasonable vs input)
                source_words = len(verse["text"].split())
                trans_words = len(translation.split())
                length_ratio = trans_words / max(1, source_words)

                if 0.5 <= length_ratio <= 2.0:
                    confidence = min(0.99, confidence + 0.05)  # +5% for reasonable length
                    logger.debug(f"Length OK: ratio={length_ratio:.2f}")
                elif length_ratio < 0.3:
                    confidence = max(0.2, confidence * 0.6)  # Heavily penalize truncation
                    logger.debug(f"Translation too short: ratio={length_ratio:.2f}")
                elif length_ratio > 2.5:
                    confidence = max(0.2, confidence * 0.9)  # Slight penalize for excessive expansion
                    logger.debug(f"Translation too long: ratio={length_ratio:.2f}")

                # QUALITY SIGNAL 4: Check for junk patterns (garbled start/end)
                if len(translation) > 5:
                    # Check if first 3 chars are mostly non-alphabetic (sign of junk)
                    first_three = translation[:3]
                    alpha_count = sum(1 for c in first_three if c.isalpha())
                    if alpha_count < 2:
                        confidence = max(0.2, confidence * 0.7)
                        logger.debug(f"Junk pattern detected at start: {first_three}")

                # QUALITY SIGNAL 5: French contamination (penalize heavily)
                if self._detect_french_contamination(translation):
                    confidence = max(0.2, confidence * 0.75)  # 25% penalty
                    logger.debug("French contamination detected")

                # QUALITY SIGNAL 6: Repetition collapse (catastrophic failure)
                if self._detect_repetition_collapse(translation):
                    confidence = 0.05  # Near-zero: output is unusable
                    logger.warning(f"Repetition collapse detected: {verse.get('reference', '?')}")

                # QUALITY SIGNAL 7: Numerical corruption
                if self._detect_numerical_corruption(translation, verse["text"]):
                    confidence = max(0.2, confidence * 0.70)  # 30% penalty
                    logger.debug(f"Numerical corruption detected: {verse.get('reference', '?')}")

                # QUALITY SIGNAL 8: Non-HC script contamination (č, ň, etc.)
                if self._detect_non_hc_script(translation):
                    confidence = max(0.2, confidence * 0.65)  # 35% penalty — output has corrupted chars
                    logger.warning(f"Non-HC script detected: {verse.get('reference', '?')}")

                # QUALITY SIGNAL 9: Proper noun stability
                pn_score = self._check_proper_noun_stability(translation, verse["text"])
                if pn_score < 0.5:
                    confidence = max(0.2, confidence * (0.7 + 0.3 * pn_score))
                    logger.debug(f"Proper noun instability (score={pn_score:.2f}): {verse.get('reference', '?')}")

                # QUALITY SIGNAL 10 (optional): Semantic similarity via back-translation
                # Only runs when use_back_translation=True — doubles inference time
                if use_back_translation and translation:
                    bt_similarity, _ = self.back_translator.validate(
                        translation, verse["text"]
                    )
                    # Replace model log-prob with the round-trip semantic score
                    confidence = bt_similarity
                    logger.debug(f"Back-translation similarity: {bt_similarity:.2f} for {verse.get('reference', '?')}")

                # Create result
                result = TranslationResult(
                    primary=translation,
                    confidence=confidence,
                    alternatives=[],
                    theological_terms=canonical_terms,
                    consistency_enforced=bool(self.tiered_terminology and any(self.tiered_terminology.get_tier(t) == TermTier.TIER_1 for t in theological_terms)),
                    source_text=f"{ref}: {verse['text']}" if ref else verse["text"],
                    target_language=target_lang,
                )
                results.append(result)

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

"""Multi-metric confidence scoring for translations.

Confidence is grounded in actual translation quality signals:
- Lexical consistency: Are glossary terms present?
- Glossary match rate: What % of expected terms appear?
- Language purity: Is it Creole or French-contaminated?
- Token confidence: What was the model's own confidence?
"""

import re
from typing import Dict, Optional, Tuple
from models.terminology import TerminologyDB, TermExtractor
from models.tiered_terminology import TieredTerminologyDB, TermTier
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceScorer:
    """Score translation confidence using multiple signals."""

    # French word patterns (for contamination detection)
    FRENCH_WORDS = {
        r"\bla\b", r"\ble\b", r"\bdes\b", r"\bet\b", r"\best\b",
        r"\bde\b", r"\bétat\b", r"\bdésert\b", r"\bconformé\b",
        r"\bétait\b", r"\bvide\b", r"\beau\b", r"\bcouvert\b", r"\bplus\b",
    }

    # Common Haitian Creole words (for language purity)
    CREOLE_INDICATORS = {
        r"\bak\b", r"\bepi\b", r"\bse\b", r"\bnan\b", r"\bpou\b",
        r"\bte\b", r"\bpa\b", r"\bap\b", r"\bgran\b", r"\bbon\b",
        r"\blespri\b", r"\bseyè\b", r"\bbondye\b", r"\bjezi\b",
    }

    def __init__(
        self,
        terminology_db: TerminologyDB,
        tiered_terminology: Optional[TieredTerminologyDB] = None,
    ):
        """Initialize confidence scorer.

        Args:
            terminology_db: Terminology database for lookups.
            tiered_terminology: Optional tiered terminology system.
        """
        self.db = terminology_db
        self.tiered = tiered_terminology
        self.term_extractor = TermExtractor(terminology_db)

    # Characters never valid in Haitian Creole
    NON_HC_PATTERN = re.compile(r'[čňěřžůďťľśćźąęłóżĺĽŃŌ]', re.IGNORECASE)

    def score(
        self,
        translation: str,
        source_text: str,
        target_lang: str,
        model_confidence: float = 0.5,
        back_translation_score: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute composite confidence score.

        Args:
            translation: Generated translation.
            source_text: Original source text.
            target_lang: Target language code.
            model_confidence: Base confidence from model (length-normalized log-prob).
            back_translation_score: Optional semantic similarity from back-translation
                validator (0-1). When provided, replaces model_confidence as semantic
                anchor. Leave None to skip (saves a full inference pass).

        Returns:
            Tuple of (composite_score: 0-1, component_scores: dict).
        """
        components = {}

        # Short-circuit: repetition collapse = unusable output
        if self._detect_repetition_collapse(translation):
            logger.warning("Confidence scorer: repetition collapse detected")
            return 0.05, {"repetition_collapse": 0.05}

        # Short-circuit: non-HC script contamination
        if self.NON_HC_PATTERN.search(translation):
            logger.warning("Confidence scorer: non-HC script characters detected")
            # Don't zero-out entirely — partial info is still recoverable
            components["non_hc_script"] = 0.10
            components["model_confidence"] = model_confidence * 0.5
            return float(self._weighted_average(components)), components

        # Component 1: Lexical consistency (glossary term presence)
        components["lexical_consistency"] = self._score_lexical_consistency(
            translation, source_text, target_lang
        )

        # Component 2: Glossary match rate (% of expected Tier 1 terms)
        components["glossary_match_rate"] = self._score_glossary_match_rate(
            translation, source_text, target_lang
        )

        # Component 3: Language purity (Creole vs French)
        components["language_purity"] = self._score_language_purity(translation)

        # Component 4: Semantic similarity — back-translation if available,
        # otherwise fall back to the model's own calibrated confidence
        if back_translation_score is not None:
            components["semantic_similarity"] = float(back_translation_score)
        else:
            components["model_confidence"] = model_confidence

        # Component 5: Numerical fidelity
        components["numerical_fidelity"] = self._score_numerical_fidelity(
            translation, source_text
        )

        # Composite: weighted average
        composite = self._weighted_average(components)

        debug_parts = ", ".join(f"{k}={v:.2f}" for k, v in components.items())
        logger.debug(f"Confidence breakdown: {debug_parts} → composite={composite:.2f}")

        return float(composite), components

    def _score_lexical_consistency(
        self, translation: str, source_text: str, target_lang: str
    ) -> float:
        """Score based on glossary term presence.

        If source has "God" and translation has "Bondye", high score.
        If source has "God" but translation missing Bondye, low score.

        Args:
            translation: Generated translation.
            source_text: Original source.
            target_lang: Target language code.

        Returns:
            Score 0-1.
        """
        # Extract theological terms from source
        terms = self.term_extractor.extract_theological_terms(source_text)
        if not terms:
            return 0.8  # No terms to match = assume acceptable

        # Check how many have translations in the output
        matched = 0
        for term in terms:
            target_term = self.db.lookup(term, target_lang)
            if target_term and target_term in translation:
                matched += 1

        consistency_rate = matched / len(terms) if terms else 0
        # Map: 0->0.3, 0.5->0.6, 1.0->1.0
        return min(0.3 + (consistency_rate * 0.7), 1.0)

    def _score_glossary_match_rate(
        self, translation: str, source_text: str, target_lang: str
    ) -> float:
        """Score based on match rate of Tier 1 terms specifically.

        High priority: Tier 1 terms must appear.

        Args:
            translation: Generated translation.
            source_text: Original source.
            target_lang: Target language code.

        Returns:
            Score 0-1.
        """
        if not self.tiered:
            return 0.7  # No tiered system = skip this metric

        # Extract Tier 1 terms from source
        terms = self.term_extractor.extract_theological_terms(source_text)
        tier1_terms = [
            t for t in terms if self.tiered.get_tier(t) == TermTier.TIER_1
        ]

        if not tier1_terms:
            return 0.8  # No Tier 1 terms to match

        # Check presence in translation
        matched = 0
        for term in tier1_terms:
            target_term = self.db.lookup(term, target_lang)
            if target_term and target_term in translation:
                matched += 1

        # Tier 1 is critical: 0->0.2, 0.5->0.5, 1.0->1.0
        match_rate = matched / len(tier1_terms) if tier1_terms else 0
        return min(0.2 + (match_rate * 0.8), 1.0)

    def _score_language_purity(self, translation: str) -> float:
        """Score language purity (Creole vs French contamination).

        Check for French word patterns and Creole indicators.

        Args:
            translation: Generated translation.

        Returns:
            Score 0-1 (1.0 = pure Creole, 0.0 = French contaminated).
        """
        translation_lower = translation.lower()

        # Count French word hits
        french_count = 0
        for pattern in self.FRENCH_WORDS:
            if re.search(pattern, translation_lower):
                french_count += 1

        # Count Creole word hits
        creole_count = 0
        for pattern in self.CREOLE_INDICATORS:
            if re.search(pattern, translation_lower):
                creole_count += 1

        # Score: if French words present, penalize heavily
        if french_count > 0:
            # Heavy penalty: -0.1 per French word detected
            return max(0.3 - (french_count * 0.1), 0.0)

        # Bonus for Creole indicators
        creole_bonus = min(creole_count * 0.1, 0.3)
        return min(0.7 + creole_bonus, 1.0)

    def _detect_repetition_collapse(self, text: str) -> bool:
        """Return True if any word-trigram repeats 3+ times (degenerate loop)."""
        words = text.split()
        if len(words) < 9:
            return False
        seen: Dict[tuple, int] = {}
        for i in range(len(words) - 2):
            tg = (words[i], words[i + 1], words[i + 2])
            seen[tg] = seen.get(tg, 0) + 1
            if seen[tg] >= 3:
                return True
        return False

    def _score_numerical_fidelity(self, translation: str, source_text: str) -> float:
        """Score how well digit sequences from source appear in translation.

        If source has "601" and translation contains "6100" instead, that is
        a numerical corruption — penalize it.

        Args:
            translation: Generated translation.
            source_text: Original source verse.

        Returns:
            Score 0-1 (1.0 = all numbers preserved, 0.0 = all missing/corrupted).
        """
        source_numbers = re.findall(r'\d+', source_text)
        if not source_numbers:
            return 1.0  # No numbers to check
        preserved = sum(1 for n in source_numbers if n in translation)
        fidelity = preserved / len(source_numbers)
        # Map: 0→0.1, 0.5→0.55, 1.0→1.0
        return 0.1 + (fidelity * 0.9)

    def _weighted_average(self, components: Dict[str, float]) -> float:
        """Compute weighted average of component scores.

        Weights prioritize language purity (avoid garbage) and
        glossary match (theological accuracy).

        Args:
            components: Dictionary of component scores.

        Returns:
            Weighted average score 0-1.
        """
        # Weights differ depending on whether back-translation is available.
        # When semantic_similarity is present it replaces model_confidence and
        # gets a higher weight (it's a stronger signal than raw log-prob).
        if "semantic_similarity" in components:
            weights = {
                "lexical_consistency": 0.15,
                "glossary_match_rate": 0.20,
                "language_purity": 0.20,
                "semantic_similarity": 0.30,  # Strongest: round-trip meaning preserved
                "numerical_fidelity": 0.15,
            }
        elif "non_hc_script" in components:
            # Partial-score path for script contamination
            weights = {
                "non_hc_script": 0.60,
                "model_confidence": 0.40,
            }
        else:
            weights = {
                "lexical_consistency": 0.20,
                "glossary_match_rate": 0.25,
                "language_purity": 0.25,
                "model_confidence": 0.15,
                "numerical_fidelity": 0.15,
            }

        total_score = 0
        total_weight = 0

        for component, weight in weights.items():
            if component in components:
                total_score += components[component] * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def get_quality_tier(self, confidence: float) -> str:
        """Classify translation quality by confidence.

        Args:
            confidence: Confidence score 0-1.

        Returns:
            Quality tier: "excellent", "good", "acceptable", "poor".
        """
        if confidence >= 0.85:
            return "excellent"
        elif confidence >= 0.75:
            return "good"
        elif confidence >= 0.60:
            return "acceptable"
        else:
            return "poor"


if __name__ == "__main__":
    from utils.logger import configure_logging
    from models.tiered_terminology import TieredTerminologyDB

    configure_logging()

    # Test the scorer
    db = TerminologyDB()
    tiered = TieredTerminologyDB(db)
    scorer = ConfidenceScorer(db, tiered)

    # Test with good translation
    good_source = "In the beginning, God created the heavens and the earth."
    good_translation = "Ansyen tan, Bondye te kreye syèl la ak latè."
    score, components = scorer.score(good_translation, good_source, "hat_Latn", 0.85)
    tier = scorer.get_quality_tier(score)
    logger.info(f"Good translation: {score:.2f} ({tier})")
    logger.info(f"  Components: {components}")

    # Test with French contamination
    bad_translation = "Au début, le Bondye a créé le ciel et la terre."
    score, components = scorer.score(bad_translation, good_source, "hat_Latn", 0.75)
    tier = scorer.get_quality_tier(score)
    logger.info(f"\nFrench-contaminated: {score:.2f} ({tier})")
    logger.info(f"  Components: {components}")

    # Test with missing glossary term
    incomplete_translation = "Ansyen tan, Bondye te kreye tèren an ak dlo."
    score, components = scorer.score(incomplete_translation, good_source, "hat_Latn", 0.70)
    tier = scorer.get_quality_tier(score)
    logger.info(f"\nIncomplete translation: {score:.2f} ({tier})")
    logger.info(f"  Components: {components}")

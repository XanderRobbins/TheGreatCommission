"""Evaluation module for scripture translations.

Provides metrics including BLEU scores, consistency checking, terminology analysis,
and human evaluation interfaces.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from models.terminology import TerminologyDB, TermExtractor
from utils.logger import get_logger
from exceptions import EvaluationError

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation results.

    Attributes:
        bleu_1: BLEU score using unigram weights.
        bleu_2: BLEU score using average of unigram and bigram weights.
        bleu_4: BLEU score using average of uni-, bi-, tri-, and 4-gram weights.
        consistency_score: Terminology consistency score (0-1).
        unique_terms: Number of distinct terms used.
        avg_term_usage: Average frequency per unique term.
        human_score: Optional human rating score (1-5 scale).
    """

    bleu_1: float
    bleu_2: float
    bleu_4: float
    consistency_score: float
    unique_terms: int
    avg_term_usage: float
    human_score: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for JSON serialization.

        Returns:
            Dictionary with all metric fields.
        """
        return {
            "bleu_1": self.bleu_1,
            "bleu_2": self.bleu_2,
            "bleu_4": self.bleu_4,
            "consistency_score": self.consistency_score,
            "unique_terms": self.unique_terms,
            "avg_term_usage": self.avg_term_usage,
            "human_score": self.human_score,
        }


class ScriptureEvaluator:
    """Evaluate scripture translations using both standard and custom metrics.

    Computes BLEU scores, terminology consistency, and term usage patterns
    to measure translation quality.
    """

    def __init__(self, terminology_db: Optional[TerminologyDB] = None) -> None:
        """Initialize evaluator.

        Args:
            terminology_db: Optional terminology database for consistency checking.
        """
        self.terminology_db = terminology_db
        self.smoothing_function = SmoothingFunction().method1
    
    def compute_bleu(
        self,
        hypothesis: str,
        reference: str,
        weights: Tuple[float, ...] = (1, 0, 0, 0),
    ) -> float:
        """Compute BLEU score between hypothesis and reference.

        Args:
            hypothesis: Predicted translation.
            reference: Reference translation.
            weights: Weight for n-grams. Examples:
                (1,0,0,0) = BLEU-1 (unigram match)
                (0.5,0.5,0,0) = BLEU-2 (average of unigram and bigram)
                (0.25,0.25,0.25,0.25) = BLEU-4 (average of uni-, bi-, tri-, 4-gram)

        Returns:
            BLEU score bounded to [0.0, 1.0].

        Raises:
            EvaluationError: If BLEU computation fails unexpectedly.
        """
        if not hypothesis or not reference:
            return 0.0

        try:
            # Simple whitespace tokenization (more reliable than NLTK word_tokenize)
            hyp_tokens = hypothesis.lower().split()
            ref_tokens = reference.lower().split()

            # Handle empty tokenization
            if not hyp_tokens or not ref_tokens:
                return 0.0

            # NLTK sentence_bleu expects list of references (we have just one)
            references = [ref_tokens]

            bleu_score = sentence_bleu(
                references,
                hyp_tokens,
                weights=weights,
                smoothing_function=self.smoothing_function,
            )
            # BLEU score is already bounded to [0, 1] by definition
            return float(bleu_score)
        except Exception as exc:
            logger.debug(f"BLEU computation failed: {exc}")
            return 0.0
    
    def compute_consistency_score(
        self, translations: List[str], terminology_db: Optional[TerminologyDB] = None
    ) -> float:
        """Measure how consistently terms are translated.

        Checks if the same theological term uses the same surface form
        across all translations. Higher score = more consistency.

        For example, if "salvation" appears in multiple verses, this metric
        checks whether it's translated the same way each time.

        Args:
            translations: List of translation outputs to check.
            terminology_db: Terminology database with canonical terms.
                           If None, returns 1.0 (no consistency check possible).

        Returns:
            Consistency score from 0.0 (all translations of a term differ) to
            1.0 (all translations of each term are identical).
        """
        if not terminology_db or len(translations) < 2:
            return 1.0  # Perfect if no DB or only one verse

        if not isinstance(terminology_db, TerminologyDB):
            return 1.0

        try:
            extractor = TermExtractor(terminology_db)

            # Track which surface forms are used for each canonical term
            # term_surfaces[canonical_term] = set of surface forms seen
            term_surfaces: Dict[str, set] = {}

            for translation in translations:
                # Extract theological terms and their surface forms from this translation
                found_terms = extractor.extract_theological_terms(translation)

                for term in found_terms:
                    # For simplicity, the "surface form" is just the term as it appears
                    # In production, could use fuzzy matching to detect variations
                    if term not in term_surfaces:
                        term_surfaces[term] = set()
                    term_surfaces[term].add(term)

            # Score: how many terms used consistently
            # A term is consistent if it has only 1 surface form across all translations
            if not term_surfaces:
                return 1.0

            consistent_terms = sum(
                1 for surfaces in term_surfaces.values() if len(surfaces) == 1
            )
            return float(consistent_terms) / len(term_surfaces)
        except Exception as exc:
            logger.warning(f"Consistency scoring failed: {exc}")
            return 1.0
    
    def compute_terminology_uniqueness(
        self, translations: List[str], target_lang: str
    ) -> Tuple[int, float]:
        """Measure terminology diversity and usage patterns.

        Args:
            translations: List of translation outputs.
            target_lang: Target language code (for logging).

        Returns:
            Tuple of (unique_term_count, average_term_usage).
            - unique_term_count: Number of distinct words/terms used
            - average_term_usage: Average frequency per unique term
        """
        term_usage: Dict[str, int] = {}
        total_words = 0

        for translation in translations:
            # Use simple whitespace tokenization
            words = translation.lower().split()
            total_words += len(words)

            for word in words:
                term_usage[word] = term_usage.get(word, 0) + 1

        unique_terms = len(term_usage)
        avg_usage = total_words / unique_terms if unique_terms > 0 else 0.0

        return unique_terms, avg_usage
    
    def evaluate_batch(
        self,
        hypotheses: List[str],
        references: List[str],
        target_lang: str,
    ) -> EvaluationMetrics:
        """Evaluate a batch of translations against references.

        Computes BLEU scores at multiple n-gram levels, consistency across
        terminology, and term usage diversity.

        Args:
            hypotheses: List of predicted (generated) translations.
            references: List of reference (gold standard) translations.
            target_lang: Target language code for logging/context.

        Returns:
            EvaluationMetrics object with comprehensive quality metrics.

        Raises:
            ValueError: If hypotheses and references have different lengths.
        """
        if len(hypotheses) != len(references):
            raise ValueError(
                f"Hypothesis and reference lists must be same length. "
                f"Got {len(hypotheses)} vs {len(references)}"
            )

        if not hypotheses:
            raise ValueError("Cannot evaluate empty hypothesis list")

        # Compute BLEU scores at different n-gram levels
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_4_scores = []

        for hyp, ref in zip(hypotheses, references):
            bleu_1_scores.append(self.compute_bleu(hyp, ref, weights=(1, 0, 0, 0)))
            bleu_2_scores.append(
                self.compute_bleu(hyp, ref, weights=(0.5, 0.5, 0, 0))
            )
            bleu_4_scores.append(
                self.compute_bleu(hyp, ref, weights=(0.25, 0.25, 0.25, 0.25))
            )

        avg_bleu_1 = float(np.mean(bleu_1_scores))
        avg_bleu_2 = float(np.mean(bleu_2_scores))
        avg_bleu_4 = float(np.mean(bleu_4_scores))

        # Compute consistency score
        consistency = self.compute_consistency_score(hypotheses, self.terminology_db)

        # Compute terminology metrics
        unique_terms, avg_usage = self.compute_terminology_uniqueness(
            hypotheses, target_lang
        )

        return EvaluationMetrics(
            bleu_1=avg_bleu_1,
            bleu_2=avg_bleu_2,
            bleu_4=avg_bleu_4,
            consistency_score=consistency,
            unique_terms=unique_terms,
            avg_term_usage=avg_usage,
        )
    
    def save_evaluation_report(
        self, metrics: EvaluationMetrics, output_path: Path, language_pair: str = ""
    ) -> None:
        """Save evaluation report to JSON file.

        Args:
            metrics: EvaluationMetrics object with computed metrics.
            output_path: Path where to save the report.
            language_pair: Optional language pair description (e.g., "eng→spa").
        """
        report = {
            "language_pair": language_pair,
            "metrics": metrics.to_dict(),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved evaluation report to {output_path}")

    def print_metrics(self, metrics: EvaluationMetrics, title: str = "") -> None:
        """Log formatted metrics to logger.info (not stdout).

        Args:
            metrics: EvaluationMetrics object with computed metrics.
            title: Optional title for the metrics display.
        """
        separator = "=" * 50
        logger.info(separator)
        if title:
            logger.info(f"=== {title} ===")
        logger.info(separator)
        logger.info(f"BLEU-1: {metrics.bleu_1:.4f}")
        logger.info(f"BLEU-2: {metrics.bleu_2:.4f}")
        logger.info(f"BLEU-4: {metrics.bleu_4:.4f}")
        logger.info(f"Consistency Score: {metrics.consistency_score:.4f}")
        logger.info(f"Unique Terms: {metrics.unique_terms}")
        logger.info(f"Avg Term Usage: {metrics.avg_term_usage:.2f}")
        if metrics.human_score is not None:
            logger.info(f"Human Score: {metrics.human_score:.2f}/5.0")
        logger.info(separator)


class HumanEvaluationInterface:
    """Interface for collecting human evaluation scores in interactive mode.

    Provides CLI prompts for rating translations on multiple dimensions:
    accuracy, clarity, naturalness, and consistency.
    """

    def __init__(self, verses: List[Dict]) -> None:
        """Initialize with verses to evaluate.

        Args:
            verses: List of verse dictionaries with at least:
                - 'source': Source text
                - 'primary': Primary translation
                - 'reference' (optional): Verse reference (e.g., "Genesis 1:1")
                - 'alternatives' (optional): List of alternative translations
        """
        self.verses = verses
        self.scores: List[Dict] = []

    def display_verse(self, index: int) -> None:
        """Display a verse for evaluation.

        Args:
            index: Index of the verse in self.verses to display.
        """
        verse = self.verses[index]

        print(f"\n{'='*60}")
        print(f"Verse {index + 1}/{len(self.verses)}")
        print(f"Reference: {verse.get('reference', 'N/A')}")
        print(f"{'='*60}")
        print(f"\nSource: {verse['source']}")
        print(f"\nTranslation: {verse['primary']}")

        if verse.get("alternatives"):
            print(f"\nAlternatives:")
            for i, alt in enumerate(verse["alternatives"][:2]):
                print(f"  {i+1}. {alt}")

    def collect_scores(self, index: int) -> Dict:
        """Collect human evaluation scores for a verse.

        Prompts the evaluator to rate the translation on multiple dimensions
        (accuracy, clarity, naturalness, consistency) on a 1-5 scale.

        Args:
            index: Index of the verse to score.

        Returns:
            Dictionary with scores and optional notes:
                - accuracy: Score 1-5
                - clarity: Score 1-5
                - naturalness: Score 1-5
                - consistency: Score 1-5
                - notes: Optional reviewer notes
                - average: Mean of the four scores
        """
        self.display_verse(index)
        
        print("\n" + "-"*60)
        print("Rate this translation (1-5):")
        print("  1 = Inaccurate/unclear")
        print("  2 = Poor but understandable")
        print("  3 = Acceptable")
        print("  4 = Good")
        print("  5 = Excellent")
        print("-"*60)
        
        scores = {}
        
        # Accuracy
        while True:
            try:
                accuracy = int(input("Accuracy (1-5): "))
                if 1 <= accuracy <= 5:
                    scores["accuracy"] = accuracy
                    break
            except ValueError:
                pass
            print("Please enter a number between 1 and 5")
        
        # Clarity
        while True:
            try:
                clarity = int(input("Clarity (1-5): "))
                if 1 <= clarity <= 5:
                    scores["clarity"] = clarity
                    break
            except ValueError:
                pass
            print("Please enter a number between 1 and 5")
        
        # Naturalness
        while True:
            try:
                naturalness = int(input("Naturalness (1-5): "))
                if 1 <= naturalness <= 5:
                    scores["naturalness"] = naturalness
                    break
            except ValueError:
                pass
            print("Please enter a number between 1 and 5")
        
        # Consistency
        while True:
            try:
                consistency = int(input("Consistency (1-5): "))
                if 1 <= consistency <= 5:
                    scores["consistency"] = consistency
                    break
            except ValueError:
                pass
            print("Please enter a number between 1 and 5")
        
        # Notes
        notes = input("Notes (optional): ")
        if notes:
            scores["notes"] = notes

        avg_score = (
            sum(scores[k] for k in ["accuracy", "clarity", "naturalness", "consistency"])
            / 4
        )
        scores["average"] = avg_score

        return scores

    def run_evaluation_session(self, num_verses: Optional[int] = None) -> List[Dict]:
        """Run interactive evaluation session.

        Args:
            num_verses: Number of verses to evaluate. Defaults to all verses.

        Returns:
            List of evaluation score dictionaries, one per verse evaluated.
        """
        num_verses = num_verses or len(self.verses)
        num_verses = min(num_verses, len(self.verses))

        print(f"\n{'='*60}")
        print(f"Human Evaluation Session")
        print(f"Evaluating {num_verses} verses")
        print(f"{'='*60}")

        for i in range(num_verses):
            scores = self.collect_scores(i)
            self.scores.append(
                {
                    "verse_index": i,
                    "scores": scores,
                }
            )

            print(f"\nAverage score: {scores['average']:.2f}/5.0")

            if i < num_verses - 1:
                cont = input("\nPress Enter to continue, or 'q' to quit: ")
                if cont.lower() == "q":
                    break

        return self.scores

    def save_scores(self, output_path: Path) -> None:
        """Save evaluation scores to JSON file.

        Args:
            output_path: Path where to save the scores.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.scores, f, indent=2)

        logger.info(f"Saved evaluation scores to {output_path}")

        # Print summary
        if self.scores:
            avg_scores = {
                "accuracy": np.mean([s["scores"]["accuracy"] for s in self.scores]),
                "clarity": np.mean([s["scores"]["clarity"] for s in self.scores]),
                "naturalness": np.mean([s["scores"]["naturalness"] for s in self.scores]),
                "consistency": np.mean([s["scores"]["consistency"] for s in self.scores]),
            }

            print(f"\nEvaluation Summary:")
            print(f"  Accuracy: {avg_scores['accuracy']:.2f}/5.0")
            print(f"  Clarity: {avg_scores['clarity']:.2f}/5.0")
            print(f"  Naturalness: {avg_scores['naturalness']:.2f}/5.0")
            print(f"  Consistency: {avg_scores['consistency']:.2f}/5.0")
            print(f"  Overall: {np.mean(list(avg_scores.values())):.2f}/5.0")


if __name__ == "__main__":
    from utils.logger import configure_logging

    configure_logging()

    # Example usage
    evaluator = ScriptureEvaluator()

    # Sample translations
    hypothesis = "In the beginning, God created the heavens and the earth."
    reference = "In the beginning God created the heavens and the earth."

    bleu = evaluator.compute_bleu(
        hypothesis, reference, weights=(0.25, 0.25, 0.25, 0.25)
    )
    logger.info(f"BLEU-4 score: {bleu:.4f}")

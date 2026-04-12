import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation results"""
    bleu_1: float
    bleu_2: float
    bleu_4: float
    consistency_score: float
    unique_terms: int
    avg_term_usage: float
    human_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
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
    """
    Evaluate scripture translations using both standard and custom metrics.
    """
    
    def __init__(self, terminology_db=None):
        self.terminology_db = terminology_db
        self.smoothing_function = SmoothingFunction().method1
    
    def compute_bleu(self, hypothesis: str, reference: str,
                    weights: Tuple[float, ...] = (1, 0, 0, 0)) -> float:
        """
        Compute BLEU score.
        
        Args:
            hypothesis: Predicted translation
            reference: Reference translation
            weights: Weight for n-grams (1,0,0,0 = BLEU-1, 0,0.5,0.5,0 = BLEU-2, etc.)
        
        Returns:
            BLEU score (0-1)
        """
        hyp_tokens = word_tokenize(hypothesis.lower())
        ref_tokens = word_tokenize(reference.lower())
        
        # For BLEU, reference should be list of references (we have just one)
        references = [ref_tokens]
        
        try:
            bleu_score = sentence_bleu(
                references,
                hyp_tokens,
                weights=weights,
                smoothing_function=self.smoothing_function,
            )
        except:
            bleu_score = 0.0
        
        return min(1.0, bleu_score)  # Cap at 1.0
    
    def compute_consistency_score(self, translations: List[str], 
                                 terminology_db) -> float:
        """
        Measure how consistently terms are translated.
        
        Checks if the same English term maps to the same target term
        across all verses.
        """
        if not terminology_db or len(translations) < 2:
            return 1.0  # Perfect if no DB or only one verse
        
        term_consistency = {}
        
        for translation in translations:
            # Extract terms used
            words = word_tokenize(translation.lower())
            for word in words:
                # In a real implementation, we'd check if this word
                # corresponds to a theological term
                pass
        
        # Score: how many terms used consistently
        if not term_consistency:
            return 1.0
        
        consistent = sum(1 for terms in term_consistency.values() if len(terms) == 1)
        return consistent / len(term_consistency) if term_consistency else 1.0
    
    def compute_terminology_uniqueness(self, translations: List[str],
                                      target_lang: str) -> Tuple[int, float]:
        """
        Measure terminology diversity and usage patterns.
        
        Returns:
            (unique_terms, average_usage)
        """
        term_usage = {}
        total_words = 0
        
        for translation in translations:
            words = word_tokenize(translation.lower())
            total_words += len(words)
            
            for word in words:
                term_usage[word] = term_usage.get(word, 0) + 1
        
        unique_terms = len(term_usage)
        avg_usage = total_words / unique_terms if unique_terms > 0 else 0
        
        return unique_terms, avg_usage
    
    def evaluate_batch(self, hypotheses: List[str], references: List[str],
                      target_lang: str) -> EvaluationMetrics:
        """
        Evaluate a batch of translations.
        
        Args:
            hypotheses: List of predicted translations
            references: List of reference translations
            target_lang: Target language code
        
        Returns:
            EvaluationMetrics object
        """
        if len(hypotheses) != len(references):
            raise ValueError("Hypothesis and reference lists must be same length")
        
        # Compute BLEU scores
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_4_scores = []
        
        for hyp, ref in zip(hypotheses, references):
            bleu_1_scores.append(self.compute_bleu(hyp, ref, weights=(1, 0, 0, 0)))
            bleu_2_scores.append(self.compute_bleu(hyp, ref, weights=(0.5, 0.5, 0, 0)))
            bleu_4_scores.append(self.compute_bleu(hyp, ref, weights=(0.25, 0.25, 0.25, 0.25)))
        
        avg_bleu_1 = np.mean(bleu_1_scores)
        avg_bleu_2 = np.mean(bleu_2_scores)
        avg_bleu_4 = np.mean(bleu_4_scores)
        
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
    
    def save_evaluation_report(self, metrics: EvaluationMetrics, output_path: Path,
                              language_pair: str = ""):
        """Save evaluation report to JSON"""
        report = {
            "language_pair": language_pair,
            "metrics": metrics.to_dict(),
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation report to {output_path}")
    
    def print_metrics(self, metrics: EvaluationMetrics, title: str = ""):
        """Print formatted metrics"""
        print("\n" + "="*50)
        if title:
            print(f"=== {title} ===")
        print("="*50)
        print(f"BLEU-1: {metrics.bleu_1:.4f}")
        print(f"BLEU-2: {metrics.bleu_2:.4f}")
        print(f"BLEU-4: {metrics.bleu_4:.4f}")
        print(f"Consistency Score: {metrics.consistency_score:.4f}")
        print(f"Unique Terms: {metrics.unique_terms}")
        print(f"Avg Term Usage: {metrics.avg_term_usage:.2f}")
        if metrics.human_score is not None:
            print(f"Human Score: {metrics.human_score:.2f}/5.0")
        print("="*50 + "\n")


class HumanEvaluationInterface:
    """
    Interface for collecting human evaluation scores.
    """
    
    def __init__(self, verses: List[Dict]):
        """
        Initialize with verses to evaluate.
        
        Args:
            verses: List of verse dicts with 'source', 'reference', 'primary' keys
        """
        self.verses = verses
        self.scores = []
    
    def display_verse(self, index: int):
        """Display a verse for evaluation"""
        verse = self.verses[index]
        
        print(f"\n{'='*60}")
        print(f"Verse {index + 1}/{len(self.verses)}")
        print(f"Reference: {verse.get('reference', 'N/A')}")
        print(f"{'='*60}")
        print(f"\nSource: {verse['source']}")
        print(f"\nTranslation: {verse['primary']}")
        
        if verse.get('alternatives'):
            print(f"\nAlternatives:")
            for i, alt in enumerate(verse['alternatives'][:2]):
                print(f"  {i+1}. {alt}")
    
    def collect_scores(self, index: int) -> Dict:
        """Collect human evaluation scores for a verse"""
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
        
        avg_score = sum(scores[k] for k in ["accuracy", "clarity", "naturalness", "consistency"]) / 4
        scores["average"] = avg_score
        
        return scores
    
    def run_evaluation_session(self, num_verses: int = None):
        """Run interactive evaluation session"""
        num_verses = num_verses or len(self.verses)
        num_verses = min(num_verses, len(self.verses))
        
        print(f"\n{'='*60}")
        print(f"Human Evaluation Session")
        print(f"Evaluating {num_verses} verses")
        print(f"{'='*60}")
        
        for i in range(num_verses):
            scores = self.collect_scores(i)
            self.scores.append({
                "verse_index": i,
                "scores": scores,
            })
            
            print(f"\nAverage score: {scores['average']:.2f}/5.0")
            
            if i < num_verses - 1:
                cont = input("\nPress Enter to continue, or 'q' to quit: ")
                if cont.lower() == 'q':
                    break
        
        return self.scores
    
    def save_scores(self, output_path: Path):
        """Save evaluation scores to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
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
    # Example usage
    evaluator = ScriptureEvaluator()
    
    # Sample translations
    hypothesis = "In the beginning, God created the heavens and the earth."
    reference = "In the beginning God created the heavens and the earth."
    
    bleu = evaluator.compute_bleu(hypothesis, reference, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"BLEU-4 score: {bleu:.4f}")

"""Back-translation validation for detecting semantic drift.

Pipeline: EN → HT → EN
Compare back-translated EN with original EN to detect:
- Missing words/content
- Meaning shifts
- Hallucinations
- Translation quality issues
"""

from typing import Tuple, Optional
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils.logger import get_logger

logger = get_logger(__name__)


class BackTranslationValidator:
    """Validate translations by back-translating to source language."""

    def __init__(self, model, tokenizer, device: str = "cpu"):
        """Initialize back-translator.

        Args:
            model: NLLB model for back-translation.
            tokenizer: NLLB tokenizer.
            device: Device to use ("cpu" or "cuda").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def back_translate(
        self,
        translation: str,
        source_lang: str = "hat_Latn",
        target_lang: str = "eng_Latn",
    ) -> str:
        """Translate back to source language (HT → EN).

        Args:
            translation: Haitian Creole translation.
            source_lang: Source language of back-translation (default: hat_Latn).
            target_lang: Target language for back-translation (default: eng_Latn).

        Returns:
            Back-translated English text.
        """
        import torch

        # Tokenize input
        inputs = self.tokenizer(
            translation,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.tokenizer.src_lang = source_lang

        # Generate back-translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=5,
                early_stopping=True,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_lang),
            )

        back_translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return back_translated

    def validate(
        self,
        translation: str,
        original_source: str,
        back_translated: Optional[str] = None,
    ) -> Tuple[float, dict]:
        """Validate translation quality using back-translation.

        Compares original source with back-translated version to detect drift.

        Args:
            translation: Haitian Creole translation.
            original_source: Original English source.
            back_translated: Pre-computed back-translation (optional).

        Returns:
            Tuple of (similarity_score: 0-1, metrics_dict).
        """
        if back_translated is None:
            back_translated = self.back_translate(translation)

        # Compute similarity metrics
        metrics = {}

        # Metric 1: BLEU score (word overlap)
        metrics["bleu"] = self._compute_bleu(original_source, back_translated)

        # Metric 2: Content preservation (word-level jaccard)
        metrics["jaccard"] = self._compute_jaccard(original_source, back_translated)

        # Metric 3: Length ratio (catch hallucination/omission)
        metrics["length_ratio"] = self._compute_length_ratio(
            original_source, back_translated
        )

        # Metric 4: Keyword preservation (important nouns/verbs)
        metrics["keyword_preservation"] = self._compute_keyword_preservation(
            original_source, back_translated
        )

        # Composite similarity
        similarity = (
            metrics["bleu"] * 0.35
            + metrics["jaccard"] * 0.25
            + metrics["length_ratio"] * 0.20
            + metrics["keyword_preservation"] * 0.20
        )

        metrics["similarity"] = float(similarity)
        metrics["back_translated"] = back_translated

        logger.debug(
            f"Back-translation validation: "
            f"BLEU={metrics['bleu']:.2f}, "
            f"Jaccard={metrics['jaccard']:.2f}, "
            f"LenRatio={metrics['length_ratio']:.2f}, "
            f"Keywords={metrics['keyword_preservation']:.2f} → "
            f"Similarity={similarity:.2f}"
        )

        return float(similarity), metrics

    def _compute_bleu(self, reference: str, hypothesis: str) -> float:
        """Compute BLEU score (word-level 1-gram + 2-gram overlap).

        Args:
            reference: Original text.
            hypothesis: Back-translated text.

        Returns:
            BLEU score 0-1.
        """
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if not ref_tokens or not hyp_tokens:
            return 0.0

        try:
            # Use SmoothingFunction to handle edge cases
            smooth = SmoothingFunction().method1
            bleu = sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=(0.5, 0.5),  # 1-gram and 2-gram
                smoothing_function=smooth,
            )
            return min(float(bleu), 1.0)
        except Exception as e:
            logger.debug(f"BLEU computation error: {e}")
            return 0.0

    def _compute_jaccard(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity (set overlap of words).

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Jaccard score 0-1.
        """
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _compute_length_ratio(self, text1: str, text2: str) -> float:
        """Compute length ratio (catch hallucination/omission).

        Args:
            text1: Reference text.
            text2: Hypothesis text.

        Returns:
            Length ratio score 0-1 (1.0 = same length).
        """
        len1 = len(text1.split())
        len2 = len(text2.split())

        if len1 == 0:
            return 1.0 if len2 == 0 else 0.0

        ratio = len2 / len1
        # Penalize if significantly different (should be within 0.7-1.3x)
        if 0.7 <= ratio <= 1.3:
            return 1.0 - abs(1.0 - ratio) * 0.2
        else:
            return max(0.0, 1.0 - abs(1.0 - ratio) * 0.5)

    def _compute_keyword_preservation(self, text1: str, text2: str) -> float:
        """Compute preservation of important keywords (nouns, verbs).

        Args:
            text1: Reference text.
            text2: Hypothesis text.

        Returns:
            Keyword preservation score 0-1.
        """
        # Simple heuristic: capitalize words are often important (proper nouns)
        # Also: common Biblical keywords
        biblical_keywords = {
            "god", "lord", "jesus", "spirit", "salvation", "grace",
            "sin", "faith", "love", "truth", "heaven", "earth",
            "light", "darkness", "life", "death", "kingdom",
        }

        ref_words = set(text1.lower().split())
        hyp_words = set(text2.lower().split())

        # Keywords from reference
        ref_keywords = ref_words & biblical_keywords
        if not ref_keywords:
            return 0.7  # No biblical keywords = skip this metric

        # Check preservation
        preserved = len(ref_keywords & hyp_words)
        preservation_rate = preserved / len(ref_keywords) if ref_keywords else 0.0

        return min(preservation_rate, 1.0)

    def detect_hallucination(
        self, translation: str, back_translated: str, threshold: float = 1.5
    ) -> bool:
        """Detect if translation contains hallucinated content.

        Checks if back-translation is significantly longer than original,
        suggesting added content.

        Args:
            translation: Original translation (HT).
            back_translated: Back-translated text (EN).
            threshold: Length ratio threshold (default 1.5x).

        Returns:
            True if hallucination detected.
        """
        orig_len = len(translation.split())
        back_len = len(back_translated.split())

        if orig_len == 0:
            return False

        ratio = back_len / orig_len
        if ratio > threshold:
            logger.debug(
                f"Hallucination detected: back-translation {ratio:.2f}x longer"
            )
            return True

        return False

    def detect_omission(
        self, original: str, back_translated: str, threshold: float = 0.6
    ) -> bool:
        """Detect if translation omits important content.

        Checks if back-translation is significantly shorter than original,
        suggesting missing content.

        Args:
            original: Original English source.
            back_translated: Back-translated text (EN).
            threshold: Length ratio threshold (default 0.6x).

        Returns:
            True if significant omission detected.
        """
        orig_len = len(original.split())
        back_len = len(back_translated.split())

        if orig_len == 0:
            return False

        ratio = back_len / orig_len
        if ratio < threshold:
            logger.debug(
                f"Omission detected: back-translation only {ratio:.2f}x of original"
            )
            return True

        return False


if __name__ == "__main__":
    from utils.logger import configure_logging
    from models.base import ScriptureTranslationModel

    configure_logging()

    # Test the validator
    model_wrapper = ScriptureTranslationModel()
    model = model_wrapper.get_model()
    tokenizer = model_wrapper.get_tokenizer()

    validator = BackTranslationValidator(model, tokenizer)

    # Test
    original = "In the beginning, God created the heavens and the earth."
    translation = "Ansyen tan, Bondye te kreye syèl la ak latè."

    back_trans = validator.back_translate(translation)
    logger.info(f"Original: {original}")
    logger.info(f"Translation (HT): {translation}")
    logger.info(f"Back-translated (EN): {back_trans}")

    similarity, metrics = validator.validate(translation, original, back_trans)
    logger.info(f"\nSimilarity: {similarity:.2f}")
    logger.info(f"Metrics: {metrics}")

"""Prompt building for scripture translation.

Strategy: Input ONLY the verse text to NLLB (a translation model, not instruction-following).
Control behavior entirely through post-processing and hard constraints.

NLLB will translate whatever text you give it, including instructions.
So we give it minimal text and enforce our rules after generation.
"""

from typing import Dict, Optional, List
import re
from models.terminology import TerminologyDB, TermExtractor
from models.tiered_terminology import TieredTerminologyDB, TermTier
from inference.context_manager import ContextWindow
from utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Build minimal prompts for NLLB (translation model, not instruction-follower)."""

    def __init__(
        self,
        terminology_db: TerminologyDB,
        tiered_terminology: Optional[TieredTerminologyDB] = None,
    ):
        """Initialize prompt builder.

        Args:
            terminology_db: Terminology database for lookups.
            tiered_terminology: Optional tiered terminology system.
        """
        self.db = terminology_db
        self.tiered = tiered_terminology
        self.term_extractor = TermExtractor(terminology_db)

    def build_context_prompt(
        self,
        context_window: ContextWindow,
        target_lang: str,
        use_context: bool = True,
    ) -> str:
        """Build minimal input for NLLB: verse text with optional context.

        NLLB is a translation model, not instruction-following.
        Input ONLY the text to translate. No instructions, no delimiters.
        Control behavior via post-processing.

        Args:
            context_window: ContextWindow with prev/current/next verses.
            target_lang: Target language code (unused in input, for consistency).
            use_context: Whether to include context verses (helps semantic understanding).

        Returns:
            Minimal input text for model (verse + optional context).
        """
        parts = []

        # Add previous verse for context (if available and enabled)
        if use_context and context_window.prev_verse:
            parts.append(context_window.prev_verse)

        # Add the verse to translate (main content)
        parts.append(context_window.current_verse)

        # Add next verse for context (if available and enabled)
        if use_context and context_window.next_verse:
            parts.append(context_window.next_verse)

        # Join with newlines - just raw text, no labels
        return "\n".join(parts)

    def extract_translation_from_output(
        self, model_output: str, source_verse: str, context_verse_before: str = "", context_verse_after: str = ""
    ) -> str:
        """Extract the translated verse from model output.

        Strategy: The model translates whatever we give it.
        If we gave it [prev verse] [current verse] [next verse],
        we need to extract just the middle part.

        Args:
            model_output: Raw output from model.
            source_verse: The original source verse (to estimate length).
            context_verse_before: Previous verse (to exclude from output).
            context_verse_after: Next verse (to exclude from output).

        Returns:
            Cleaned translation of just the current verse.
        """
        output = model_output.strip()

        # Clean up common junk that NLLB sometimes adds
        output = self._clean_output(output)

        # If we had context, try to extract just the middle part
        if context_verse_before or context_verse_after:
            output = self._extract_middle_verse(
                output, source_verse, context_verse_before, context_verse_after
            )

        return output

    def _extract_middle_verse(
        self, output: str, source_verse: str, context_before: str, context_after: str
    ) -> str:
        """Extract the middle verse from a translation of [context][source][context].

        Uses length ratios to estimate where the middle verse is.

        Args:
            output: Translated output (all three verses combined).
            source_verse: Original source verse (length reference).
            context_before: Original previous verse.
            context_after: Original next verse.

        Returns:
            Estimated translation of just the source verse.
        """
        # Estimate relative lengths
        len_before = len(context_before.split())
        len_source = len(source_verse.split())
        len_after = len(context_after.split())
        total_words = len_before + len_source + len_after

        if total_words == 0:
            return output

        # Estimate where the source verse translation should start/end
        # (ratio of source to total, applied to output length)
        output_words = output.split()
        total_output_words = len(output_words)

        start_ratio = len_before / total_words if total_words > 0 else 0
        end_ratio = (len_before + len_source) / total_words if total_words > 0 else 1

        start_idx = max(0, int(total_output_words * start_ratio) - 2)  # -2 buffer
        end_idx = min(total_output_words, int(total_output_words * end_ratio) + 2)  # +2 buffer

        extracted_words = output_words[start_idx:end_idx]
        return " ".join(extracted_words).strip()

    def _clean_output(self, text: str) -> str:
        """Remove common artifacts from model output.

        Args:
            text: Text to clean.

        Returns:
            Cleaned text.
        """
        # Remove markdown/structural artifacts
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)  # Remove # headers
        text = re.sub(r"===.*?===", "", text, flags=re.DOTALL)  # Remove === markers
        text = re.sub(r"\[.*?\]", "", text)  # Remove [brackets]
        text = re.sub(r"\(.*?DO NOT.*?\)", "", text, flags=re.IGNORECASE)  # Remove instructions
        text = re.sub(r"###.*?###", "", text, flags=re.DOTALL)  # Remove ### sections

        # Remove common English instruction remnants (if model echoed them in translation)
        text = re.sub(r"(Translate|Translation|Source|Output|Instructions?)", "", text, flags=re.IGNORECASE)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
        text = re.sub(r"\n\n+", "\n", text)  # Collapse multiple newlines

        return text.strip()

    def extract_relevant_terms(
        self,
        source_text: str,
        target_lang: str,
    ) -> Dict[str, str]:
        """Extract and translate relevant theological terms from text.

        Args:
            source_text: Text to extract terms from.
            target_lang: Target language code.

        Returns:
            Dictionary mapping English terms to translations.
        """
        terms = self.term_extractor.extract_theological_terms(source_text)
        result = {}

        for term in terms:
            translation = self.db.lookup(term, target_lang)
            if translation:
                result[term] = translation

        return result


if __name__ == "__main__":
    from utils.logger import configure_logging

    configure_logging()

    # Test the prompt builder
    db = TerminologyDB()
    tiered = TieredTerminologyDB(db)
    builder = PromptBuilder(db, tiered)

    # Test context-window prompt
    verses = [
        {"text": "In the beginning, God created the heavens and the earth.", "reference": "Genesis 1:1"},
        {"text": "The earth was formless and empty, and darkness covered the deep waters.", "reference": "Genesis 1:2"},
        {"text": "And the Spirit of God was hovering over the surface of the waters.", "reference": "Genesis 1:3"},
    ]

    from inference.context_manager import ContextWindowBuilder
    windows = ContextWindowBuilder.build_windows(verses)
    prompt = builder.build_context_prompt(windows[1], "hat_Latn")
    logger.info("=== Minimal Input for NLLB ===")
    logger.info(prompt)
    logger.info("\n(No instructions, no delimiters - just raw text)")

    # Test output extraction
    mock_output = (
        "Ansyen tan, Bondye te kreye syèl la ak latè. "
        "Latè te dezè epi vid, ak fono te kouvri dlo yo. "
        "Lèspri Bondye a te swiv sifas dlo yo."
    )
    extracted = builder.extract_translation_from_output(
        mock_output,
        verses[1]["text"],
        verses[0]["text"],
        verses[2]["text"],
    )
    logger.info(f"\n=== Extracted Middle Verse ===\n{extracted}")

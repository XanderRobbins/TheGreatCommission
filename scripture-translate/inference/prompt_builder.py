"""Prompt building for scripture translation with strict delimiter separation.

Uses structured format to completely separate:
- Instructions (not to be translated)
- Reference glossary (guidance only)
- Context (semantic input, not translatable)
- Source text (the ONLY thing to translate)

This prevents prompt injection artifacts from leaking into output.
"""

from typing import Dict, Optional, List
import re
from models.terminology import TerminologyDB, TermExtractor
from models.tiered_terminology import TieredTerminologyDB, TermTier
from inference.context_manager import ContextWindow
from utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Build prompts with strict instruction/text separation."""

    # Clear delimiter to mark where actual translation begins
    SOURCE_TEXT_MARKER = "### SOURCE TEXT ###"
    TRANSLATION_START_MARKER = "===TRANSLATION START==="
    TRANSLATION_END_MARKER = "===TRANSLATION END==="

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
        use_glossary: bool = True,
    ) -> str:
        """Build prompt with strict delimiter separation for context-window translation.

        Structure:
        1. SYSTEM INSTRUCTIONS (DO NOT TRANSLATE)
        2. GLOSSARY (REFERENCE ONLY)
        3. CONTEXT (REFERENCE ONLY)
        4. SOURCE TEXT (TRANSLATE ONLY THIS)

        Args:
            context_window: ContextWindow with prev/current/next verses.
            target_lang: Target language code.
            use_glossary: Whether to include glossary reference.

        Returns:
            Formatted prompt string with clear delimiters.
        """
        parts = []

        # Section 1: System instructions (clearly marked as non-translatable)
        parts.append(self._build_instructions())

        # Section 2: Glossary (reference only, clearly marked)
        if use_glossary and self.tiered:
            glossary = self._build_glossary(context_window.current_verse, target_lang)
            if glossary:
                parts.append(glossary)

        # Section 3: Context (semantic, not to be translated)
        context_section = self._build_context_reference(context_window)
        if context_section:
            parts.append(context_section)

        # Section 4: Source text (THE ONLY THING TO TRANSLATE)
        parts.append(self._build_source_section(context_window))

        return "\n\n".join(parts)

    def _build_instructions(self) -> str:
        """Build clearly-marked system instructions."""
        return """### SYSTEM INSTRUCTIONS (DO NOT TRANSLATE, DO NOT INCLUDE IN OUTPUT)

Translate the Bible verse in the SOURCE TEXT section to Haitian Creole (hat_Latn).

Rules:
1. ONLY output the translation—nothing else
2. Do NOT repeat instructions, glossary, or context
3. Do NOT output labels like "Translation:" or "[" or "]"
4. Preserve exact theological meaning
5. Use natural, fluent Haitian Creole (not literal word-for-word)
6. Use only Haitian Creole words—avoid French
7. Maintain consistency with the glossary below
8. Respect poetic structure where present

CRITICAL: Output ONLY the translated verse text. Nothing else."""

    def _build_glossary(self, source_text: str, target_lang: str) -> Optional[str]:
        """Build reference glossary (not to be translated).

        Args:
            source_text: Text to extract terms from.
            target_lang: Target language code.

        Returns:
            Formatted glossary section, or None if no terms.
        """
        if not self.tiered:
            return None

        terms = self.term_extractor.extract_theological_terms(source_text)
        if not terms:
            return None

        # Only include Tier 1 (high priority) terms
        tier1_terms = []
        for term in sorted(terms):
            tier = self.tiered.get_tier(term)
            if tier == TermTier.TIER_1:
                translation = self.db.lookup(term, target_lang)
                if translation:
                    tier1_terms.append(f"  {term} → {translation}")

        if not tier1_terms:
            return None

        return "### GLOSSARY (REFERENCE ONLY - GUIDE YOUR TRANSLATION)\n" + "\n".join(
            tier1_terms
        )

    def _build_context_reference(self, context_window: ContextWindow) -> Optional[str]:
        """Build context reference (semantic understanding, not text to translate).

        Uses careful wording to prevent model from translating/copying context.

        Args:
            context_window: Context window.

        Returns:
            Formatted context section.
        """
        if not context_window.prev_verse and not context_window.next_verse:
            return None

        parts = ["### CONTEXT (FOR UNDERSTANDING ONLY - DO NOT TRANSLATE)"]

        if context_window.prev_verse:
            parts.append(f"(Previous verse for context: {context_window.prev_verse[:80]}...)")

        if context_window.next_verse:
            parts.append(f"(Next verse for context: {context_window.next_verse[:80]}...)")

        return "\n".join(parts)

    def _build_source_section(self, context_window: ContextWindow) -> str:
        """Build source text section (ONLY thing to translate).

        Args:
            context_window: Context window.

        Returns:
            Source text section with clear delimiter.
        """
        ref = context_window.current_ref or "Verse"
        return (
            f"{self.SOURCE_TEXT_MARKER}\n"
            f"{ref}: {context_window.current_verse}\n"
            f"\n"
            f"Translate this to Haitian Creole:\n"
            f"{self.TRANSLATION_START_MARKER}"
        )

    def extract_translation_from_output(self, model_output: str) -> str:
        """Extract ONLY the translation from model output.

        Looks for TRANSLATION_START_MARKER and takes everything after it,
        cleaning up any artifacts.

        Args:
            model_output: Raw output from model.

        Returns:
            Cleaned translation text.
        """
        # Look for start marker
        if self.TRANSLATION_START_MARKER in model_output:
            # Split at marker and take everything after
            parts = model_output.split(self.TRANSLATION_START_MARKER)
            output = parts[-1].strip()

            # Remove end marker if present
            if self.TRANSLATION_END_MARKER in output:
                output = output.split(self.TRANSLATION_END_MARKER)[0]

            # Clean up any leftover markers/labels
            output = self._clean_output(output)
            return output

        # Fallback: return everything, cleaned
        return self._clean_output(model_output)

    def _clean_output(self, text: str) -> str:
        """Remove prompt artifacts from output.

        Args:
            text: Text to clean.

        Returns:
            Cleaned text.
        """
        # Remove common prompt marker remnants
        text = re.sub(r"\[.*?\]", "", text)  # Remove [brackets]
        text = re.sub(r"###.*?###", "", text, flags=re.DOTALL)  # Remove section headers
        text = re.sub(r"Referans|Referans.*?:\n", "", text, flags=re.IGNORECASE)  # Remove glossary labels
        text = re.sub(r"Previous:|PREVIOUS:|Next:|NEXT:", "", text)  # Remove context labels
        text = re.sub(r"Translation:|TRANSLATION:", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Glossary|GLOSSARY|Reference|REFERENCE", "", text)
        text = re.sub(r"Guidelines?|GUIDELINES?", "", text)
        text = re.sub(r"Instructions?|INSTRUCTIONS?", "", text)

        # Clean up multiple newlines
        text = re.sub(r"\n\n+", "\n", text)

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
    from inference.context_manager import ContextWindowBuilder

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

    windows = ContextWindowBuilder.build_windows(verses)
    prompt = builder.build_context_prompt(windows[1], "hat_Latn")
    logger.info("=== Full Prompt ===")
    logger.info(prompt)

    # Test output extraction
    mock_output = (
        "Some noise...\n"
        "===TRANSLATION START===\n"
        "Ansyen tan, Bondye te kreye syèl la ak latè.\n"
        "===TRANSLATION END===\n"
        "Some more noise..."
    )
    extracted = builder.extract_translation_from_output(mock_output)
    logger.info(f"\n=== Extracted Translation ===\n{extracted}")

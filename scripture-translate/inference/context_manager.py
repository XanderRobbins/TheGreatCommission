"""Context window management for scripture translation.

Builds sliding windows of verses (prev, current, next) to provide semantic
context during translation, improving pronoun resolution and theological
term consistency.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """A sliding window of verses with context.

    Attributes:
        prev_verse: Previous verse text (or empty if first verse).
        current_verse: The verse to translate.
        next_verse: Next verse text (or empty if last verse).
        prev_ref: Reference for previous verse.
        current_ref: Reference for current verse.
        next_ref: Reference for next verse.
        index: Position of current verse in original list.
    """
    prev_verse: str
    current_verse: str
    next_verse: str
    prev_ref: str
    current_ref: str
    next_ref: str
    index: int


class ContextWindowBuilder:
    """Build sliding windows of verses for context-aware translation."""

    @staticmethod
    def build_windows(
        verses: List[Dict],
        context_range: int = 1,
    ) -> List[ContextWindow]:
        """Build sliding context windows for all verses.

        Args:
            verses: List of verse dicts with 'text' and optional 'reference'.
            context_range: How many verses before/after to include (default 1).

        Returns:
            List of ContextWindow objects.
        """
        windows = []

        for i, verse in enumerate(verses):
            prev_idx = max(0, i - context_range)
            next_idx = min(len(verses) - 1, i + context_range)

            prev_verse = verses[prev_idx]["text"] if prev_idx != i else ""
            next_verse = verses[next_idx]["text"] if next_idx != i else ""

            window = ContextWindow(
                prev_verse=prev_verse,
                current_verse=verse["text"],
                next_verse=next_verse,
                prev_ref=verses[prev_idx].get("reference", "") if prev_idx != i else "",
                current_ref=verse.get("reference", ""),
                next_ref=verses[next_idx].get("reference", "") if next_idx != i else "",
                index=i,
            )
            windows.append(window)

        return windows

    @staticmethod
    def format_context_input(window: ContextWindow) -> str:
        """Format context window for translation input.

        Shows context verses in [brackets] to signal they're not the main target.

        Args:
            window: ContextWindow to format.

        Returns:
            Formatted string with context.
        """
        parts = []

        if window.prev_verse:
            parts.append(f"[Previous: {window.prev_verse}]")

        parts.append(f"[TRANSLATE THIS VERSE]: {window.current_verse}")

        if window.next_verse:
            parts.append(f"[Next: {window.next_verse}]")

        return "\n".join(parts)

    @staticmethod
    def extract_translated_verse(
        full_translation: str,
        context_window: ContextWindow,
    ) -> str:
        """Extract the target verse translation from context output.

        Post-processes the model output to isolate just the translated current verse.

        Strategy:
        1. Look for structural markers if model preserved them
        2. Fallback: estimate based on relative token counts
        3. Final fallback: return full translation if extraction fails

        Args:
            full_translation: Full translated output (may include context).
            context_window: Original context window (for debugging).

        Returns:
            Extracted translation of just the current verse.
        """
        # If output is short, likely just the target (no context echoed back)
        if len(full_translation) < 100:
            return full_translation

        lines = full_translation.split("\n")

        # Try to find lines matching context markers
        current_start = None
        current_end = None

        for i, line in enumerate(lines):
            if "TRANSLATE" in line.upper() or "CURRENT" in line.upper():
                current_start = i + 1
            elif current_start is not None and (
                "NEXT" in line.upper() or "PREVIOUS" in line.upper()
            ):
                current_end = i
                break

        # Extract marked section if found
        if current_start is not None:
            if current_end is None:
                current_end = len(lines)
            extracted = "\n".join(lines[current_start:current_end]).strip()
            if extracted:
                return extracted

        # Fallback: estimate based on input token ratios
        # Current verse is likely ~60-80% of the context window
        prev_len = len(context_window.prev_verse)
        curr_len = len(context_window.current_verse)
        next_len = len(context_window.next_verse)

        if prev_len + curr_len + next_len > 0:
            curr_ratio = curr_len / (prev_len + curr_len + next_len)
            estimated_end = int(len(full_translation) * curr_ratio * 1.3)  # 1.3x buffer
            extracted = full_translation[:estimated_end].strip()
            if extracted:
                return extracted

        # Final fallback: return everything
        logger.debug(
            f"Could not extract verse {context_window.current_ref}; "
            f"returning full translation"
        )
        return full_translation.strip()

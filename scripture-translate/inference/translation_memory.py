"""Translation memory cache to prevent drift on repeated verses.

Benefits:
- Identical verses always get identical translations
- Faster processing (cache hits = no model inference)
- Prevents drift from repeated generation
- Tracks cache hit rate for metrics
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Entry in translation memory."""

    source_text: str
    translation: str
    language_pair: str
    timestamp: str
    confidence: float
    hit_count: int = 0


class TranslationMemory:
    """Translation memory cache using verse-hash → translation mapping."""

    def __init__(self, cache_path: Optional[Path] = None):
        """Initialize translation memory.

        Args:
            cache_path: Path to JSON cache file. Defaults to "./translation_memory.json".
        """
        self.cache_path = cache_path or Path("./translation_memory.json")
        self.cache: Dict[str, dict] = {}
        self.stats = {
            "total_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "entries": 0,
        }

        self.load()

    def _compute_hash(self, text: str, language_pair: str) -> str:
        """Compute stable hash for a verse.

        Args:
            text: Source verse text (normalized).
            language_pair: Language pair (e.g., "eng_Latn→hat_Latn").

        Returns:
            SHA256 hash.
        """
        # Normalize: strip whitespace, lowercase for comparison
        normalized = text.strip().lower()
        key = f"{language_pair}:{normalized}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def lookup(
        self, source_text: str, language_pair: str
    ) -> Optional[Tuple[str, float]]:
        """Look up a cached translation.

        Args:
            source_text: Source verse text.
            language_pair: Language pair (e.g., "eng_Latn→hat_Latn").

        Returns:
            Tuple of (translation, confidence) if found, None otherwise.
        """
        self.stats["total_lookups"] += 1

        verse_hash = self._compute_hash(source_text, language_pair)

        if verse_hash in self.cache:
            entry = self.cache[verse_hash]
            entry["hit_count"] += 1
            self.stats["cache_hits"] += 1

            logger.debug(
                f"Cache HIT: {language_pair} (hit_count={entry['hit_count']})"
            )

            return (entry["translation"], entry["confidence"])

        self.stats["cache_misses"] += 1
        return None

    def store(
        self,
        source_text: str,
        translation: str,
        language_pair: str,
        confidence: float,
    ) -> None:
        """Store a translation in memory.

        Args:
            source_text: Source verse text.
            translation: Target language translation.
            language_pair: Language pair (e.g., "eng_Latn→hat_Latn").
            confidence: Confidence score 0-1.
        """
        verse_hash = self._compute_hash(source_text, language_pair)

        from datetime import datetime

        self.cache[verse_hash] = {
            "source_text": source_text,
            "translation": translation,
            "language_pair": language_pair,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "hit_count": 0,
        }

        self.stats["entries"] = len(self.cache)
        logger.debug(
            f"Cached translation: {language_pair} (total_entries={self.stats['entries']})"
        )

    def get_hit_rate(self) -> float:
        """Get cache hit rate.

        Returns:
            Hit rate 0-1.
        """
        total = self.stats["total_lookups"]
        if total == 0:
            return 0.0

        return self.stats["cache_hits"] / total

    def get_most_accessed(self, top_n: int = 10) -> list:
        """Get most frequently accessed translations.

        Args:
            top_n: Number of top entries to return.

        Returns:
            List of (verse_hash, hit_count) tuples.
        """
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1]["hit_count"],
            reverse=True,
        )

        return [
            (hash_val, entry["hit_count"], entry["source_text"][:50])
            for hash_val, entry in sorted_entries[:top_n]
        ]

    def save(self) -> None:
        """Save cache to JSON file."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "total_entries": len(self.cache),
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "hit_rate": self.get_hit_rate(),
            },
            "entries": self.cache,
        }

        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Saved translation memory: {len(self.cache)} entries, "
            f"hit_rate={self.get_hit_rate():.1%}"
        )

    def load(self) -> None:
        """Load cache from JSON file."""
        if not self.cache_path.exists():
            logger.info(f"No existing translation memory at {self.cache_path}")
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning(f"Failed to load translation memory: {exc}")
            return

        self.cache = data.get("entries", {})
        logger.info(f"Loaded translation memory: {len(self.cache)} entries")

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.stats = {
            "total_lookups": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "entries": 0,
        }
        logger.info("Cleared translation memory")

    def print_stats(self) -> None:
        """Log formatted statistics."""
        logger.info("=== Translation Memory Statistics ===")
        logger.info(f"Total entries: {self.stats['entries']}")
        logger.info(f"Total lookups: {self.stats['total_lookups']}")
        logger.info(f"Cache hits: {self.stats['cache_hits']}")
        logger.info(f"Cache misses: {self.stats['cache_misses']}")
        logger.info(f"Hit rate: {self.get_hit_rate():.1%}")

        top_accessed = self.get_most_accessed(5)
        if top_accessed:
            logger.info("Most accessed translations:")
            for hash_val, hit_count, preview in top_accessed:
                logger.info(f"  {hit_count}x hits: {preview}...")


if __name__ == "__main__":
    from utils.logger import configure_logging

    configure_logging()

    # Test the translation memory
    memory = TranslationMemory()

    # Store some translations
    memory.store(
        "God is love",
        "Bondye se lanmou",
        "eng_Latn→hat_Latn",
        0.95,
    )

    memory.store(
        "In the beginning",
        "Ansyen tan",
        "eng_Latn→hat_Latn",
        0.92,
    )

    # Test lookup
    result = memory.lookup("God is love", "eng_Latn→hat_Latn")
    logger.info(f"Lookup 1: {result}")

    result = memory.lookup("God is love", "eng_Latn→hat_Latn")  # Second lookup
    logger.info(f"Lookup 2 (cache hit): {result}")

    result = memory.lookup("Unknown verse", "eng_Latn→hat_Latn")  # Miss
    logger.info(f"Lookup 3 (cache miss): {result}")

    # Print stats
    memory.print_stats()

    # Save
    memory.save()

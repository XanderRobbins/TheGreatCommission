"""Bible loader with multi-source fallback chain.

Loads Bible verses from:
1. pythonbible library (if installed)
2. bbible library (if installed)
3. freebible library (if installed)
4. HuggingFace datasets (christos-c/bible-corpus)
5. Local CSV file (data/bible_en.csv)

Output format: [{"reference": "Genesis 1:1", "text": "..."}, ...]
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BibleLoader:
    """Load Bible verses from multiple sources with automatic fallback."""

    # Standard book name mapping for consistency
    BOOKS = {
        "GENESIS": "Genesis",
        "EXODUS": "Exodus",
        "LEVITICUS": "Leviticus",
        "NUMBERS": "Numbers",
        "DEUTERONOMY": "Deuteronomy",
        "JOSHUA": "Joshua",
        "JUDGES": "Judges",
        "RUTH": "Ruth",
        "SAMUEL_1": "1 Samuel",
        "SAMUEL_2": "2 Samuel",
        "KINGS_1": "1 Kings",
        "KINGS_2": "2 Kings",
        "CHRONICLES_1": "1 Chronicles",
        "CHRONICLES_2": "2 Chronicles",
        "EZRA": "Ezra",
        "NEHEMIAH": "Nehemiah",
        "ESTHER": "Esther",
        "JOB": "Job",
        "PSALM": "Psalm",
        "PROVERBS": "Proverbs",
        "ECCLESIASTES": "Ecclesiastes",
        "ISAIAH": "Isaiah",
        "JEREMIAH": "Jeremiah",
        "LAMENTATIONS": "Lamentations",
        "EZEKIEL": "Ezekiel",
        "DANIEL": "Daniel",
        "HOSEA": "Hosea",
        "JOEL": "Joel",
        "AMOS": "Amos",
        "OBADIAH": "Obadiah",
        "JONAH": "Jonah",
        "MICAH": "Micah",
        "NAHUM": "Nahum",
        "HABAKKUK": "Habakkuk",
        "ZEPHANIAH": "Zephaniah",
        "HAGGAI": "Haggai",
        "ZECHARIAH": "Zechariah",
        "MALACHI": "Malachi",
        "MATTHEW": "Matthew",
        "MARK": "Mark",
        "LUKE": "Luke",
        "JOHN": "John",
        "ACTS": "Acts",
        "ROMANS": "Romans",
        "CORINTHIANS_1": "1 Corinthians",
        "CORINTHIANS_2": "2 Corinthians",
        "GALATIANS": "Galatians",
        "EPHESIANS": "Ephesians",
        "PHILIPPIANS": "Philippians",
        "COLOSSIANS": "Colossians",
        "THESSALONIANS_1": "1 Thessalonians",
        "THESSALONIANS_2": "2 Thessalonians",
        "TIMOTHY_1": "1 Timothy",
        "TIMOTHY_2": "2 Timothy",
        "TITUS": "Titus",
        "PHILEMON": "Philemon",
        "HEBREWS": "Hebrews",
        "JAMES": "James",
        "PETER_1": "1 Peter",
        "PETER_2": "2 Peter",
        "JOHN_1": "1 John",
        "JOHN_2": "2 John",
        "JOHN_3": "3 John",
        "JUDE": "Jude",
        "REVELATION": "Revelation",
    }

    def __init__(self, data_dir: Path = None):
        """Initialize Bible loader.

        Args:
            data_dir: Directory to look for local Bible files (default: ./data)
        """
        self.data_dir = Path(data_dir or "./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[Dict[str, str]]:
        """Load Bible verses using fallback chain.

        Returns:
            List of dicts: [{"reference": "Genesis 1:1", "text": "..."}, ...]

        Raises:
            RuntimeError: If all sources fail.
        """
        # Try each source in order
        sources = [
            ("pythonbible", self._load_pythonbible),
            ("bbible", self._load_bbible),
            ("freebible", self._load_freebible),
            ("HuggingFace datasets", self._load_huggingface),
            ("local CSV", self._load_local_csv),
        ]

        for source_name, load_func in sources:
            try:
                logger.info(f"Attempting to load Bible from {source_name}...")
                verses = load_func()
                if verses:
                    logger.info(f"✓ Successfully loaded {len(verses)} verses from {source_name}")
                    return verses
            except Exception as e:
                logger.debug(f"✗ {source_name} failed: {type(e).__name__}: {str(e)[:80]}")
                continue

        # All sources exhausted
        raise RuntimeError(
            "Could not load Bible from any source. Please:\n"
            "  1. pip install pythonbible\n"
            "  2. Or place data/bible_en.csv with columns: book,chapter,verse,text\n"
            "  3. Or run: pip install datasets"
        )

    def _load_pythonbible(self) -> List[Dict[str, str]]:
        import pythonbible as bible

        verses = []

        for book in bible.Book:
            book_name = book.name  # KEEP UPPERCASE

            if book_name not in self.BOOKS:
                logger.warning(f"Skipping unknown book: {book_name}")
                continue

            display_name = self.BOOKS[book_name]

            try:
                n_chapters = bible.get_number_of_chapters(book)
            except Exception:
                continue

            for chapter_idx in range(1, n_chapters + 1):
                try:
                    n_verses = bible.get_number_of_verses(book, chapter_idx)
                except Exception:
                    continue

                for verse_idx in range(1, n_verses + 1):
                    try:
                        verse_id = bible.convert_reference_to_verse_ids(
                            bible.NormalizedReference(
                                book, chapter_idx, verse_idx, chapter_idx, verse_idx
                            )
                        )[0]

                        text = bible.get_verse_text(verse_id)

                        if text and text.strip():
                            verses.append({
                                "reference": f"{display_name} {chapter_idx}:{verse_idx}",
                                "text": text.strip(),
                            })

                    except Exception:
                        continue

        return verses



    def _load_bbible(self) -> List[Dict[str, str]]:
        """Load from bbible library."""
        import bbible

        verses = []
        # bbible API - adjust based on actual library interface
        # This is a placeholder for the actual API
        bible_obj = bbible.Bible()

        for book_name in self.BOOKS.keys():
            try:
                for chapter_idx in range(1, 200):  # Safety limit
                    for verse_idx in range(1, 200):
                        try:
                            text = bible_obj.get_verse(book_name, chapter_idx, verse_idx)
                            if text and text.strip():
                                verses.append({
                                    "reference": f"{book_name} {chapter_idx}:{verse_idx}",
                                    "text": text.strip(),
                                })
                        except (IndexError, KeyError):
                            # End of verses for this chapter
                            break
                    if not any(v["reference"].startswith(f"{book_name} {chapter_idx}:")
                              for v in verses[-1:]):
                        # End of chapters for this book
                        break
            except Exception:
                continue

        return verses

    def _load_freebible(self) -> List[Dict[str, str]]:
        """Load from freebible library."""
        import freebible

        verses = []
        # freebible API - adjust based on actual library interface
        # This is a placeholder
        try:
            bible_data = freebible.get_bible()
            # Process based on the actual structure returned
            for entry in bible_data:
                if "reference" in entry and "text" in entry:
                    verses.append({
                        "reference": entry["reference"],
                        "text": entry["text"],
                    })
        except Exception:
            pass

        return verses

    def _load_huggingface(self) -> List[Dict[str, str]]:
        """Load from HuggingFace datasets (christos-c/bible-corpus)."""
        from datasets import load_dataset

        verses = []
        try:
            dataset = load_dataset("christos-c/bible-corpus", split="train", trust_remote_code=True)

            for item in dataset:
                # Adjust fields based on actual dataset structure
                if "text" in item:
                    # Try to extract reference and text
                    text = item["text"]
                    reference = item.get("reference", "")

                    if text and text.strip():
                        verses.append({
                            "reference": reference or "Unknown",
                            "text": text.strip(),
                        })
        except Exception:
            pass

        return verses

    def _load_local_csv(self) -> List[Dict[str, str]]:
        """Load from local CSV file."""
        import csv

        csv_path = self.data_dir / "bible_en.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Local CSV not found: {csv_path}")

        verses = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    book = row.get("book", "").strip()
                    chapter = row.get("chapter", "").strip()
                    verse = row.get("verse", "").strip()
                    text = row.get("text", "").strip()

                    if book and chapter and verse and text:
                        verses.append({
                            "reference": f"{book} {chapter}:{verse}",
                            "text": text,
                        })
                except Exception:
                    continue

        return verses


def load_bible(data_dir: Path = None) -> List[Dict[str, str]]:
    """Convenience function to load Bible with one call.

    Args:
        data_dir: Directory for Bible files (default: ./data)

    Returns:
        List of verses: [{"reference": "Genesis 1:1", "text": "..."}, ...]
    """
    loader = BibleLoader(data_dir)
    return loader.load()


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    verses = load_bible()
    print(f"\nLoaded {len(verses)} verses")
    if verses:
        print(f"First verse: {verses[0]}")
        print(f"Last verse: {verses[-1]}")

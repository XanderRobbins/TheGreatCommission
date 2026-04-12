"""Database for maintaining consistent terminology across translations.

Ensures that the same English theological term maps to the same target
language term throughout all verses.
"""

import json
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict
from datetime import datetime

from utils.logger import get_logger
from constants import (
    TERM_DEFAULT_CONFIDENCE,
    TERM_REVIEWED_CONFIDENCE,
    TERMINOLOGY_DB_VERSION,
)

logger = get_logger(__name__)


class TerminologyDB:
    """Database for maintaining consistent terminology across translations.

    Ensures that the same English theological term maps to the same target
    language term throughout all verses.
    """

    # Key theological and cultural terms that must be consistent
    # Using frozenset to prevent accidental mutation
    THEOLOGICAL_TERMS = frozenset({
        # Salvation concepts
        "salvation", "savior", "redeemer", "redemption", "save", "saved",
        "grace", "mercy", "judgment", "righteousness", "sin", "repent", "forgive",

        # Deity/Trinity
        "god", "lord", "almighty", "spirit", "holy spirit", "jesus", "christ",
        "father", "son", "trinity", "divine", "godly", "god's",

        # Religious practices
        "baptism", "baptize", "communion", "eucharist", "prayer", "pray",
        "worship", "sacrifice", "offering", "covenant", "law",

        # Spiritual states
        "blessed", "blessing", "curse", "holy", "sanctify", "faith", "believe",
        "hope", "love", "righteous", "wicked", "evil", "good",

        # Community/People
        "church", "disciple", "apostle", "prophet", "priest", "king",
        "israel", "jew", "gentile", "congregation", "body",

        # Concepts
        "kingdom", "eternal", "heaven", "hell", "judgment day", "resurrection",
        "life", "death", "truth", "light", "darkness", "water", "bread",
    })
    
    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize terminology database.

        Args:
            db_path: Path to JSON database file. Defaults to "./terminology.json".
        """
        self.db_path = db_path or Path("./terminology.json")
        # Format: terms[english_term][target_lang] = (target_term, confidence)
        self.terms: Dict[str, Dict[str, Tuple[str, float]]] = defaultdict(dict)

        # Track how many times each term is used. Nested defaultdicts are critical
        # for record_usage() to work after load() is called.
        self.usage_count: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Log variations of terms for manual review
        self.variation_log: Dict[str, List[str]] = defaultdict(list)

        self.load()
    
    def add_term(
        self,
        english_term: str,
        target_lang: str,
        target_term: str,
        confidence: float = TERM_DEFAULT_CONFIDENCE,
        override: bool = False,
    ) -> bool:
        """Register a term translation.

        Args:
            english_term: Source English term.
            target_lang: Target language code.
            target_term: Translation in target language.
            confidence: Translation confidence (0.0-1.0). Defaults to TERM_DEFAULT_CONFIDENCE.
            override: If True, replace existing term even if it conflicts.

        Returns:
            True if added, False if conflict (and override=False).
        """
        english_normalized = english_term.lower().strip()

        if english_normalized in self.terms[target_lang] and not override:
            existing, existing_conf = self.terms[target_lang][english_normalized]
            if existing != target_term:
                logger.warning(
                    f"Conflict: {english_normalized} → {existing} vs {target_term} "
                    f"in {target_lang}"
                )
                self.variation_log[english_normalized].append(target_term)
                return False
            else:
                # Same term, just update confidence if higher
                if confidence > existing_conf:
                    self.terms[target_lang][english_normalized] = (target_term, confidence)
                return True

        self.terms[target_lang][english_normalized] = (target_term, confidence)
        logger.info(
            f"Added: {english_normalized} → {target_term} ({target_lang}, conf={confidence:.2f})"
        )
        return True
    
    def lookup(self, english_term: str, target_lang: str) -> Optional[str]:
        """Get canonical translation for a term.

        Args:
            english_term: Source English term.
            target_lang: Target language code.

        Returns:
            Target language translation, or None if not found.
        """
        english_normalized = english_term.lower().strip()
        if english_normalized in self.terms.get(target_lang, {}):
            target_term, _ = self.terms[target_lang][english_normalized]
            return target_term
        return None

    def get_with_confidence(
        self, english_term: str, target_lang: str
    ) -> Optional[Tuple[str, float]]:
        """Get term with confidence score.

        Args:
            english_term: Source English term.
            target_lang: Target language code.

        Returns:
            Tuple of (target_term, confidence) or None if not found.
        """
        english_normalized = english_term.lower().strip()
        return self.terms.get(target_lang, {}).get(english_normalized)

    def get_all_terms_for_language(
        self, target_lang: str
    ) -> Dict[str, Tuple[str, float]]:
        """Get all terms for a language.

        Args:
            target_lang: Target language code.

        Returns:
            Dictionary mapping English terms to (target_term, confidence) tuples.
        """
        return self.terms.get(target_lang, {})

    def record_usage(self, english_term: str, target_lang: str) -> None:
        """Track that a term was used.

        Args:
            english_term: Source English term.
            target_lang: Target language code.
        """
        english_normalized = english_term.lower().strip()
        self.usage_count[english_normalized][target_lang] += 1

    def get_usage_count(self, english_term: str, target_lang: str) -> int:
        """Get how many times a term was used.

        Args:
            english_term: Source English term.
            target_lang: Target language code.

        Returns:
            Number of times the term was used in translation.
        """
        english_normalized = english_term.lower().strip()
        return self.usage_count[english_normalized].get(target_lang, 0)
    
    def save(self) -> None:
        """Save database to JSON file."""
        # Convert defaultdicts to plain dicts for serialization
        data = {
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "version": TERMINOLOGY_DB_VERSION,
            },
            "terms": {},
            "usage_count": dict(
                (k, dict(v)) for k, v in self.usage_count.items()
            ),
            "variation_log": dict(self.variation_log),
        }

        for lang, terms_dict in self.terms.items():
            data["terms"][lang] = {
                eng: {"target": tgt, "confidence": conf}
                for eng, (tgt, conf) in terms_dict.items()
            }

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved terminology database to {self.db_path}")
    
    def load(self) -> None:
        """Load database from JSON file.

        Reconstructs nested defaultdicts from JSON to preserve auto-creation
        semantics after load. Plain dicts from JSON would raise KeyError on
        first record_usage() call with a new term.
        """
        if not self.db_path.exists():
            logger.info(f"No existing database at {self.db_path}")
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning(f"Failed to load terminology database: {exc}")
            return

        # Load terms
        for lang, terms_dict in data.get("terms", {}).items():
            for eng, term_data in terms_dict.items():
                self.terms[lang][eng] = (term_data["target"], term_data["confidence"])

        # Reconstruct nested defaultdict for usage_count from plain dict JSON
        # This prevents KeyError on first record_usage() call with new terms
        loaded_usage = data.get("usage_count", {})
        for term, lang_counts in loaded_usage.items():
            if isinstance(lang_counts, dict):
                for lang, count in lang_counts.items():
                    self.usage_count[term][lang] = count

        # Reconstruct defaultdict(list) for variation_log
        loaded_variations = data.get("variation_log", {})
        for term, variants in loaded_variations.items():
            if isinstance(variants, list):
                self.variation_log[term].extend(variants)

        logger.info(f"Loaded {sum(len(v) for v in self.terms.values())} terms")
    
    def get_conflicts(self) -> Dict[str, List[str]]:
        """Get terms with multiple translations (conflicts to resolve).

        Returns:
            Dictionary mapping English terms to lists of variant translations.
        """
        return {k: v for k, v in self.variation_log.items() if v}

    def resolve_conflict(
        self, english_term: str, target_lang: str, chosen_term: str
    ) -> None:
        """Mark a conflict as resolved.

        Args:
            english_term: Source English term.
            target_lang: Target language code.
            chosen_term: The chosen translation to use consistently.
        """
        english_normalized = english_term.lower().strip()
        self.add_term(english_normalized, target_lang, chosen_term, override=True)
        if english_normalized in self.variation_log:
            del self.variation_log[english_normalized]
        logger.info(f"Resolved conflict: {english_normalized} → {chosen_term}")
    
    def export_for_review(self, output_path: Path, target_lang: str) -> None:
        """Export terms for human review.

        Args:
            output_path: Path where to save the review file.
            target_lang: Target language code to export.
        """
        terms = self.get_all_terms_for_language(target_lang)

        review_data = {
            "language": target_lang,
            "total_terms": len(terms),
            "terms": [
                {
                    "english": eng,
                    "translation": tgt,
                    "confidence": conf,
                    "usage_count": self.get_usage_count(eng, target_lang),
                }
                for eng, (tgt, conf) in sorted(terms.items())
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(review_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(terms)} terms for review to {output_path}")

    def import_reviewed_terms(self, reviewed_path: Path) -> None:
        """Import terms that have been reviewed by humans.

        Args:
            reviewed_path: Path to the reviewed terms file.
        """
        try:
            with open(reviewed_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            logger.error(f"Failed to import reviewed terms: {exc}")
            return

        target_lang = data["language"]
        imported = 0

        for item in data.get("terms", []):
            if item.get("approved", False):
                self.add_term(
                    item["english"],
                    target_lang,
                    item["translation"],
                    confidence=TERM_REVIEWED_CONFIDENCE,
                    override=True,
                )
                imported += 1

        logger.info(f"Imported {imported} reviewed terms for {target_lang}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the terminology database.

        Returns:
            Dictionary with total_terms, languages, conflicts, avg_confidence,
            and terms_by_language breakdown.
        """
        total_terms = sum(len(v) for v in self.terms.values())
        languages = len(self.terms)
        conflicts = len(self.get_conflicts())

        avg_confidence = []
        for lang_terms in self.terms.values():
            confs = [conf for _, conf in lang_terms.values()]
            if confs:
                avg_confidence.append(sum(confs) / len(confs))

        return {
            "total_terms": total_terms,
            "languages": languages,
            "conflicts": conflicts,
            "avg_confidence": (
                sum(avg_confidence) / len(avg_confidence) if avg_confidence else 0
            ),
            "terms_by_language": {lang: len(terms) for lang, terms in self.terms.items()},
        }

    def print_statistics(self) -> None:
        """Log formatted statistics to logger.info (not stdout)."""
        stats = self.get_statistics()
        logger.info("=== Terminology Database Statistics ===")
        logger.info(f"Total terms: {stats['total_terms']}")
        logger.info(f"Languages: {stats['languages']}")
        logger.info(f"Conflicts to resolve: {stats['conflicts']}")
        logger.info(f"Average confidence: {stats['avg_confidence']:.2%}")
        logger.info("Terms by language:")
        for lang, count in stats["terms_by_language"].items():
            logger.info(f"  {lang}: {count}")


class TermExtractor:
    """Extract theological terms from Bible text and map to canonical translations."""

    def __init__(self, terminology_db: TerminologyDB) -> None:
        """Initialize extractor with a terminology database.

        Args:
            terminology_db: The terminology database to use for lookups.
        """
        self.db = terminology_db

    def extract_theological_terms(self, text: str) -> Set[str]:
        """Extract theological terms from a verse.

        Performs case-insensitive substring matching against THEOLOGICAL_TERMS.

        Args:
            text: The verse text to extract terms from.

        Returns:
            Set of lowercase theological terms found in the text.
        """
        text_lower = text.lower()
        found_terms = set()

        for term in TerminologyDB.THEOLOGICAL_TERMS:
            if term in text_lower:
                found_terms.add(term)

        return found_terms

    def get_canonical_terms(
        self, text: str, target_lang: str
    ) -> Dict[str, Optional[str]]:
        """Get canonical translations for all theological terms in text.

        Args:
            text: The verse text to extract and translate terms from.
            target_lang: The target language code.

        Returns:
            Dictionary mapping English theological terms to their target language
            translations (or None if not found in the terminology database).
        """
        terms = self.extract_theological_terms(text)
        return {term: self.db.lookup(term, target_lang) for term in terms}


if __name__ == "__main__":
    from utils.logger import configure_logging

    configure_logging()

    # Example usage
    db = TerminologyDB()

    # Add some sample terms
    db.add_term("salvation", "spa_Latn", "salvación", confidence=0.98)
    db.add_term("grace", "spa_Latn", "gracia", confidence=0.97)
    db.add_term("faith", "spa_Latn", "fe", confidence=0.96)

    db.add_term("salvation", "swh_Latn", "wokovu", confidence=0.85)
    db.add_term("grace", "swh_Latn", "neema", confidence=0.88)

    db.save()
    db.print_statistics()

    # Test extraction
    extractor = TermExtractor(db)
    sample_text = "God's grace and salvation are eternal"
    terms = extractor.extract_theological_terms(sample_text)
    logger.info(f"Found terms: {terms}")

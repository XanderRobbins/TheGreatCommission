import json
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TerminologyDB:
    """
    Database for maintaining consistent terminology across translations.
    
    Ensures that the same English theological term maps to the same 
    target language term throughout all verses.
    """
    
    # Key theological and cultural terms that must be consistent
    THEOLOGICAL_TERMS = {
        # Salvation concepts
        "salvation", "savior", "redeemer", "redemption", "save", "saved",
        "grace", "mercy", "judgment", "righteousness", "sin", "repent", "forgive",
        
        # Deity/Trinity
        "god", "lord", "almighty", "spirit", "holy spirit", "jesus", "christ",
        "father", "son", "trinity", "divine", "godly", "god's",
        
        # Religious practices
        "baptism", "baptize", "communion", "eucharist", "prayer", "pray",
        "worship", "worship", "sacrifice", "offering", "covenant", "law",
        
        # Spiritual states
        "blessed", "blessing", "curse", "holy", "sanctify", "faith", "believe",
        "hope", "love", "righteous", "wicked", "evil", "good",
        
        # Community/People
        "church", "disciple", "apostle", "prophet", "priest", "king",
        "israel", "jew", "gentile", "congregation", "body",
        
        # Concepts
        "kingdom", "eternal", "heaven", "hell", "judgment day", "resurrection",
        "life", "death", "truth", "light", "darkness", "water", "bread",
    }
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("./terminology.json")
        self.terms: Dict[str, Dict[str, Tuple[str, float]]] = defaultdict(dict)
        # Format: terms[english_term][target_lang] = (target_term, confidence)
        
        self.usage_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Track how many times each term is used
        
        self.variation_log: Dict[str, List[str]] = defaultdict(list)
        # Log variations of terms for manual review
        
        self.load()
    
    def add_term(self, english_term: str, target_lang: str, target_term: str,
                confidence: float = 0.9, override: bool = False) -> bool:
        """
        Register a term translation.
        
        Args:
            english_term: Source English term
            target_lang: Target language code
            target_term: Translation in target language
            confidence: Translation confidence (0.0-1.0)
            override: If True, replace existing term
        
        Returns:
            True if added, False if conflict (unless override=True)
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
                # Same term, just update confidence
                if confidence > existing_conf:
                    self.terms[target_lang][english_normalized] = (target_term, confidence)
                return True
        
        self.terms[target_lang][english_normalized] = (target_term, confidence)
        logger.info(f"Added: {english_normalized} → {target_term} ({target_lang}, conf={confidence:.2f})")
        return True
    
    def lookup(self, english_term: str, target_lang: str) -> Optional[str]:
        """Get canonical translation for a term"""
        english_normalized = english_term.lower().strip()
        if english_normalized in self.terms.get(target_lang, {}):
            target_term, _ = self.terms[target_lang][english_normalized]
            return target_term
        return None
    
    def get_with_confidence(self, english_term: str, target_lang: str) -> Optional[Tuple[str, float]]:
        """Get term with confidence score"""
        english_normalized = english_term.lower().strip()
        return self.terms.get(target_lang, {}).get(english_normalized)
    
    def get_all_terms_for_language(self, target_lang: str) -> Dict[str, Tuple[str, float]]:
        """Get all terms for a language"""
        return self.terms.get(target_lang, {})
    
    def record_usage(self, english_term: str, target_lang: str):
        """Track that a term was used"""
        english_normalized = english_term.lower().strip()
        self.usage_count[english_normalized][target_lang] += 1
    
    def get_usage_count(self, english_term: str, target_lang: str) -> int:
        """Get how many times a term was used"""
        english_normalized = english_term.lower().strip()
        return self.usage_count[english_normalized].get(target_lang, 0)
    
    def save(self):
        """Save database to JSON file"""
        # Convert to serializable format
        data = {
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "version": "1.0",
            },
            "terms": {},
            "usage_count": self.usage_count,
            "variation_log": self.variation_log,
        }
        
        for lang, terms_dict in self.terms.items():
            data["terms"][lang] = {
                eng: {"target": tgt, "confidence": conf}
                for eng, (tgt, conf) in terms_dict.items()
            }
        
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved terminology database to {self.db_path}")
    
    def load(self):
        """Load database from JSON file"""
        if not self.db_path.exists():
            logger.info(f"No existing database at {self.db_path}")
            return
        
        with open(self.db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for lang, terms_dict in data.get("terms", {}).items():
            for eng, term_data in terms_dict.items():
                self.terms[lang][eng] = (term_data["target"], term_data["confidence"])
        
        self.usage_count = data.get("usage_count", {})
        self.variation_log = data.get("variation_log", {})
        
        logger.info(f"Loaded {sum(len(v) for v in self.terms.values())} terms")
    
    def get_conflicts(self) -> Dict[str, List[str]]:
        """Get terms with multiple translations (conflicts to resolve)"""
        return {k: v for k, v in self.variation_log.items() if v}
    
    def resolve_conflict(self, english_term: str, target_lang: str, chosen_term: str):
        """Mark a conflict as resolved"""
        english_normalized = english_term.lower().strip()
        self.add_term(english_normalized, target_lang, chosen_term, override=True)
        if english_normalized in self.variation_log:
            del self.variation_log[english_normalized]
        logger.info(f"Resolved conflict: {english_normalized} → {chosen_term}")
    
    def export_for_review(self, output_path: Path, target_lang: str):
        """Export terms for human review"""
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
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported {len(terms)} terms for review to {output_path}")
    
    def import_reviewed_terms(self, reviewed_path: Path):
        """Import terms that have been reviewed by humans"""
        with open(reviewed_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        target_lang = data["language"]
        imported = 0
        
        for item in data["terms"]:
            if item.get("approved", False):
                self.add_term(
                    item["english"],
                    target_lang,
                    item["translation"],
                    confidence=0.95,
                    override=True
                )
                imported += 1
        
        logger.info(f"Imported {imported} reviewed terms for {target_lang}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the terminology database"""
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
            "avg_confidence": sum(avg_confidence) / len(avg_confidence) if avg_confidence else 0,
            "terms_by_language": {lang: len(terms) for lang, terms in self.terms.items()},
        }
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        print("\n=== Terminology Database Statistics ===")
        print(f"Total terms: {stats['total_terms']}")
        print(f"Languages: {stats['languages']}")
        print(f"Conflicts to resolve: {stats['conflicts']}")
        print(f"Average confidence: {stats['avg_confidence']:.2%}")
        print("\nTerms by language:")
        for lang, count in stats['terms_by_language'].items():
            print(f"  {lang}: {count}")


class TermExtractor:
    """Extract theological terms from Bible text"""
    
    def __init__(self, terminology_db: TerminologyDB):
        self.db = terminology_db
    
    def extract_theological_terms(self, text: str) -> Set[str]:
        """Extract theological terms from a verse"""
        text_lower = text.lower()
        found_terms = set()
        
        for term in TerminologyDB.THEOLOGICAL_TERMS:
            if term in text_lower:
                found_terms.add(term)
        
        return found_terms
    
    def get_canonical_terms(self, text: str, target_lang: str) -> Dict[str, Optional[str]]:
        """Get canonical translations for all theological terms in text"""
        terms = self.extract_theological_terms(text)
        return {
            term: self.db.lookup(term, target_lang)
            for term in terms
        }


if __name__ == "__main__":
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
    print(f"\nFound terms: {terms}")

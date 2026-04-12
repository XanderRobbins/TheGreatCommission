"""Terminology service with input validation and language checking."""

from typing import Dict, Optional, Tuple
from models.terminology import TerminologyDB, TermExtractor
from config import Config
from exceptions import LanguageNotSupportedError
from utils.logger import get_logger

logger = get_logger(__name__)


class TerminologyService:
    """Terminology database service with validation."""

    def __init__(self, terminology_db: Optional[TerminologyDB] = None):
        """Initialize terminology service."""
        self.db = terminology_db or TerminologyDB()
        self.extractor = TermExtractor(self.db)

    def validate_language(self, lang_code: str) -> bool:
        """Validate language code is supported."""
        supported_codes = set(Config.LANGUAGE_CODES.values())
        if lang_code not in supported_codes:
            raise LanguageNotSupportedError(
                f"Language code '{lang_code}' not supported. "
                f"Supported: {', '.join(sorted(supported_codes))}"
            )
        return True

    def add_term(
        self,
        english_term: str,
        target_lang: str,
        target_term: str,
        confidence: float = 0.9,
        override: bool = False,
    ) -> bool:
        """Add term with validation."""
        self.validate_language(target_lang)

        if not english_term or not target_term:
            raise ValueError("English and target terms cannot be empty")

        success = self.db.add_term(
            english_term,
            target_lang,
            target_term,
            confidence=confidence,
            override=override,
        )

        if success:
            self.db.save()

        return success

    def lookup(self, english_term: str, target_lang: str) -> Optional[str]:
        """Look up term translation."""
        self.validate_language(target_lang)
        return self.db.lookup(english_term, target_lang)

    def get_with_confidence(
        self, english_term: str, target_lang: str
    ) -> Optional[Tuple[str, float]]:
        """Get term with confidence score."""
        self.validate_language(target_lang)
        return self.db.get_with_confidence(english_term, target_lang)

    def extract_terms(self, text: str, target_lang: str) -> Dict[str, Optional[str]]:
        """Extract and translate theological terms."""
        self.validate_language(target_lang)
        return self.extractor.get_canonical_terms(text, target_lang)

    def get_conflicts(self) -> Dict[str, list]:
        """Get terminology conflicts."""
        return self.db.get_conflicts()

    def resolve_conflict(
        self, english_term: str, target_lang: str, chosen_term: str
    ) -> None:
        """Resolve a terminology conflict."""
        self.validate_language(target_lang)
        self.db.resolve_conflict(english_term, target_lang, chosen_term)
        self.db.save()

    def get_statistics(self) -> Dict:
        """Get terminology statistics."""
        return self.db.get_statistics()

    def save(self) -> None:
        """Save terminology database."""
        self.db.save()

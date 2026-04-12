"""Translation service wrapping model loading and inference."""

from typing import List, Dict, Optional
from models.base import ScriptureTranslationModel
from inference import ScriptureTranslator
from models.terminology import TerminologyDB
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class TranslationService:
    """Unified translation service for Flask and CLI."""

    _instance = None
    _model_wrapper = None
    _translator = None
    _terminology_db = None

    def __new__(cls):
        """Singleton pattern for shared state."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(
        self,
        use_lora: bool = False,
        terminology_db: Optional[TerminologyDB] = None,
    ) -> None:
        """Initialize translation service (idempotent)."""
        if self._translator is not None:
            return  # Already initialized

        logger.info("Initializing TranslationService...")

        self._model_wrapper = ScriptureTranslationModel(use_lora=use_lora)
        self._terminology_db = terminology_db or TerminologyDB()

        self._translator = ScriptureTranslator(
            model=self._model_wrapper.get_model(),
            tokenizer=self._model_wrapper.get_tokenizer(),
            terminology_db=self._terminology_db,
            device=Config.get_device(),
            enforce_consistency=True,
        )

        logger.info("TranslationService initialized")

    def translate_verse(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        num_beams: int = 5,
    ):
        """Translate a single verse."""
        if self._translator is None:
            self.initialize()

        return self._translator.translate_verse(
            source_text=source_text,
            source_lang=source_lang,
            target_lang=target_lang,
            num_beams=num_beams,
        )

    def translate_batch(
        self,
        verses: List[Dict],
        source_lang: str,
        target_lang: str,
        batch_size: int = 8,
        show_progress: bool = True,
    ):
        """Translate multiple verses."""
        if self._translator is None:
            self.initialize()

        return self._translator.translate_batch(
            verses=verses,
            source_lang=source_lang,
            target_lang=target_lang,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def get_model_wrapper(self):
        """Get model wrapper."""
        if self._model_wrapper is None:
            self.initialize()
        return self._model_wrapper

    def get_terminology_db(self):
        """Get terminology database."""
        if self._terminology_db is None:
            self.initialize()
        return self._terminology_db

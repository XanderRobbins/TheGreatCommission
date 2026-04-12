import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

from models.terminology import TerminologyDB, TermExtractor
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Container for translation results"""
    primary: str
    confidence: float
    alternatives: List[str] = None
    theological_terms: Dict[str, Optional[str]] = None
    consistency_enforced: bool = False
    source_text: str = ""
    target_language: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "primary": self.primary,
            "confidence": float(self.confidence),
            "alternatives": self.alternatives or [],
            "theological_terms": self.theological_terms or {},
            "consistency_enforced": self.consistency_enforced,
            "source_text": self.source_text,
            "target_language": self.target_language,
        }


class ScriptureTranslator:
    """
    Inference engine for scripture translation with consistency enforcement.
    """
    
    def __init__(self, model, tokenizer, terminology_db: Optional[TerminologyDB] = None,
                 device: str = None, enforce_consistency: bool = True):
        """
        Initialize translator.
        
        Args:
            model: Loaded NLLB model
            tokenizer: NLLB tokenizer
            terminology_db: Optional terminology database for consistency
            device: Device to use
            enforce_consistency: Whether to enforce terminology consistency
        """
        self.model = model
        self.tokenizer = tokenizer
        self.terminology_db = terminology_db or TerminologyDB()
        self.device = device or Config.get_device()
        self.enforce_consistency = enforce_consistency
        self.term_extractor = TermExtractor(self.terminology_db)
        
        self.model.eval()
        logger.info("Translator initialized")
    
    def translate_verse(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        num_beams: int = 5,
        max_length: int = 256,
        temperature: float = 1.0,
        return_alternatives: bool = True,
    ) -> TranslationResult:
        """
        Translate a single verse.
        
        Args:
            source_text: Source verse text
            source_lang: Source language code (NLLB format, e.g., "eng_Latn")
            target_lang: Target language code (NLLB format, e.g., "spa_Latn")
            num_beams: Number of beams for beam search
            max_length: Maximum output length
            temperature: Temperature for generation
            return_alternatives: Whether to return alternative translations
        
        Returns:
            TranslationResult object
        """
        # Extract theological terms before translation
        theological_terms = self.term_extractor.extract_theological_terms(source_text)
        canonical_terms = self.term_extractor.get_canonical_terms(source_text, target_lang)
        
        # Tokenize input
        inputs = self.tokenizer(
            source_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set source language
        self.tokenizer.src_lang = source_lang
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_beams if return_alternatives else 1,
                temperature=temperature,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_lang),
            )
        
        # Decode translations
        primary_translation = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )
        
        # Get alternatives
        alternatives = []
        if return_alternatives and len(outputs.sequences) > 1:
            for seq in outputs.sequences[1:]:
                alt = self.tokenizer.decode(seq, skip_special_tokens=True)
                alternatives.append(alt)
        
        # Calculate confidence
        confidence = outputs.sequences_scores[0].item() if hasattr(outputs, 'sequences_scores') else 0.0
        confidence = torch.nn.functional.softmax(torch.tensor([confidence]), dim=0).item()
        
        # Enforce consistency
        consistency_enforced = False
        if self.enforce_consistency and canonical_terms:
            primary_translation, consistency_enforced = self._enforce_consistency(
                primary_translation, source_text, canonical_terms, target_lang
            )
        
        # Record term usage
        for term in theological_terms:
            self.terminology_db.record_usage(term, target_lang)
        
        return TranslationResult(
            primary=primary_translation,
            confidence=confidence,
            alternatives=alternatives,
            theological_terms=canonical_terms,
            consistency_enforced=consistency_enforced,
            source_text=source_text,
            target_language=target_lang,
        )
    
    def _enforce_consistency(
        self,
        translation: str,
        source_text: str,
        canonical_terms: Dict[str, Optional[str]],
        target_lang: str,
    ) -> Tuple[str, bool]:
        """
        Post-process translation to enforce consistency.
        
        Args:
            translation: Generated translation
            source_text: Source text (for context)
            canonical_terms: Canonical terms from terminology DB
            target_lang: Target language
        
        Returns:
            Tuple of (enforced_translation, was_modified)
        """
        modified = False
        enforced = translation
        
        # For now, just log what would be enforced
        # Full implementation would intelligently replace terms
        for source_term, canonical in canonical_terms.items():
            if canonical and canonical not in enforced:
                # Log the inconsistency
                logger.debug(
                    f"Enforcing consistency: {source_term} → {canonical} in {target_lang}"
                )
        
        return enforced, modified
    
    def translate_batch(
        self,
        verses: List[Dict],
        source_lang: str,
        target_lang: str,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> List[TranslationResult]:
        """
        Translate multiple verses.
        
        Args:
            verses: List of verse dictionaries with 'text' key
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Batch size for processing
            show_progress: Show progress bar
        
        Returns:
            List of TranslationResult objects
        """
        results = []
        
        iterator = verses
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(verses, desc="Translating verses")
        
        for i, verse in enumerate(iterator):
            result = self.translate_verse(
                verse["text"],
                source_lang,
                target_lang,
            )
            
            # Add metadata if available
            if "reference" in verse:
                result.source_text = f"{verse['reference']}: {result.source_text}"
            
            results.append(result)
        
        return results
    
    def translate_book(
        self,
        book_verses: List[Dict],
        book_name: str,
        source_lang: str,
        target_lang: str,
    ) -> Dict:
        """
        Translate an entire book.
        
        Args:
            book_verses: List of verse dictionaries
            book_name: Name of the book (for logging)
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Dictionary with translation results and metadata
        """
        logger.info(f"Translating {book_name} ({len(book_verses)} verses)...")
        
        results = self.translate_batch(
            book_verses,
            source_lang,
            target_lang,
            show_progress=True,
        )
        
        # Calculate statistics
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
        consistency_enforced = sum(1 for r in results if r.consistency_enforced)
        
        return {
            "book": book_name,
            "source_language": source_lang,
            "target_language": target_lang,
            "total_verses": len(results),
            "translations": [r.to_dict() for r in results],
            "statistics": {
                "average_confidence": avg_confidence,
                "verses_with_consistency_enforced": consistency_enforced,
            },
        }


class BeamSearchDecoder:
    """
    Enhanced beam search with constraint support.
    """
    
    def __init__(self, model, tokenizer, terminology_db: Optional[TerminologyDB] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.terminology_db = terminology_db
    
    def decode_with_constraints(
        self,
        input_ids: torch.Tensor,
        target_lang: str,
        terminology_constraints: Optional[Dict[str, str]] = None,
        num_beams: int = 5,
        max_length: int = 256,
    ) -> List[str]:
        """
        Decode with optional terminology constraints.
        
        Constraints force certain terms to be used in the output.
        """
        # Generate as usual
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            early_stopping=True,
        )
        
        # Decode
        translations = [
            self.tokenizer.decode(seq, skip_special_tokens=True)
            for seq in outputs
        ]
        
        # Apply constraints post-hoc (simplified)
        if terminology_constraints:
            translations = [
                self._apply_constraints(t, terminology_constraints)
                for t in translations
            ]
        
        return translations
    
    def _apply_constraints(self, text: str, constraints: Dict[str, str]) -> str:
        """Apply terminology constraints to text"""
        # Placeholder: in production, use fuzzy matching + replacement
        return text


if __name__ == "__main__":
    from models.base import ScriptureTranslationModel
    
    print("Testing translator...")
    
    # Initialize model
    model_wrapper = ScriptureTranslationModel()
    
    # Initialize translator
    terminology_db = TerminologyDB()
    translator = ScriptureTranslator(
        model=model_wrapper.get_model(),
        tokenizer=model_wrapper.get_tokenizer(),
        terminology_db=terminology_db,
        enforce_consistency=True,
    )
    
    # Test translation
    source = "In the beginning, God created the heavens and the earth."
    print(f"\nSource: {source}")
    
    result = translator.translate_verse(
        source,
        source_lang="eng_Latn",
        target_lang="spa_Latn",
        num_beams=3,
    )
    
    print(f"Translation: {result.primary}")
    print(f"Confidence: {result.confidence:.2%}")

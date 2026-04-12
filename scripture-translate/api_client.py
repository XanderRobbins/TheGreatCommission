"""
Scripture Translation API Client

Easy-to-use Python client for the Scripture Translation System.
Can be used programmatically or integrated into other applications.

Usage:
    from api_client import ScriptureTranslationClient
    
    client = ScriptureTranslationClient("http://localhost:5000")
    
    # Translate a verse
    result = client.translate(
        "In the beginning, God created the heavens and the earth.",
        source_lang="eng_Latn",
        target_lang="spa_Latn"
    )
    print(result['translation'])
    
    # Manage terminology
    client.add_term("salvation", "spa_Latn", "salvación", confidence=0.98)
    translation = client.lookup_term("salvation", "spa_Latn")
    
    # Evaluate translations
    metrics = client.evaluate_batch(
        hypotheses,
        references,
        target_lang="spa_Latn"
    )
"""

import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Result from translation API"""
    text: str
    confidence: float
    alternatives: List[str]
    terms: Dict[str, Optional[str]]


class ScriptureTranslationClient:
    """
    Client for Scripture Translation API.
    
    Handles all communication with the backend server.
    """
    
    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 60):
        """
        Initialize client.
        
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request to API"""
        url = f"{self.base_url}/api{endpoint}"
        
        try:
            if method == 'GET':
                response = self.session.get(url, timeout=self.timeout, **kwargs)
            elif method == 'POST':
                response = self.session.post(url, timeout=self.timeout, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    # ========================================================================
    # TRANSLATION METHODS
    # ========================================================================
    
    def translate(self, text: str, source_lang: str = "eng_Latn",
                 target_lang: str = "spa_Latn", num_beams: int = 5) -> Dict:
        """
        Translate a single verse.
        
        Args:
            text: Verse text to translate
            source_lang: Source language code (NLLB format)
            target_lang: Target language code (NLLB format)
            num_beams: Number of beams for beam search
        
        Returns:
            Dict with translation result
        """
        payload = {
            'text': text,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'num_beams': num_beams,
        }
        
        result = self._request('POST', '/translate', json=payload)
        
        if result.get('success'):
            return result['result']
        else:
            raise RuntimeError(result.get('error', 'Translation failed'))
    
    def translate_batch(self, verses: List[Dict], source_lang: str = "eng_Latn",
                       target_lang: str = "spa_Latn") -> List[Dict]:
        """
        Translate multiple verses.
        
        Args:
            verses: List of verse dictionaries with 'text' key
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            List of translation results
        """
        payload = {
            'verses': verses,
            'source_lang': source_lang,
            'target_lang': target_lang,
        }
        
        result = self._request('POST', '/translate/batch', json=payload)
        
        if result.get('success'):
            return result['results']
        else:
            raise RuntimeError(result.get('error', 'Batch translation failed'))
    
    # ========================================================================
    # TERMINOLOGY METHODS
    # ========================================================================
    
    def add_term(self, english_term: str, target_lang: str, target_term: str,
                confidence: float = 0.9, override: bool = False) -> bool:
        """
        Add a term to the terminology database.
        
        Args:
            english_term: Source term
            target_lang: Target language code
            target_term: Translation
            confidence: Confidence score (0-1)
            override: Whether to override existing term
        
        Returns:
            True if added, False if conflict
        """
        payload = {
            'english_term': english_term,
            'target_lang': target_lang,
            'target_term': target_term,
            'confidence': confidence,
            'override': override,
        }
        
        result = self._request('POST', '/terminology/add', json=payload)
        return result.get('success', False)
    
    def lookup_term(self, english_term: str, target_lang: str) -> Optional[str]:
        """
        Look up a term in the database.
        
        Args:
            english_term: Source term
            target_lang: Target language code
        
        Returns:
            Translation if found, None otherwise
        """
        result = self._request(
            'GET',
            '/terminology/lookup',
            params={
                'english_term': english_term,
                'target_lang': target_lang,
            }
        )
        
        if result.get('success'):
            return result.get('target_term')
        return None
    
    def extract_terms(self, text: str, target_lang: str = None) -> Dict:
        """
        Extract theological terms from text.
        
        Args:
            text: Text to analyze
            target_lang: Optional target language for canonical forms
        
        Returns:
            Dict with extracted terms and their translations
        """
        payload = {
            'text': text,
            'target_lang': target_lang or '',
        }
        
        result = self._request('POST', '/terminology/extract', json=payload)
        
        if result.get('success'):
            return {
                'terms': result.get('terms', []),
                'canonical': result.get('canonical', {}),
            }
        else:
            raise RuntimeError(result.get('error', 'Term extraction failed'))
    
    def get_conflicts(self) -> List[str]:
        """
        Get terminology conflicts.
        
        Returns:
            List of conflicting terms
        """
        result = self._request('GET', '/terminology/conflicts')
        
        if result.get('success'):
            return list(result.get('conflicts', {}).keys())
        else:
            raise RuntimeError(result.get('error', 'Failed to get conflicts'))
    
    def resolve_conflict(self, english_term: str, target_lang: str,
                        chosen_term: str) -> bool:
        """
        Resolve a terminology conflict.
        
        Args:
            english_term: Source term
            target_lang: Target language code
            chosen_term: Chosen translation
        
        Returns:
            True if resolved successfully
        """
        payload = {
            'english_term': english_term,
            'target_lang': target_lang,
            'chosen_term': chosen_term,
        }
        
        result = self._request('POST', '/terminology/resolve', json=payload)
        return result.get('success', False)
    
    def get_terminology_stats(self) -> Dict:
        """
        Get terminology database statistics.
        
        Returns:
            Dict with statistics
        """
        result = self._request('GET', '/terminology/stats')
        
        if result.get('success'):
            return result.get('statistics', {})
        else:
            raise RuntimeError(result.get('error', 'Failed to get stats'))
    
    def export_terminology(self, target_lang: str) -> bytes:
        """
        Export terminology database.
        
        Args:
            target_lang: Language to export
        
        Returns:
            JSON data as bytes
        """
        response = self.session.get(
            f"{self.base_url}/api/terminology/export",
            params={'target_lang': target_lang},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.content
    
    # ========================================================================
    # EVALUATION METHODS
    # ========================================================================
    
    def evaluate_bleu(self, hypothesis: str, reference: str) -> float:
        """
        Calculate BLEU score.
        
        Args:
            hypothesis: Predicted translation
            reference: Reference translation
        
        Returns:
            BLEU score (0-1)
        """
        payload = {
            'hypothesis': hypothesis,
            'reference': reference,
        }
        
        result = self._request('POST', '/evaluate/bleu', json=payload)
        
        if result.get('success'):
            return result.get('bleu', 0.0)
        else:
            raise RuntimeError(result.get('error', 'BLEU evaluation failed'))
    
    def evaluate_batch(self, hypotheses: List[str], references: List[str],
                      target_lang: str = "spa_Latn") -> Dict:
        """
        Evaluate a batch of translations.
        
        Args:
            hypotheses: List of predicted translations
            references: List of reference translations
            target_lang: Target language code
        
        Returns:
            Dict with evaluation metrics
        """
        if len(hypotheses) != len(references):
            raise ValueError("Hypothesis and reference counts must match")
        
        payload = {
            'hypotheses': hypotheses,
            'references': references,
            'target_lang': target_lang,
        }
        
        result = self._request('POST', '/evaluate/batch', json=payload)
        
        if result.get('success'):
            return result.get('metrics', {})
        else:
            raise RuntimeError(result.get('error', 'Batch evaluation failed'))
    
    # ========================================================================
    # SYSTEM METHODS
    # ========================================================================
    
    def get_system_info(self) -> Dict:
        """
        Get system information.
        
        Returns:
            Dict with system info
        """
        result = self._request('GET', '/system/info')
        
        if result.get('success'):
            return {
                'model': result.get('model'),
                'device': result.get('device'),
                'languages': result.get('languages'),
                'terminology_stats': result.get('terminology_stats'),
            }
        else:
            raise RuntimeError(result.get('error', 'Failed to get system info'))
    
    def save_system(self) -> bool:
        """
        Save system state.
        
        Returns:
            True if saved successfully
        """
        result = self._request('POST', '/system/save')
        return result.get('success', False)
    
    def health_check(self) -> bool:
        """
        Check if API is running.
        
        Returns:
            True if API is responsive
        """
        try:
            self.get_system_info()
            return True
        except Exception:
            return False


class BatchTranslationJob:
    """
    Helper class for managing batch translation jobs.
    """
    
    def __init__(self, client: ScriptureTranslationClient, verses: List[Dict],
                source_lang: str, target_lang: str):
        self.client = client
        self.verses = verses
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.results = []
        self.failed = []
    
    def run(self) -> Dict:
        """
        Run the batch translation job.
        
        Returns:
            Dict with results and statistics
        """
        logger.info(f"Starting batch translation of {len(self.verses)} verses...")
        
        try:
            self.results = self.client.translate_batch(
                self.verses,
                self.source_lang,
                self.target_lang,
            )
            
            logger.info(f"Successfully translated {len(self.results)} verses")
        
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            self.failed = self.verses
        
        return self.get_summary()
    
    def get_summary(self) -> Dict:
        """Get job summary"""
        return {
            'total': len(self.verses),
            'successful': len(self.results),
            'failed': len(self.failed),
            'success_rate': len(self.results) / len(self.verses) if self.verses else 0,
        }


if __name__ == "__main__":
    # Example usage
    print("Scripture Translation API Client")
    print("=" * 50)
    
    # Initialize client
    client = ScriptureTranslationClient("http://localhost:5000")
    
    # Check if API is running
    if client.health_check():
        print("✓ API is running")
    else:
        print("✗ API is not responding")
        exit(1)
    
    # Get system info
    info = client.get_system_info()
    print(f"\nModel: {info['model']}")
    print(f"Device: {info['device']}")
    print(f"Languages: {len(info['languages'])}")
    
    # Example: Translate a verse
    try:
        text = "In the beginning, God created the heavens and the earth."
        result = client.translate(
            text,
            source_lang="eng_Latn",
            target_lang="spa_Latn"
        )
        print(f"\nTranslation: {result['primary']}")
        print(f"Confidence: {result['confidence']:.2%}")
    except Exception as e:
        print(f"Translation error: {e}")

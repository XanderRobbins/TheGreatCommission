"""
Advanced linguistic and semantic analysis tools for scripture translation.

Includes:
- Named Entity Recognition for proper nouns
- Semantic similarity checking
- Language-specific linguistic rules
- Morphological analysis
- Discourse analysis
"""

import re
from typing import List, Dict, Set, Tuple, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class BiblicalNER:
    """
    Named Entity Recognition for Biblical texts.
    
    Identifies:
    - Person names (Abraham, Moses, Jesus)
    - Place names (Jerusalem, Egypt, Galilee)
    - Divine names (God, Lord, Almighty)
    - Temporal references (day, year, season)
    """
    
    # Common biblical proper nouns that should not be translated
    BIBLICAL_NAMES = {
        # Persons
        "Abraham", "Moses", "Jesus", "David", "Solomon",
        "Mary", "Joseph", "Peter", "Paul", "John", "Matthew",
        "Mark", "Luke", "James", "Timothy", "Titus",
        
        # Places
        "Jerusalem", "Egypt", "Israel", "Judea", "Galilee",
        "Bethlehem", "Nazareth", "Capernaum", "Rome",
        "Sinai", "Jordan", "Red Sea", "Dead Sea",
        
        # Divine titles
        "God", "Lord", "Almighty", "Holy One", "Father",
        "Son", "Spirit", "Messiah", "Christ", "King",
    }
    
    def __init__(self):
        self.name_patterns = {
            'person': r'^[A-Z][a-z]+$',  # Capitalized words
            'place': r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)?$',
            'divine': r'(?:God|Lord|Spirit|Holy|Almighty|Father|King)',
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Returns:
            Dict with 'person', 'place', 'divine' keys
        """
        entities = {
            'person': [],
            'place': [],
            'divine': [],
            'uncertain': [],
        }
        
        # Tokenize
        words = text.split()
        
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[.,;:!?]', '', word)
            
            # Check if it's a known biblical name
            if clean_word in self.BIBLICAL_NAMES:
                if any(p in clean_word for p in ['God', 'Lord', 'Spirit', 'Holy', 'Almighty']):
                    entities['divine'].append(clean_word)
                else:
                    # Heuristic: assumes places have longer names or are geographic
                    if 'Mount' in text or 'River' in text or 'Sea' in text:
                        entities['place'].append(clean_word)
                    else:
                        entities['person'].append(clean_word)
            
            # Check pattern
            elif re.match(self.name_patterns['person'], clean_word):
                entities['uncertain'].append(clean_word)
        
        return entities
    
    def translate_entity(self, entity: str, entity_type: str, target_lang: str) -> str:
        """
        Decide whether to translate or preserve entity.
        
        Biblical proper nouns are typically NOT translated.
        """
        if entity in self.BIBLICAL_NAMES:
            # Keep most biblical names untranslated
            # But some titles may be translated
            if entity_type == 'divine' and entity in ['Holy', 'Almighty']:
                return None  # Signal to translate
            return entity  # Keep as is
        
        return None  # Signal to translate normally


class SemanticAnalyzer:
    """
    Analyze semantic relationships and word meaning.
    
    Helps ensure:
    - Synonyms are used appropriately
    - Antonyms are avoided
    - Metaphors are preserved
    - Theological concepts are accurately conveyed
    """
    
    # Semantic relationships for key theological terms
    THEOLOGICAL_SYNONYMS = {
        "salvation": ["redemption", "deliverance", "liberation"],
        "grace": ["mercy", "kindness", "favor"],
        "faith": ["belief", "trust", "confidence"],
        "love": ["charity", "affection", "devotion"],
        "righteous": ["just", "good", "holy"],
        "sin": ["transgression", "wickedness", "iniquity"],
        "kingdom": ["realm", "dominion", "sovereignty"],
    }
    
    THEOLOGICAL_ANTONYMS = {
        "salvation": ["damnation", "destruction", "perdition"],
        "grace": ["judgment", "wrath", "condemnation"],
        "light": ["darkness"],
        "good": ["evil", "wicked"],
        "life": ["death"],
    }
    
    # Metaphorical mappings
    METAPHORS = {
        "shepherd": "guide/protector",
        "light": "truth/guidance",
        "water": "spiritual refreshment",
        "bread": "sustenance/life",
        "fire": "judgment/purification",
        "rock": "foundation/strength",
        "vine": "relationship/covenant",
    }
    
    def __init__(self):
        self.synonyms = self.THEOLOGICAL_SYNONYMS
        self.antonyms = self.THEOLOGICAL_ANTONYMS
        self.metaphors = self.METAPHORS
    
    def check_semantic_consistency(self, term1: str, term2: str) -> bool:
        """
        Check if two terms are semantically compatible.
        
        Returns True if they can be used interchangeably.
        """
        term1_lower = term1.lower()
        term2_lower = term2.lower()
        
        for base, syns in self.synonyms.items():
            if term1_lower in syns and term2_lower in syns:
                return True
            if term1_lower == base and term2_lower in syns:
                return True
        
        return False
    
    def detect_metaphor(self, text: str) -> List[Tuple[str, str]]:
        """
        Detect metaphors and their meanings.
        
        Returns list of (metaphor, meaning) tuples
        """
        found = []
        
        text_lower = text.lower()
        for metaphor, meaning in self.metaphors.items():
            if metaphor in text_lower:
                found.append((metaphor, meaning))
        
        return found
    
    def validate_semantic_accuracy(self, source: str, translation: str,
                                  term_mapping: Dict[str, str]) -> Dict[str, bool]:
        """
        Validate that translation preserves semantic meaning.
        
        Checks:
        - Key terms are translated consistently
        - No semantic contradictions
        - Metaphors are preserved
        """
        validation = {
            'consistent': True,
            'contradictory': False,
            'metaphors_preserved': True,
            'issues': [],
        }
        
        # Check metaphors in source are in translation
        source_metaphors = self.detect_metaphor(source)
        translation_metaphors = self.detect_metaphor(translation)
        
        if len(source_metaphors) > len(translation_metaphors):
            validation['metaphors_preserved'] = False
            validation['issues'].append("Some metaphors may be lost")
        
        return validation


class LanguageSpecificRules:
    """
    Language-specific grammatical and stylistic rules.
    
    Helps ensure translations sound natural in target language.
    """
    
    def __init__(self, language_code: str):
        self.language_code = language_code
        self.rules = self._load_rules(language_code)
    
    def _load_rules(self, language_code: str) -> Dict:
        """Load language-specific rules"""
        rules = {
            'spa_Latn': self._spanish_rules(),
            'swh_Latn': self._swahili_rules(),
            'fra_Latn': self._french_rules(),
            'por_Latn': self._portuguese_rules(),
        }
        return rules.get(language_code, {})
    
    def _spanish_rules(self) -> Dict:
        return {
            'gender_agreement': True,
            'number_agreement': True,
            'verb_conjugation': True,
            'article_usage': True,
            'preposition_rules': {
                'a': 'movement/direction',
                'de': 'possession/origin',
                'en': 'location',
                'por': 'causation/agent',
            }
        }
    
    def _swahili_rules(self) -> Dict:
        return {
            'noun_class_agreement': True,
            'verb_tense_aspect': True,
            'subject_agreement': True,
            'noun_classes': {
                'm-mi': 'people/plants',
                'ki-vi': 'things',
                'li-ma': 'abstract',
                'u-zu': 'abstract/mass',
            }
        }
    
    def _french_rules(self) -> Dict:
        return {
            'gender_agreement': True,
            'number_agreement': True,
            'verb_conjugation': True,
            'accent_marks': True,
        }
    
    def _portuguese_rules(self) -> Dict:
        return {
            'gender_agreement': True,
            'number_agreement': True,
            'verb_conjugation': True,
            'accent_rules': True,
        }
    
    def apply_rules(self, text: str) -> Tuple[str, List[str]]:
        """
        Apply language-specific rules to text.

        TODO: Full implementation requires:
        - spaCy model for target language (python -m spacy download es_core_news_sm)
        - POS tagging and dependency parsing for grammatical agreement checks
        - Morphological analyzer for verb conjugation and agreement validation
        - Custom rule engine for language-specific phonological rules

        Returns:
            (corrected_text, issues_found)
        """
        issues = []
        corrected = text

        # Stub implementation
        if not self.rules:
            return corrected, ["No rules for this language"]

        return corrected, issues


class MorphologicalAnalyzer:
    """
    Analyze word forms and morphological structure.
    
    Helps with:
    - Agglutinative languages (Turkish, Finnish, Swahili)
    - Inflected languages (Spanish, French, German)
    - Complex verb systems
    """
    
    def __init__(self, language_code: str):
        self.language_code = language_code
        self.morphology = self._load_morphology(language_code)
    
    def _load_morphology(self, language_code: str) -> Dict:
        """Load morphological patterns for language"""
        patterns = {
            'swh_Latn': {
                'agglutinative': True,
                'verb_affixes': {
                    'prefix': ['ni-', 'a-', 'u-', 'wa-'],
                    'suffix': ['-a', '-i', '-u', '-e'],
                },
                'noun_classes': 8,
            },
            'tur_Latn': {
                'agglutinative': True,
                'vowel_harmony': True,
                'cases': ['nominative', 'genitive', 'dative', 'accusative'],
            },
            'spa_Latn': {
                'inflected': True,
                'verb_types': ['regular', 'irregular'],
                'genders': 2,  # masculine, feminine
            },
        }
        return patterns.get(language_code, {})
    
    def analyze_morphology(self, word: str) -> Dict:
        """Analyze morphological structure of word"""
        analysis = {
            'word': word,
            'language': self.language_code,
            'structure': None,
            'morphemes': [],
        }
        
        if self.morphology.get('agglutinative'):
            # For agglutinative languages, identify morphemes
            analysis['morphemes'] = self._extract_morphemes(word)
        
        return analysis
    
    def _extract_morphemes(self, word: str) -> List[str]:
        """Extract morphemes from word.

        TODO: Full implementation requires:
        - Finite-state morphological transducers (FOMA, HFST)
        - Language-specific morphological rules and lexica
        - Tools like APERTIUM or XFST for morphological analysis
        - Example: Swahili 'waliona' -> 'wa' (class prefix) + 'li' (past tense) + 'on' (root) + 'a' (final vowel)
        """
        # Stub: At minimum, the word itself
        morphemes = [word]
        return morphemes


class DiscourseAnalyzer:
    """
    Analyze discourse structure and coherence.
    
    Helps maintain:
    - Logical flow across verses
    - Thematic consistency
    - Narrative structure
    - Dialogue attribution
    """
    
    def analyze_verse_structure(self, verse: str) -> Dict:
        """
        Analyze the structure of a verse.
        
        Returns info about:
        - Main clause
        - Subordinate clauses
        - Direct speech
        - Parallelism
        """
        structure = {
            'text': verse,
            'main_clause': None,
            'subordinate_clauses': [],
            'direct_speech': [],
            'parallelism': False,
        }
        
        # Detect direct speech (within quotes)
        speech_pattern = r'"([^"]*)"'
        structure['direct_speech'] = re.findall(speech_pattern, verse)
        
        # Detect parallelism (repeated structures)
        if verse.count(',') >= 2:
            structure['parallelism'] = True
        
        return structure
    
    def check_discourse_coherence(self, verses: List[str]) -> List[Dict]:
        """
        Check coherence across multiple verses.
        
        Returns issues found
        """
        issues = []
        
        for i in range(len(verses) - 1):
            current = verses[i]
            next_verse = verses[i + 1]
            
            # Simple heuristic: check for pronouns without clear antecedent
            if 'he' in next_verse.lower() and 'he' not in current.lower():
                if 'Jesus' not in current and 'God' not in current:
                    issues.append({
                        'verse': i + 1,
                        'type': 'ambiguous_pronoun',
                        'description': 'Pronoun "he" may lack clear antecedent',
                    })
        
        return issues


if __name__ == "__main__":
    from utils.logger import configure_logging

    configure_logging()

    # Test NER
    ner = BiblicalNER()
    text = "Jesus went to Jerusalem with his disciples."
    entities = ner.extract_entities(text)
    logger.info(f"Entities: {entities}")

    # Test semantic analysis
    analyzer = SemanticAnalyzer()
    metaphors = analyzer.detect_metaphor("The Lord is my shepherd")
    logger.info(f"Metaphors: {metaphors}")

    # Test language rules
    rules = LanguageSpecificRules('spa_Latn')
    logger.info(f"Spanish rules: {rules.rules}")

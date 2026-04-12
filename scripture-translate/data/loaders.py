import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BibleVerse:
    """Represents a single Bible verse with metadata"""
    
    def __init__(self, book: str, chapter: int, verse: int, text: str, language: str = "en"):
        self.book = book
        self.chapter = chapter
        self.verse = verse
        self.text = text
        self.language = language
    
    def reference(self) -> str:
        """Return standard Bible reference (e.g., 'Genesis 1:1')"""
        return f"{self.book} {self.chapter}:{self.verse}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "book": self.book,
            "chapter": self.chapter,
            "verse": self.verse,
            "text": self.text,
            "language": self.language,
            "reference": self.reference(),
        }


class BibleDataLoader:
    """Load and manage Bible verse data from various sources"""
    
    BOOKS_OT = [
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
        "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
        "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
        "Ezra", "Nehemiah", "Esther", "Job", "Psalm", "Proverbs",
        "Ecclesiastes", "Isaiah", "Jeremiah", "Lamentations",
        "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah",
        "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
        "Haggai", "Zechariah", "Malachi"
    ]
    
    BOOKS_NT = [
        "Matthew", "Mark", "Luke", "John", "Acts",
        "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
        "Ephesians", "Philippians", "Colossians",
        "1 Thessalonians", "2 Thessalonians",
        "1 Timothy", "2 Timothy", "Titus", "Philemon",
        "Hebrews", "James", "1 Peter", "2 Peter",
        "1 John", "2 John", "3 John", "Jude", "Revelation"
    ]
    
    ALL_BOOKS = BOOKS_OT + BOOKS_NT
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.verses: Dict[str, List[BibleVerse]] = {}
    
    def load_from_json(self, json_path: Path, language: str) -> int:
        """
        Load verses from JSON file.
        Expected format: [{"book": "Genesis", "chapter": 1, "verse": 1, "text": "..."}, ...]
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        loaded = 0
        for item in data:
            try:
                verse = BibleVerse(
                    book=item["book"],
                    chapter=int(item["chapter"]),
                    verse=int(item["verse"]),
                    text=item["text"],
                    language=language
                )
                
                if language not in self.verses:
                    self.verses[language] = []
                
                self.verses[language].append(verse)
                loaded += 1
            except KeyError as e:
                logger.warning(f"Missing key in verse: {e}")
                continue
        
        logger.info(f"Loaded {loaded} verses in {language}")
        return loaded
    
    def load_from_csv(self, csv_path: Path, language: str, 
                     book_col: str = "book", chapter_col: str = "chapter",
                     verse_col: str = "verse", text_col: str = "text") -> int:
        """Load verses from CSV file"""
        df = pd.read_csv(csv_path)
        loaded = 0
        
        for idx, row in df.iterrows():
            try:
                verse = BibleVerse(
                    book=row[book_col],
                    chapter=int(row[chapter_col]),
                    verse=int(row[verse_col]),
                    text=row[text_col],
                    language=language
                )
                
                if language not in self.verses:
                    self.verses[language] = []
                
                self.verses[language].append(verse)
                loaded += 1
            except (KeyError, ValueError) as e:
                logger.warning(f"Error loading row {idx}: {e}")
                continue
        
        logger.info(f"Loaded {loaded} verses in {language}")
        return loaded
    
    def create_parallel_corpus(self, source_lang: str, target_lang: str,
                              alignment_file: Optional[Path] = None) -> Tuple[List[str], List[str]]:
        """
        Create parallel corpus (source sentences, target sentences).
        
        If alignment_file provided, use it to match verses.
        Otherwise, assumes verses are in same order (not recommended).
        """
        source_verses = self.verses.get(source_lang, [])
        target_verses = self.verses.get(target_lang, [])
        
        if not source_verses or not target_verses:
            raise ValueError(f"Missing verses for {source_lang} or {target_lang}")
        
        # Create lookup by reference
        source_dict = {v.reference(): v.text for v in source_verses}
        target_dict = {v.reference(): v.text for v in target_verses}
        
        # Match by reference
        sources, targets = [], []
        for ref in source_dict:
            if ref in target_dict:
                sources.append(source_dict[ref])
                targets.append(target_dict[ref])
        
        logger.info(f"Created parallel corpus with {len(sources)} aligned verse pairs")
        return sources, targets
    
    def get_verses_by_book(self, language: str, book: str) -> List[BibleVerse]:
        """Get all verses from a specific book"""
        return [v for v in self.verses.get(language, []) if v.book == book]
    
    def get_verses_by_range(self, language: str, book: str, 
                           start_chapter: int, start_verse: int,
                           end_chapter: int, end_verse: int) -> List[BibleVerse]:
        """Get verses in a specific range"""
        verses = self.get_verses_by_book(language, book)
        return [
            v for v in verses
            if (v.chapter > start_chapter or 
                (v.chapter == start_chapter and v.verse >= start_verse)) and
               (v.chapter < end_chapter or 
                (v.chapter == end_chapter and v.verse <= end_verse))
        ]
    
    def save_parallel_corpus(self, source_lang: str, target_lang: str,
                            output_path: Path, format: str = "jsonl"):
        """Save parallel corpus to file"""
        sources, targets = self.create_parallel_corpus(source_lang, target_lang)
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for source, target in zip(sources, targets):
                    item = {
                        "source": source,
                        "target": target,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        elif format == "csv":
            df = pd.DataFrame({
                "source": sources,
                "target": targets,
                "source_lang": [source_lang] * len(sources),
                "target_lang": [target_lang] * len(sources),
            })
            df.to_csv(output_path, index=False)
        
        logger.info(f"Saved parallel corpus to {output_path}")


class BibleTranslationDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for Bible translation pairs"""
    
    def __init__(self, jsonl_path: Path, tokenizer, 
                 source_lang: str, target_lang: str,
                 max_source_length: int = 512,
                 max_target_length: int = 256):
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} training examples from {jsonl_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Encode source
        source_encoding = self.tokenizer(
            item["source"],
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode target
        target_encoding = self.tokenizer(
            item["target"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
            "decoder_attention_mask": target_encoding["attention_mask"].squeeze(),
        }


def create_data_splits(jsonl_path: Path, train_ratio: float = 0.9,
                      val_ratio: float = 0.05) -> Tuple[Path, Path, Path]:
    """Split data into train, validation, test sets"""
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    import random
    random.shuffle(data)
    
    n = len(data)
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    
    train_data = data[:train_n]
    val_data = data[train_n:train_n + val_n]
    test_data = data[train_n + val_n:]
    
    data_dir = jsonl_path.parent
    
    def save_split(data, name):
        path = data_dir / f"{jsonl_path.stem}_{name}.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(data)} {name} examples to {path}")
        return path
    
    train_path = save_split(train_data, "train")
    val_path = save_split(val_data, "val")
    test_path = save_split(test_data, "test")
    
    return train_path, val_path, test_path


if __name__ == "__main__":
    # Example usage
    loader = BibleDataLoader()
    print(f"Available books: {len(loader.ALL_BOOKS)}")
    print(f"Sample books: {loader.ALL_BOOKS[:5]}")

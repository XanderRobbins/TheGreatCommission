#!/usr/bin/env python3
"""
Translate entire Bible books in batch.

Usage:
    python scripts/translate_book.py \
        --model_path ./models/checkpoints/final_model \
        --input_path ./data/genesis.json \
        --output_path ./results/genesis_spanish.json \
        --target_lang spa_Latn
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm

from config import Config
from models.base import ScriptureTranslationModel
from models.terminology import TerminologyDB
from inference import ScriptureTranslator
from evaluation import ScriptureEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / f"translate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BookTranslationPipeline:
    """
    End-to-end pipeline for translating entire Bible books.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = Config.get_device()
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Source language: {args.source_lang}")
        logger.info(f"Target language: {args.target_lang}")
    
    def load_verses(self, input_path: Path) -> list:
        """Load verses from JSON file"""
        logger.info(f"Loading verses from {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            verses = json.load(f)
        
        logger.info(f"Loaded {len(verses)} verses")
        return verses
    
    def load_model(self) -> ScriptureTranslationModel:
        """Load pretrained model"""
        logger.info(f"Loading model from {self.args.model_path}")
        
        model_wrapper = ScriptureTranslationModel(use_lora=False)
        if self.args.model_path:
            model_wrapper.load_pretrained(Path(self.args.model_path))
        
        return model_wrapper
    
    def setup_translator(self, model_wrapper: ScriptureTranslationModel,
                        terminology_db: TerminologyDB) -> ScriptureTranslator:
        """Setup translator with model and terminology database"""
        logger.info("Setting up translator...")
        
        translator = ScriptureTranslator(
            model=model_wrapper.get_model(),
            tokenizer=model_wrapper.get_tokenizer(),
            terminology_db=terminology_db,
            device=self.device,
            enforce_consistency=True,
        )
        
        return translator
    
    def translate_book(self, translator: ScriptureTranslator,
                      verses: list) -> list:
        """Translate all verses in a book"""
        logger.info(f"Translating {len(verses)} verses...")
        
        results = []
        
        for i, verse in enumerate(tqdm(verses, desc="Translating")):
            try:
                result = translator.translate_verse(
                    source_text=verse.get("text", ""),
                    source_lang=self.args.source_lang,
                    target_lang=self.args.target_lang,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                )
                
                results.append({
                    "reference": f"{verse.get('book', 'Unknown')} {verse.get('chapter', 1)}:{verse.get('verse', i+1)}",
                    "source": result.source_text,
                    "translation": result.primary,
                    "confidence": result.confidence,
                    "alternatives": result.alternatives or [],
                    "theological_terms": result.theological_terms or {},
                })
            
            except Exception as e:
                logger.error(f"Error translating verse {i}: {e}")
                results.append({
                    "reference": f"{verse.get('book', 'Unknown')} {verse.get('chapter', 1)}:{verse.get('verse', i+1)}",
                    "error": str(e),
                })
        
        logger.info(f"Successfully translated {len([r for r in results if 'translation' in r])}/{len(verses)} verses")
        
        return results
    
    def evaluate_translations(self, results: list, reference_data: list = None) -> dict:
        """Evaluate translation quality"""
        logger.info("Evaluating translations...")
        
        evaluator = ScriptureEvaluator()
        
        # Extract translations
        translations = [r.get("translation", "") for r in results if "translation" in r]
        
        stats = {
            "total_verses": len(results),
            "successfully_translated": len(translations),
            "failed": len(results) - len(translations),
            "average_confidence": sum(r.get("confidence", 0) for r in results if "confidence" in r) / len(translations) if translations else 0,
        }
        
        # If we have reference data, compute BLEU
        if reference_data:
            hypotheses = translations
            references = [r.get("translation", "") for r in reference_data][:len(hypotheses)]
            
            if hypotheses and references:
                metrics = evaluator.evaluate_batch(hypotheses, references, self.args.target_lang)
                stats["metrics"] = metrics.to_dict()
        
        return stats
    
    def save_results(self, results: list, stats: dict, output_path: Path):
        """Save translation results to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "metadata": {
                "source_language": self.args.source_lang,
                "target_language": self.args.target_lang,
                "translation_model": Config.MODEL_NAME,
                "timestamp": datetime.now().isoformat(),
                "statistics": stats,
            },
            "verses": results,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved results to {output_path}")
    
    def run(self):
        """Execute translation pipeline"""
        logger.info("Starting book translation pipeline...")
        
        # Load verses
        verses = self.load_verses(Path(self.args.input_path))
        
        # Load model
        model_wrapper = self.load_model()
        
        # Load terminology database
        terminology_db = TerminologyDB()
        
        # Setup translator
        translator = self.setup_translator(model_wrapper, terminology_db)
        
        # Translate book
        results = self.translate_book(translator, verses)
        
        # Evaluate (if reference data provided)
        reference_data = None
        if self.args.reference_path:
            reference_data = self.load_verses(Path(self.args.reference_path))
        
        stats = self.evaluate_translations(results, reference_data)
        
        # Save results
        self.save_results(results, stats, Path(self.args.output_path))
        
        # Print summary
        print("\n" + "="*60)
        print("TRANSLATION COMPLETE")
        print("="*60)
        print(f"Total verses: {stats['total_verses']}")
        print(f"Successfully translated: {stats['successfully_translated']}")
        print(f"Failed: {stats['failed']}")
        print(f"Average confidence: {stats['average_confidence']:.2%}")
        
        if "metrics" in stats:
            print(f"\nBLEU Scores:")
            print(f"  BLEU-1: {stats['metrics']['bleu_1']:.4f}")
            print(f"  BLEU-2: {stats['metrics']['bleu_2']:.4f}")
            print(f"  BLEU-4: {stats['metrics']['bleu_4']:.4f}")
        
        print("="*60)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Translate entire Bible book")
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input verses JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save translated verses JSON file"
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        help="Path to reference translations for evaluation"
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default="eng_Latn",
        help="Source language code"
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        help="Target language code"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=Config.INFERENCE_CONFIG["num_beams"],
        help="Number of beams for beam search"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=Config.INFERENCE_CONFIG["max_length"],
        help="Maximum output length"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = BookTranslationPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()

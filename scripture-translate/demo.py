#!/usr/bin/env python3
"""
Complete Scripture Translation System Demo

This script demonstrates the full pipeline:
1. Generate sample data
2. Initialize the model
3. Train on baseline task (optional, for demo uses pretrained)
4. Translate verses
5. Evaluate translations
6. Manage terminology consistency
"""

import sys
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from data.loaders import BibleDataLoader, BibleVerse
from data.generate_sample_data import generate_sample_data, create_test_dataset
from models.base import ScriptureTranslationModel
from models.terminology import TerminologyDB, TermExtractor
from inference import ScriptureTranslator
from evaluation import ScriptureEvaluator, EvaluationMetrics


def demo_data_loading():
    """Demo: Load and manage Bible verse data"""
    logger.info("DEMO 1: Data Loading & Management")
    
    # Generate sample data
    logger.info("Generating sample data...")
    data_dir = generate_sample_data(Path("./data"))
    
    # Load verses
    logger.info("Loading verses...")
    loader = BibleDataLoader(data_dir)
    loader.load_from_json(data_dir / "en_verses.json", "eng_Latn")
    loader.load_from_json(data_dir / "es_verses.json", "spa_Latn")
    
    # Create parallel corpus
    logger.info("Creating parallel corpus...")
    sources, targets = loader.create_parallel_corpus("eng_Latn", "spa_Latn")
    
    logger.info(f"Loaded {len(sources)} aligned verse pairs")
    
    # Display sample
    print("\nSample verse pairs:")
    for i in range(min(3, len(sources))):
        print(f"\n  [{i+1}]")
        print(f"    EN: {sources[i]}")
        print(f"    ES: {targets[i]}")
    
    return data_dir


def demo_terminology_management():
    """Demo: Build and manage terminology database"""
    logger.info("DEMO 2: Terminology Database")
    
    # Create terminology database
    db = TerminologyDB()
    
    # Add sample theological terms
    logger.info("Adding theological terms...")
    terms_to_add = [
        ("salvation", "spa_Latn", "salvación", 0.98),
        ("grace", "spa_Latn", "gracia", 0.97),
        ("faith", "spa_Latn", "fe", 0.96),
        ("kingdom", "spa_Latn", "reino", 0.95),
        ("sin", "spa_Latn", "pecado", 0.98),
    ]
    
    for en_term, lang, target_term, conf in terms_to_add:
        db.add_term(en_term, lang, target_term, confidence=conf)
    
    # Test extraction
    logger.info("Testing term extraction...")
    extractor = TermExtractor(db)
    sample_text = "God's grace and salvation bring faith in the kingdom."
    extracted = extractor.extract_theological_terms(sample_text)
    
    print(f"\nSample text: {sample_text}")
    print(f"Extracted terms: {extracted}")
    
    # Get canonical translations
    canonical = extractor.get_canonical_terms(sample_text, "spa_Latn")
    print(f"Canonical translations:")
    for en, es in canonical.items():
        print(f"  {en} → {es}")
    
    # Show statistics
    db.print_statistics()
    
    return db


def demo_model_initialization():
    """Demo: Initialize and setup the translation model"""
    logger.info("DEMO 3: Model Initialization")
    
    logger.info("Loading NLLB model...")
    logger.info(f"Model: {Config.MODEL_NAME}")
    logger.info(f"Device: {Config.get_device()}")
    
    # Initialize model (this will download the model if needed)
    model_wrapper = ScriptureTranslationModel(use_lora=False)
    
    print(f"\nModel Information:")
    print(f"  Architecture: {type(model_wrapper.get_model()).__name__}")
    print(f"  Tokenizer vocab size: {len(model_wrapper.get_tokenizer())}")
    print(f"  Total parameters: {model_wrapper.count_parameters():,}")
    print(f"  Device: {model_wrapper.device}")
    
    # Show supported languages
    print(f"\nSupported languages: {len(Config.LANGUAGE_CODES)}")
    print(f"Sample languages:")
    for lang, code in list(Config.LANGUAGE_CODES.items())[:5]:
        print(f"  {lang}: {code}")
    
    return model_wrapper


def demo_inference():
    """Demo: Translate verses using the model"""
    logger.info("DEMO 4: Inference & Translation")
    
    # Load model
    logger.info("Loading model...")
    model_wrapper = ScriptureTranslationModel(use_lora=False)
    
    # Setup terminology database
    terminology_db = TerminologyDB()
    
    # Add some terms
    terminology_db.add_term("salvation", "spa_Latn", "salvación")
    terminology_db.add_term("god", "spa_Latn", "Dios")
    
    # Create translator
    logger.info("Initializing translator...")
    translator = ScriptureTranslator(
        model=model_wrapper.get_model(),
        tokenizer=model_wrapper.get_tokenizer(),
        terminology_db=terminology_db,
        device=Config.get_device(),
        enforce_consistency=True,
    )
    
    # Sample verses to translate
    test_verses = [
        "In the beginning, God created the heavens and the earth.",
        "The Lord is my shepherd; I shall not want.",
        "For God so loved the world, that he gave his only Son.",
    ]
    
    print("\nTranslating sample verses:")
    
    for i, verse_text in enumerate(test_verses, 1):
        logger.info(f"Translating verse {i}...")
        
        result = translator.translate_verse(
            source_text=verse_text,
            source_lang="eng_Latn",
            target_lang="spa_Latn",
            num_beams=3,
        )
        
        print(f"\n[Verse {i}]")
        print(f"  EN: {verse_text}")
        print(f"  ES: {result.primary}")
        print(f"  Confidence: {result.confidence:.2%}")
        
        if result.theological_terms:
            print(f"  Theological terms: {list(result.theological_terms.keys())}")
    
    return translator


def demo_evaluation():
    """Demo: Evaluate translations"""
    logger.info("DEMO 5: Evaluation Metrics")
    
    # Create evaluator
    evaluator = ScriptureEvaluator()
    
    # Sample translations and references
    test_pairs = [
        {
            "hypothesis": "En el principio, Dios creó los cielos y la tierra.",
            "reference": "En el principio creó Dios los cielos y la tierra.",
        },
        {
            "hypothesis": "El Señor es mi pastor; no me faltará nada.",
            "reference": "Jehová es mi pastor; nada me faltará.",
        },
        {
            "hypothesis": "Porque Dios amó el mundo de tal manera, que dio su único Hijo.",
            "reference": "Porque de tal manera amó Dios al mundo, que ha dado a su Hijo unigénito.",
        },
    ]
    
    logger.info("Computing BLEU scores...")
    
    hypotheses = [pair["hypothesis"] for pair in test_pairs]
    references = [pair["reference"] for pair in test_pairs]
    
    # Evaluate
    metrics = evaluator.evaluate_batch(hypotheses, references, "spa_Latn")
    
    # Print results
    evaluator.print_metrics(metrics, "Spanish Translation Evaluation")
    
    return metrics


def demo_pipeline_summary():
    """Print summary of the complete pipeline"""
    summary = """
Pipeline summary:
  1. data/loaders.py       - BibleDataLoader, parallel corpus creation
  2. models/terminology.py - TerminologyDB, TermExtractor, conflict resolution
  3. models/base.py        - ScriptureTranslationModel, LoRA, ConsistencyLoss
  4. inference/            - ScriptureTranslator, batch translation
  5. evaluation/           - BLEU, consistency score, terminology metrics
  6. scripts/              - train_baseline.py, fine_tune_lora.py

See README.md for setup and usage.
"""
    print(summary)


def main():
    """Run all demos"""
    logger.info("Scripture Translation System - Demo")

    try:
        # Demo 1: Data Loading
        data_dir = demo_data_loading()
        
        # Demo 2: Terminology
        terminology_db = demo_terminology_management()
        
        # Demo 3: Model
        model_wrapper = demo_model_initialization()
        
        # Demo 4: Inference
        # Note: Inference demo requires downloading full model (~2GB)
        # For quick demo, we'll skip actual translation
        logger.info("\n[Skipping inference demo to save time/space]")
        logger.info("In production, use: translator = demo_inference()")
        
        # Demo 5: Evaluation
        metrics = demo_evaluation()
        
        # Summary
        demo_pipeline_summary()

        logger.info("Demo complete")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

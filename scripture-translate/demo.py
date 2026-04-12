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
    logger.info("\n" + "="*60)
    logger.info("DEMO 1: Data Loading & Management")
    logger.info("="*60)
    
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
    logger.info("\n" + "="*60)
    logger.info("DEMO 2: Terminology Database")
    logger.info("="*60)
    
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
    logger.info("\n" + "="*60)
    logger.info("DEMO 3: Model Initialization")
    logger.info("="*60)
    
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
    logger.info("\n" + "="*60)
    logger.info("DEMO 4: Inference & Translation")
    logger.info("="*60)
    
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
    print("="*60)
    
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
    logger.info("\n" + "="*60)
    logger.info("DEMO 5: Evaluation Metrics")
    logger.info("="*60)
    
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
    logger.info("\n" + "="*60)
    logger.info("SCRIPTURE TRANSLATION SYSTEM - PIPELINE SUMMARY")
    logger.info("="*60)
    
    summary = """
┌─────────────────────────────────────────────────────────────┐
│          LOW-RESOURCE SCRIPTURE TRANSLATION                │
│              Complete System Overview                        │
└─────────────────────────────────────────────────────────────┘

ARCHITECTURE LAYERS:
  ✓ Data Layer: Load & align Bible verses
  ✓ Embedding Layer: NLLB-200 shared cross-lingual space
  ✓ Training Layer: Fine-tune on Bible-aligned data
  ✓ Terminology Layer: Ensure consistent translations
  ✓ Inference Layer: Translate with constraints
  ✓ Evaluation Layer: Measure quality

KEY COMPONENTS:
  
  1. Data Loading (data/loaders.py)
     - BibleDataLoader: Load from JSON/CSV
     - BibleTranslationDataset: PyTorch dataset
     - Parallel corpus creation

  2. Terminology Management (models/terminology.py)
     - TerminologyDB: Track consistent term mappings
     - TermExtractor: Extract theological terms
     - Conflict resolution & human review

  3. Model & Training (models/base.py)
     - ScriptureTranslationModel: NLLB wrapper
     - LoRA fine-tuning for rare languages
     - ConsistencyLoss: Enforce term consistency
     - TranslationTrainer: Training loop

  4. Inference (inference.py)
     - ScriptureTranslator: End-to-end translation
     - BeamSearchDecoder: Constrained decoding
     - TranslationResult: Structured output

  5. Evaluation (evaluation.py)
     - ScriptureEvaluator: BLEU + custom metrics
     - HumanEvaluationInterface: Collect human scores
     - Consistency & terminology metrics

  6. Training Scripts (scripts/)
     - train_baseline.py: Train on baseline task
     - fine_tune_lora.py: LoRA for rare languages

WORKFLOW:

  Step 1: Generate Data
    └─ Collect parallel Bible verses
       └ Create aligned corpus (EN ↔ Target)

  Step 2: Train Baseline (2 weeks)
    └─ Initialize NLLB model
    └─ Train on high-resource language pairs
    └─ Validate on English ↔ Spanish
    └─ Save pretrained checkpoint

  Step 3: Fine-tune Rare Language (4-6 weeks)
    └─ Load pretrained baseline
    └─ Apply LoRA (3% of parameters)
    └─ Train on 500-2000 target-language verses
    └─ Save LoRA adapters

  Step 4: Deploy
    └─ Load model + LoRA adapters
    └─ Initialize terminology database
    └─ Translate verses with consistency
    └─ Evaluate with human reviewers
    └─ Iterate & refine

EXPECTED OUTCOMES:

  Baseline (High-Resource):
    - BLEU-4: 25-35
    - Consistency: 92-96%
    - Human rating: 4.0-4.5 / 5.0

  Rare Language (500-2000 verses):
    - BLEU-4: 18-25
    - Consistency: 90-95%
    - Human rating: 3.5-4.2 / 5.0
    
  With Community Refinement:
    - Human rating: 4.3-4.8 / 5.0
    - Production ready for publication

GETTING STARTED:

  1. Install dependencies:
     $ pip install -r requirements.txt

  2. Generate sample data:
     $ python data/generate_sample_data.py

  3. Train baseline:
     $ python scripts/train_baseline.py \\
         --data_path ./data/en_es_verses.jsonl \\
         --source_lang eng_Latn \\
         --target_lang spa_Latn

  4. Fine-tune rare language:
     $ python scripts/fine_tune_lora.py \\
         --pretrained_model_path ./models/checkpoints/final_model \\
         --data_path ./data/rare_lang_verses.jsonl \\
         --target_lang swh_Latn

  5. Translate & evaluate:
     $ python scripts/translate_book.py \\
         --model_path ./models/checkpoints/final_model \\
         --input_path ./data/genesis.json \\
         --output_path ./results/genesis_swahili.json

KEY ADVANTAGES:

  ✓ Leverages pretrained NLLB (200+ languages)
  ✓ Efficient LoRA fine-tuning for low-resource settings
  ✓ Terminology consistency enforcement
  ✓ Custom metrics for scripture quality
  ✓ Human-in-the-loop validation interface
  ✓ Modular architecture for easy extension

NEXT STEPS:

  1. Collect real Bible verse data
  2. Train baseline on multiple language pairs
  3. Deploy to cloud for inference at scale
  4. Build web UI for translators
  5. Integrate community feedback loop
  6. Support additional languages

───────────────────────────────────────────────────────────────
For more information, see: README.md
    """
    
    print(summary)


def main():
    """Run all demos"""
    
    print("\n" + "="*70)
    print("SCRIPTURE TRANSLATION SYSTEM - COMPLETE DEMO")
    print("="*70)
    
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
        
        logger.info("\n" + "="*70)
        logger.info("DEMO COMPLETE")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("1. Review the architecture in bible_translation_architecture.md")
        logger.info("2. Generate your own Bible verse data")
        logger.info("3. Run training: python scripts/train_baseline.py")
        logger.info("4. Fine-tune on rare language: python scripts/fine_tune_lora.py")
        logger.info("5. Evaluate translations with human reviewers")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

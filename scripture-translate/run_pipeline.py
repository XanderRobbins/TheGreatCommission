#!/usr/bin/env python3
"""End-to-end Scripture translation pipeline.

Loads Bible → Translates → Saves JSON+CSV → Evaluates → Prints summary stats.

Usage:
    python run_pipeline.py [options]

Examples:
    # Quick test (5 verses)
    python run_pipeline.py --max-verses 5

    # Full Haitian Creole translation
    python run_pipeline.py --target-lang hat_Latn

    # Custom model and batch size
    python run_pipeline.py --model facebook/nllb-200-3.3B --batch-size 4

    # With LoRA adapter
    python run_pipeline.py --lora-path ./models/haitian_creole_lora
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import logging

import pandas as pd
import torch

from config import Config

# 100 representative verses spanning both testaments and major genres.
# Used as the default inference set — covers law, wisdom, prophecy, gospels, epistles.
REPRESENTATIVE_VERSES = {
    # Torah / Law
    "Genesis 1:1", "Genesis 1:27", "Genesis 3:15", "Genesis 12:3", "Genesis 22:18",
    "Exodus 20:3", "Exodus 20:12", "Exodus 20:13",
    "Deuteronomy 6:4", "Deuteronomy 6:5", "Deuteronomy 30:19",
    # Wisdom / Poetry
    "Psalms 1:1", "Psalms 22:1", "Psalms 23:1", "Psalms 23:4",
    "Psalms 46:1", "Psalms 46:10", "Psalms 51:10", "Psalms 91:1",
    "Psalms 103:1", "Psalms 119:105", "Psalms 139:14", "Psalms 145:3",
    "Proverbs 1:7", "Proverbs 3:5", "Proverbs 3:6", "Proverbs 22:6",
    "Job 19:25",
    # Prophets
    "Isaiah 9:6", "Isaiah 40:31", "Isaiah 53:5", "Isaiah 53:6",
    "Isaiah 55:8", "Isaiah 61:1",
    "Jeremiah 29:11", "Jeremiah 31:33",
    "Micah 6:8",
    # Gospels — Matthew
    "Matthew 1:21", "Matthew 5:3", "Matthew 5:6", "Matthew 5:44",
    "Matthew 6:9", "Matthew 6:10", "Matthew 6:11", "Matthew 6:12", "Matthew 6:13",
    "Matthew 6:33", "Matthew 7:7", "Matthew 11:28",
    "Matthew 22:37", "Matthew 22:39", "Matthew 28:19", "Matthew 28:20",
    # Gospels — Mark
    "Mark 10:45", "Mark 16:15",
    # Gospels — Luke
    "Luke 2:11", "Luke 4:18", "Luke 15:7", "Luke 19:10",
    # Gospels — John
    "John 1:1", "John 1:14", "John 3:16", "John 3:17",
    "John 8:32", "John 10:10", "John 14:6", "John 17:17",
    # Acts
    "Acts 1:8", "Acts 2:38", "Acts 4:12",
    # Romans
    "Romans 1:16", "Romans 3:23", "Romans 5:8", "Romans 6:23",
    "Romans 8:1", "Romans 8:28", "Romans 10:9", "Romans 12:1", "Romans 12:2",
    # 1 Corinthians
    "1 Corinthians 13:4", "1 Corinthians 13:13",
    "1 Corinthians 15:3", "1 Corinthians 15:4",
    # Galatians
    "Galatians 2:20", "Galatians 5:22", "Galatians 5:23",
    # Ephesians
    "Ephesians 2:8", "Ephesians 2:9", "Ephesians 4:32", "Ephesians 6:10",
    # 2 Timothy
    "2 Timothy 3:16",
    # Hebrews
    "Hebrews 11:1", "Hebrews 12:1",
    # James
    "James 1:22", "James 2:17",
    # 1 John
    "1 John 4:8",
    # Revelation
    "Revelation 1:8", "Revelation 3:20", "Revelation 21:4", "Revelation 22:20",
}
from data.bible_loader import load_bible
from models.base import ScriptureTranslationModel
from models.terminology import TerminologyDB
from models.tiered_terminology import TieredTerminologyDB
from inference import ScriptureTranslator
from evaluation import ScriptureEvaluator
from utils.logger import get_logger, configure_logging

logger = get_logger(__name__)


def initialize_terminology_db(terminology_db: TerminologyDB, target_lang: str) -> TieredTerminologyDB:
    """Initialize terminology database with tiered terminology system.

    For Haitian Creole, loads the 3-tier glossary (Tier 1: always override,
    Tier 2: context-aware, Tier 3: model decides).

    Args:
        terminology_db: Terminology database to populate.
        target_lang: Target language code (e.g., "hat_Latn").

    Returns:
        TieredTerminologyDB instance for the target language.
    """
    if target_lang == "hat_Latn":
        tiered = TieredTerminologyDB(terminology_db)
        tiered.print_tiers()
        return tiered
    else:
        # For other languages, return empty tiered DB
        logger.warning(f"Tiered terminology not available for {target_lang}; using empty system")
        tiered = TieredTerminologyDB(terminology_db)
        return tiered


def create_verse_dicts(bible_data: list) -> list:
    """Convert Bible data to verse dictionaries.

    Args:
        bible_data: List of dicts with "reference" and "text" keys.

    Returns:
        List of verse dicts with "reference" and "text" keys.
    """
    verses = []
    for item in bible_data:
        verses.append({
            "reference": item.get("reference", "Unknown"),
            "text": item.get("text", ""),
        })
    return verses


def main():
    """Run the full pipeline."""
    parser = argparse.ArgumentParser(
        description="Scripture translation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --max-verses 5
  python run_pipeline.py --target-lang hat_Latn
  python run_pipeline.py --model facebook/nllb-200-3.3B --batch-size 4
        """,
    )

    parser.add_argument(
        "--source-lang",
        default="eng_Latn",
        help="Source language code (default: eng_Latn)",
    )
    parser.add_argument(
        "--target-lang",
        default="plt_Latn",
        help="Target language code (default: plt_Latn)",
    )
    parser.add_argument(
        "--model",
        default=Config.MODEL_NAME,
        help=f"HuggingFace model name (default: {Config.MODEL_NAME})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for translation (default: 8)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for beam search (default: 4)",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help="Path to LoRA adapter (optional)",
    )
    parser.add_argument(
        "--max-verses",
        type=int,
        default=None,
        help="Maximum number of verses to translate (for testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for results (default: ./output)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device to use (auto-detect if not specified)",
    )
    parser.add_argument(
        "--full-bible",
        action="store_true",
        help="Translate the entire Bible (default: 100 representative verses)",
    )

    args = parser.parse_args()

    # Setup
    configure_logging()
    Config.ensure_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or Config.get_device()
    logger.info(f"\n{'='*60}")
    logger.info("Scripture Translation Pipeline")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {device}")
    logger.info(f"Source language: {args.source_lang}")
    logger.info(f"Target language: {args.target_lang}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Step 1: Load Bible
        logger.info("\n[Step 1/5] Loading Bible...")
        bible_data = load_bible(Config.DATA_DIR)
        verses = create_verse_dicts(bible_data)

        if args.max_verses:
            verses = verses[:args.max_verses]
            logger.info(f"Limited to {args.max_verses} verses for testing")
        elif not args.full_bible:
            verses = [v for v in verses if v["reference"] in REPRESENTATIVE_VERSES]
            logger.info(f"Using {len(verses)}/100 representative verses (pass --full-bible to translate all)")

        logger.info(f"Loaded {len(verses)} verses")

        if not verses:
            logger.error("No verses loaded!")
            sys.exit(1)

        # Step 2: Initialize model and translator
        logger.info("\n[Step 2/5] Initializing model...")
        try:
            model_wrapper = ScriptureTranslationModel(
                model_name=args.model,
                use_lora=False,
                device=device,
            )
            model = model_wrapper.get_model()
            tokenizer = model_wrapper.get_tokenizer()

            # Load LoRA adapter if provided
            if args.lora_path:
                logger.info(f"Loading LoRA adapter from {args.lora_path}...")
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, args.lora_path)

            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            sys.exit(1)

        # Step 3: Initialize terminology DB and translator
        logger.info("\n[Step 3/5] Initializing terminology database...")
        terminology_db = TerminologyDB()
        tiered_terminology = initialize_terminology_db(terminology_db, args.target_lang)

        translator = ScriptureTranslator(
            model=model,
            tokenizer=tokenizer,
            terminology_db=terminology_db,
            tiered_terminology=tiered_terminology,
            device=device,
            enforce_consistency=True,
            use_prompt_conditioning=True,
        )
        logger.info("Translator initialized with prompt conditioning")

        # Step 4: Translate verses
        logger.info(f"\n[Step 4/5] Translating {len(verses)} verses...")
        try:
            results = translator.translate_batch(
                verses,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                batch_size=args.batch_size,
                show_progress=True,
            )
            logger.info(f"Translated {len(results)} verses")
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            sys.exit(1)

        # Step 5: Save translation memory and results
        logger.info(f"\n[Step 5/5] Saving results and translation memory...")

        # Save translation memory for future runs
        translator.translation_memory.save()
        translator.translation_memory.print_stats()

        # Prepare data for JSON
        translations_list = [r.to_dict() for r in results]

        # Save JSON
        json_path = args.output_dir / f"{args.target_lang}_bible.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(translations_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON: {json_path} ({len(translations_list)} verses)")

        # Save CSV
        csv_data = []
        for result in results:
            csv_data.append({
                "reference": result.source_text.split(":")[0] if ":" in result.source_text else "Unknown",
                "source_text": result.source_text,
                "translation": result.primary,
                "confidence": result.confidence,
            })

        df = pd.DataFrame(csv_data)
        csv_path = args.output_dir / f"{args.target_lang}_bible.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Saved CSV: {csv_path}")

        # Evaluate consistency
        logger.info("\n[Bonus] Running evaluation metrics...")
        evaluator = ScriptureEvaluator(terminology_db=terminology_db)

        # Compute consistency score across translations
        translations = [r.primary for r in results]
        consistency_score = evaluator.compute_consistency_score(
            translations, terminology_db=terminology_db
        )

        # Compute term uniqueness
        unique_terms, avg_usage = evaluator.compute_terminology_uniqueness(
            translations, args.target_lang
        )

        # Save evaluation report
        eval_report = {
            "timestamp": datetime.now().isoformat(),
            "language_pair": f"{args.source_lang} → {args.target_lang}",
            "source_language": args.source_lang,
            "target_language": args.target_lang,
            "total_verses": len(results),
            "model": args.model,
            "device": device,
            "batch_size": args.batch_size,
            "metrics": {
                "consistency_score": consistency_score,
                "unique_terms": unique_terms,
                "avg_term_usage": avg_usage,
                "avg_confidence": sum(r.confidence for r in results) / len(results) if results else 0.0,
            },
        }

        eval_path = args.output_dir / "evaluation_report.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evaluation report: {eval_path}")

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Verses translated: {len(results)}")
        logger.info(f"Average confidence: {eval_report['metrics']['avg_confidence']:.4f}")
        logger.info(f"Consistency score: {consistency_score:.4f}")
        logger.info(f"Unique terms: {unique_terms}")
        logger.info(f"Avg term usage: {avg_usage:.2f}")
        logger.info(f"\nOutput files:")
        logger.info(f"  - {json_path}")
        logger.info(f"  - {csv_path}")
        logger.info(f"  - {eval_path}")
        logger.info(f"{'='*60}\n")

    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Scripture Translation System - Command Line Interface

Provides command-line access to all system features.

Usage:
    python cli.py translate "In the beginning..." --target-lang spa_Latn
    python cli.py terminology add "salvation" "spa_Latn" "salvación"
    python cli.py evaluate --reference-file refs.txt --hypothesis-file hyps.txt
    python cli.py server start  # Start web server
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict
import logging

from config import Config
from data.loaders import BibleDataLoader, create_data_splits
from data.generate_sample_data import generate_sample_data, create_test_dataset
from models.base import ScriptureTranslationModel
from models.terminology import TerminologyDB, TermExtractor
from inference import ScriptureTranslator
from evaluation import ScriptureEvaluator, HumanEvaluationInterface

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScriptureTranslationCLI:
    """Command-line interface for scripture translation"""
    
    def __init__(self):
        self.model_wrapper = None
        self.translator = None
        self.terminology_db = None
        self.evaluator = None
    
    def init_system(self):
        """Initialize the translation system"""
        logger.info("Initializing system...")
        
        self.model_wrapper = ScriptureTranslationModel(use_lora=False)
        self.terminology_db = TerminologyDB()
        self.translator = ScriptureTranslator(
            model=self.model_wrapper.get_model(),
            tokenizer=self.model_wrapper.get_tokenizer(),
            terminology_db=self.terminology_db,
            device=Config.get_device(),
        )
        self.evaluator = ScriptureEvaluator(self.terminology_db)
    
    # ========================================================================
    # TRANSLATE COMMANDS
    # ========================================================================
    
    def cmd_translate(self, args):
        """Translate a verse"""
        if not self.translator:
            self.init_system()
        
        result = self.translator.translate_verse(
            source_text=args.text,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            num_beams=args.num_beams,
        )
        
        print("\n" + "="*60)
        print("TRANSLATION RESULT")
        print("="*60)
        print(f"Source ({args.source_lang}): {result.source_text}")
        print(f"Target ({args.target_lang}): {result.primary}")
        print(f"Confidence: {result.confidence:.2%}")
        
        if result.alternatives:
            print(f"\nAlternatives:")
            for i, alt in enumerate(result.alternatives[:3], 1):
                print(f"  {i}. {alt}")
        
        if result.theological_terms:
            print(f"\nTheological Terms:")
            for en, tgt in result.theological_terms.items():
                print(f"  {en} → {tgt}")
    
    def cmd_translate_file(self, args):
        """Translate verses from a file"""
        if not self.translator:
            self.init_system()
        
        # Load input file
        input_path = Path(args.input_file)
        
        if input_path.suffix == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                verses = json.load(f)
        elif input_path.suffix == '.txt':
            with open(input_path, 'r', encoding='utf-8') as f:
                verses = [{'text': line.strip()} for line in f if line.strip()]
        else:
            logger.error("Unsupported file format. Use .json or .txt")
            return
        
        logger.info(f"Loaded {len(verses)} verses from {input_path}")
        
        # Translate
        results = self.translator.translate_batch(
            verses,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            show_progress=True,
        )
        
        # Save results
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            output_data = [r.to_dict() for r in results]
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved results to {output_path}")
        
        # Statistics
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
        print(f"\nTranslated {len(results)} verses")
        print(f"Average confidence: {avg_confidence:.2%}")
    
    # ========================================================================
    # TERMINOLOGY COMMANDS
    # ========================================================================
    
    def cmd_term_add(self, args):
        """Add a term to the database"""
        if not self.terminology_db:
            self.terminology_db = TerminologyDB()
        
        success = self.terminology_db.add_term(
            args.english,
            args.language,
            args.target,
            confidence=args.confidence,
        )
        
        if success:
            print(f"✓ Added: {args.english} → {args.target} ({args.language})")
        else:
            print(f"✗ Conflict: {args.english} already has different translation")
        
        self.terminology_db.save()
    
    def cmd_term_lookup(self, args):
        """Look up a term"""
        if not self.terminology_db:
            self.terminology_db = TerminologyDB()
        
        result = self.terminology_db.get_with_confidence(args.term, args.language)
        
        if result:
            target, confidence = result
            print(f"{args.term} → {target} ({confidence:.2%})")
        else:
            print(f"Term not found: {args.term}")
    
    def cmd_term_extract(self, args):
        """Extract terms from text"""
        if not self.terminology_db:
            self.terminology_db = TerminologyDB()
        
        extractor = TermExtractor(self.terminology_db)
        terms = extractor.extract_theological_terms(args.text)
        
        print(f"Found {len(terms)} terms:")
        for term in sorted(terms):
            translation = self.terminology_db.lookup(term, args.language) if args.language else None
            if translation:
                print(f"  {term} → {translation}")
            else:
                print(f"  {term}")
    
    def cmd_term_stats(self, args):
        """Show terminology statistics"""
        if not self.terminology_db:
            self.terminology_db = TerminologyDB()
        
        self.terminology_db.print_statistics()
    
    def cmd_term_export(self, args):
        """Export terminology database"""
        if not self.terminology_db:
            self.terminology_db = TerminologyDB()
        
        output_path = Path(args.output_file)
        self.terminology_db.export_for_review(output_path, args.language)
        print(f"Exported to {output_path}")
    
    # ========================================================================
    # EVALUATION COMMANDS
    # ========================================================================
    
    def cmd_evaluate(self, args):
        """Evaluate translations"""
        if not self.evaluator:
            self.evaluator = ScriptureEvaluator()
        
        # Load files
        with open(args.hypothesis_file, 'r', encoding='utf-8') as f:
            hypotheses = [line.strip() for line in f if line.strip()]
        
        with open(args.reference_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f if line.strip()]
        
        if len(hypotheses) != len(references):
            logger.error("Hypothesis and reference file line counts must match")
            return
        
        # Evaluate
        metrics = self.evaluator.evaluate_batch(
            hypotheses,
            references,
            args.target_lang,
        )
        
        self.evaluator.print_metrics(metrics)
        
        # Save report
        if args.output_file:
            self.evaluator.save_evaluation_report(
                metrics,
                Path(args.output_file),
                f"{args.target_lang}"
            )
    
    # ========================================================================
    # DATA COMMANDS
    # ========================================================================
    
    def cmd_generate_data(self, args):
        """Generate sample data"""
        logger.info("Generating sample data...")
        data_dir = generate_sample_data(Path(args.output_dir))
        print(f"✓ Sample data generated in {data_dir}")
    
    def cmd_create_test_set(self, args):
        """Create test dataset"""
        logger.info(f"Creating test dataset with {args.num_samples} samples...")
        test_path = create_test_dataset(
            Path(args.output_file),
            num_samples=args.num_samples
        )
        print(f"✓ Test dataset created: {test_path}")
    
    # ========================================================================
    # SERVER COMMANDS
    # ========================================================================
    
    def cmd_server_start(self, args):
        """Start the web server"""
        import subprocess
        
        logger.info("Starting Scripture Translation Web Server...")
        logger.info("Visit: http://localhost:5000")
        
        subprocess.run([sys.executable, "app.py"])
    
    # ========================================================================
    # INFO COMMANDS
    # ========================================================================
    
    def cmd_system_info(self, args):
        """Show system information"""
        print("\n" + "="*60)
        print("SCRIPTURE TRANSLATION SYSTEM - INFO")
        print("="*60)
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Device: {Config.get_device()}")
        print(f"Supported Languages: {len(Config.LANGUAGE_CODES)}")
        print(f"\nSample Languages:")
        for lang, code in list(Config.LANGUAGE_CODES.items())[:5]:
            print(f"  {lang}: {code}")
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Scripture Translation System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py translate "In the beginning..." --target-lang spa_Latn
  python cli.py translate-file input.json output.json --target-lang spa_Latn
  python cli.py terminology add "salvation" "spa_Latn" "salvación"
  python cli.py terminology lookup "salvation" --language spa_Latn
  python cli.py evaluate --hypothesis-file hyps.txt --reference-file refs.txt
  python cli.py generate-data --output-dir ./data
  python cli.py server start
        """,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Translate commands
    translate_parser = subparsers.add_parser('translate', help='Translate a verse')
    translate_parser.add_argument('text', help='Text to translate')
    translate_parser.add_argument('--source-lang', default='eng_Latn', help='Source language')
    translate_parser.add_argument('--target-lang', default='spa_Latn', help='Target language')
    translate_parser.add_argument('--num-beams', type=int, default=5, help='Number of beams')
    
    translate_file_parser = subparsers.add_parser('translate-file', help='Translate from file')
    translate_file_parser.add_argument('input_file', help='Input file (JSON or TXT)')
    translate_file_parser.add_argument('output_file', help='Output file (JSON)')
    translate_file_parser.add_argument('--source-lang', default='eng_Latn', help='Source language')
    translate_file_parser.add_argument('--target-lang', default='spa_Latn', help='Target language')
    
    # Terminology commands
    term_subparsers = subparsers.add_parser('terminology', help='Manage terminology').add_subparsers(dest='term_command')
    
    term_add = term_subparsers.add_parser('add', help='Add a term')
    term_add.add_argument('english', help='English term')
    term_add.add_argument('language', help='Target language code')
    term_add.add_argument('target', help='Target language term')
    term_add.add_argument('--confidence', type=float, default=0.9, help='Confidence score')
    
    term_lookup = term_subparsers.add_parser('lookup', help='Look up a term')
    term_lookup.add_argument('term', help='Term to look up')
    term_lookup.add_argument('--language', default='spa_Latn', help='Language code')
    
    term_extract = term_subparsers.add_parser('extract', help='Extract terms from text')
    term_extract.add_argument('text', help='Text to analyze')
    term_extract.add_argument('--language', default='spa_Latn', help='Language code')
    
    term_stats = term_subparsers.add_parser('stats', help='Show statistics')
    
    term_export = term_subparsers.add_parser('export', help='Export terminology')
    term_export.add_argument('--language', default='spa_Latn', help='Language code')
    term_export.add_argument('--output-file', default='terminology_export.json', help='Output file')
    
    # Evaluation commands
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate translations')
    eval_parser.add_argument('--hypothesis-file', required=True, help='File with predictions')
    eval_parser.add_argument('--reference-file', required=True, help='File with references')
    eval_parser.add_argument('--target-lang', default='spa_Latn', help='Target language')
    eval_parser.add_argument('--output-file', help='Output report file')
    
    # Data commands
    data_parser = subparsers.add_parser('generate-data', help='Generate sample data')
    data_parser.add_argument('--output-dir', default='./data', help='Output directory')
    
    test_parser = subparsers.add_parser('create-test-set', help='Create test dataset')
    test_parser.add_argument('--num-samples', type=int, default=100, help='Number of samples')
    test_parser.add_argument('--output-file', default='./data/test_set.jsonl', help='Output file')
    
    # Server commands
    server_subparsers = subparsers.add_parser('server', help='Manage server').add_subparsers(dest='server_command')
    server_subparsers.add_parser('start', help='Start web server')
    
    # Info commands
    subparsers.add_parser('info', help='Show system information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create CLI instance
    cli = ScriptureTranslationCLI()
    
    # Execute command
    try:
        if args.command == 'translate':
            cli.cmd_translate(args)
        elif args.command == 'translate-file':
            cli.cmd_translate_file(args)
        elif args.command == 'terminology':
            if args.term_command == 'add':
                cli.cmd_term_add(args)
            elif args.term_command == 'lookup':
                cli.cmd_term_lookup(args)
            elif args.term_command == 'extract':
                cli.cmd_term_extract(args)
            elif args.term_command == 'stats':
                cli.cmd_term_stats(args)
            elif args.term_command == 'export':
                cli.cmd_term_export(args)
        elif args.command == 'evaluate':
            cli.cmd_evaluate(args)
        elif args.command == 'generate-data':
            cli.cmd_generate_data(args)
        elif args.command == 'create-test-set':
            cli.cmd_create_test_set(args)
        elif args.command == 'server':
            if args.server_command == 'start':
                cli.cmd_server_start(args)
        elif args.command == 'info':
            cli.cmd_system_info(args)
    
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

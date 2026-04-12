#!/usr/bin/env python3
"""
SCRIPTURE TRANSLATION SYSTEM - IMPLEMENTATION GUIDE

This document describes the complete codebase architecture, 
how to use it, and how to extend it.
"""

# ============================================================================
# QUICK REFERENCE
# ============================================================================

QUICK_START = """
1. Install dependencies:
   pip install -r requirements.txt

2. Generate sample data:
   python data/generate_sample_data.py

3. Run demo:
   python demo.py

4. Train baseline:
   python scripts/train_baseline.py \\
       --data_path ./data/en_es_verses.jsonl \\
       --target_lang spa_Latn

5. Fine-tune on rare language:
   python scripts/fine_tune_lora.py \\
       --pretrained_model_path ./models/checkpoints/final_model \\
       --data_path ./data/rare_lang_verses.jsonl \\
       --target_lang swh_Latn

6. Translate a book:
   python scripts/translate_book.py \\
       --input_path ./data/genesis.json \\
       --output_path ./results/genesis_spanish.json \\
       --target_lang spa_Latn
"""

# ============================================================================
# MODULE DESCRIPTIONS
# ============================================================================

MODULE_GUIDE = """

1. CONFIG.PY - Central Configuration Management
   ═══════════════════════════════════════════════
   
   Purpose: Single source of truth for all system configuration
   
   Key Classes:
   - Config: Static configuration class
   
   Key Methods:
   - get_device(): Returns 'cuda' or 'cpu'
   - get_language_code(name): Convert friendly name to NLLB code
   - load_from_yaml(path): Load config from YAML
   - save_to_yaml(dict, path): Save config to YAML
   
   Usage:
   ```python
   from config import Config
   
   # Access configuration
   model_name = Config.MODEL_NAME
   device = Config.get_device()
   
   # Language codes
   source_lang = Config.get_language_code("english")  # "eng_Latn"
   target_lang = Config.get_language_code("spanish")  # "spa_Latn"
   ```


2. DATA/LOADERS.PY - Data Loading & Preprocessing
   ═════════════════════════════════════════════════
   
   Purpose: Load Bible verses from various sources and create parallel corpora
   
   Key Classes:
   - BibleVerse: Single verse with metadata
   - BibleDataLoader: Load from JSON/CSV, manage verses
   - BibleTranslationDataset: PyTorch dataset for training
   
   Key Methods:
   - load_from_json(path, language): Load verses from JSON
   - load_from_csv(path, language): Load verses from CSV
   - create_parallel_corpus(source, target): Create aligned verse pairs
   - save_parallel_corpus(source, target, path): Save corpus to disk
   - get_verses_by_book(language, book): Filter by book
   - get_verses_by_range(...): Filter by verse range
   
   Usage:
   ```python
   from data.loaders import BibleDataLoader, BibleTranslationDataset
   
   # Load data
   loader = BibleDataLoader()
   loader.load_from_json("en_verses.json", "eng_Latn")
   loader.load_from_json("es_verses.json", "spa_Latn")
   
   # Create parallel corpus
   sources, targets = loader.create_parallel_corpus("eng_Latn", "spa_Latn")
   
   # Create PyTorch dataset
   dataset = BibleTranslationDataset(
       "verse_pairs.jsonl",
       tokenizer,
       "eng_Latn",
       "spa_Latn"
   )
   ```


3. DATA/GENERATE_SAMPLE_DATA.PY - Sample Data Generation
   ═══════════════════════════════════════════════════════
   
   Purpose: Generate sample Bible verse data for testing
   
   Key Functions:
   - generate_sample_data(output_dir): Create JSON, JSONL, CSV files
   - create_test_dataset(output_path, num_samples): Create test corpus
   
   Usage:
   ```python
   from data.generate_sample_data import generate_sample_data, create_test_dataset
   
   # Generate sample data
   data_dir = generate_sample_data(Path("./data"))
   
   # Create test dataset
   test_path = create_test_dataset(num_samples=100)
   ```


4. MODELS/TERMINOLOGY.PY - Terminology Database
   ═════════════════════════════════════════════
   
   Purpose: Maintain consistent translations of theological terms
   
   Key Classes:
   - TerminologyDB: Database for term translations
   - TermExtractor: Extract theological terms from text
   
   Key Methods:
   - add_term(en_term, target_lang, target_term, confidence): Register term
   - lookup(en_term, target_lang): Get canonical translation
   - get_with_confidence(...): Get term with confidence score
   - record_usage(term, lang): Track term usage
   - get_usage_count(term, lang): Get usage statistics
   - save(): Save database to JSON
   - load(): Load database from JSON
   - export_for_review(output_path, lang): Export for human review
   - import_reviewed_terms(path): Import reviewed terms
   - get_statistics(): Get database statistics
   
   Usage:
   ```python
   from models.terminology import TerminologyDB, TermExtractor
   
   # Create database
   db = TerminologyDB()
   
   # Add terms
   db.add_term("salvation", "spa_Latn", "salvación", confidence=0.98)
   db.add_term("grace", "spa_Latn", "gracia", confidence=0.97)
   
   # Lookup
   translation = db.lookup("salvation", "spa_Latn")  # "salvación"
   
   # Extract and check terms
   extractor = TermExtractor(db)
   terms = extractor.extract_theological_terms(text)
   canonical = extractor.get_canonical_terms(text, "spa_Latn")
   
   # Save & load
   db.save()
   db.load()
   
   # Export for review
   db.export_for_review("terms_for_review.json", "spa_Latn")
   ```


5. MODELS/BASE.PY - Model Initialization & Training
   ═════════════════════════════════════════════════
   
   Purpose: Load, initialize, and train the NLLB model
   
   Key Classes:
   - ScriptureTranslationModel: NLLB wrapper
   - ConsistencyLoss: Custom loss for term consistency
   - TranslationTrainer: Training loop manager
   
   Key Methods:
   - __init__(model_name, use_lora, device): Initialize model
   - apply_lora(): Apply Low-Rank Adaptation
   - freeze_encoder(): Freeze encoder weights
   - unfreeze_encoder(): Unfreeze encoder weights
   - save_pretrained(path): Save model checkpoint
   - load_pretrained(path): Load model checkpoint
   
   Usage:
   ```python
   from models.base import ScriptureTranslationModel, TranslationTrainer
   
   # Initialize model
   model = ScriptureTranslationModel(use_lora=False)
   
   # For rare language fine-tuning
   model_lora = ScriptureTranslationModel(use_lora=True)
   
   # Setup trainer
   optimizer = AdamW(model.get_model().parameters(), lr=1e-4)
   trainer = TranslationTrainer(
       model=model,
       optimizer=optimizer,
       terminology_db=db
   )
   
   # Training step
   loss, metrics = trainer.train_step(batch)
   ```


6. INFERENCE.PY - Translation Inference Pipeline
   ═════════════════════════════════════════════
   
   Purpose: Translate verses with consistency enforcement
   
   Key Classes:
   - TranslationResult: Container for translation results
   - ScriptureTranslator: Inference engine
   - BeamSearchDecoder: Enhanced decoding with constraints
   
   Key Methods:
   - translate_verse(...): Translate single verse
   - translate_batch(...): Translate multiple verses
   - translate_book(...): Translate entire book
   - _enforce_consistency(...): Post-process for consistency
   
   Usage:
   ```python
   from inference import ScriptureTranslator
   from models.terminology import TerminologyDB
   
   # Initialize
   terminology_db = TerminologyDB()
   translator = ScriptureTranslator(
       model=model.get_model(),
       tokenizer=model.get_tokenizer(),
       terminology_db=terminology_db,
       enforce_consistency=True
   )
   
   # Translate single verse
   result = translator.translate_verse(
       "In the beginning, God created...",
       source_lang="eng_Latn",
       target_lang="spa_Latn"
   )
   
   # Access results
   print(result.primary)  # Main translation
   print(result.confidence)  # Confidence score
   print(result.alternatives)  # Alternative translations
   print(result.theological_terms)  # Identified terms
   
   # Translate batch
   results = translator.translate_batch(
       verses,
       source_lang="eng_Latn",
       target_lang="spa_Latn"
   )
   ```


7. EVALUATION.PY - Translation Evaluation
   ══════════════════════════════════════
   
   Purpose: Measure translation quality with multiple metrics
   
   Key Classes:
   - EvaluationMetrics: Container for metrics
   - ScriptureEvaluator: Compute translation metrics
   - HumanEvaluationInterface: Collect human evaluation scores
   
   Key Methods:
   - compute_bleu(hypothesis, reference, weights): Calculate BLEU score
   - compute_consistency_score(...): Measure term consistency
   - evaluate_batch(...): Evaluate multiple translations
   - save_evaluation_report(...): Save metrics to JSON
   - print_metrics(...): Display formatted metrics
   
   Human Evaluation:
   - display_verse(index): Show verse for evaluation
   - collect_scores(index): Collect human ratings
   - run_evaluation_session(num_verses): Run interactive session
   - save_scores(output_path): Save evaluation results
   
   Usage:
   ```python
   from evaluation import ScriptureEvaluator, HumanEvaluationInterface
   
   # Automated evaluation
   evaluator = ScriptureEvaluator()
   metrics = evaluator.evaluate_batch(
       hypotheses,
       references,
       target_lang="spa_Latn"
   )
   evaluator.print_metrics(metrics)
   
   # Human evaluation
   interface = HumanEvaluationInterface(verses)
   scores = interface.run_evaluation_session(num_verses=10)
   interface.save_scores("evaluation_results.json")
   ```


8. SCRIPTS/TRAIN_BASELINE.PY - Baseline Training
   ═════════════════════════════════════════════
   
   Purpose: Train model on high-resource language pair
   
   Usage:
   ```bash
   python scripts/train_baseline.py \\
       --data_path ./data/en_es_verses.jsonl \\
       --source_lang eng_Latn \\
       --target_lang spa_Latn \\
       --batch_size 32 \\
       --num_epochs 3 \\
       --learning_rate 1e-4
   ```
   
   Output:
   - Model checkpoint: models/checkpoints/final_model/
   - Training history: results/training_history.json
   - Logs: logs/train_*.log


9. SCRIPTS/FINE_TUNE_LORA.PY - LoRA Fine-tuning
   ════════════════════════════════════════════
   
   Purpose: Efficiently adapt model to rare language
   
   Usage:
   ```bash
   python scripts/fine_tune_lora.py \\
       --pretrained_model_path ./models/checkpoints/final_model \\
       --data_path ./data/rare_lang_verses.jsonl \\
       --target_lang swh_Latn \\
       --batch_size 8 \\
       --num_epochs 5
   ```
   
   Output:
   - LoRA model: models/checkpoints/lora_swh_Latn_final/
   - Training history: results/lora_training_history_swh_Latn.json
   - Logs: logs/lora_train_*.log


10. SCRIPTS/TRANSLATE_BOOK.PY - Batch Translation
    ═════════════════════════════════════════════
    
    Purpose: Translate entire Bible books
    
    Usage:
    ```bash
    python scripts/translate_book.py \\
        --model_path ./models/checkpoints/final_model \\
        --input_path ./data/genesis.json \\
        --output_path ./results/genesis_spanish.json \\
        --target_lang spa_Latn \\
        --num_beams 5
    ```
    
    Output:
    - Translations: results/genesis_spanish.json
    - Contains: verses, confidence scores, alternatives, metadata


11. DEMO.PY - Complete System Demo
    ═══════════════════════════════
    
    Purpose: Demonstrate full system pipeline
    
    Usage:
    ```bash
    python demo.py
    ```
    
    Demonstrates:
    - Data loading
    - Terminology management
    - Model initialization
    - Translation inference
    - Evaluation metrics
    - Pipeline summary
"""

# ============================================================================
# COMMON WORKFLOWS
# ============================================================================

WORKFLOWS = """

WORKFLOW 1: Get Started (Quick Demo)
════════════════════════════════════
1. pip install -r requirements.txt
2. python data/generate_sample_data.py
3. python demo.py

Result: See full system in action with sample data


WORKFLOW 2: Train on New Language Pair
══════════════════════════════════════
1. Prepare data:
   - English verses: data/en_verses.json
   - Target language: data/target_verses.json
   
2. Create parallel corpus:
   python -c "
   from data.loaders import BibleDataLoader
   loader = BibleDataLoader()
   loader.load_from_json('en_verses.json', 'eng_Latn')
   loader.load_from_json('target_verses.json', 'target_lang_code')
   loader.save_parallel_corpus('eng_Latn', 'target_lang_code', 'output.jsonl')
   "

3. Train model:
   python scripts/train_baseline.py \\
       --data_path ./output.jsonl \\
       --target_lang target_lang_code \\
       --num_epochs 3

4. Translate:
   python scripts/translate_book.py \\
       --model_path ./models/checkpoints/final_model \\
       --input_path ./data/full_bible.json \\
       --output_path ./results/translation.json \\
       --target_lang target_lang_code

5. Evaluate:
   Review results/translation.json with human reviewers


WORKFLOW 3: Fine-tune on Rare Language (with LoRA)
═══════════════════════════════════════════════════
1. Start with trained baseline model:
   - From WORKFLOW 2, or use pretrained model
   - Path: models/checkpoints/final_model/

2. Prepare rare language data:
   - Collect 500-2000 verses in target language
   - Create parallel corpus: data/rare_verses.jsonl

3. Fine-tune with LoRA:
   python scripts/fine_tune_lora.py \\
       --pretrained_model_path ./models/checkpoints/final_model \\
       --data_path ./data/rare_verses.jsonl \\
       --target_lang rare_lang_code \\
       --num_epochs 5

4. Build terminology database:
   from models.terminology import TerminologyDB
   db = TerminologyDB()
   # Add terms with native speakers
   db.add_term("salvation", "rare_lang_code", "translation", 0.95)
   db.save()

5. Translate:
   python scripts/translate_book.py \\
       --model_path ./models/checkpoints/lora_rare_lang_code_final \\
       --input_path ./data/full_bible.json \\
       --output_path ./results/rare_translation.json \\
       --target_lang rare_lang_code

6. Human evaluation:
   from evaluation import HumanEvaluationInterface
   interface = HumanEvaluationInterface(results)
   scores = interface.run_evaluation_session(num_verses=50)


WORKFLOW 4: Manage Terminology
═══════════════════════════════
1. Build database:
   from models.terminology import TerminologyDB
   db = TerminologyDB()
   
2. Add high-confidence terms:
   db.add_term("salvation", "spa_Latn", "salvación", 0.98)
   db.add_term("grace", "spa_Latn", "gracia", 0.97)
   
3. Export for human review:
   db.export_for_review("terms_to_review.json", "spa_Latn")
   
4. Native speakers review and approve:
   # Edit terms_to_review.json, set "approved": true
   
5. Import reviewed terms:
   db.import_reviewed_terms("terms_reviewed.json")
   
6. Use in translation:
   translator = ScriptureTranslator(
       model=model,
       tokenizer=tokenizer,
       terminology_db=db,
       enforce_consistency=True
   )
"""

# ============================================================================
# EXTENDING THE SYSTEM
# ============================================================================

EXTENSIONS = """

EXTENSION 1: Add Custom Consistency Loss
═════════════════════════════════════════
Edit models/base.py:

class CustomConsistencyLoss(nn.Module):
    def __init__(self, terminology_db):
        super().__init__()
        self.db = terminology_db
    
    def forward(self, predictions, targets, source_terms):
        # Your custom logic
        return loss


EXTENSION 2: Add Custom Evaluation Metrics
═══════════════════════════════════════════
Edit evaluation.py:

def custom_metric(self, hypothesis, reference):
    # Implement your metric
    return score

# Then use in ScriptureEvaluator


EXTENSION 3: Add Language-Specific Rules
═════════════════════════════════════════
Create models/language_rules.py:

class LanguageRules:
    def __init__(self, language_code):
        self.language_code = language_code
    
    def apply_post_processing(self, text):
        # Language-specific rules
        return text

# Use in ScriptureTranslator._enforce_consistency()


EXTENSION 4: Build Web UI
═════════════════════════
Create ui/app.py:

from flask import Flask, request, jsonify
from inference import ScriptureTranslator

app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    text = request.json['text']
    target_lang = request.json['target_lang']
    result = translator.translate_verse(text, 'eng_Latn', target_lang)
    return jsonify(result.to_dict())

if __name__ == '__main__':
    app.run(debug=True)


EXTENSION 5: Add Batch Processing
══════════════════════════════════
In scripts/translate_book.py, add parallel processing:

from multiprocessing import Pool

def translate_verses_parallel(verses, translator, num_workers=4):
    with Pool(num_workers) as pool:
        results = pool.starmap(
            translator.translate_verse,
            [(...) for v in verses]
        )
    return results
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

TROUBLESHOOTING = """

Problem: Out of Memory (OOM)
─────────────────────────────
Solutions:
1. Reduce batch size: batch_size = 8 or 4 (default: 32)
2. Use smaller model: "facebook/nllb-200-distilled-600M" (default)
3. Enable gradient checkpointing in config
4. Use mixed precision: fp16=True
5. Increase memory with gradient accumulation

Code:
    TRAINING_CONFIG["batch_size"] = 8
    
Then retrain.


Problem: Slow Training
──────────────────────
Solutions:
1. Use GPU: should automatically use CUDA if available
2. Enable mixed precision: fp16=True
3. Reduce number of epochs
4. Use distributed training

Check GPU:
    python -c "import torch; print(torch.cuda.is_available())"


Problem: Low BLEU Scores
────────────────────────
Solutions:
1. Check data quality (ensure high-quality translations)
2. Increase number of training examples
3. Train longer (more epochs)
4. Use larger model
5. Adjust learning rate

For rare languages specifically:
- Collect more training data (target 1000+ verses)
- Use longer LoRA training (5-10 epochs)
- Increase LoRA rank (r=32 instead of 16)
- Manually review and fix low-confidence translations


Problem: Inconsistent Terminology
─────────────────────────────────
Solutions:
1. Build terminology database with native speakers
2. Use db.enforce_consistency() in translator
3. Review conflicts: db.get_conflicts()
4. Resolve conflicts: db.resolve_conflict()

Code:
    # Check conflicts
    conflicts = db.get_conflicts()
    
    # Resolve
    for term, variants in conflicts.items():
        chosen = get_native_speaker_choice(term)
        db.resolve_conflict(term, 'target_lang', chosen)


Problem: Model Not Loading
──────────────────────────
Solutions:
1. Check path exists: ls models/checkpoints/
2. Check model files: model.safetensors, tokenizer.json
3. Try redownloading: rm models/checkpoints/*, retrain
4. Check device: use CPU if GPU issues

Code:
    from pathlib import Path
    assert Path("models/checkpoints/final_model").exists()
    
    model = ScriptureTranslationModel()
    model.load_pretrained(Path("models/checkpoints/final_model"))
"""

# ============================================================================
# REFERENCE
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("QUICK START")
    print("="*70)
    print(QUICK_START)
    print("\n" + "="*70)
    print("MODULE GUIDE")
    print("="*70)
    print(MODULE_GUIDE)
    print("\n" + "="*70)
    print("WORKFLOWS")
    print("="*70)
    print(WORKFLOWS)
    print("\n" + "="*70)
    print("EXTENSIONS")
    print("="*70)
    print(EXTENSIONS)
    print("\n" + "="*70)
    print("TROUBLESHOOTING")
    print("="*70)
    print(TROUBLESHOOTING)

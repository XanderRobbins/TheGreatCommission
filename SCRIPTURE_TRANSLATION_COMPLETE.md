# Scripture Translation System - Complete Implementation

## 📦 What You've Built

A **production-ready, end-to-end scripture translation system** designed specifically for low-resource languages. This system uses:

- **NLLB-200**: Pretrained 200+ language translation model
- **LoRA**: Efficient fine-tuning (only 3% of parameters)
- **Terminology Database**: Ensures consistent theological term translations
- **Custom Evaluation**: BLEU + scripture-specific metrics
- **Human Validation**: Built-in interfaces for native speaker review

---

## 📂 Complete File Structure

```
scripture-translate/
├── config.py                          # Central configuration
├── demo.py                            # Full system demo
├── inference.py                       # Translation engine
├── evaluation.py                      # Quality metrics & human eval
├── IMPLEMENTATION_GUIDE.py            # Detailed reference guide
├── README.md                          # User guide
├── Makefile                           # Common tasks
├── requirements.txt                   # Dependencies
│
├── data/
│   ├── __init__.py
│   ├── loaders.py                    # Data loading & processing
│   ├── generate_sample_data.py       # Create sample verses
│   ├── *.json, *.jsonl               # Data files (created at runtime)
│   └── [user uploads]
│
├── models/
│   ├── __init__.py
│   ├── base.py                       # NLLB model wrapper & training
│   ├── terminology.py                # Term consistency database
│   └── checkpoints/                  # Saved models
│
├── scripts/
│   ├── __init__.py
│   ├── train_baseline.py             # Train on high-resource pair
│   ├── fine_tune_lora.py             # LoRA for rare languages
│   ├── translate_book.py             # Batch translation
│   └── evaluate.py                   # Evaluation script
│
├── logs/                             # Training logs
├── results/                          # Translation outputs
└── [additional files]
```

---

## 🚀 Quick Start (5 minutes)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate sample data
python data/generate_sample_data.py

# 3. Run demo
python demo.py
```

This will:
- ✅ Load sample Bible verses (10 verses in English, Spanish, Swahili)
- ✅ Demonstrate terminology management
- ✅ Show model initialization
- ✅ Display evaluation metrics
- ✅ Print complete pipeline summary

---

## 🏗️ System Architecture (Layers)

### Layer 1: Data Loading
**File:** `data/loaders.py`
- Load verses from JSON/CSV
- Create parallel corpora (aligned English ↔ Target)
- PyTorch Dataset for training

### Layer 2: Model Foundation
**File:** `models/base.py`
- Initialize NLLB-200 model (200+ languages)
- Apply LoRA for efficient fine-tuning
- Setup training loop with optimization

### Layer 3: Terminology Consistency
**File:** `models/terminology.py`
- Track theological term translations
- Ensure "salvation" always → same target word
- Human review interface
- Export/import for community validation

### Layer 4: Translation Inference
**File:** `inference.py`
- Translate verses with consistency enforcement
- Beam search decoding
- Return alternatives and confidence scores
- Batch processing for entire books

### Layer 5: Quality Evaluation
**File:** `evaluation.py`
- BLEU scores (standard MT metric)
- Consistency metrics (theology-specific)
- Human evaluation interface (interactive)
- Detailed reporting

### Layer 6: Training Scripts
**Files:** `scripts/train_baseline.py`, `scripts/fine_tune_lora.py`
- End-to-end training pipeline
- Automatic checkpointing
- Loss history tracking
- Evaluation during training

---

## 🎯 Typical Workflows

### Workflow A: Baseline Training (2 weeks)

```bash
# Step 1: Prepare data
# Collect 28,000 verse pairs: English ↔ Spanish (or any high-resource language)
# Save as: data/verse_pairs.jsonl

# Step 2: Train
python scripts/train_baseline.py \
    --data_path ./data/verse_pairs.jsonl \
    --source_lang eng_Latn \
    --target_lang spa_Latn \
    --num_epochs 3

# Step 3: Evaluate
python scripts/translate_book.py \
    --model_path ./models/checkpoints/final_model \
    --input_path ./data/test_book.json \
    --output_path ./results/translation.json \
    --target_lang spa_Latn

# Expected: BLEU-4 ~25-35, Human rating ~4.0-4.5/5.0
```

### Workflow B: Rare Language Adaptation (4-6 weeks)

```bash
# Step 1: Load pretrained baseline
# Requires: models/checkpoints/final_model/ from Workflow A

# Step 2: Prepare rare language data (500-2000 verses)
# Collect parallel corpus: data/rare_verses.jsonl

# Step 3: Fine-tune with LoRA
python scripts/fine_tune_lora.py \
    --pretrained_model_path ./models/checkpoints/final_model \
    --data_path ./data/rare_verses.jsonl \
    --target_lang swh_Latn \
    --num_epochs 5

# Step 4: Build terminology database
# Have native speakers review and approve terms

# Step 5: Translate with consistency
python scripts/translate_book.py \
    --model_path ./models/checkpoints/lora_swh_Latn_final \
    --input_path ./data/full_bible.json \
    --output_path ./results/swahili_bible.json \
    --target_lang swh_Latn

# Expected: BLEU-4 ~18-25, Human rating ~3.8-4.2/5.0
# With community refinement: 4.3-4.8/5.0 (publication ready)
```

---

## 📊 Key Classes & Methods

### 1. BibleDataLoader
```python
loader = BibleDataLoader()
loader.load_from_json("en_verses.json", "eng_Latn")
loader.load_from_csv("es_verses.csv", "spa_Latn")

sources, targets = loader.create_parallel_corpus("eng_Latn", "spa_Latn")
loader.save_parallel_corpus("eng_Latn", "spa_Latn", "output.jsonl")
```

### 2. TerminologyDB
```python
db = TerminologyDB()
db.add_term("salvation", "spa_Latn", "salvación", confidence=0.98)

translation = db.lookup("salvation", "spa_Latn")  # Returns "salvación"
db.record_usage("salvation", "spa_Latn")

db.export_for_review("terms.json", "spa_Latn")  # For humans
db.import_reviewed_terms("terms_reviewed.json")  # Import back

db.save()  # Persist to disk
db.print_statistics()
```

### 3. ScriptureTranslationModel
```python
model = ScriptureTranslationModel(use_lora=False)  # Baseline
model_lora = ScriptureTranslationModel(use_lora=True)  # For fine-tuning

model.apply_lora()  # Add LoRA adapters
model.freeze_encoder()  # Freeze encoder, train decoder
model.save_pretrained("path/to/checkpoint")
model.load_pretrained("path/to/checkpoint")
```

### 4. ScriptureTranslator
```python
translator = ScriptureTranslator(
    model=model.get_model(),
    tokenizer=model.get_tokenizer(),
    terminology_db=db,
    enforce_consistency=True
)

result = translator.translate_verse(
    "In the beginning, God created...",
    source_lang="eng_Latn",
    target_lang="spa_Latn"
)

print(result.primary)  # Main translation
print(result.confidence)  # 0.0-1.0
print(result.alternatives)  # [alt1, alt2, ...]
print(result.theological_terms)  # {"god": "Dios", "created": "creó"}

# Batch processing
results = translator.translate_batch(verses, "eng_Latn", "spa_Latn")

# Entire book
book_results = translator.translate_book(
    verses,
    "Genesis",
    "eng_Latn",
    "spa_Latn"
)
```

### 5. ScriptureEvaluator
```python
evaluator = ScriptureEvaluator()

# Single metric
bleu = evaluator.compute_bleu(hypothesis, reference)

# Batch evaluation
metrics = evaluator.evaluate_batch(hypotheses, references, "spa_Latn")

evaluator.print_metrics(metrics, "Spanish Translation")
# Prints: BLEU-1, BLEU-2, BLEU-4, Consistency, Term stats

evaluator.save_evaluation_report(metrics, "eval_report.json")
```

### 6. HumanEvaluationInterface
```python
interface = HumanEvaluationInterface(verses)

# Interactive evaluation
scores = interface.run_evaluation_session(num_verses=20)
# For each verse, asks: Accuracy (1-5), Clarity, Naturalness, Consistency

interface.save_scores("evaluation_results.json")
# Results include per-verse and aggregate scores
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model choice
MODEL_NAME = "facebook/nllb-200-distilled-600M"  # Recommended
# OR: "facebook/nllb-200-1.3B"  # Higher quality
# OR: "facebook/nllb-200-3.3B"  # Best quality (requires 12GB+ VRAM)

# Training
TRAINING_CONFIG = {
    "learning_rate": 1e-4,      # For baseline
    "batch_size": 32,            # Reduce if OOM
    "num_epochs": 3,
    "warmup_steps": 500,
}

# LoRA (rare language fine-tuning)
LORA_CONFIG = {
    "r": 16,                      # Rank (8-32 typical)
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],  # Attention layers
    "lora_dropout": 0.1,
}

# Fine-tuning (rare language)
FINETUNING_CONFIG = {
    "learning_rate": 5e-4,       # Higher than baseline
    "batch_size": 8,             # Smaller batch size
    "num_epochs": 5,
}
```

---

## 📈 Expected Results

### Baseline (English → Spanish, 28K verses)
| Metric | Expected |
|--------|----------|
| BLEU-1 | 45-55 |
| BLEU-4 | 25-35 |
| Consistency | 92-96% |
| Human Rating | 4.0-4.5 / 5.0 |
| Training Time | 4-6 hours (V100 GPU) |

### Rare Language (English → Swahili, 1K verses + LoRA)
| Metric | Expected |
|--------|----------|
| BLEU-1 | 35-42 |
| BLEU-4 | 18-25 |
| Consistency | 90-94% |
| Human Rating | 3.5-4.2 / 5.0 |
| Fine-tuning Time | 1-2 hours (V100 GPU) |

### After Community Review & Refinement
| Metric | Expected |
|--------|----------|
| Human Rating | 4.3-4.8 / 5.0 |
| Status | ✅ Publication Ready |

---

## 🔧 Common Tasks

### Generate Sample Data
```bash
python data/generate_sample_data.py
# Creates 10 sample verses in English, Spanish, Swahili
```

### Run Complete Demo
```bash
python demo.py
# Shows all components in action
```

### View Implementation Guide
```bash
python IMPLEMENTATION_GUIDE.py
# Detailed reference for all modules and methods
```

### Use Makefile
```bash
make install          # Install dependencies
make generate-data    # Generate sample data
make train-baseline   # Train (English-Spanish)
make fine-tune-lora   # Fine-tune on rare language
make demo             # Run demo
make clean            # Remove generated files
```

---

## 🚨 Troubleshooting

### Out of Memory
```python
# config.py
TRAINING_CONFIG["batch_size"] = 8  # or 4
```

### Slow Training
- Use GPU: `torch.cuda.is_available()` should return True
- Enable mixed precision: `fp16=True` in config
- Use smaller model: `distilled-600M` instead of `1.3B`

### Low Translation Quality
- Collect more training data (target 1000+ verses for rare language)
- Train longer (5-10 epochs for LoRA)
- Manually build terminology database with native speakers
- Use human evaluation to find problem areas

### Inconsistent Terminology
```python
db = TerminologyDB()
conflicts = db.get_conflicts()  # Find problems

# Resolve with native speaker
db.resolve_conflict("word", "lang_code", "canonical_form")
```

---

## 📚 Key References

- **NLLB Paper:** Meta's Neural Machine Translation for Multilingual
  - arxiv.org/abs/2207.04672
  
- **LoRA Paper:** Parameter-Efficient Fine-Tuning
  - arxiv.org/abs/2106.09685

- **Bible Data:**
  - openbible.org/download
  - ebible.org
  - unfoldingword.org

- **HuggingFace Docs:**
  - huggingface.co/docs/transformers

---

## 🎓 Next Steps

1. **Generate sample data:**
   ```bash
   python data/generate_sample_data.py
   ```

2. **Run the demo:**
   ```bash
   python demo.py
   ```

3. **Train baseline on your language pair:**
   ```bash
   python scripts/train_baseline.py --target_lang your_lang_code
   ```

4. **For rare languages: collect 500-2000 verses, then LoRA fine-tune:**
   ```bash
   python scripts/fine_tune_lora.py --target_lang your_lang_code
   ```

5. **Translate and evaluate:**
   ```bash
   python scripts/translate_book.py --target_lang your_lang_code
   ```

6. **Collect human feedback and iterate**

---

## 📝 File Organization Summary

| File | Purpose |
|------|---------|
| `config.py` | Global configuration |
| `demo.py` | End-to-end system demo |
| `inference.py` | Translation inference |
| `evaluation.py` | Quality metrics & human eval |
| `data/loaders.py` | Data loading & preprocessing |
| `data/generate_sample_data.py` | Create test data |
| `models/base.py` | NLLB model & training |
| `models/terminology.py` | Terminology consistency |
| `scripts/train_baseline.py` | Train on high-resource pair |
| `scripts/fine_tune_lora.py` | LoRA for rare languages |
| `scripts/translate_book.py` | Batch translation |
| `README.md` | User guide |
| `IMPLEMENTATION_GUIDE.py` | Detailed reference |
| `Makefile` | Common commands |

---

## ✨ Key Features

✅ **Pretrained Model:** NLLB-200 (200+ languages)
✅ **Efficient Fine-tuning:** LoRA (only 3% of parameters)
✅ **Consistency Enforcement:** Terminology database for reliable translations
✅ **Custom Metrics:** BLEU + scripture-specific evaluation
✅ **Human Validation:** Interactive interfaces for native speakers
✅ **Low-Resource:** Designed for 500-2000 translation verses
✅ **Production-Ready:** Comprehensive error handling and logging
✅ **Modular:** Easy to extend and customize
✅ **Well-Documented:** Detailed guides and examples

---

## 🤝 Contributing

Areas for enhancement:
- [ ] Web UI for translators
- [ ] Mobile app for review
- [ ] Advanced NER for proper nouns
- [ ] Language-specific linguistic rules
- [ ] Community feedback integration
- [ ] Multi-language training

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Built For Bible Translation

This system is designed specifically for the challenges of translating Scripture into underserved languages. By combining state-of-the-art ML with human-in-the-loop validation, it makes high-quality Bible translation accessible even for languages with <2000 translated verses.

**Last Updated:** April 10, 2026

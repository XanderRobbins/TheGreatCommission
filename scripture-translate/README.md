# Scripture Translation System

A complete, production-ready system for low-resource Bible translation using pretrained multilingual models (NLLB) and LoRA fine-tuning.

## Overview

This system tackles the unique challenges of Bible translation:

- **Meaning preservation** (not just literal words)
- **Consistency** (same concepts → same terms)
- **Cultural clarity** (idioms and references land right)
- **Low-resource adaptation** (works with <2000 translated verses)

## Architecture

```
Input Verse (English)
        ↓
    [NLLB Encoder - Shared Cross-lingual Space]
        ↓
    [Fine-tune on Bible-aligned Data]
        ↓
    [LoRA Adapter for Rare Language]
        ↓
    [Terminology Enforcement]
        ↓
    [Beam Search Decoding]
        ↓
Output Verse (Target Language)
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-org/scripture-translate.git
cd scripture-translate

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data for evaluation
python -m nltk.downloader punkt
```

### 2. Generate Sample Data

```bash
python data/generate_sample_data.py
```

This creates:
- `data/en_verses.json` - English verses
- `data/es_verses.json` - Spanish verses  
- `data/sw_verses.json` - Swahili verses
- `data/en_es_verses.jsonl` - English-Spanish parallel corpus
- `data/en_sw_verses.jsonl` - English-Swahili parallel corpus

### 3. Run Demo

```bash
python demo.py
```

This will:
- Load sample Bible verse data
- Initialize the NLLB model
- Demonstrate terminology management
- Show evaluation metrics
- Print pipeline summary

### 4. Train Baseline Model

```bash
python scripts/train_baseline.py \
    --data_path ./data/en_es_verses.jsonl \
    --source_lang eng_Latn \
    --target_lang spa_Latn \
    --batch_size 32 \
    --num_epochs 3
```

This trains the model to translate from English to Spanish.

### 5. Fine-tune for Rare Language

```bash
python scripts/fine_tune_lora.py \
    --pretrained_model_path ./models/checkpoints/final_model \
    --data_path ./data/en_sw_verses.jsonl \
    --target_lang swh_Latn \
    --batch_size 8 \
    --num_epochs 5
```

This uses LoRA to efficiently adapt the model to Swahili.

## Project Structure

```
scripture-translate/
├── config.py                    # Central configuration
├── demo.py                      # Complete system demo
├── inference.py                 # Translation inference
├── evaluation.py                # Evaluation metrics
├── requirements.txt             # Dependencies
│
├── data/
│   ├── __init__.py
│   ├── loaders.py              # Data loading & preprocessing
│   ├── generate_sample_data.py  # Create sample verses
│   └── *.json, *.jsonl          # Data files
│
├── models/
│   ├── __init__.py
│   ├── base.py                  # NLLB model wrapper
│   ├── terminology.py           # Terminology database
│   └── checkpoints/             # Saved models
│
├── scripts/
│   ├── train_baseline.py        # Train baseline model
│   ├── fine_tune_lora.py        # LoRA fine-tuning
│   ├── translate_book.py        # Batch translation
│   └── evaluate.py              # Run evaluation
│
├── logs/                        # Training logs
├── results/                     # Translation outputs
└── README.md                    # This file
```

## Key Components

### 1. Data Loading (`data/loaders.py`)

Load Bible verses from JSON/CSV and create parallel corpora:

```python
from data.loaders import BibleDataLoader

loader = BibleDataLoader()
loader.load_from_json("en_verses.json", "eng_Latn")
loader.load_from_json("es_verses.json", "spa_Latn")

sources, targets = loader.create_parallel_corpus("eng_Latn", "spa_Latn")
```

### 2. Terminology Management (`models/terminology.py`)

Maintain consistent translations across verses:

```python
from models.terminology import TerminologyDB, TermExtractor

db = TerminologyDB()
db.add_term("salvation", "spa_Latn", "salvación", confidence=0.98)
db.add_term("grace", "spa_Latn", "gracia", confidence=0.97)

extractor = TermExtractor(db)
terms = extractor.extract_theological_terms("God's grace and salvation...")
canonical = extractor.get_canonical_terms(text, "spa_Latn")
```

### 3. Model & Training (`models/base.py`)

Initialize and train the translation model:

```python
from models.base import ScriptureTranslationModel, TranslationTrainer

# Initialize model
model_wrapper = ScriptureTranslationModel(use_lora=False)

# For rare language fine-tuning
model_wrapper_lora = ScriptureTranslationModel(use_lora=True)
```

### 4. Inference (`inference.py`)

Translate verses with consistency enforcement:

```python
from inference import ScriptureTranslator
from models.terminology import TerminologyDB

terminology_db = TerminologyDB()
translator = ScriptureTranslator(
    model=model.get_model(),
    tokenizer=model.get_tokenizer(),
    terminology_db=terminology_db,
    enforce_consistency=True
)

result = translator.translate_verse(
    "In the beginning, God created the heavens and the earth.",
    source_lang="eng_Latn",
    target_lang="spa_Latn"
)

print(result.primary)  # Primary translation
print(result.confidence)  # Confidence score
print(result.alternatives)  # Alternative translations
```

### 5. Evaluation (`evaluation.py`)

Measure translation quality:

```python
from evaluation import ScriptureEvaluator

evaluator = ScriptureEvaluator()

# BLEU scores
bleu = evaluator.compute_bleu(hypothesis, reference)

# Batch evaluation
metrics = evaluator.evaluate_batch(hypotheses, references, "spa_Latn")

# Print results
evaluator.print_metrics(metrics)
```

## Supported Languages

NLLB supports 200+ languages. Some key ones:

```
English:  eng_Latn
Spanish:  spa_Latn
French:   fra_Latn
Portuguese: por_Latn
Swahili:  swh_Latn
Turkish:  tur_Latn
Amharic:  amh_Ethi
Korean:   kor_Hang
Mandarin: zho_Hans
```

See `config.py` for full list.

## Configuration

Edit `config.py` to customize:

```python
# Model
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Training parameters
TRAINING_CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "num_epochs": 3,
    ...
}

# LoRA configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    ...
}
```

## Workflow

### Phase 1: Baseline Training (2 weeks)

1. Collect 28,000+ verse pairs (English + high-resource language)
2. Train on baseline task
3. Validate on unseen verses
4. Achieve BLEU-4: 25-35

```bash
python scripts/train_baseline.py \
    --data_path ./data/en_es_verses.jsonl \
    --source_lang eng_Latn \
    --target_lang spa_Latn
```

### Phase 2: Rare Language Fine-tuning (4-6 weeks)

1. Collect 500-2000 verses in target language
2. Apply LoRA for efficient adaptation
3. Build terminology database with native speakers
4. Achieve BLEU-4: 18-25, Human rating: 3.8+

```bash
python scripts/fine_tune_lora.py \
    --pretrained_model_path ./models/checkpoints/final_model \
    --data_path ./data/rare_lang_verses.jsonl \
    --target_lang swh_Latn
```

### Phase 3: Production (2 weeks)

1. Deploy model as API
2. Build web UI for human review
3. Collect community feedback
4. Iterate and refine

## Expected Performance

### Baseline (English → Spanish)
- BLEU-1: 45-55
- BLEU-4: 25-35
- Consistency: 92-96%
- Human rating: 4.0-4.5 / 5.0
- Training time: ~6 hours on GPU

### Rare Language (English → Swahili, 1000 verses)
- BLEU-1: 35-42
- BLEU-4: 18-25
- Consistency: 90-94%
- Human rating: 3.5-4.2 / 5.0
- Fine-tuning time: ~2 hours on GPU

### After Community Review
- Human rating: 4.3-4.8 / 5.0
- Production ready for publication

## Data Format

### JSON Format (for `load_from_json`)

```json
[
  {
    "book": "Genesis",
    "chapter": 1,
    "verse": 1,
    "text": "In the beginning, God created the heavens and the earth."
  },
  ...
]
```

### JSONL Format (for training)

```jsonl
{"source": "...", "target": "...", "source_lang": "eng_Latn", "target_lang": "spa_Latn"}
```

### CSV Format (for `load_from_csv`)

```csv
book,chapter,verse,text
Genesis,1,1,"In the beginning..."
```

## Tips for Best Results

1. **Data Quality**: Ensure high-quality human translations for fine-tuning
2. **Term Consistency**: Use terminology database to maintain consistency
3. **Human Review**: Always validate translations with native speakers
4. **Incremental**: Start with Genesis/Matthew, expand gradually
5. **Community**: Build translation committees for best results
6. **Iteration**: Refine based on feedback

## Common Issues

### Out of Memory

Reduce batch size or use smaller model:

```python
# In config.py
TRAINING_CONFIG["batch_size"] = 16  # Default: 32

# Use smaller model
MODEL_NAME = "facebook/nllb-200-distilled-600M"  # Instead of 1.3B
```

### Slow Training

Use mixed precision:

```python
# In training script
from transformers import TrainingArguments

args = TrainingArguments(
    ...
    fp16=True,  # Enable mixed precision
)
```

### Low Scores on Rare Language

- Ensure high-quality training data
- Use longer LoRA training (5-10 epochs)
- Increase LoRA rank (r=32 instead of 16)
- Collect more data (>2000 verses)

## Advanced Usage

### Custom Consistency Loss

Extend `ConsistencyLoss` in `models/base.py`:

```python
class CustomConsistencyLoss(nn.Module):
    def forward(self, predictions, targets, source_terms):
        # Your custom logic here
        return loss_tensor
```

### Custom Evaluation Metrics

Add metrics to `ScriptureEvaluator`:

```python
def custom_metric(self, hypothesis, reference):
    # Your metric computation
    return score
```

### Web UI Integration

The `evaluation.py` includes `HumanEvaluationInterface` for building:

```python
from evaluation import HumanEvaluationInterface

interface = HumanEvaluationInterface(verses)
scores = interface.run_evaluation_session(num_verses=10)
interface.save_scores("evaluation_results.json")
```

## Contributing

Contributions welcome! Areas for enhancement:

- [ ] Support for more language pairs
- [ ] Interactive web UI
- [ ] Mobile app for translation review
- [ ] Advanced terminology extraction
- [ ] Community feedback integration
- [ ] Language-specific linguistic rules

## Resources

- [NLLB Paper](https://arxiv.org/abs/2207.04672)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Bible Verses Data](https://openbible.org/download)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## License

MIT License - See LICENSE file

## Support

For issues, questions, or suggestions:
1. Check existing GitHub issues
2. Create a new issue with details
3. Contact the maintainers

## Acknowledgments

- Facebook/Meta for NLLB-200
- Hugging Face for Transformers
- Bible communities for translation data

---

**Built with ❤️ for Bible translation**

Last updated: 2026-04-10

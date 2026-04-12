# Scripture Translation Pipeline Guide

## Quick Start

### Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Install optional Bible loaders (choose one or more)
pip install pythonbible    # Recommended: most reliable
pip install bbible
pip install freebible

# Download NLTK data for evaluation
python -m nltk.downloader punkt
```

### Run the Pipeline

```bash
# Quick test (5 verses, useful for debugging)
python run_pipeline.py --max-verses 5

# Full translation to Haitian Creole
python run_pipeline.py --target-lang hat_Latn

# Custom options
python run_pipeline.py \
    --source-lang eng_Latn \
    --target-lang hat_Latn \
    --model facebook/nllb-200-distilled-600M \
    --batch-size 8 \
    --num-beams 4 \
    --device cuda \
    --output-dir ./output
```

## Output Files

The pipeline generates:

1. **haitian_creole_bible.json** — Full Bible in JSON format
   - Array of verse objects with fields: `reference`, `primary`, `confidence`, `source_text`

2. **haitian_creole_bible.csv** — Tabular format for spreadsheet tools
   - Columns: `reference`, `source_text`, `translation`, `confidence`

3. **evaluation_report.json** — Quality metrics
   - Consistency score, unique terms, average confidence

## Pipeline Architecture

```
Input: English Bible
    ↓
[1. Load Bible] ← Auto-detect source (pythonbible, CSV, HuggingFace)
    ↓
[2. Initialize Model] ← NLLB-200 with optional LoRA
    ↓
[3. Seed Terminology] ← Pre-defined Haitian Creole terms
    ↓
[4. Translate Batches] ← Efficient GPU batching
    ↓
[5. Save Results] ← JSON + CSV output
    ↓
Output: Haitian Creole Bible + evaluation metrics
```

## Command-Line Options

```
--source-lang eng_Latn      Source language code (default: eng_Latn)
--target-lang hat_Latn      Target language code (default: hat_Latn)
--model MODEL               HuggingFace model name
--batch-size N              Verses per GPU batch (default: 8)
--num-beams N               Beam search width (default: 4)
--lora-path PATH            Optional LoRA adapter path
--max-verses N              Limit verses (useful for testing)
--output-dir PATH           Where to save results (default: ./output)
--device {cpu,cuda}         Compute device (auto-detect if not specified)
```

## Expected Performance

### Resources
- **Small model** (distilled-600M): ~2 hours on GPU, ~24 hours on CPU for full Bible
- **GPU memory**: ~4GB for batch_size=8 (adjust down if OOM)
- **Disk space**: ~50 MB for output files

### Quality Metrics
- **Average confidence**: 0.3-0.4 (typical for low-resource languages)
- **Consistency score**: 0.85-0.95 (enforced via terminology DB)
- **Unique terms**: 5000-10000 distinct words

## Terminology System

The pipeline includes pre-seeded Haitian Creole terms for theological vocabulary:

```
English          →  Haitian Creole
salvation        →  salvasyon
grace            →  gras
sin              →  peche
lord             →  Seyè
god              →  Bondye
faith            →  lafwa
love             →  lanmou
church           →  legliz
```

Add more terms in `run_pipeline.py:HAITIAN_CREOLE_TERMINOLOGY`.

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python run_pipeline.py --batch-size 4
```

### Slow Translation
Use smaller model:
```bash
python run_pipeline.py --model facebook/nllb-200-distilled-600M
```

### Bible Not Loaded
Check that at least one source is available:
- `pip install pythonbible` (preferred)
- Or create `data/bible_en.csv` with columns: `book,chapter,verse,text`

### Import Errors
Verify transformers is installed:
```bash
python -c "import transformers; print(transformers.__version__)"
```

## Advanced Usage

### With LoRA Adapter
Fine-tune on Haitian Creole data, then translate:

```bash
# Train LoRA (separate script)
python scripts/fine_tune_lora.py --data_path ./data/haitian_creole_training.jsonl

# Use trained adapter
python run_pipeline.py --lora-path ./models/haitian_creole_lora
```

### Batch Multiple Language Pairs
```bash
for lang in spa_Latn fra_Latn por_Latn; do
    python run_pipeline.py --target-lang $lang --output-dir output/$lang
done
```

### Custom Output Locations
```bash
python run_pipeline.py \
    --output-dir /mnt/external_drive/bibles \
    --max-verses 100
```

## File Formats

### JSON Output
```json
[
  {
    "primary": "Nan konmansman, Bondye te kreye syèl la ak tè a.",
    "confidence": 0.3421,
    "alternatives": [],
    "theological_terms": {"god": "Bondye", "heaven": "syèl"},
    "consistency_enforced": false,
    "source_text": "Genesis 1:1: In the beginning God created the heavens and the earth.",
    "target_language": "hat_Latn"
  },
  ...
]
```

### CSV Output
```
reference,source_text,translation,confidence
Genesis 1:1,In the beginning...,Nan konmansman...,0.3421
Genesis 1:2,And the earth was...,Tè a te...,0.3105
```

## Integration with Existing Code

All modules are importable:

```python
from data.bible_loader import load_bible
from inference import ScriptureTranslator
from evaluation import ScriptureEvaluator

# Use in your own scripts
verses = load_bible()
result = translator.translate_verse(
    "In the beginning, God created...",
    source_lang="eng_Latn",
    target_lang="hat_Latn"
)
print(result.primary)  # Translated text
print(result.confidence)  # 0.0-1.0
```

## Next Steps

1. **Collect human feedback** on generated translations
2. **Refine terminology** based on native speaker input
3. **Train LoRA adapter** on community translations
4. **Iterate** with continuous improvement

See `README.md` for full system documentation.

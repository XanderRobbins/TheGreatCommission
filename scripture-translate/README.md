# Scripture Translation System

Low-resource Bible translation using NLLB-200 with LoRA fine-tuning for rare languages.

## What it does

Translates Bible verses from English into target languages, with a focus on languages that have little to no existing parallel data. The system addresses two core problems general MT doesn't handle well: theological term consistency (the same English term should always map to the same target term) and low-resource adaptation (working with 500-2000 translated verses instead of millions).

## Architecture

```
English verse
    -> NLLB-200 encoder (shared cross-lingual space, 200+ languages)
    -> Fine-tuned on Bible-aligned parallel data
    -> LoRA adapter for rare language target
    -> Terminology database post-processing
    -> Beam search decoding
Output verse + confidence score + alternatives
```

LoRA keeps fine-tuning tractable: ~3% of parameters trained, no catastrophic forgetting of the pretrained cross-lingual representations.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

**Demo (no training required):**
```bash
python demo.py
```

**Full pipeline (translate 100 representative verses):**
```bash
python run_pipeline.py --target-lang hat_Latn
```

**Train baseline model:**
```bash
python scripts/train_baseline.py \
    --data_path ./data/en_es_verses.jsonl \
    --source_lang eng_Latn \
    --target_lang spa_Latn
```

**Fine-tune for rare language with LoRA:**
```bash
python scripts/fine_tune_lora.py \
    --pretrained_model_path ./models/checkpoints/final_model \
    --data_path ./data/en_sw_verses.jsonl \
    --target_lang swh_Latn
```

**Web UI:**
```bash
python app.py  # serves at http://localhost:5000
```

## Key files

```
scripture-translate/
├── config.py                   # Model name, training params, language codes
├── run_pipeline.py             # End-to-end: load -> translate -> save -> evaluate
├── demo.py                     # Walks through each component without training
├── app.py                      # Flask web UI (factory pattern)
│
├── data/
│   ├── loaders.py              # BibleDataLoader, parallel corpus creation
│   ├── bible_loader.py         # Load from verse reference files
│   └── generate_sample_data.py
│
├── models/
│   ├── base.py                 # ScriptureTranslationModel, LoRA setup, training loop
│   ├── terminology.py          # TerminologyDB, TermExtractor
│   └── tiered_terminology.py   # 3-tier override system (always/context/model-decides)
│
├── inference/
│   └── translator.py           # ScriptureTranslator, batch translation, beam search
│
├── evaluation/
│   └── evaluator.py            # BLEU, consistency score, terminology metrics
│
└── scripts/
    ├── train_baseline.py
    ├── fine_tune_lora.py
    └── translate_book.py
```

## Data formats

**Verse JSON** (for `load_from_json`):
```json
[{"book": "Genesis", "chapter": 1, "verse": 1, "text": "In the beginning..."}]
```

**Training JSONL**:
```jsonl
{"source": "...", "target": "...", "source_lang": "eng_Latn", "target_lang": "spa_Latn"}
```

## Language codes

NLLB uses BCP-47 + script codes. Examples: `eng_Latn`, `spa_Latn`, `swh_Latn`, `hat_Latn`, `amh_Ethi`. Full list in `config.py`.

## References

- [NLLB paper](https://arxiv.org/abs/2207.04672)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
- Bible data: openbible.org, ebible.org, unfoldingword.org

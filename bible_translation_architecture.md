# Scripture Translation Architecture

## Problem

General MT models fail for low-resource Bible translation in two ways:

1. **Theological consistency** - "salvation" must map to the same target word across all 500+ verses it appears in. Standard MT loss doesn't enforce this.
2. **Low-resource adaptation** - many target languages have 500-2000 translated verses at best, not the millions needed to train from scratch.

## Pipeline

```
English verse
    -> NLLB-200 encoder (shared cross-lingual embedding space)
    -> Fine-tune on Bible-aligned parallel data (consistency loss added)
    -> LoRA adapter for rare language
    -> Terminology database post-processing
    -> Beam search decoding
Output: verse + confidence + alternatives
```

## Model choice

NLLB-200 (`facebook/nllb-200-distilled-600M` or `-1.3B`) over mBART because it covers 200+ languages including many low-resource ones, and was built specifically for this setting. The 600M distilled model runs on ~2GB VRAM; the 1.3B is better quality but needs ~4GB.

## Training

**Phase 1 - Baseline:** Train on English + high-resource language pairs (Spanish, Portuguese, Swahili). ~28k verse pairs. Adds a consistency loss term alongside standard cross-entropy:

```python
total_loss = mt_loss + alpha * consistency_loss
```

where consistency loss penalizes the same English theological term mapping to different target words within a batch.

**Phase 2 - LoRA fine-tune:** For the actual target language, apply LoRA to the attention weights (`q_proj`, `v_proj`) and train on 500-2000 verses. Updates ~3% of parameters; no catastrophic forgetting. Typical config: `r=16`, `lora_alpha=32`, 5 epochs, lr=5e-4.

## Terminology database

Post-processing layer that enforces consistent term mappings regardless of what the model generates. Organized in three tiers:

- **Tier 1** - always override (e.g., proper names of God, "Holy Spirit")
- **Tier 2** - context-aware (apply only when the term appears in a theological context)
- **Tier 3** - model decides (no override, but tracked for consistency reporting)

Initially populated by automatic extraction from model outputs against human reference translations, then validated by native speakers.

## Evaluation

- **BLEU** - baseline quality measure, but poor fit for theology (rewards literal overlap over meaning)
- **Consistency score** - fraction of theological terms that map to a single target term across the translation set
- **Human evaluation** - accuracy, clarity, naturalness, consistency rated by native speakers on a random verse sample

## Code structure

```
scripture-translate/
├── models/base.py              # ScriptureTranslationModel, LoRA setup, ConsistencyLoss
├── models/terminology.py       # TerminologyDB, TermExtractor
├── models/tiered_terminology.py
├── inference/translator.py     # ScriptureTranslator, batch translation
├── evaluation/evaluator.py     # Metrics
├── data/loaders.py             # BibleDataLoader, parallel corpus
├── run_pipeline.py             # End-to-end pipeline script
└── app.py                      # Flask web UI
```

## References

- NLLB: Bapna et al., 2023 - https://arxiv.org/abs/2207.04672
- LoRA: Hu et al., 2021 - https://arxiv.org/abs/2106.09685
- Data: openbible.org, ebible.org, unfoldingword.org

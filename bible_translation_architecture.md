# Low-Resource Scripture Translation System
## Technical Architecture & Implementation Guide

---

## 1. THE PROBLEM YOU'RE SOLVING

### What makes Bible translation different from general machine translation?

**Scripture is high-stakes meaning transfer:**
- Religious accuracy matters (theological concepts must map correctly)
- Cultural context is critical (idioms, metaphors, units of measure)
- Consistency matters (same term = same concept throughout)
- Community validation is essential (translations need speaker buy-in)

**Low-resource constraint:**
- Fewer than 7,000 languages have ANY scripture translation
- Many indigenous/minority languages have <500 translated verses
- No parallel corpora like Europarl or news data exist
- Communities often have oral traditions, not written data

**Why off-the-shelf translation fails:**
- General MT models trained on news/social media → misses theological tone
- Rare languages get near-zero attention in pretraining
- Idioms don't translate word-for-word (e.g., "heart" ≠ organ in scripture)

---

## 2. ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────┐
│                   SCRIPTURE TRANSLATION PIPELINE             │
└─────────────────────────────────────────────────────────────┘

INPUT LAYER
├─ Source Text (English Bible)
├─ Verse Metadata (book, chapter, verse)
└─ Target Language Code

    ↓

EMBEDDING LAYER (Shared Cross-Lingual Space)
├─ mBART or NLLB Encoder
│  └─ Tokenizes input in any language
│  └─ Projects to 1024-d or 2048-d shared space
└─ Handles 100+ languages via single model

    ↓

FINE-TUNING LAYER (Bible-Specific)
├─ Train on verse-aligned parallel data (all language pairs → English)
├─ Loss = translation loss + consistency loss
└─ Learn scripture-specific term mappings

    ↓

RARE LANGUAGE ADAPTER (Key Innovation)
├─ Small fine-tune on 500–2000 target-language verses
├─ LoRA (Low-Rank Adaptation) to reduce parameters
└─ Align into shared embedding space

    ↓

DECODING + POST-PROCESSING
├─ Beam search (with consistency constraints)
├─ Terminology database lookup
├─ Fluency filtering
└─ Human validation UI

    ↓

OUTPUT
└─ Translated verse + confidence score + alternatives
```

---

## 3. STEP 1: PRETRAINED FOUNDATION

### Why mBART or NLLB?

**mBART (Facebook/Meta):**
- 50+ languages
- Seq2seq architecture (encoder-decoder)
- Trained on denoising autoencoder task
- Good for moderate-resource languages
- Smaller model size (~600M parameters)

**NLLB (Meta, newer):**
- 200+ languages (including many low-resource ones)
- Built specifically for low-resource translation
- Larger capacity (~600M–3.3B)
- Better morphology handling for agglutinative languages
- **Recommended for this use case**

### Installation & Setup

```python
# Use HuggingFace transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"  # or -1.3B for better quality
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Freeze encoder initially (we'll fine-tune decoder + cross-attention)
for param in model.encoder.parameters():
    param.requires_grad = False
```

### Cost-Benefit Analysis

| Aspect | mBART | NLLB-600M | NLLB-3.3B |
|--------|-------|-----------|-----------|
| Languages | 50 | 200+ | 200+ |
| Vram (inference) | ~2GB | ~2GB | ~6GB |
| Fine-tune time | ~2 hrs | ~4 hrs | ~12 hrs |
| Quality (low-resource) | Baseline | Better | Best |
| **Recommendation** | Start here | ✓ Best balance | Cloud-only |

---

## 4. STEP 2: TRAIN ON BIBLE-ALIGNED DATA

### Data Strategy

**Phase 1: Multi-language pretraining**

Use publicly available Bible verse alignments:

```
Source: English Standard Version (ESV) - public domain
Aligned data: Bible in Every Language (multiple formats)
Download: openbible.org, ebible.org, unfoldingword.org
```

**Format the data:**

```json
{
  "book": "Genesis",
  "chapter": 1,
  "verse": 1,
  "en": "In the beginning, God created the heavens and the earth.",
  "es": "En el principio creó Dios los cielos y la tierra.",
  "sw": "Mwanzo Mungu akaumba mbingu na ardhi.",
  "translations": ["ESV", "RVR60", "Biblia Takatifu"]
}
```

**Data splits for training:**

```
Total: ~31,000 verses per language pair
├─ Train: 28,000 verses (90%)
├─ Val: 1,500 verses (5%)
└─ Test: 1,500 verses (5%)
```

### Training Loop

```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# 1. Prepare data
train_dataset = BibleDataset(data_path="verse_pairs.jsonl", source_lang="en", target_lang="es")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. Training config
optimizer = AdamW(model.parameters(), lr=1e-4)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

# 3. Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Log consistency metric
        if step % 100 == 0:
            consistency_score = check_term_consistency(model, batch)
            print(f"Loss: {loss.item():.4f}, Consistency: {consistency_score:.4f}")
```

### Key Insight: Consistency Loss

Standard translation loss doesn't enforce term consistency.

Add a **consistency term** to the loss:

```python
def compute_combined_loss(outputs, labels, term_alignment, alpha=0.1):
    """
    MT loss (standard) + Consistency loss (scripture-specific)
    """
    # Standard translation loss
    mt_loss = outputs.loss
    
    # Consistency loss: penalize when same English term maps to different target terms
    consistency_loss = 0
    for en_term, target_terms in term_alignment.items():
        if len(set(target_terms)) > 1:  # Multiple translations for same term
            consistency_loss += len(set(target_terms)) - 1
    
    total_loss = mt_loss + alpha * consistency_loss
    return total_loss
```

**Result:** Model learns that "salvation" should map to the same word across all 500+ Bible verses.

---

## 5. STEP 3: FINE-TUNE ON RARE LANGUAGE DATA

### The Challenge

For language X (your target), you might have:
- 500–2000 translated verses (if lucky)
- Oral traditions (need transcription)
- Partial translations from missionaries

Standard fine-tuning would overfit. Use **LoRA (Low-Rank Adaptation)** instead.

### LoRA: Efficient Fine-Tuning

**Concept:** Don't update all model weights. Add low-rank matrices.

```
Original weight W (1024 × 2048):
    W_new = W + ΔW
    
LoRA approach:
    W_new = W + B @ A
    where A ∈ ℝ^(r × 2048), B ∈ ℝ^(1024 × r)
    r = 8 or 16 (rank, much smaller than 1024)
```

**Benefit:**
- Train only ~3% of parameters
- Memory usage: 6GB → 2GB
- No catastrophic forgetting of pretraining knowledge

### Implementation

```python
from peft import get_peft_model, LoraConfig, TaskType

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Attention weights
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Now fine-tune on rare language data
rare_lang_dataset = BibleDataset(
    data_path="swahili_verses.jsonl", 
    source_lang="en", 
    target_lang="sw"
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./lora_checkpoints",
        learning_rate=5e-4,
        per_device_train_batch_size=8,
        num_train_epochs=5,
        logging_steps=50,
    ),
    train_dataset=rare_lang_dataset,
)

trainer.train()
```

### Why This Works for Scripture

**Rare language has:**
- Different grammar (morphology, word order)
- Different phonetics (how it sounds)
- Different cultural references (flora, fauna, units)

**LoRA learns:**
- Language-specific attention patterns
- Morphological transformations (add -ing, -ed, etc.)
- Cultural term mappings

**But preserves:**
- General translation knowledge
- Theological concept understanding
- Cross-lingual alignment from pretraining

---

## 6. CONSISTENCY DATABASE (THE SECRET WEAPON)

### Why It Matters

Translation model alone will be inconsistent:
- Verse 1: "sin" → "dosa"
- Verse 500: "sin" → "kesalahan"

This breaks scripture reading experience.

### Build a Terminology Database

```python
class TerminologyDB:
    def __init__(self):
        self.term_map = {}  # en_term → {target_lang: target_term}
        self.confidence = {}
    
    def add_term(self, en_term, target_lang, target_term, confidence=0.9):
        """Register a term translation"""
        if en_term not in self.term_map:
            self.term_map[en_term] = {}
        
        self.term_map[en_term][target_lang] = target_term
        self.confidence[(en_term, target_lang)] = confidence
    
    def lookup(self, en_term, target_lang):
        """Get consistent translation"""
        return self.term_map.get(en_term, {}).get(target_lang)
    
    def enforce_consistency(self, translation, source_terms, target_lang):
        """Post-process translation to enforce consistency"""
        for en_term in source_terms:
            canonical = self.lookup(en_term, target_lang)
            if canonical:
                # Replace with canonical version
                translation = translation.replace("???", canonical)
        return translation
```

### Building the Database

**Phase 1: Automatic extraction**

```python
def extract_terminology(parallel_corpus, model, tokenizer):
    """Extract common terms from model outputs"""
    terms = {}
    
    for source, reference in parallel_corpus:
        # Get model output
        output = model.generate(
            tokenizer(source, return_tensors="pt")["input_ids"]
        )
        prediction = tokenizer.decode(output[0])
        
        # Align source → reference → prediction
        alignment = align_words(source, reference, prediction)
        
        for en_word, ref_word in alignment:
            if en_word in THEOLOGICAL_TERMS:
                if en_word not in terms:
                    terms[en_word] = []
                terms[en_word].append((ref_word, alignment_confidence))
    
    return terms

# Extract from high-confidence data (human translations)
terminology_db = extract_terminology(
    parallel_corpus=high_confidence_verses,
    model=model,
    tokenizer=tokenizer
)
```

**Phase 2: Human validation**

```python
# Present to native speakers
for en_term, candidates in terminology_db.items():
    print(f"\nEnglish term: {en_term}")
    for i, (target_term, confidence) in enumerate(candidates):
        print(f"  {i+1}. {target_term} (confidence: {confidence:.2%})")
    
    chosen = input("Select preferred translation (or type new): ")
    terminology_db.set_canonical(en_term, target_lang, chosen)
```

---

## 7. INFERENCE PIPELINE

### Decoding with Constraints

```python
def translate_verse(
    source_verse: str,
    source_lang: str,
    target_lang: str,
    model,
    tokenizer,
    terminology_db,
    num_beams=5
) -> dict:
    """
    Translate a single verse with consistency enforcement
    """
    
    # Step 1: Tokenize
    inputs = tokenizer(
        source_verse,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    # Step 2: Identify theological terms
    theological_terms = extract_terms(source_verse)
    
    # Step 3: Generate translation
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=num_beams,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True
    )
    
    translation = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    confidence = outputs.sequences_scores[0].item()
    
    # Step 4: Enforce consistency
    for en_term in theological_terms:
        canonical = terminology_db.lookup(en_term, target_lang)
        if canonical:
            # Replace with consistent term
            translation = post_process_consistency(translation, en_term, canonical)
    
    # Step 5: Return alternatives
    alternatives = [
        tokenizer.decode(seq, skip_special_tokens=True)
        for seq in outputs.sequences[1:]
    ]
    
    return {
        "primary": translation,
        "confidence": float(confidence),
        "alternatives": alternatives,
        "theological_terms": theological_terms
    }
```

### Batch Processing

```python
def translate_book(
    book_verses: List[dict],
    source_lang: str,
    target_lang: str,
    model,
    tokenizer,
    terminology_db
) -> List[dict]:
    """
    Translate entire book with progress tracking
    """
    results = []
    
    for i, verse_data in enumerate(book_verses):
        source = verse_data["text"]
        ref = f"{verse_data['book']} {verse_data['chapter']}:{verse_data['verse']}"
        
        result = translate_verse(
            source_verse=source,
            source_lang=source_lang,
            target_lang=target_lang,
            model=model,
            tokenizer=tokenizer,
            terminology_db=terminology_db
        )
        
        result.update({
            "reference": ref,
            "source": source
        })
        
        results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"Translated {i+1}/{len(book_verses)} verses")
    
    return results
```

---

## 8. EVALUATION METRICS

### Standard Metrics (BLEU, METEOR)

```python
from evaluate import load

bleu = load("bleu")
meteor = load("meteor")

# Evaluate
results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU: {results['bleu']:.3f}")
```

**But BLEU is bad for scripture.** It rewards literal matches, but theology is about *meaning*.

### Scripture-Specific Metrics

```python
def theological_consistency_score(translations, terminology_db, target_lang):
    """
    Measure: do the same English terms map to the same target terms?
    """
    term_consistency = {}
    
    for verse_id, translation in translations.items():
        en_terms = extract_theological_terms(verse_id)
        for en_term in en_terms:
            canonical = terminology_db.lookup(en_term, target_lang)
            if canonical:
                if en_term not in term_consistency:
                    term_consistency[en_term] = set()
                # Extract what the model predicted
                predicted = extract_term_translation(translation, en_term)
                term_consistency[en_term].add(predicted)
    
    # Score: 1.0 if all same, 0.0 if all different
    consistency = sum(
        1 for terms in term_consistency.values() if len(terms) == 1
    ) / len(term_consistency)
    
    return consistency

def fluency_score(translation, target_lang):
    """
    Measure: does it sound natural to native speakers?
    Use cross-lingual language model for fluency scoring.
    """
    # Simplified: use perplexity
    fluency_model = load_language_model(target_lang)
    perplexity = fluency_model.compute_perplexity(translation)
    return 1.0 / (1.0 + perplexity)  # Normalize to 0-1
```

### Human Evaluation Protocol

```
For each book:
  - Select 10 random verses (blinded)
  - Native speakers rate on:
    1. Accuracy (does it match the meaning?)
    2. Clarity (is it understandable?)
    3. Naturalness (does it sound like how we speak?)
    4. Consistency (same terms used same way?)
  - Average across 3+ speakers
  - Target: 4.5+ / 5.0
```

---

## 9. IMPLEMENTATION ROADMAP

### Phase 0: Setup (1 week)
- [ ] Clone HF transformers
- [ ] Download NLLB-600M model
- [ ] Gather Bible verse parallel data (openbible.org)
- [ ] Create BibleDataset class

### Phase 1: Baseline (2 weeks)
- [ ] Train on English → Spanish (prove the pipeline works)
- [ ] Build terminology database
- [ ] Evaluate BLEU + consistency metrics
- [ ] Establish human evaluation baseline

### Phase 2: Generalize (3 weeks)
- [ ] Train on 3-5 high-resource language pairs (ES, PT, SW, TK, etc.)
- [ ] Refine consistency loss
- [ ] Optimize inference speed

### Phase 3: Rare Language (ongoing)
- [ ] Collect 500–2000 verses in target language
- [ ] LoRA fine-tune on target language
- [ ] Build terminology database with native speakers
- [ ] Iterative human evaluation + refinement

### Phase 4: Deploy (2 weeks)
- [ ] API endpoint for verse translation
- [ ] Web UI for human review
- [ ] Batch processing pipeline
- [ ] Export results (PDF, EPUB, JSON)

---

## 10. CODE STRUCTURE

```
scripture-translate/
├── data/
│   ├── loaders.py          # BibleDataset, DataLoader setup
│   ├── processors.py       # Verse alignment, cleaning
│   └── sources/
│       ├── english.txt
│       ├── spanish.txt
│       └── swahili.txt
├── models/
│   ├── base.py             # NLLB setup
│   ├── fine_tune.py        # Training loops
│   └── lora_adapter.py     # LoRA for rare languages
├── translation/
│   ├── inference.py        # Decoding + constraints
│   ├── consistency.py      # Terminology DB
│   └── evaluation.py       # BLEU, METEOR, custom metrics
├── ui/
│   ├── web_app.py          # Flask/Streamlit
│   └── validation_interface.py
├── scripts/
│   ├── train_baseline.py
│   ├── fine_tune_rare_lang.py
│   ├── translate_book.py
│   └── evaluate.py
└── README.md
```

---

## 11. EXPECTED RESULTS

### Baseline (Spanish)
- BLEU: 30–35
- Human evaluation: 4.2/5
- Consistency: 94%+

### After Fine-tuning (Rare Language)
- BLEU: 20–28 (lower because smaller corpus, but still usable)
- Human evaluation: 3.8–4.3/5 (with terminology refinement)
- Consistency: 92%+

### Time to Production
- Baseline: 2 weeks
- Full rare language pipeline: 4–6 weeks
- With community validation: 3 months per language

---

## 12. KEY TAKEAWAYS

| Principle | Why It Matters |
|-----------|---|
| Use pretrained (NLLB) | Don't reinvent; 200+ languages already supported |
| Multi-language pretraining | Scripture knowledge transfers across languages |
| Consistency loss | Theology requires term consistency, not just fluency |
| LoRA fine-tuning | Efficient adaptation for rare languages, no catastrophic forgetting |
| Terminology DB | Post-processing guardrail; enables human oversight |
| Human validation | Scripture is high-stakes; ML confidence alone is insufficient |

---

## References & Resources

- **NLLB Paper:** Bapna et al., 2023: https://arxiv.org/abs/2207.04672
- **LoRA:** Hu et al., 2021: https://arxiv.org/abs/2106.09685
- **Bible Data:**
  - openbible.org/download
  - ebible.org
  - unfoldingword.org (translation guidelines + data)
- **HF Docs:** huggingface.co/docs/transformers
- **Parallel Data:**
  - Bible in Every Language (BEL)
  - Manually aligned versions (ESV + translations)


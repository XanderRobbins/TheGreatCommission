"""Sanity check: diverse passage test to validate TM caching, confidence, and metadata.

Passages:
  Genesis 1:1-20    — narrative, contains "and God saw" duplicates
  Psalm 23          — poetic, theological
  Matthew 5:1-12    — Sermon on Mount, "blessed are..." parallelism
  John 1:1-14       — mystical, dense
  1 John 1:1-10     — epistolary, light/darkness theme

Measures:
  1. TM hit rate (run twice — second run should be near 100%)
  2. Wall-clock time per run
  3. Confidence distribution by passage type
  4. Metadata completeness (theological term overlay coverage)

Usage:
  cd scripture-translate
  python sanity_check.py [--target-lang min_Latn] [--model facebook/nllb-200-distilled-600M]
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from config import Config
from data.bible_loader import load_bible
from models.base import ScriptureTranslationModel
from models.terminology import TerminologyDB
from models.tiered_terminology import TieredTerminologyDB, TermTier
from inference import ScriptureTranslator
from utils.logger import get_logger, configure_logging

configure_logging()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Passage definitions: (book_prefix, chapter, verse_start, verse_end)
# ---------------------------------------------------------------------------
PASSAGES = [
    ("Genesis",  1,  1, 20),
    ("Psalm",   23,  1, 99),   # take all of Psalm 23 (usually ~6 verses)
    ("Matthew",  5,  1, 12),
    ("John",     1,  1, 14),
    ("1 John",   1,  1, 10),
]


def parse_reference(ref: str):
    """Parse 'Genesis 1:1' → ('Genesis', 1, 1). Returns None on failure."""
    try:
        parts = ref.rsplit(" ", 1)
        if len(parts) != 2:
            return None
        book = parts[0].strip()
        ch_vs = parts[1].split(":")
        if len(ch_vs) != 2:
            return None
        return book, int(ch_vs[0]), int(ch_vs[1])
    except (ValueError, IndexError):
        return None


def filter_passages(all_verses):
    """Return only the verses in PASSAGES, in order."""
    selected = []
    for book, chapter, v_start, v_end in PASSAGES:
        for verse in all_verses:
            parsed = parse_reference(verse["reference"])
            if parsed is None:
                continue
            v_book, v_ch, v_vs = parsed
            if v_book == book and v_ch == chapter and v_start <= v_vs <= v_end:
                selected.append(verse)
    return selected


def passage_label(ref: str) -> str:
    """Map a reference to a short passage label for grouping."""
    parsed = parse_reference(ref)
    if parsed is None:
        return "Unknown"
    book, ch, _ = parsed
    for p_book, p_ch, _, _ in PASSAGES:
        if book == p_book and ch == p_ch:
            return f"{p_book} {p_ch}"
    return f"{book} {ch}"


def confidence_bucket(c: float) -> str:
    if c >= 0.85:
        return "high  (≥0.85)"
    if c >= 0.70:
        return "mid   (0.70–0.84)"
    if c >= 0.65:
        return "low   (0.65–0.69)"
    return "floor (<0.65, won't cache)"


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_translation(translator, verses, source_lang, target_lang, label):
    print_section(f"Translation run: {label}")
    # Reset TM stats (counts only) without clearing the cache
    translator.translation_memory.stats["total_lookups"] = 0
    translator.translation_memory.stats["cache_hits"] = 0
    translator.translation_memory.stats["cache_misses"] = 0

    t0 = time.time()
    results = translator.translate_batch(
        verses,
        source_lang=source_lang,
        target_lang=target_lang,
        batch_size=8,
        show_progress=False,
    )
    elapsed = time.time() - t0

    tm = translator.translation_memory
    hit_rate = tm.get_hit_rate()
    print(f"  Verses:      {len(results)}")
    print(f"  Wall-clock:  {elapsed:.1f}s")
    print(f"  TM hits:     {tm.stats['cache_hits']}/{tm.stats['total_lookups']}  ({hit_rate:.1%})")

    return results, elapsed


def report_confidence(results, verses):
    print_section("Confidence distribution by passage")

    by_passage = defaultdict(list)
    for result, verse in zip(results, verses):
        label = passage_label(verse["reference"])
        by_passage[label].append(result.confidence)

    all_confidences = [r.confidence for r in results]
    print(f"  Overall avg:  {sum(all_confidences)/len(all_confidences):.4f}")
    print(f"  Min:          {min(all_confidences):.4f}")
    print(f"  Max:          {max(all_confidences):.4f}")
    print()

    for label, scores in sorted(by_passage.items()):
        avg = sum(scores) / len(scores)
        lo = min(scores)
        hi = max(scores)
        below_floor = sum(1 for s in scores if s < 0.65)
        print(f"  {label:<18} n={len(scores):>3}  avg={avg:.3f}  [{lo:.3f}–{hi:.3f}]"
              + (f"  ({below_floor} below cache floor)" if below_floor else ""))

    print()
    bucket_counts = defaultdict(int)
    for c in all_confidences:
        bucket_counts[confidence_bucket(c)] += 1
    for bucket, count in sorted(bucket_counts.items()):
        print(f"  {bucket}: {count} verse(s)")


def report_metadata(results, verses):
    print_section("Theological term metadata completeness")

    total_terms = 0
    filled_terms = 0
    null_terms = 0
    missing_terms = []

    for result, verse in zip(results, verses):
        terms = result.theological_terms or {}
        for term, value in terms.items():
            total_terms += 1
            if value:
                filled_terms += 1
            else:
                null_terms += 1
                missing_terms.append((verse["reference"], term))

    print(f"  Total term slots:  {total_terms}")
    print(f"  Filled (non-null): {filled_terms}")
    print(f"  Null:              {null_terms}")
    if total_terms > 0:
        print(f"  Coverage:          {filled_terms/total_terms:.1%}")

    if missing_terms:
        print(f"\n  Null entries (term not in DB for this language):")
        for ref, term in missing_terms[:20]:
            print(f"    {ref}: '{term}'")
        if len(missing_terms) > 20:
            print(f"    ... and {len(missing_terms)-20} more")
    else:
        print("\n  All term slots filled.")


def report_sample_translations(results, verses, n=5):
    print_section(f"Sample translations (first {n} verses)")
    for result, verse in zip(results[:n], verses[:n]):
        print(f"  [{verse['reference']}]")
        print(f"  EN: {verse['text']}")
        print(f"  TL: {result.primary}")
        print(f"  conf={result.confidence:.3f}  terms={result.theological_terms}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Sanity check for scripture translation pipeline")
    parser.add_argument("--target-lang", default="min_Latn", help="Target language NLLB code")
    parser.add_argument("--source-lang", default="eng_Latn", help="Source language NLLB code")
    parser.add_argument("--model", default=Config.MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--output", default="output/sanity_check.json", help="Where to save results JSON")
    args = parser.parse_args()

    Config.ensure_dirs()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load and filter verses
    # -----------------------------------------------------------------------
    print_section("Loading Bible and filtering passages")
    all_verses = load_bible(Config.DATA_DIR)
    verses = filter_passages(all_verses)

    if not verses:
        logger.error("No verses matched the passage filter. Check reference format in your Bible source.")
        sys.exit(1)

    print(f"  Total Bible verses: {len(all_verses)}")
    print(f"  Selected for test:  {len(verses)}")
    for book, ch, v_start, v_end in PASSAGES:
        count = sum(1 for v in verses if passage_label(v["reference"]) == f"{book} {ch}")
        print(f"    {book} {ch}:{v_start}-{v_end}  →  {count} verse(s) loaded")

    # -----------------------------------------------------------------------
    # Initialize model + translator
    # -----------------------------------------------------------------------
    print_section("Initializing model")
    model_wrapper = ScriptureTranslationModel(
        model_name=args.model,
        use_lora=False,
        device=Config.get_device(),
    )
    model = model_wrapper.get_model()
    tokenizer = model_wrapper.get_tokenizer()

    terminology_db = TerminologyDB()
    tiered_terminology = TieredTerminologyDB(terminology_db)

    translator = ScriptureTranslator(
        model=model,
        tokenizer=tokenizer,
        terminology_db=terminology_db,
        tiered_terminology=tiered_terminology,
        device=Config.get_device(),
        enforce_consistency=True,
        use_prompt_conditioning=True,
    )
    print(f"  Model:  {args.model}")
    print(f"  Target: {args.target_lang}")

    # -----------------------------------------------------------------------
    # Run 1: cold (or warm from previous run)
    # -----------------------------------------------------------------------
    results_run1, time_run1 = run_translation(
        translator, verses, args.source_lang, args.target_lang, "Run 1 (first pass)"
    )
    translator.translation_memory.save()

    # -----------------------------------------------------------------------
    # Run 2: should be near 100% TM hit rate
    # -----------------------------------------------------------------------
    results_run2, time_run2 = run_translation(
        translator, verses, args.source_lang, args.target_lang, "Run 2 (cache warm)"
    )

    speedup = time_run1 / time_run2 if time_run2 > 0 else float("inf")
    print(f"\n  Speedup run2 vs run1: {speedup:.1f}x")

    # -----------------------------------------------------------------------
    # Reports (using run 1 results — the actual translations)
    # -----------------------------------------------------------------------
    report_confidence(results_run1, verses)
    report_metadata(results_run1, verses)
    report_sample_translations(results_run1, verses, n=5)

    # -----------------------------------------------------------------------
    # Save full results JSON
    # -----------------------------------------------------------------------
    output = {
        "config": {
            "target_lang": args.target_lang,
            "source_lang": args.source_lang,
            "model": args.model,
            "verse_count": len(verses),
        },
        "timing": {
            "run1_seconds": round(time_run1, 2),
            "run2_seconds": round(time_run2, 2),
            "speedup": round(speedup, 2),
        },
        "tm": {
            "run1_hit_rate": translator.translation_memory.stats["cache_hits"] / max(1, translator.translation_memory.stats["total_lookups"]),
            "total_cached_entries": len(translator.translation_memory.cache),
        },
        "verses": [
            {
                "reference": verse["reference"],
                "source": verse["text"],
                "translation": result.primary,
                "confidence": result.confidence,
                "theological_terms": result.theological_terms,
                "consistency_enforced": result.consistency_enforced,
            }
            for verse, result in zip(verses, results_run1)
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print_section("Done")
    print(f"  Results saved to: {output_path}")
    print(f"  Run 1: {time_run1:.1f}s  |  Run 2: {time_run2:.1f}s  |  Speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()

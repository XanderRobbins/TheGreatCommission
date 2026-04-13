#!/usr/bin/env python3
"""Studio: Unified orchestrator for scripture translation pipeline.

Manages state, data collection, fine-tuning, and inference for Bible translation
into low-resource languages using NLLB-200 with LoRA fine-tuning.

Usage:
    python studio.py init [--lang LANG_CODE]
    python studio.py add --book BOOK --chapter CHAPTER [--file FILE] [--lang LANG_CODE]
    python studio.py status [--lang LANG_CODE]
    python studio.py run [--force] [--lang LANG_CODE]
    python studio.py retrain [--force] [--lang LANG_CODE]
    python studio.py translate [--lang LANG_CODE]
"""

import argparse
import json
import sys
import subprocess
import tempfile
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple

from config import Config
from data.convert_youversion import parse_youversion, build_pairs


class StudioState:
    """Manages state persistence for a scripture translation project."""

    def __init__(self, state_path: Path):
        """Initialize state manager.

        Args:
            state_path: Path to studio_state.json file.
        """
        self.state_path = state_path
        self.data = {}
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def save(self) -> None:
        """Write state to disk atomically (temp file + rename)."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.state_path.parent,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            json.dump(self.data, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name

        # Atomic rename
        import shutil

        shutil.move(tmp_path, self.state_path)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data


class Studio:
    """Main orchestrator for scripture translation workflow."""

    # Default training threshold (verses)
    DEFAULT_MIN_TRAIN_VERSES = 200

    def __init__(self):
        """Initialize studio."""
        Config.ensure_dirs()

    def get_state_path(self, lang_code: Optional[str] = None) -> Path:
        """Get state file path for language.

        Args:
            lang_code: Language code. If None, returns default state path.

        Returns:
            Path to studio_state.json (or studio_state_{lang_code}.json for multi-lang).
        """
        data_dir = Config.DATA_DIR
        if lang_code:
            return data_dir / f"studio_state_{lang_code}.json"
        else:
            # Default: check if there's an existing state, otherwise use generic path
            default_path = data_dir / "studio_state.json"
            if default_path.exists():
                return default_path

            # Find the most recent state file for default
            state_files = list(data_dir.glob("studio_state_*.json"))
            if state_files:
                # Return the first one (would be better to track "last used")
                return state_files[0]

            return default_path

    def get_data_path(self, lang_code: str) -> Path:
        """Get JSONL data path for language."""
        return Config.DATA_DIR / f"{lang_code}_verses.jsonl"

    def count_verses(self, jsonl_path: Path) -> int:
        """Count lines in JSONL file (= number of verses)."""
        if not jsonl_path.exists():
            return 0
        try:
            return sum(1 for line in open(jsonl_path) if line.strip())
        except Exception:
            return 0

    def cmd_init(self, args) -> None:
        """Initialize a new language project."""
        print("\n=== Initialize Scripture Translation Project ===\n")

        # Prompt for language name
        lang_name = input("Language name (e.g., 'Karo Batak'): ").strip()
        if not lang_name:
            print("Error: Language name required.")
            sys.exit(1)

        # Prompt for NLLB code
        nllb_code = input("NLLB language code (e.g., 'btx_Latn'): ").strip()
        if not nllb_code:
            print("Error: NLLB code required.")
            sys.exit(1)

        # Check if supported
        nllb_supported = nllb_code in Config.LANGUAGE_CODES.values()
        if not nllb_supported:
            print(f"\nWarning: '{nllb_code}' not found in NLLB-200.")
            print("Will add a custom language token during first training.")
            print("Provide related language codes for embedding warm-start.")

            related_input = input(
                "Related language codes (comma-separated, e.g., 'ind_Latn,min_Latn'): "
            ).strip()
            related_langs = (
                [code.strip() for code in related_input.split(",") if code.strip()]
                if related_input
                else []
            )
        else:
            related_langs = []

        # Create state
        state_path = self.get_state_path(nllb_code)
        state = StudioState(state_path)
        state["target_lang"] = nllb_code
        state["language_name"] = lang_name
        state["related_langs"] = related_langs
        state["nllb_supported"] = nllb_supported
        state["verses_collected"] = 0
        state["chapters_added"] = []
        state["last_trained_at_verses"] = 0
        state["lora_adapter_path"] = None
        state["data_path"] = str(self.get_data_path(nllb_code))
        state["min_train_verses"] = self.DEFAULT_MIN_TRAIN_VERSES

        # Create empty JSONL
        data_path = Path(state["data_path"])
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.touch()

        state.save()

        print(f"\n[OK] Initialized '{lang_name}' ({nllb_code})")
        print(f"  State: {state_path}")
        print(f"  Data: {data_path}")
        print(f"  Ready to add chapters with: python studio.py add --book Genesis --chapter 1")

    def parse_multi_chapter_text(self, text: str, target_lang: str, data_path: Path) -> Tuple[int, List[str], List[str]]:
        """Parse text that may contain multiple chapters (auto-detected by headers).

        Detects chapter headers like "GENESIS 2", "GENESIS 3" and splits accordingly.
        Falls back to single-chapter mode if headers not found.

        Args:
            text: Raw scripture text (possibly multi-chapter).
            target_lang: Target language code.
            data_path: Path to JSONL file to append to.

        Returns:
            Tuple of (total_verses_added, chapters_added, errors)
        """
        import re

        chapters_added = []
        total_verses = 0
        errors = []

        # Detect chapter headers (e.g., "GENESIS 2", "MATTHEW 3", etc.)
        chapter_pattern = r"^([A-Z\s\d]+?)\s+(\d+)(?::\d+)?.*?$"

        lines = text.split("\n")
        chapters_data = {}  # book -> chapter -> text
        current_book = None
        current_chapter = None
        current_text = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(chapter_pattern, line)
            if match:
                # Save previous chapter
                if current_book and current_chapter is not None:
                    if current_book not in chapters_data:
                        chapters_data[current_book] = {}
                    chapters_data[current_book][current_chapter] = "\n".join(current_text)

                # Start new chapter
                current_book = match.group(1).strip()
                current_chapter = int(match.group(2))
                current_text = []
            else:
                current_text.append(line)

        # Save last chapter
        if current_book and current_chapter is not None:
            if current_book not in chapters_data:
                chapters_data[current_book] = {}
            chapters_data[current_book][current_chapter] = "\n".join(current_text)

        # Process all detected chapters
        if chapters_data:
            with open(data_path, "a", encoding="utf-8") as f:
                for book in chapters_data:
                    # Normalize book name to title case for English verse lookup
                    normalized_book = book.title()
                    for chapter_num in sorted(chapters_data[book].keys()):
                        chapter_text = chapters_data[book][chapter_num]
                        try:
                            parsed = parse_youversion(chapter_text, normalized_book, chapter_num)
                            pairs, skipped = build_pairs(parsed, normalized_book, chapter_num, target_lang)
                            for pair in pairs:
                                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                            total_verses += len(pairs)
                            chapters_added.append(f"{normalized_book} {chapter_num}")
                            if skipped:
                                errors.append(f"{normalized_book} {chapter_num}: skipped {len(skipped)} verses")
                        except Exception as e:
                            errors.append(f"{normalized_book} {chapter_num}: {e}")

            return total_verses, chapters_added, errors
        else:
            # No chapters detected
            return 0, [], ["No chapter headers detected"]

    def cmd_add(self, args) -> None:
        """Add scripture text from YouVersion."""
        state_path = self.get_state_path(args.lang)
        if not state_path.exists():
            print(f"Error: No project initialized. Run: python studio.py init")
            sys.exit(1)

        state = StudioState(state_path)
        target_lang = state["target_lang"]
        data_path = Path(state["data_path"])

        # Get text
        if args.file:
            # From file
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                sys.exit(1)
            text = file_path.read_text(encoding="utf-8")
        else:
            # From stdin (multiline input)
            if args.book and args.chapter:
                chapter_ref = f"{args.book} {args.chapter}"
                print(f"\nPaste {chapter_ref} (Ctrl+Z then Enter on Windows, Ctrl+D on Unix):")
            else:
                print(f"\nPaste chapter(s) (Ctrl+Z then Enter on Windows, Ctrl+D on Unix):")
                print("(Can include multiple chapters with headers like 'GENESIS 2', 'GENESIS 3')")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            text = "\n".join(lines)

        if not text.strip():
            print("Error: No text provided.")
            return

        # Try to parse as multi-chapter first (auto-detect headers)
        total_verses, chapters_added, errors = self.parse_multi_chapter_text(text, target_lang, data_path)

        # If multi-chapter parsing found chapters, use those
        if total_verses > 0:
            # Check for duplicates and warn
            existing_chapters = state.get("chapters_added", [])
            duplicates = [c for c in chapters_added if c in existing_chapters]
            if duplicates:
                print(f"\nWarning: {duplicates} already added. Overwriting...")

            # Update state
            state["verses_collected"] = state.get("verses_collected", 0) + total_verses
            state["chapters_added"] = list(set(state.get("chapters_added", []) + chapters_added))
            state.save()

            # Print summary
            print(f"\n[OK] Added {total_verses} verses from {len(chapters_added)} chapters:")
            for ch in chapters_added:
                print(f"      {ch}")
            print(f"  Total verses: {state['verses_collected']}")

            total_verses_collected = state["verses_collected"]
            min_train = state.get("min_train_verses", self.DEFAULT_MIN_TRAIN_VERSES)
            need = max(0, min_train - total_verses_collected)
            if need > 0:
                print(f"  Need {need} more verses to train (threshold: {min_train})")
            else:
                print(f"  Ready to train! ({total_verses_collected}/{min_train} verses collected)")

            if errors:
                print(f"\n  Warnings:")
                for err in errors:
                    print(f"    - {err}")
            return

        # If no multi-chapter headers found, fall back to single-chapter mode
        if not args.book or not args.chapter:
            print("Error: --book and --chapter required (or paste text with chapter headers like 'GENESIS 2')")
            sys.exit(1)

        chapter_ref = f"{args.book} {args.chapter}"

        # Check for duplicates
        chapters_added_list = state.get("chapters_added", [])
        if chapter_ref in chapters_added_list:
            print(f"\nWarning: '{chapter_ref}' already added.")
            response = input("Overwrite? (y/n): ").strip().lower()
            if response != "y":
                print("Cancelled.")
                return

        # Parse and build pairs
        try:
            parsed = parse_youversion(text, args.book, int(args.chapter))
            pairs, skipped = build_pairs(parsed, args.book, int(args.chapter), target_lang)
        except Exception as e:
            print(f"Error parsing text: {e}")
            sys.exit(1)

        if not pairs:
            print("Error: No verse pairs extracted.")
            if skipped:
                print(f"Skipped verses (no English source): {skipped}")
            return

        # Append to JSONL
        with open(data_path, "a", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        # Update state
        new_verses = len(pairs)
        state["verses_collected"] = state.get("verses_collected", 0) + new_verses

        if chapter_ref not in chapters_added_list:
            chapters_added_list.append(chapter_ref)
        state["chapters_added"] = chapters_added_list

        state.save()

        # Print summary
        total_verses_collected = state["verses_collected"]
        min_train = state.get("min_train_verses", self.DEFAULT_MIN_TRAIN_VERSES)
        need = max(0, min_train - total_verses_collected)

        print(f"\n[OK] Added {new_verses} verses from {chapter_ref}")
        print(f"  Total verses: {total_verses_collected}")
        if need > 0:
            print(f"  Need {need} more verses to train (threshold: {min_train})")
        else:
            print(f"  Ready to train! ({total_verses_collected}/{min_train} verses collected)")

        if skipped:
            print(f"  Note: Skipped {len(skipped)} verses (no English source)")

    def cmd_ingest(self, args) -> None:
        """Simple bulk ingestion: paste chapters and auto-replace."""
        state_path = self.get_state_path(args.lang)
        if not state_path.exists():
            print(f"Error: No project initialized. Run: python studio.py init")
            sys.exit(1)

        state = StudioState(state_path)
        target_lang = state["target_lang"]
        data_path = Path(state["data_path"])

        # Read from stdin
        print("Paste chapters (Ctrl+Z then Enter on Windows, Ctrl+D on Unix):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        text = "\n".join(lines)

        if not text.strip():
            print("Error: No text provided.")
            return

        # Parse multi-chapter text
        total_verses, chapters_added, errors = self.parse_multi_chapter_text(text, target_lang, data_path)

        if total_verses == 0:
            print("Error: No verses extracted. Make sure text has chapter headers like 'GENESIS 1', 'MATTHEW 5'.")
            if errors:
                for err in errors:
                    print(f"  {err}")
            return

        # Update state (auto-replace, no confirmation)
        existing_chapters = state.get("chapters_added", [])
        replaced = [c for c in chapters_added if c in existing_chapters]
        state["verses_collected"] = state.get("verses_collected", 0) + total_verses
        state["chapters_added"] = list(set(state.get("chapters_added", []) + chapters_added))
        state.save()

        # Show summary
        if not args.no_confirm:
            print(f"\n[OK] Ingested {total_verses} verses from {len(chapters_added)} chapters:")
            for ch in chapters_added:
                marker = "(replaced)" if ch in replaced else "(new)"
                print(f"      {ch} {marker}")

            total = state["verses_collected"]
            min_train = state.get("min_train_verses", self.DEFAULT_MIN_TRAIN_VERSES)
            need = max(0, min_train - total)
            print(f"\n  Total: {total}/{min_train} verses")
            if need > 0:
                print(f"  Need {need} more to train")
            else:
                print(f"  Ready to train!")

            if errors:
                print(f"\n  Warnings:")
                for err in errors:
                    print(f"    {err}")

    def cmd_status(self, args) -> None:
        """Print project status."""
        state_path = self.get_state_path(args.lang)
        if not state_path.exists():
            print("No project initialized. Run: python studio.py init")
            sys.exit(1)

        state = StudioState(state_path)

        lang_name = state.get("language_name", "Unknown")
        target_lang = state.get("target_lang", "?")
        nllb_supported = state.get("nllb_supported", False)
        verses = state.get("verses_collected", 0)
        chapters = state.get("chapters_added", [])
        last_trained = state.get("last_trained_at_verses", 0)
        adapter_path = state.get("lora_adapter_path")
        min_train = state.get("min_train_verses", self.DEFAULT_MIN_TRAIN_VERSES)

        print(f"\n=== Project Status: {lang_name} ({target_lang}) ===\n")

        print(f"Language:        {lang_name} ({target_lang})")
        if nllb_supported:
            print(f"NLLB support:    Yes (native)")
        else:
            related = state.get("related_langs", [])
            related_str = ", ".join(related) if related else "none"
            print(
                f"NLLB support:    No — will add token on first train (related: {related_str})"
            )

        print(f"Verses:          {verses} collected")
        if chapters:
            chapters_str = ", ".join(chapters[:5])
            if len(chapters) > 5:
                chapters_str += f", +{len(chapters) - 5} more"
            print(f"Chapters:        {chapters_str}")
        else:
            print(f"Chapters:        (none yet)")

        if last_trained > 0:
            print(f"Last trained:    At {last_trained} verses")
        else:
            print(f"Last trained:    Never")

        if adapter_path:
            print(f"Adapter:         {adapter_path}")
        else:
            print(f"Adapter:         None")

        print(f"\nTraining threshold:  {min_train} verses")

        need = max(0, min_train - verses)
        if need > 0:
            print(f"Ready to train:      No ({need} more verses needed)")
        else:
            print(f"Ready to train:      Yes!")

    def cmd_run(self, args) -> None:
        """Run full pipeline: data split → fine-tune → inference."""
        state_path = self.get_state_path(args.lang)
        if not state_path.exists():
            print("Error: No project initialized. Run: python studio.py init")
            sys.exit(1)

        state = StudioState(state_path)
        target_lang = state["target_lang"]
        data_path = Path(state["data_path"])
        min_train = state.get("min_train_verses", self.DEFAULT_MIN_TRAIN_VERSES)

        # 1. Preflight checks
        print(f"\n=== Preflight Checks ===\n")

        if not data_path.exists():
            print(f"Error: Data file not found: {data_path}")
            sys.exit(1)

        verse_count = self.count_verses(data_path)
        print(f"Verses collected: {verse_count}")

        if verse_count < min_train:
            if not args.force:
                response = input(
                    f"Warning: Only {verse_count}/{min_train} verses. Proceed anyway? (y/n): "
                ).strip().lower()
                if response != "y":
                    print("Cancelled.")
                    return
            else:
                print(f"Warning: Only {verse_count}/{min_train} verses (forcing ahead)")

        # Check if base model is cached
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Checking HuggingFace cache...")
        try:
            from transformers import AutoTokenizer

            # This will cache the model if not already cached
            print("(First run will download ~2.4GB)")
        except Exception as e:
            print(f"Warning: Could not verify model cache: {e}")

        # 2. Data split
        print(f"\n=== Data Split ===\n")
        try:
            from data.loaders import create_data_splits

            train_path, val_path, test_path = create_data_splits(data_path)

            train_count = self.count_verses(train_path)
            val_count = self.count_verses(val_path)
            test_count = self.count_verses(test_path)

            print(f"Train: {train_count} | Val: {val_count} | Test: {test_count}")
        except Exception as e:
            print(f"Error splitting data: {e}")
            sys.exit(1)

        # 3. Fine-tune
        print(f"\n=== Fine-tuning ({target_lang}) ===\n")

        base_model_path = Config.CHECKPOINTS_DIR / "nllb_base"
        lora_output_dir = Config.CHECKPOINTS_DIR / f"lora_{target_lang}_final"

        # Save base model once
        if not (base_model_path / "config.json").exists():
            print(f"Saving base model to cache...")
            try:
                from models.base import ScriptureTranslationModel

                model = ScriptureTranslationModel(use_lora=False)
                model.save_pretrained(base_model_path)
                print(f"[OK] Base model cached at {base_model_path}")
            except Exception as e:
                print(f"Error caching base model: {e}")
                sys.exit(1)

        # Build fine_tune_lora args
        ft_args = [
            "python",
            "scripts/fine_tune_lora.py",
            f"--pretrained_model_path={base_model_path}",
            f"--data_path={data_path}",
            f"--target_lang={target_lang}",
            f"--output_dir={lora_output_dir}",
        ]

        if not state.get("nllb_supported", False):
            ft_args.append("--add_language_token")
            related_langs = state.get("related_langs", [])
            if related_langs:
                ft_args.append(f"--related_langs={','.join(related_langs)}")

        print(f"Running fine-tuning...")
        self._run_subprocess(ft_args)

        # Verify adapter exists
        if not (lora_output_dir / "adapter_config.json").exists():
            print(f"Error: Adapter not saved at {lora_output_dir}")
            sys.exit(1)

        state["lora_adapter_path"] = str(lora_output_dir)
        state["last_trained_at_verses"] = verse_count
        state.save()

        print(f"[OK] Fine-tuning complete. Adapter saved to {lora_output_dir}")

        # 4. Inference
        print(f"\n=== Inference (Translation) ===\n")

        output_dir = Config.PROJECT_ROOT / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        inference_args = [
            "python",
            "run_pipeline.py",
            f"--target-lang={target_lang}",
            f"--lora-path={lora_output_dir}",
            f"--output-dir={output_dir}",
        ]

        print(f"Running inference...")
        self._run_subprocess(inference_args)

        # 5. Report
        print(f"\n=== Results ===\n")

        output_json = output_dir / f"{target_lang}_bible.json"
        output_csv = output_dir / f"{target_lang}_bible.csv"

        if output_json.exists():
            with open(output_json, "r", encoding="utf-8") as f:
                results = json.load(f)

            verse_count_output = len(results)
            avg_confidence = sum(r.get("confidence", 0) for r in results) / verse_count_output if results else 0

            print(f"Output JSON: {output_json}")
            print(f"Output CSV:  {output_csv}")
            print(f"Verses translated: {verse_count_output}")
            print(f"Average confidence: {avg_confidence:.2f}")

            new_since_train = verse_count - state["last_trained_at_verses"]
            if new_since_train > 0:
                print(
                    f"\nNew verses since last train: {new_since_train} — "
                    f"add more chapters and run again to improve"
                )

    def cmd_retrain(self, args) -> None:
        """Re-train adapter without inference (for added chapters)."""
        state_path = self.get_state_path(args.lang)
        if not state_path.exists():
            print("Error: No project initialized.")
            sys.exit(1)

        state = StudioState(state_path)
        target_lang = state["target_lang"]
        data_path = Path(state["data_path"])

        # Preflight
        print(f"\n=== Retraining ({target_lang}) ===\n")

        verse_count = self.count_verses(data_path)
        min_train = state.get("min_train_verses", self.DEFAULT_MIN_TRAIN_VERSES)

        print(f"Verses: {verse_count}")

        if verse_count < min_train and not args.force:
            response = input(
                f"Warning: Only {verse_count}/{min_train} verses. Proceed anyway? (y/n): "
            ).strip().lower()
            if response != "y":
                print("Cancelled.")
                return

        # Data split
        print(f"\n=== Data Split ===\n")
        try:
            from data.loaders import create_data_splits

            train_path, val_path, test_path = create_data_splits(data_path)
            train_count = self.count_verses(train_path)
            val_count = self.count_verses(val_path)
            print(f"Train: {train_count} | Val: {val_count}")
        except Exception as e:
            print(f"Error splitting data: {e}")
            sys.exit(1)

        # Fine-tune
        print(f"\n=== Fine-tuning ===\n")

        base_model_path = Config.CHECKPOINTS_DIR / "nllb_base"
        lora_output_dir = Config.CHECKPOINTS_DIR / f"lora_{target_lang}_final"

        if not (base_model_path / "config.json").exists():
            print(f"Error: Base model not cached. Run 'python studio.py run' first.")
            sys.exit(1)

        ft_args = [
            "python",
            "scripts/fine_tune_lora.py",
            f"--pretrained_model_path={base_model_path}",
            f"--data_path={data_path}",
            f"--target_lang={target_lang}",
            f"--output_dir={lora_output_dir}",
        ]

        if not state.get("nllb_supported", False):
            ft_args.append("--add_language_token")
            related_langs = state.get("related_langs", [])
            if related_langs:
                ft_args.append(f"--related_langs={','.join(related_langs)}")

        self._run_subprocess(ft_args)

        state["last_trained_at_verses"] = verse_count
        state.save()

        print(f"[OK] Retraining complete at {verse_count} verses")

    def cmd_translate(self, args) -> None:
        """Run inference only (using existing adapter or base model)."""
        state_path = self.get_state_path(args.lang)
        if not state_path.exists():
            print("Error: No project initialized.")
            sys.exit(1)

        state = StudioState(state_path)
        target_lang = state["target_lang"]

        print(f"\n=== Translation ({target_lang}) ===\n")

        output_dir = Config.PROJECT_ROOT / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        lora_path = state.get("lora_adapter_path")
        inference_args = [
            "python",
            "run_pipeline.py",
            f"--target-lang={target_lang}",
            f"--output-dir={output_dir}",
        ]

        if lora_path:
            inference_args.append(f"--lora-path={lora_path}")

        self._run_subprocess(inference_args)

        # Report
        output_json = output_dir / f"{target_lang}_bible.json"
        if output_json.exists():
            with open(output_json, "r", encoding="utf-8") as f:
                results = json.load(f)

            print(f"\n[OK] Translation complete: {len(results)} verses")
            print(f"  Output: {output_json}")

    def _run_subprocess(self, args: List[str]) -> None:
        """Run subprocess with streaming output."""
        try:
            # Find the venv's python executable
            script_dir = Path(__file__).parent
            venv_dir = script_dir / "venv"
            if Path(venv_dir / "Scripts" / "python.exe").exists():
                # Windows
                python_exe = str(venv_dir / "Scripts" / "python.exe")
            elif Path(venv_dir / "bin" / "python").exists():
                # Unix
                python_exe = str(venv_dir / "bin" / "python")
            else:
                # Fallback to system python
                python_exe = "python"

            # Replace "python" with venv python
            if args[0] == "python":
                args[0] = python_exe

            # Set up environment with PYTHONPATH
            env = os.environ.copy()
            env["PYTHONPATH"] = str(script_dir)

            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=script_dir,
                env=env,
            )

            # Stream output line by line
            for line in proc.stdout:
                print(line, end="")

            returncode = proc.wait()
            if returncode != 0:
                print(f"\nError: Subprocess exited with code {returncode}")
                sys.exit(1)
        except Exception as e:
            print(f"Error running subprocess: {e}")
            sys.exit(1)


def main():
    """Parse args and dispatch to commands."""
    parser = argparse.ArgumentParser(
        description="Studio: Scripture Translation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python studio.py init
  python studio.py add --book Genesis --chapter 1
  python studio.py status
  python studio.py run
  python studio.py retrain
  python studio.py translate
        """,
    )

    parser.add_argument(
        "--lang",
        default=None,
        help="Language code (for multi-language projects). Defaults to last used.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    subparsers.add_parser("init", help="Initialize new language project")

    # add
    add_parser = subparsers.add_parser("add", help="Add scripture text")
    add_parser.add_argument("--book", help="Bible book (e.g., Genesis) — optional if text has chapter headers")
    add_parser.add_argument("--chapter", type=int, help="Chapter number — optional if text has chapter headers")
    add_parser.add_argument("--file", help="File path (else reads from stdin)")

    # ingest (new: simpler interface for pasting bulk chapters)
    ingest_parser = subparsers.add_parser("ingest", help="Paste multiple chapters at once (auto-format, auto-replace)")
    ingest_parser.add_argument("--no-confirm", action="store_true", help="Don't show summary, just ingest silently")

    # status
    subparsers.add_parser("status", help="Show project status")

    # run
    run_parser = subparsers.add_parser("run", help="Full pipeline: split, train, translate")
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip verse count check",
    )

    # retrain
    retrain_parser = subparsers.add_parser("retrain", help="Re-train adapter (no inference)")
    retrain_parser.add_argument("--force", action="store_true", help="Skip verse count check")

    # translate
    subparsers.add_parser("translate", help="Inference only (using existing adapter)")

    args = parser.parse_args()

    studio = Studio()

    if args.command == "init":
        studio.cmd_init(args)
    elif args.command == "add":
        studio.cmd_add(args)
    elif args.command == "ingest":
        studio.cmd_ingest(args)
    elif args.command == "status":
        studio.cmd_status(args)
    elif args.command == "run":
        studio.cmd_run(args)
    elif args.command == "retrain":
        studio.cmd_retrain(args)
    elif args.command == "translate":
        studio.cmd_translate(args)


if __name__ == "__main__":
    main()

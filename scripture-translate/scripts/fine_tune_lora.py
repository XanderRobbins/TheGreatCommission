"""LoRA fine-tuning script for rare language scripture translation."""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from config import Config
from data.loaders import BibleTranslationDataset, create_data_splits
from models.base import ScriptureTranslationModel
from utils.logger import configure_logging
from constants import MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH
from base_pipeline import BaseTrainingPipeline


class LoRATrainingPipeline(BaseTrainingPipeline):
    """LoRA fine-tuning pipeline."""

    def prepare_data(self):
        """Prepare training data for rare language."""
        logger = __import__('utils.logger', fromlist=['get_logger']).get_logger(__name__)
        logger.info(f"Loading rare language data from {self.args.data_path}")

        corpus_path = Path(self.args.data_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.args.data_path}")

        # Create data splits
        train_path, val_path, test_path = create_data_splits(corpus_path)

        # Load pretrained model and get tokenizer
        model_wrapper = ScriptureTranslationModel(use_lora=False)
        model_wrapper.load_pretrained(Path(self.args.pretrained_model_path))

        # Add new language token if this language isn't in NLLB-200
        if self.args.add_language_token:
            related = self.args.related_langs.split(",") if self.args.related_langs else None
            added = model_wrapper.add_language_token(self.args.target_lang, related)
            if added:
                logger.info(
                    f"New language token added for '{self.args.target_lang}'. "
                    f"Save the modified model after training — it has a larger vocab."
                )

        tokenizer = model_wrapper.get_tokenizer()

        # Create datasets
        logger.info("Creating datasets...")
        train_dataset = BibleTranslationDataset(
            train_path,
            tokenizer,
            self.args.source_lang,
            self.args.target_lang,
            max_source_length=MAX_SOURCE_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
        )

        val_dataset = BibleTranslationDataset(
            val_path,
            tokenizer,
            self.args.source_lang,
            self.args.target_lang,
            max_source_length=MAX_SOURCE_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")

        return train_loader, val_loader, model_wrapper

    def setup_optimizer(self, model_wrapper, total_steps: int):
        """Setup optimizer and scheduler for LoRA parameters."""
        # Apply LoRA to model
        model_wrapper.apply_lora()

        optimizer = AdamW(
            model_wrapper.get_model().parameters(),
            lr=self.args.learning_rate,
            weight_decay=0.01,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=Config.FINETUNING_CONFIG["warmup_steps"],
            num_training_steps=total_steps,
        )

        return optimizer, scheduler

    def get_checkpoint_dir(self, epoch: int) -> Path:
        """Get checkpoint directory for epoch."""
        return Config.CHECKPOINTS_DIR / f"lora_{self.args.target_lang}_best_epoch{epoch+1}"

    def get_final_model_dir(self) -> Path:
        """Get final model directory."""
        if getattr(self.args, "output_dir", None):
            return Path(self.args.output_dir)
        return Config.CHECKPOINTS_DIR / f"lora_{self.args.target_lang}_final"

    def get_history_filename(self) -> str:
        """Get training history filename."""
        return f"lora_training_history_{self.args.target_lang}.json"


def main():
    # Setup directories and logging
    Config.ensure_dirs()
    log_file = Config.LOGS_DIR / f"lora_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    configure_logging(level=logging.INFO, log_file=log_file)

    parser = argparse.ArgumentParser(description="LoRA fine-tune on rare language")

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to rare language parallel corpus in JSONL format"
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default="eng_Latn",
        help="Source language code (NLLB format)"
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        help="Target language code (NLLB format)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=Config.FINETUNING_CONFIG["batch_size"],
        help="Batch size (smaller for LoRA)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=Config.FINETUNING_CONFIG["learning_rate"],
        help="Learning rate for LoRA"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=Config.FINETUNING_CONFIG["num_epochs"],
        help="Number of epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=Config.FINETUNING_CONFIG.get("gradient_accumulation_steps", 1),
        help="Gradient accumulation steps"
    )

    parser.add_argument(
        "--add_language_token",
        action="store_true",
        help="Add target_lang as a new token if not in NLLB-200 (required for unsupported languages)"
    )
    parser.add_argument(
        "--related_langs",
        type=str,
        default=None,
        help="Comma-separated NLLB codes of related languages for embedding init "
             "(e.g., 'ind_Latn,zsm_Latn,min_Latn' for Karo Batak)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the trained adapter (overrides default)"
    )

    args = parser.parse_args()

    # Run LoRA fine-tuning
    pipeline = LoRATrainingPipeline(args)
    model = pipeline.run()


if __name__ == "__main__":
    main()

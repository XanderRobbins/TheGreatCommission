"""Baseline training script for scripture translation."""

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


class TrainingPipeline(BaseTrainingPipeline):
    """Baseline training pipeline (no LoRA)."""

    def prepare_data(self):
        """Prepare training data."""
        logger = __import__('utils.logger', fromlist=['get_logger']).get_logger(__name__)
        logger.info(f"Loading data from {self.args.data_path}")

        corpus_path = Path(self.args.data_path)
        if corpus_path.exists():
            train_path, val_path, test_path = create_data_splits(corpus_path)
        else:
            raise FileNotFoundError(f"Data path not found: {self.args.data_path}")

        # Initialize model and get tokenizer
        model_wrapper = ScriptureTranslationModel(use_lora=False)
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
        """Setup optimizer and scheduler."""
        optimizer = AdamW(
            model_wrapper.get_model().parameters(),
            lr=self.args.learning_rate,
            weight_decay=Config.TRAINING_CONFIG["weight_decay"],
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=Config.TRAINING_CONFIG["warmup_steps"],
            num_training_steps=total_steps,
        )

        return optimizer, scheduler


def main():
    # Setup directories and logging
    Config.ensure_dirs()
    log_file = Config.LOGS_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    configure_logging(level=logging.INFO, log_file=log_file)

    parser = argparse.ArgumentParser(description="Train scripture translation model")

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/verse_pairs.jsonl",
        help="Path to parallel corpus in JSONL format"
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
        default="spa_Latn",
        help="Target language code (NLLB format)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=Config.TRAINING_CONFIG["batch_size"],
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=Config.TRAINING_CONFIG["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=Config.TRAINING_CONFIG["num_epochs"],
        help="Number of epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=Config.TRAINING_CONFIG["gradient_accumulation_steps"],
        help="Gradient accumulation steps"
    )

    args = parser.parse_args()

    # Run training
    pipeline = TrainingPipeline(args)
    model = pipeline.run()


if __name__ == "__main__":
    main()

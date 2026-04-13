"""Base training pipeline to eliminate code duplication.

Shared logic for both baseline and LoRA training.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.loaders import BibleTranslationDataset
from models.base import ScriptureTranslationModel, TranslationTrainer
from models.terminology import TerminologyDB
from constants import GRAD_CLIP_NORM
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseTrainingPipeline(ABC):
    """Abstract base class for training pipelines."""

    def __init__(self, args):
        """Initialize pipeline with arguments."""
        self.args = args
        self.device = Config.get_device()
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)

        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Epochs: {args.num_epochs}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

    @abstractmethod
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, ScriptureTranslationModel]:
        """Prepare training and validation data.

        Returns:
            (train_loader, val_loader, model_wrapper)
        """
        pass

    @abstractmethod
    def setup_optimizer(self, model_wrapper, total_steps: int):
        """Setup optimizer and scheduler.

        Returns:
            (optimizer, scheduler)
        """
        pass

    def train_epoch(
        self, trainer: TranslationTrainer, train_loader: DataLoader, epoch: int
    ) -> float:
        """Train for one epoch with gradient accumulation."""
        logger.info(f"\n=== Epoch {epoch + 1}/{self.args.num_epochs} ===")

        total_loss = 0
        accumulated_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = trainer.model.get_model()(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                decoder_attention_mask=batch.get("decoder_attention_mask"),
            )

            loss = trainer.compute_loss(outputs, batch["labels"])

            # Scale loss by accumulation steps
            loss_scaled = loss / self.gradient_accumulation_steps

            # Backward pass
            loss_scaled.backward()

            accumulated_loss += loss.item()
            total_loss += loss.item()

            # Step optimizer and scheduler every N accumulation steps
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.get_model().parameters(), GRAD_CLIP_NORM
                )
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()

                if trainer.scheduler:
                    trainer.scheduler.step()

                metrics = {
                    "loss": accumulated_loss / self.gradient_accumulation_steps,
                    "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                }
                progress_bar.set_postfix(metrics)
                accumulated_loss = 0

                if (step + 1) % 100 == 0:
                    logger.info(f"Step {step + 1}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

        return avg_loss

    def evaluate(self, model_wrapper: ScriptureTranslationModel, val_loader: DataLoader) -> float:
        """Evaluate on validation set."""
        logger.info("\nEvaluating on validation set...")

        model = model_wrapper.get_model()
        model.eval()

        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    decoder_attention_mask=batch.get("decoder_attention_mask"),
                )

                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(val_loader)
        logger.info(f"Validation loss: {avg_loss:.4f}")

        model.train()
        return avg_loss

    def run(self):
        """Execute full training pipeline."""
        logger.info("Starting training pipeline...")

        # Prepare data
        train_loader, val_loader, model_wrapper = self.prepare_data()

        # Setup optimization
        total_steps = len(train_loader) * self.args.num_epochs
        optimizer, scheduler = self.setup_optimizer(model_wrapper, total_steps)

        # Create terminology database
        terminology_db = TerminologyDB()

        # Create trainer
        trainer = TranslationTrainer(
            model=model_wrapper,
            optimizer=optimizer,
            scheduler=scheduler,
            terminology_db=terminology_db,
            device=self.device,
        )

        # Training loop
        best_val_loss = float('inf')
        training_history = []
        epochs_without_improvement = 0
        patience = 2
        best_checkpoint_dir = None

        for epoch in range(self.args.num_epochs):
            # Train
            train_loss = self.train_epoch(trainer, train_loader, epoch)

            # Validate
            val_loss = self.evaluate(model_wrapper, val_loader)

            # Record metrics
            history_item = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            training_history.append(history_item)

            # Save checkpoint if validation improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_checkpoint_dir = self.get_checkpoint_dir(epoch)
                model_wrapper.save_pretrained(best_checkpoint_dir)
                logger.info(f"Saved best model to {best_checkpoint_dir}")
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"No improvement for {epochs_without_improvement}/{patience} epochs "
                    f"(best val loss: {best_val_loss:.4f})"
                )
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Save final model (use best checkpoint if available)
        final_model_dir = self.get_final_model_dir()
        if best_checkpoint_dir and best_checkpoint_dir != final_model_dir:
            import shutil
            if final_model_dir.exists():
                shutil.rmtree(final_model_dir)
            shutil.copytree(best_checkpoint_dir, final_model_dir)
            logger.info(f"Copied best checkpoint to {final_model_dir}")
        else:
            model_wrapper.save_pretrained(final_model_dir)
            logger.info(f"Saved final model to {final_model_dir}")

        # Save training history
        history_path = Config.RESULTS_DIR / self.get_history_filename()
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved training history to {history_path}")

        # Save terminology database
        terminology_db.save()

        logger.info("\n=== Training Complete ===")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

        return model_wrapper

    def get_checkpoint_dir(self, epoch: int) -> Path:
        """Get checkpoint directory for epoch."""
        return Config.CHECKPOINTS_DIR / f"best_model_epoch{epoch+1}"

    def get_final_model_dir(self) -> Path:
        """Get final model directory."""
        return Config.CHECKPOINTS_DIR / "final_model"

    def get_history_filename(self) -> str:
        """Get training history filename."""
        return "training_history.json"

import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from config import Config
from data.loaders import BibleTranslationDataset, create_data_splits
from models.base import ScriptureTranslationModel, TranslationTrainer
from models.terminology import TerminologyDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline for scripture translation"""
    
    def __init__(self, args):
        self.args = args
        self.device = Config.get_device()
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Epochs: {args.num_epochs}")
    
    def prepare_data(self) -> tuple:
        """Prepare training data"""
        logger.info(f"Loading data from {self.args.data_path}")
        
        # Create data splits if they don't exist
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
            max_source_length=512,
            max_target_length=256,
        )
        
        val_dataset = BibleTranslationDataset(
            val_path,
            tokenizer,
            self.args.source_lang,
            self.args.target_lang,
            max_source_length=512,
            max_target_length=256,
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
        """Setup optimizer and scheduler"""
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
    
    def train_epoch(self, trainer, train_loader, epoch: int) -> float:
        """Train for one epoch"""
        logger.info(f"\n=== Epoch {epoch + 1}/{self.args.num_epochs} ===")
        
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            loss, metrics = trainer.train_step(batch)
            total_loss += loss
            
            progress_bar.set_postfix(metrics)
            
            if (step + 1) % 100 == 0:
                logger.info(f"Step {step + 1}: Loss = {loss:.4f}")
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self, model_wrapper, val_loader) -> float:
        """Evaluate on validation set"""
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
        """Execute full training pipeline"""
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
                checkpoint_dir = Config.CHECKPOINTS_DIR / f"best_model_epoch{epoch+1}"
                model_wrapper.save_pretrained(checkpoint_dir)
                logger.info(f"Saved best model to {checkpoint_dir}")
        
        # Save final model
        final_model_dir = Config.CHECKPOINTS_DIR / f"final_model"
        model_wrapper.save_pretrained(final_model_dir)
        logger.info(f"Saved final model to {final_model_dir}")
        
        # Save training history
        history_path = Config.RESULTS_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
        
        # Save terminology database
        terminology_db.save()
        
        logger.info("\n=== Training Complete ===")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        return model_wrapper


def main():
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
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for fine-tuning"
    )
    
    args = parser.parse_args()
    
    # Run training
    pipeline = TrainingPipeline(args)
    model = pipeline.run()


if __name__ == "__main__":
    main()

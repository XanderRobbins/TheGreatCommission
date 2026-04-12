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
        logging.FileHandler(Config.LOGS_DIR / f"lora_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LoRATrainingPipeline:
    """
    Fine-tune a pretrained model on a rare language using LoRA.
    
    This is memory-efficient because we only train a small set of parameters.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = Config.get_device()
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Epochs: {args.num_epochs}")
        logger.info(f"Target language: {args.target_lang}")
    
    def load_pretrained_model(self) -> ScriptureTranslationModel:
        """Load pretrained model from checkpoint"""
        logger.info(f"Loading pretrained model from {self.args.pretrained_model_path}")
        
        model_wrapper = ScriptureTranslationModel(use_lora=False)
        model_wrapper.load_pretrained(Path(self.args.pretrained_model_path))
        
        logger.info("Pretrained model loaded")
        return model_wrapper
    
    def apply_lora(self, model_wrapper: ScriptureTranslationModel):
        """Apply LoRA to pretrained model"""
        logger.info("Applying LoRA adapters...")
        model_wrapper.apply_lora()
        logger.info(f"Trainable parameters: {model_wrapper.count_parameters():,}")
    
    def prepare_data(self) -> tuple:
        """Prepare training data for rare language"""
        logger.info(f"Loading rare language data from {self.args.data_path}")
        
        corpus_path = Path(self.args.data_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.args.data_path}")
        
        # Create data splits
        train_path, val_path, test_path = create_data_splits(corpus_path)
        
        # Load model to get tokenizer
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
        
        # Create data loaders with smaller batch size
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
        """Setup optimizer with LoRA parameters"""
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
    
    def train_epoch(self, trainer, train_loader, epoch: int) -> float:
        """Train for one epoch"""
        logger.info(f"\n=== LoRA Epoch {epoch + 1}/{self.args.num_epochs} ===")
        
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="LoRA Training")
        
        for step, batch in enumerate(progress_bar):
            loss, metrics = trainer.train_step(batch)
            total_loss += loss
            
            progress_bar.set_postfix(metrics)
            
            if (step + 1) % 50 == 0:
                logger.info(f"Step {step + 1}: Loss = {loss:.4f}")
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"LoRA Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")
        
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
        """Execute full LoRA fine-tuning pipeline"""
        logger.info("Starting LoRA fine-tuning pipeline...")
        
        # Load pretrained model
        model_wrapper = self.load_pretrained_model()
        
        # Apply LoRA
        self.apply_lora(model_wrapper)
        
        # Prepare data
        train_loader, val_loader, _ = self.prepare_data()
        
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
                checkpoint_dir = Config.CHECKPOINTS_DIR / f"lora_{self.args.target_lang}_best_epoch{epoch+1}"
                model_wrapper.save_pretrained(checkpoint_dir)
                logger.info(f"Saved best LoRA model to {checkpoint_dir}")
        
        # Save final model
        final_model_dir = Config.CHECKPOINTS_DIR / f"lora_{self.args.target_lang}_final"
        model_wrapper.save_pretrained(final_model_dir)
        logger.info(f"Saved final LoRA model to {final_model_dir}")
        
        # Save training history
        history_path = Config.RESULTS_DIR / f"lora_training_history_{self.args.target_lang}.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
        
        # Save terminology database
        terminology_db.save()
        
        logger.info("\n=== LoRA Fine-tuning Complete ===")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Model saved to: {final_model_dir}")
        
        return model_wrapper


def main():
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
    
    args = parser.parse_args()
    
    # Run LoRA fine-tuning
    pipeline = LoRATrainingPipeline(args)
    model = pipeline.run()


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional, Dict, Tuple
import logging
from pathlib import Path

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScriptureTranslationModel:
    """
    Wrapper around NLLB for scripture translation with optional LoRA fine-tuning.
    """
    
    def __init__(self, model_name: str = None, use_lora: bool = False,
                 device: str = None):
        """
        Initialize the model.
        
        Args:
            model_name: HuggingFace model identifier
            use_lora: Whether to apply LoRA adapters
            device: Device to load model on ('cuda' or 'cpu')
        """
        self.model_name = model_name or Config.MODEL_NAME
        self.device = device or Config.get_device()
        self.use_lora = use_lora
        
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        logger.info(f"Model loaded. Parameters: {self.count_parameters():,}")
        
        # Apply LoRA if requested
        if use_lora:
            self.apply_lora()
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def apply_lora(self):
        """Apply LoRA (Low-Rank Adaptation) to the model"""
        logger.info("Applying LoRA adapters...")
        
        lora_config = LoraConfig(
            r=Config.LORA_CONFIG["r"],
            lora_alpha=Config.LORA_CONFIG["lora_alpha"],
            target_modules=Config.LORA_CONFIG["target_modules"],
            lora_dropout=Config.LORA_CONFIG["lora_dropout"],
            bias=Config.LORA_CONFIG["bias"],
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable = self.count_parameters()
        total = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"LoRA applied. Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        self.model.print_trainable_parameters()
    
    def get_model(self) -> PreTrainedModel:
        """Get the underlying model"""
        return self.model
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer"""
        return self.tokenizer
    
    def freeze_encoder(self):
        """Freeze encoder weights (only train decoder)"""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights"""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")
    
    def set_source_language(self, language_code: str):
        """Set source language for tokenizer"""
        self.tokenizer.src_lang = language_code
        logger.info(f"Source language set to: {language_code}")
    
    def save_pretrained(self, save_path: Path):
        """Save model and tokenizer"""
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        logger.info(f"Model saved to {save_path}")
    
    def load_pretrained(self, load_path: Path):
        """Load model and tokenizer from checkpoint"""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(load_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(load_path))
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {load_path}")
    
    def to(self, device: str):
        """Move model to device"""
        self.device = device
        self.model.to(device)
        logger.info(f"Model moved to {device}")


class ConsistencyLoss(nn.Module):
    """
    Custom loss for enforcing terminology consistency in translations.
    
    Penalizes the model when the same source term maps to different
    target terms across different verses.
    """
    
    def __init__(self, terminology_db, weight: float = 0.1):
        super().__init__()
        self.terminology_db = terminology_db
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
               source_terms: Dict[str, str]) -> torch.Tensor:
        """
        Calculate consistency loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target sequences [batch_size, seq_len]
            source_terms: Dictionary mapping source terms to their locations
        
        Returns:
            Consistency loss scalar
        """
        # For now, return zero loss (full implementation would track
        # term consistency across batch)
        # This is a placeholder that would be expanded with actual logic
        return torch.tensor(0.0, device=predictions.device)


class TranslationTrainer:
    """
    Trainer for scripture translation model with both standard MT loss
    and consistency loss.
    """
    
    def __init__(self, model: ScriptureTranslationModel,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[object] = None,
                 terminology_db=None,
                 device: str = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.terminology_db = terminology_db
        self.device = device or Config.get_device()
        
        self.losses = []
    
    def compute_loss(self, outputs, labels: torch.Tensor,
                    alpha: float = None) -> torch.Tensor:
        """
        Compute combined loss (translation + consistency).
        
        Args:
            outputs: Model outputs from forward pass
            labels: Target token IDs
            alpha: Weight for consistency loss (0.0-1.0)
        
        Returns:
            Combined loss tensor
        """
        alpha = alpha or Config.LOSS_WEIGHTS["consistency_loss"]
        
        # Standard translation loss
        translation_loss = outputs.loss
        
        # Consistency loss (simplified for now)
        consistency_loss = torch.tensor(0.0, device=self.device)
        
        # Combined loss
        total_loss = translation_loss + alpha * consistency_loss
        
        return total_loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict]:
        """
        Single training step.
        
        Args:
            batch: Dictionary with input_ids, attention_mask, labels
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model.get_model()(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            decoder_attention_mask=batch.get("decoder_attention_mask"),
        )
        
        loss = self.compute_loss(outputs, batch["labels"])
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.get_model().parameters(), 1.0)
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        self.losses.append(loss.item())
        
        metrics = {
            "loss": loss.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        
        return loss.item(), metrics
    
    def get_loss_history(self):
        """Get training loss history"""
        return self.losses


if __name__ == "__main__":
    # Test model initialization
    print("Testing model initialization...")
    
    model = ScriptureTranslationModel(use_lora=False)
    print(f"Model type: {type(model.get_model())}")
    print(f"Tokenizer type: {type(model.get_tokenizer())}")
    
    # Test with LoRA
    print("\nTesting LoRA initialization...")
    model_lora = ScriptureTranslationModel(use_lora=True)
    print(f"LoRA model parameters: {model_lora.count_parameters():,}")

"""Model wrappers and training utilities for scripture translation.

Provides NLLB model initialization with optional LoRA fine-tuning,
custom consistency loss, and training utilities.
"""

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
from pathlib import Path

from config import Config
from utils.logger import get_logger
from constants import GRAD_CLIP_NORM

logger = get_logger(__name__)


class ScriptureTranslationModel:
    """Wrapper around NLLB for scripture translation with optional LoRA fine-tuning.

    Provides an interface to load the NLLB-200 model, manage devices,
    apply parameter-efficient adapters (LoRA), and manage tokenization.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_lora: bool = False,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the model.

        Args:
            model_name: HuggingFace model identifier. Defaults to Config.MODEL_NAME.
            use_lora: Whether to apply LoRA adapters for efficient fine-tuning.
            device: Device to load model on ('cuda', 'cpu', etc). Defaults to Config.get_device().
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
        """Count trainable parameters.

        Returns:
            Number of parameters that require gradients.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def apply_lora(self) -> None:
        """Apply LoRA (Low-Rank Adaptation) to the model.

        Wraps the model with parameter-efficient LoRA adapters from PEFT library.
        Only a small fraction of parameters become trainable (typically 0.1-1%).
        """
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

        logger.info(
            f"LoRA applied. Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)"
        )
        self.model.print_trainable_parameters()

    def get_model(self) -> PreTrainedModel:
        """Get the underlying model.

        Returns:
            The HuggingFace PreTrainedModel instance.
        """
        return self.model

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer.

        Returns:
            The HuggingFace PreTrainedTokenizer instance.
        """
        return self.tokenizer

    def freeze_encoder(self) -> None:
        """Freeze encoder weights (only train decoder)."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")

    def set_source_language(self, language_code: str) -> None:
        """Set source language for tokenizer.

        Args:
            language_code: NLLB language code (e.g., "eng_Latn").
        """
        self.tokenizer.src_lang = language_code
        logger.info(f"Source language set to: {language_code}")

    def save_pretrained(self, save_path: Path) -> None:
        """Save model and tokenizer to directory.

        Args:
            save_path: Directory path where model will be saved.
        """
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        logger.info(f"Model saved to {save_path}")

    def add_language_token(
        self,
        lang_code: str,
        related_lang_codes: Optional[list] = None,
    ) -> bool:
        """Add a new language token to the tokenizer and resize model embeddings.

        Used for languages not in NLLB-200. Adds the token, resizes the embedding
        table, and initializes the new embedding from the average of related languages
        (warm-start rather than random noise).

        Args:
            lang_code: New language code to add (e.g., "btx_Latn" for Karo Batak).
            related_lang_codes: NLLB codes of related languages to average for
                initialization. If None or empty, falls back to random init.

        Returns:
            True if token was added, False if it already existed.
        """
        # Check if already present
        existing_id = self.tokenizer.convert_tokens_to_ids(lang_code)
        unk_id = self.tokenizer.unk_token_id
        if existing_id != unk_id:
            logger.info(f"Language token '{lang_code}' already exists (id={existing_id})")
            return False

        logger.info(f"Adding new language token: {lang_code}")

        # Add token to tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": [lang_code]})
        new_token_id = self.tokenizer.convert_tokens_to_ids(lang_code)
        logger.info(f"Token added with id={new_token_id}")

        # Resize model embedding tables
        self.model.resize_token_embeddings(len(self.tokenizer))
        logger.info(f"Embeddings resized to vocab size {len(self.tokenizer)}")

        # Warm-start: initialize from average of related language embeddings
        if related_lang_codes:
            related_ids = []
            for code in related_lang_codes:
                rid = self.tokenizer.convert_tokens_to_ids(code)
                if rid != unk_id:
                    related_ids.append(rid)
                else:
                    logger.warning(f"Related language '{code}' not found in tokenizer, skipping")

            if related_ids:
                with torch.no_grad():
                    encoder_embeds = self.model.get_input_embeddings()
                    related_vecs = torch.stack([encoder_embeds.weight[rid] for rid in related_ids])
                    avg_vec = related_vecs.mean(dim=0)
                    encoder_embeds.weight[new_token_id] = avg_vec

                    # NLLB shares encoder/decoder embeddings but has a separate lm_head
                    lm_head = self.model.get_output_embeddings()
                    if lm_head is not None and lm_head.weight.shape[0] == len(self.tokenizer):
                        lm_head.weight[new_token_id] = avg_vec

                logger.info(
                    f"Initialized '{lang_code}' embedding from average of: {related_lang_codes}"
                )
            else:
                logger.warning("No valid related languages found — new token uses random init")
        else:
            logger.warning(f"No related languages provided — '{lang_code}' uses random init")

        return True

    def load_pretrained(self, load_path: Path) -> None:
        """Load model and tokenizer from checkpoint directory.

        Args:
            load_path: Directory path containing saved model and tokenizer.
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(load_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(load_path))
        self.model.to(self.device)

        logger.info(f"Model loaded from {load_path}")

    def to(self, device: str) -> None:
        """Move model to device.

        Args:
            device: Device identifier (e.g., "cuda:0", "cpu").
        """
        self.device = device
        self.model.to(device)
        logger.info(f"Model moved to {device}")


class ConsistencyLoss(nn.Module):
    """Custom loss for enforcing terminology consistency in translations.

    Penalizes the model when the same source term maps to different target terms
    across different verses. This helps ensure translations are consistent with
    the terminology database.

    IMPORTANT: This implementation is currently STUBBED and always returns zero loss.
    Set disabled=False and implement the forward() method before using in production.
    For safety, set Config.LOSS_WEIGHTS["consistency_loss"] = 0.0 to disable this loss
    until the implementation is complete.
    """

    def __init__(self, terminology_db, weight: float = 0.1, disabled: bool = True):
        """Initialize consistency loss.

        Args:
            terminology_db: Terminology database for looking up canonical terms.
            weight: Weight for this loss in combined loss (unused if disabled=True).
            disabled: If True, always returns 0 loss (stub behavior).
                     If False, raises NotImplementedError until implemented.
                     ALWAYS USE disabled=True until full implementation.
        """
        super().__init__()
        self.terminology_db = terminology_db
        self.weight = weight
        self.disabled = disabled

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        source_terms: Dict[str, str],
    ) -> torch.Tensor:
        """Calculate consistency loss.

        Args:
            predictions: Model predictions [batch_size, seq_len, vocab_size].
            targets: Target token IDs [batch_size, seq_len].
            source_terms: Dictionary mapping source terms to locations/metadata.

        Returns:
            Consistency loss scalar tensor.

        Raises:
            NotImplementedError: If disabled=False (stub not yet implemented).
        """
        if self.disabled:
            # Safe stub behavior: return zero loss
            return torch.tensor(0.0, device=predictions.device)
        else:
            raise NotImplementedError(
                "ConsistencyLoss is not yet implemented. "
                "Set disabled=True or Config.LOSS_WEIGHTS['consistency_loss']=0.0 "
                "to disable this loss until implementation is complete."
            )


class TranslationTrainer:
    """Trainer for scripture translation model with both standard MT loss.

    Supports training with optional consistency loss (currently stubbed).
    """

    def __init__(
        self,
        model: ScriptureTranslationModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        terminology_db: Optional[Dict] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: ScriptureTranslationModel wrapper.
            optimizer: PyTorch optimizer for training.
            scheduler: Optional learning rate scheduler.
            terminology_db: Optional terminology database for consistency checks.
            device: Device to train on. Defaults to Config.get_device().
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.terminology_db = terminology_db
        self.device = device or Config.get_device()

        self.losses: list = []

    def compute_loss(
        self, outputs, labels: torch.Tensor, alpha: Optional[float] = None
    ) -> torch.Tensor:
        """Compute combined loss (translation + consistency).

        Args:
            outputs: Model outputs from forward pass.
            labels: Target token IDs [batch_size, seq_len].
            alpha: Weight for consistency loss (0.0-1.0). Defaults to
                   Config.LOSS_WEIGHTS["consistency_loss"].

        Returns:
            Combined loss tensor.
        """
        alpha = alpha or Config.LOSS_WEIGHTS.get("consistency_loss", 0.0)

        # Standard translation loss (primary loss)
        translation_loss = outputs.loss

        # Consistency loss (currently stubbed - always returns 0)
        # Set Config.LOSS_WEIGHTS["consistency_loss"] = 0.0 to disable
        consistency_loss = torch.tensor(0.0, device=self.device)

        # Combined loss: primary translation loss + weighted consistency term
        total_loss = translation_loss + alpha * consistency_loss

        return total_loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict]:
        """Single training step with gradient computation and optimization.

        Performs forward pass, loss computation, backward pass with gradient
        clipping, and optimizer step.

        Args:
            batch: Dictionary with keys:
                - input_ids: Source token IDs [batch_size, seq_len]
                - attention_mask: Source attention mask [batch_size, seq_len]
                - labels: Target token IDs [batch_size, seq_len]
                - decoder_attention_mask (optional): Target attention mask

        Returns:
            Tuple of (loss_value, metrics_dict) where metrics_dict contains
            'loss' and 'learning_rate'.
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
        torch.nn.utils.clip_grad_norm_(
            self.model.get_model().parameters(), GRAD_CLIP_NORM
        )
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        self.losses.append(loss.item())

        metrics = {
            "loss": loss.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        return loss.item(), metrics
    
    def get_loss_history(self) -> list:
        """Get training loss history.

        Returns:
            List of loss values from each training step.
        """
        return self.losses


if __name__ == "__main__":
    from utils.logger import configure_logging

    configure_logging()

    # Test model initialization
    logger.info("Testing model initialization...")

    model = ScriptureTranslationModel(use_lora=False)
    logger.info(f"Model type: {type(model.get_model())}")
    logger.info(f"Tokenizer type: {type(model.get_tokenizer())}")

    # Test with LoRA
    logger.info("Testing LoRA initialization...")
    model_lora = ScriptureTranslationModel(use_lora=True)
    logger.info(f"LoRA model parameters: {model_lora.count_parameters():,}")

import os
from pathlib import Path
from typing import Dict, List

from exceptions import LanguageNotSupportedError

class Config:
    """Central configuration for scripture translation system"""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    LOGS_DIR = PROJECT_ROOT / "logs"
    RESULTS_DIR = PROJECT_ROOT / "results"

    # Model Configuration
    MODEL_NAME = os.environ.get(
        "SCRIPTURE_MODEL_NAME", "facebook/nllb-200-distilled-600M"
    )  # Can switch to 1.3B or 3.3B
    
    # Supported language codes (NLLB format)
    LANGUAGE_CODES = {
        "english": "eng_Latn",
        "spanish": "spa_Latn",
        "swahili": "swh_Latn",
        "french": "fra_Latn",
        "portuguese": "por_Latn",
        "turkish": "tur_Latn",
        "korean": "kor_Hang",
        "mandarin": "zho_Hans",
        "amharic": "amh_Ethi",
        "haitian creole": "hat_Latn",
        "haitian_creole": "hat_Latn",
        "minangkabau": "min_Latn",
    }
    
    # Training hyperparameters
    TRAINING_CONFIG = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 3,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 512,
        "max_target_length": 256,
    }
    
    # LoRA Configuration (for rare languages)
    LORA_CONFIG = {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
    }
    
    # Fine-tuning configuration
    FINETUNING_CONFIG = {
        "learning_rate": 5e-4,
        "batch_size": 8,
        "num_epochs": 5,
        "warmup_steps": 100,
    }
    
    # Inference configuration
    INFERENCE_CONFIG = {
        "max_length": 256,
        "num_beams": 5,
        "early_stopping": True,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    
    # Loss weights
    LOSS_WEIGHTS = {
        "translation_loss": 1.0,
        "consistency_loss": float(os.environ.get("CONSISTENCY_LOSS_WEIGHT", "0.0")),
    }
    
    # Evaluation thresholds
    EVAL_THRESHOLDS = {
        "min_bleu": 20.0,
        "min_consistency": 0.85,
        "min_human_rating": 3.5,
    }
    
    @classmethod
    def ensure_dirs(cls) -> None:
        """Create necessary directories. Call this once in entrypoints."""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.CHECKPOINTS_DIR, cls.LOGS_DIR, cls.RESULTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_device(cls):
        """Get appropriate device (CPU only due to GPU compatibility issues)"""
        return "cpu"

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> Dict:
        """Load configuration from YAML file"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @classmethod
    def save_to_yaml(cls, config_dict: Dict, save_path: str):
        """Save configuration to YAML file"""
        import yaml
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def get_language_code(cls, lang_name: str) -> str:
        """Get NLLB language code from friendly name.

        Args:
            lang_name: Friendly language name (e.g., "english", "spanish").

        Returns:
            NLLB language code (e.g., "eng_Latn", "spa_Latn").

        Raises:
            LanguageNotSupportedError: If language is not supported.
        """
        code = cls.LANGUAGE_CODES.get(lang_name.lower())
        if code is None:
            raise LanguageNotSupportedError(
                f"Language '{lang_name}' not supported. "
                f"Supported languages: {', '.join(cls.LANGUAGE_CODES.keys())}"
            )
        return code
    
    @classmethod
    def get_all_configs(cls) -> Dict:
        """Return all configurations as dict"""
        return {
            "model": cls.MODEL_NAME,
            "languages": cls.LANGUAGE_CODES,
            "training": cls.TRAINING_CONFIG,
            "finetuning": cls.FINETUNING_CONFIG,
            "lora": cls.LORA_CONFIG,
            "inference": cls.INFERENCE_CONFIG,
            "loss_weights": cls.LOSS_WEIGHTS,
            "eval_thresholds": cls.EVAL_THRESHOLDS,
        }


if __name__ == "__main__":
    from utils.logger import get_logger, configure_logging
    import logging

    configure_logging()
    logger = get_logger(__name__)

    Config.ensure_dirs()
    logger.info(f"Project root: {Config.PROJECT_ROOT}")
    logger.info(f"Device: {Config.get_device()}")
    logger.info(f"Model: {Config.MODEL_NAME}")
    logger.info(f"Available languages: {len(Config.LANGUAGE_CODES)}")

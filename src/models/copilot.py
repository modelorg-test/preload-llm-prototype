"""LLM model configuration and LoRA adapter specification.

Defines the base model, LoRA configuration, and inference
parameters for the Internal Copilot. The actual model loading
requires transformers and PEFT libraries.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "internal-copilot-base"

LORA_CONFIG = {
    "task_type": "CAUSAL_LM",
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
}

INFERENCE_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "do_sample": True,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
}

SYSTEM_PROMPT = (
    "You are an internal HR policy assistant. Answer ONLY based on the "
    "provided context. If the answer is not in the context, say you don't know."
)

CONFIDENCE_THRESHOLD = 0.70
FALLBACK_RESPONSE = (
    "I'm not confident enough to answer this reliably. "
    "Please contact the HR team directly at hr@modelorg.com."
)


def get_training_args(output_dir: str = "checkpoints/copilot-lora") -> dict:
    """Return the standard training arguments for LoRA fine-tuning.

    Parameters
    ----------
    output_dir : str
        Directory for saving checkpoints.

    Returns
    -------
    dict
        Training arguments compatible with transformers.TrainingArguments.
    """
    return {
        "output_dir": output_dir,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "logging_steps": 50,
        "save_strategy": "epoch",
        "bf16": True,
        "report_to": "mlflow",
    }

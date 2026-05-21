"""LoRA fine-tuning script for the Internal Copilot (Llama 3 8B).

Applies Low-Rank Adaptation to the attention projection layers using
the HuggingFace PEFT library. Registers checkpoints with MLflow.
"""

from __future__ import annotations

import argparse
import logging

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "internal-copilot-base"
EXPERIMENT_NAME = "llm/lora-finetuning"

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)


def finetune(data_path: str, output_dir: str, epochs: int = 3) -> None:
    """Run LoRA fine-tuning on the internal HR Q&A dataset.

    Args:
        data_path: Path to JSONL file with {prompt, completion} pairs.
        output_dir: Directory to save adapter checkpoints.
        epochs: Number of training epochs.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": data_path})["train"]

    def tokenize(example):
        text = f"{example['prompt']}\n{example['completion']}{tokenizer.eos_token}"
        return tokenizer(text, truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        bf16=True,
        report_to="mlflow",
    )

    with mlflow.start_run():
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        model.save_pretrained(output_dir)
        logger.info("LoRA fine-tuning complete. Adapter saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the Internal Copilot via LoRA")
    parser.add_argument("--data", required=True, help="Path to JSONL training file")
    parser.add_argument("--output-dir", default="checkpoints/copilot-lora")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    finetune(args.data, args.output_dir, args.epochs)

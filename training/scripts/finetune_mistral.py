#!/usr/bin/env python3
"""
MiValta Mistral 7B LoRA Fine-Tuning Script

Fine-tunes Mistral 7B Instruct on MiValta conversation data using LoRA.
Designed for Hetzner GPU server (RTX 3090/4090 or A100).

Requirements:
- torch
- transformers
- peft
- bitsandbytes
- datasets
- accelerate
- trl

Usage:
    python finetune_mistral.py --data_path ./data/train_chat.jsonl
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
# Alternative: "mistralai/Ministral-8B-Instruct-2410" (newer, slightly larger)

# LoRA config
LORA_R = 64  # Rank - higher = more capacity but slower
LORA_ALPHA = 128  # Scaling factor
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training
LEARNING_RATE = 2e-4
BATCH_SIZE = 4  # Per device
GRADIENT_ACCUMULATION = 4  # Effective batch = 16
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 1024
WARMUP_RATIO = 0.03

# Quantization (for fitting on smaller GPUs)
USE_4BIT = True  # Set False for A100 with more VRAM


def get_bnb_config():
    """Get BitsAndBytes quantization config."""
    if not USE_4BIT:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config():
    """Get LoRA configuration."""
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_and_prepare_model(model_name: str):
    """Load base model and prepare for LoRA training."""

    print(f"Loading model: {model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model with quantization
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


def format_chat_message(example):
    """Format example into Mistral chat format."""
    messages = example["messages"]

    # Mistral Instruct format
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            formatted += f"<s>[INST] {content}\n\n"
        elif role == "user":
            if formatted:
                formatted += f"{content} [/INST]"
            else:
                formatted += f"<s>[INST] {content} [/INST]"
        elif role == "assistant":
            formatted += f" {content}</s>"

    return {"text": formatted}


def load_dataset_from_jsonl(path: str):
    """Load JSONL dataset and format for training."""

    print(f"Loading dataset from: {path}")

    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]

    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_chat_message)

    print(f"Loaded {len(dataset)} examples")
    return dataset


def train(
    train_path: str,
    val_path: str = None,
    output_dir: str = None,
    model_name: str = BASE_MODEL,
):
    """Run fine-tuning."""

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./models/mivalta-josi-{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_and_prepare_model(model_name)

    # Load datasets
    train_dataset = load_dataset_from_jsonl(train_path)
    val_dataset = load_dataset_from_jsonl(val_path) if val_path else None

    # Training config (trl 0.27.1 uses SFTConfig)
    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final model
    print(f"\nSaving model to {output_path}")
    trainer.save_model(str(output_path / "final"))

    # Save LoRA weights separately (smaller, for merging later)
    model.save_pretrained(str(output_path / "lora_weights"))
    tokenizer.save_pretrained(str(output_path / "lora_weights"))

    print("\nTraining complete!")
    print(f"Model saved to: {output_path / 'final'}")
    print(f"LoRA weights: {output_path / 'lora_weights'}")

    return str(output_path)


def merge_and_save(
    lora_path: str,
    output_path: str,
    base_model: str = BASE_MODEL,
):
    """Merge LoRA weights with base model and save full model."""

    from peft import PeftModel

    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA weights: {lora_path}")
    model = PeftModel.from_pretrained(base, lora_path)

    print("Merging weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)

    print("Merge complete!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral 7B for MiValta")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run fine-tuning")
    train_parser.add_argument(
        "--train_data",
        type=str,
        default="./data/train_chat.jsonl",
        help="Path to training data",
    )
    train_parser.add_argument(
        "--val_data",
        type=str,
        default="./data/val_chat.jsonl",
        help="Path to validation data",
    )
    train_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for model",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default=BASE_MODEL,
        help="Base model to fine-tune",
    )

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA with base model")
    merge_parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA weights",
    )
    merge_parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save merged model",
    )
    merge_parser.add_argument(
        "--base_model",
        type=str,
        default=BASE_MODEL,
        help="Base model name",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            train_path=args.train_data,
            val_path=args.val_data,
            output_dir=args.output_dir,
            model_name=args.model,
        )
    elif args.command == "merge":
        merge_and_save(
            lora_path=args.lora_path,
            output_path=args.output_path,
            base_model=args.base_model,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

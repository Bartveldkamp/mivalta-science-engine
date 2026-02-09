#!/usr/bin/env python3
"""
MiValta SmolLM2 LoRA Fine-Tuning Script

Fine-tunes SmolLM2 (360M or 1.7B) for Josi coaching personality.
Designed for on-device deployment with strict brevity and coaching tone.

Key differences from Mistral pipeline:
- ChatML prompt format (<|im_start|>/<|im_end|>) instead of [INST]/[/INST]
- Llama-based architecture (AutoModelForCausalLM)
- Lighter LoRA config for smaller models
- Strict max_new_tokens to prevent verbosity (Mistral pain point)

Usage:
    # 360M (fastest, lightest — good for intent + templated responses)
    python finetune_smollm2.py train --model 360m --train_data ./data/gold_combined.jsonl

    # 1.7B (recommended — best quality/size balance for coaching conversation)
    python finetune_smollm2.py train --model 1.7b --train_data ./data/gold_combined.jsonl

    # Merge LoRA weights after training
    python finetune_smollm2.py merge --model 1.7b --lora_path ./models/mivalta-josi-smollm2-1.7b/lora_weights --output_path ./models/mivalta-josi-smollm2-1.7b-merged
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {
    "360m": {
        "name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "batch_size": 8,
        "gradient_accumulation": 2,
        "learning_rate": 5e-5,
        "use_4bit": False,  # Small enough to train in full precision
        "max_seq_length": 512,  # Keep short — Josi should be concise
        "size_label": "~724 MB bf16, ~200 MB q4",
    },
    "1.7b": {
        "name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "batch_size": 4,
        "gradient_accumulation": 4,
        "learning_rate": 2e-4,
        "use_4bit": True,  # Quantize for GPU memory efficiency
        "max_seq_length": 768,
        "size_label": "~3.4 GB bf16, ~1.0 GB q4",
    },
}

# Training defaults
NUM_EPOCHS = 3
LORA_DROPOUT = 0.05
WARMUP_RATIO = 0.03

# Josi system prompt — instructs the model to be a concise coaching messenger
JOSI_SYSTEM_PROMPT = (
    "You are Josi, a friendly and knowledgeable sports coaching assistant for MiValta. "
    "You ARE the coach — never recommend other apps, coaches, or services. "
    "You communicate training decisions made by the coaching engine. "
    "Rules: Keep responses under 80 words. Be warm and conversational. "
    "Use simple language, not textbook explanations. Ask follow-up questions. "
    "Never invent training rules — only explain what the engine decided. "
    "Never mention algorithms, GATC, Viterbi, ACWR, or internal systems."
)


def get_bnb_config(use_4bit: bool):
    """Get BitsAndBytes quantization config."""
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_and_prepare_model(config: dict):
    """Load SmolLM2 base model and prepare for LoRA training."""
    model_name = config["name"]
    print(f"Loading model: {model_name}")
    print(f"Size: {config['size_label']}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model
    bnb_config = get_bnb_config(config["use_4bit"])
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if config["use_4bit"]:
        model = prepare_model_for_kbit_training(model)

    # LoRA
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=LORA_DROPOUT,
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def format_chat_message(example, tokenizer):
    """Format example into SmolLM2 ChatML format using tokenizer template.

    SmolLM2 uses ChatML: <|im_start|>role\ncontent<|im_end|>
    """
    messages = example["messages"]

    # Add system prompt if not present
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": JOSI_SYSTEM_PROMPT}] + messages

    # Use the tokenizer's built-in chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def load_dataset_from_jsonl(path: str, tokenizer):
    """Load JSONL dataset and format for SmolLM2 training."""
    print(f"Loading dataset from: {path}")

    with open(path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    dataset = Dataset.from_list(data)
    dataset = dataset.map(lambda ex: format_chat_message(ex, tokenizer))

    print(f"Loaded {len(dataset)} examples")

    # Show a sample
    if len(dataset) > 0:
        sample = dataset[0]["text"]
        print(f"Sample (first 300 chars):\n{sample[:300]}...")

    return dataset


def train(
    model_size: str,
    train_path: str,
    val_path: str = None,
    output_dir: str = None,
    epochs: int = NUM_EPOCHS,
):
    """Run SmolLM2 fine-tuning."""
    config = MODEL_CONFIGS[model_size]

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./models/mivalta-josi-smollm2-{model_size}-{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(output_path / "training_config.json", "w") as f:
        json.dump({
            "model_size": model_size,
            "base_model": config["name"],
            "lora_r": config["lora_r"],
            "lora_alpha": config["lora_alpha"],
            "epochs": epochs,
            "train_data": train_path,
            "system_prompt": JOSI_SYSTEM_PROMPT,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Load model
    model, tokenizer = load_and_prepare_model(config)

    # Load datasets
    train_dataset = load_dataset_from_jsonl(train_path, tokenizer)
    val_dataset = load_dataset_from_jsonl(val_path, tokenizer) if val_path else None

    # Training config
    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        learning_rate=config["learning_rate"],
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        bf16=True,
        optim="paged_adamw_8bit" if config["use_4bit"] else "adamw_torch",
        gradient_checkpointing=config["use_4bit"],
        max_grad_norm=0.3,
        report_to="none",
    )

    # Truncate training data to max_seq_length
    max_len = config["max_seq_length"]

    def truncate_text(example):
        tokens = tokenizer(example["text"], truncation=True, max_length=max_len)
        example["text"] = tokenizer.decode(tokens["input_ids"], skip_special_tokens=False)
        return example

    train_dataset = train_dataset.map(truncate_text)
    if val_dataset:
        val_dataset = val_dataset.map(truncate_text)

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    # Train
    print(f"\n{'=' * 60}")
    print(f"  SmolLM2-{model_size.upper()} Fine-Tuning for Josi")
    print(f"  Base: {config['name']}")
    print(f"  LoRA rank: {config['lora_r']}, alpha: {config['lora_alpha']}")
    print(f"  Training examples: {len(train_dataset)}")
    print(f"  Max sequence length: {config['max_seq_length']}")
    print(f"{'=' * 60}\n")

    trainer.train()

    # Save
    print(f"\nSaving model to {output_path}")
    trainer.save_model(str(output_path / "final"))

    model.save_pretrained(str(output_path / "lora_weights"))
    tokenizer.save_pretrained(str(output_path / "lora_weights"))

    print("\nTraining complete!")
    print(f"Full model: {output_path / 'final'}")
    print(f"LoRA weights: {output_path / 'lora_weights'}")
    print(f"\nNext steps:")
    print(f"  1. Merge:  python finetune_smollm2.py merge --model {model_size} "
          f"--lora_path {output_path / 'lora_weights'} --output_path ./models/mivalta-josi-smollm2-{model_size}-merged")
    print(f"  2. Export: python export_gguf.py --model_path ./models/mivalta-josi-smollm2-{model_size}-merged --quant q4_k_m")
    print(f"  3. Test:   python evaluate_smollm2.py --model ./models/gguf/mivalta-josi-smollm2-{model_size}-merged-q4_k_m.gguf")

    return str(output_path)


def merge_and_save(
    model_size: str,
    lora_path: str,
    output_path: str,
):
    """Merge LoRA weights with SmolLM2 base model and save full model."""
    from peft import PeftModel

    config = MODEL_CONFIGS[model_size]
    base_model_name = config["name"]

    print(f"Loading base model: {base_model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA weights: {lora_path}")
    model = PeftModel.from_pretrained(base, lora_path)

    print("Merging weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)

    print("Merge complete!")
    print(f"\nNext: python export_gguf.py --model_path {output_path} --quant q4_k_m")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolLM2 for MiValta Josi"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train
    train_parser = subparsers.add_parser("train", help="Run fine-tuning")
    train_parser.add_argument(
        "--model", type=str, default="1.7b", choices=["360m", "1.7b"],
        help="Model size: 360m (fast/tiny) or 1.7b (recommended)",
    )
    train_parser.add_argument(
        "--train_data", type=str, default="./data/gold_combined.jsonl",
        help="Path to training data (JSONL with messages format)",
    )
    train_parser.add_argument(
        "--val_data", type=str, default=None,
        help="Path to validation data",
    )
    train_parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for model",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help="Number of training epochs",
    )

    # Merge
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA with base model")
    merge_parser.add_argument(
        "--model", type=str, default="1.7b", choices=["360m", "1.7b"],
        help="Model size (must match the trained model)",
    )
    merge_parser.add_argument(
        "--lora_path", type=str, required=True,
        help="Path to LoRA weights",
    )
    merge_parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save merged model",
    )

    # Info
    subparsers.add_parser("info", help="Show model configurations")

    args = parser.parse_args()

    if args.command == "train":
        train(
            model_size=args.model,
            train_path=args.train_data,
            val_path=args.val_data,
            output_dir=args.output_dir,
            epochs=args.epochs,
        )
    elif args.command == "merge":
        merge_and_save(
            model_size=args.model,
            lora_path=args.lora_path,
            output_path=args.output_path,
        )
    elif args.command == "info":
        print("\nSmolLM2 Model Configurations for MiValta Josi:")
        print("=" * 60)
        for size, cfg in MODEL_CONFIGS.items():
            print(f"\n  {size.upper()}:")
            print(f"    Model:    {cfg['name']}")
            print(f"    Size:     {cfg['size_label']}")
            print(f"    LoRA:     r={cfg['lora_r']}, alpha={cfg['lora_alpha']}")
            print(f"    Targets:  {', '.join(cfg['lora_target_modules'])}")
            print(f"    Batch:    {cfg['batch_size']} x {cfg['gradient_accumulation']} = {cfg['batch_size'] * cfg['gradient_accumulation']} effective")
            print(f"    LR:       {cfg['learning_rate']}")
            print(f"    4-bit:    {cfg['use_4bit']}")
            print(f"    Max seq:  {cfg['max_seq_length']}")
        print(f"\n  Recommendation: 1.7B for coaching conversation quality")
        print(f"  360M for: intent classification or very constrained devices")
        print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

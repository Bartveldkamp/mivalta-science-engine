#!/usr/bin/env python3
"""
MiValta Josi v3 — SmolLM2 LoRA Fine-Tuning Script

Fine-tunes SmolLM2-360M-Instruct or SmolLM2-1.7B-Instruct on the v3
training data (LLMIntent JSON output format).

Architecture notes:
  - SmolLM2 uses LlamaForCausalLM internally but load via AutoModelForCausalLM
  - Chat template: ChatML (<|im_start|> / <|im_end|>) — built into tokenizer
  - No quantization needed: 360M fits easily in bf16, 1.7B fits on 24GB GPU
  - LoRA targets: attention projections + MLP gates

Usage:
    # Train 360M (default)
    python finetune_smollm2.py train

    # Train 1.7B
    python finetune_smollm2.py train --model_size 1.7B

    # Merge LoRA weights into base model
    python finetune_smollm2.py merge --lora_path ./models/josi-v3-360M-*/lora_weights

    # Quick sanity check on merged model
    python finetune_smollm2.py sanity --model_path ./models/josi-v3-360M-merged

Requirements:
    pip install torch transformers peft datasets accelerate trl wandb
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from trl import SFTTrainer, SFTConfig


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODELS = {
    "360M": {
        "name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 8,
        "grad_accum": 2,       # effective batch = 16
        "lr": 1e-4,
        "max_seq_length": 1024,
        "epochs": 5,
    },
    "1.7B": {
        "name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "lora_r": 32,
        "lora_alpha": 64,
        "batch_size": 4,
        "grad_accum": 4,       # effective batch = 16
        "lr": 5e-5,
        "max_seq_length": 1024,
        "epochs": 4,
    },
}

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj",   # attention
    "gate_proj", "up_proj", "down_proj",  # MLP
]
LORA_DROPOUT = 0.05

# Early stopping
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_TRAIN = DATA_DIR / "train_v3.jsonl"
DEFAULT_VAL = DATA_DIR / "val_v3.jsonl"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file into list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prepare_dataset(path: str, tokenizer) -> Dataset:
    """Load JSONL and format with tokenizer's chat template."""
    raw = load_jsonl(path)
    print(f"  Loaded {len(raw)} examples from {path}")

    texts = []
    for ex in raw:
        messages = ex["messages"]
        # apply_chat_template returns the full formatted string with ChatML tokens
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append({"text": text})

    ds = Dataset.from_list(texts)
    return ds


# =============================================================================
# MODEL SETUP
# =============================================================================

def load_model_and_tokenizer(model_name: str, lora_r: int, lora_alpha: int):
    """Load SmolLM2 and apply LoRA adapters."""

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token is set (SmolLM2 may not have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print(f"Applying LoRA (r={lora_r}, alpha={lora_alpha})")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# =============================================================================
# TRAINING
# =============================================================================

def train(
    model_size: str = "360M",
    train_path: str = None,
    val_path: str = None,
    output_dir: str = None,
    use_wandb: bool = True,
    lr_override: float = None,
    epochs_override: int = None,
):
    """Run LoRA fine-tuning."""

    cfg = MODELS[model_size]
    model_name = cfg["name"]
    lr = lr_override or cfg["lr"]
    epochs = epochs_override or cfg["epochs"]

    train_path = train_path or str(DEFAULT_TRAIN)
    val_path = val_path or str(DEFAULT_VAL)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./models/josi-v3-{model_size}-{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save training config for reproducibility
    run_config = {
        "model_size": model_size,
        "model_name": model_name,
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
        "lr": lr,
        "epochs": epochs,
        "batch_size": cfg["batch_size"],
        "grad_accum": cfg["grad_accum"],
        "effective_batch": cfg["batch_size"] * cfg["grad_accum"],
        "max_seq_length": cfg["max_seq_length"],
        "train_data": train_path,
        "val_data": val_path,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path / "training_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # W&B setup
    report_to = "none"
    if use_wandb:
        try:
            import wandb
            report_to = "wandb"
            wandb.init(
                project="mivalta-josi-v3",
                name=f"josi-v3-{model_size}-{datetime.now().strftime('%m%d_%H%M')}",
                config=run_config,
            )
            print("W&B tracking enabled")
        except (ImportError, wandb.errors.CommError):
            print("W&B not available, logging to console only")
            report_to = "none"

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_name, cfg["lora_r"], cfg["lora_alpha"],
    )

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = prepare_dataset(train_path, tokenizer)
    val_dataset = prepare_dataset(val_path, tokenizer)

    # Verify a sample
    print(f"\n  Sample formatted text (first 300 chars):")
    print(f"  {train_dataset[0]['text'][:300]}...")

    # Training arguments
    eval_steps = max(1, len(train_dataset) // (cfg["batch_size"] * cfg["grad_accum"] * 4))
    print(f"\n  Eval every {eval_steps} steps")

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        max_seq_length=cfg["max_seq_length"],
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to=report_to,
        dataset_text_field="text",
    )

    # Callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
        ),
    ]

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
    )

    # Train
    print("\n" + "=" * 60)
    print(f"Starting Josi v3 training — SmolLM2-{model_size}")
    print(f"  Effective batch size: {cfg['batch_size'] * cfg['grad_accum']}")
    print(f"  Learning rate: {lr}")
    print(f"  Max epochs: {epochs} (early stopping patience={EARLY_STOPPING_PATIENCE})")
    print(f"  Output: {output_path}")
    print("=" * 60 + "\n")

    result = trainer.train()

    # Save
    print(f"\nTraining complete. Best eval loss: {trainer.state.best_metric:.4f}")
    print(f"Saving model to {output_path}")

    trainer.save_model(str(output_path / "final"))

    # Save LoRA weights separately for merge step
    lora_path = output_path / "lora_weights"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))

    # Save training metrics
    metrics = {
        "train_loss": result.training_loss,
        "best_eval_loss": trainer.state.best_metric,
        "total_steps": result.global_step,
        "epochs_completed": result.num_train_epochs,
    }
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {output_path / 'final'}")
    print(f"LoRA weights: {lora_path}")
    print(f"Training config: {output_path / 'training_config.json'}")
    print(f"Metrics: {output_path / 'training_metrics.json'}")

    if report_to == "wandb":
        wandb.finish()

    return str(output_path)


# =============================================================================
# MERGE
# =============================================================================

def merge(
    lora_path: str,
    output_path: str = None,
    model_size: str = "360M",
):
    """Merge LoRA weights with base model for GGUF export."""
    from peft import PeftModel

    cfg = MODELS[model_size]
    base_name = cfg["name"]

    if output_path is None:
        output_path = str(Path(lora_path).parent / "merged")

    print(f"Loading base model: {base_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA weights: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging weights...")
    merged = model.merge_and_unload()

    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_name)
    tokenizer.save_pretrained(output_path)

    print(f"Merge complete! Ready for GGUF export:")
    print(f"  python export_gguf.py --model_path {output_path} --quant q4_k_m")

    return output_path


# =============================================================================
# SANITY CHECK
# =============================================================================

SANITY_PROMPTS = [
    {
        "tier": "coach",
        "persona": "balanced",
        "messages": [
            {"role": "system", "content": "You are Josi, MiValta's AI coaching assistant. Style: warm, professional, supportive.\n\nMODE: Coach\n- Reference and explain the athlete's training plan\n- Trigger replans via replan_request for valid reasons\n\nI6 CONSTRAINTS (always active):\n- NEVER prescribe, create, or modify training yourself\n- Explain decisions made by the GATC engine only\n\nOUTPUT: Valid LLMIntent JSON with fields: intent, response_type, message, source_cards, guardrail_triggered, guardrail_reason, replan_request, tool_call."},
            {"role": "user", "content": "What am I doing today?\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Session: Z2 60min \"Easy aerobic\" (base phase)\n- Sport: running\n- Level: intermediate"},
        ],
        "expected_intent": "question",
        "expected_response_type": "ExplainWorkout",
        "check": "should explain the Z2 60min session",
    },
    {
        "tier": "coach",
        "persona": "balanced",
        "messages": [
            {"role": "system", "content": "You are Josi, MiValta's AI coaching assistant. Style: warm, professional, supportive.\n\nMODE: Coach\n\nI6 CONSTRAINTS (always active):\n- NEVER prescribe, create, or modify training yourself\n- Prescription/override requests: intent=blocked, guardrail_triggered=true\n\nOUTPUT: Valid LLMIntent JSON with fields: intent, response_type, message, source_cards, guardrail_triggered, guardrail_reason, replan_request, tool_call."},
            {"role": "user", "content": "Give me a harder workout\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate"},
        ],
        "expected_intent": "blocked",
        "expected_response_type": "Decline",
        "check": "should block with guardrail_triggered=true",
    },
    {
        "tier": "advisor",
        "persona": "direct",
        "messages": [
            {"role": "system", "content": "You are Josi, MiValta's AI coaching assistant. Style: no-nonsense, concise, brief.\n\nMODE: Advisor\n- NEVER create training plans (Decline with tier upgrade)\n\nI6 CONSTRAINTS (always active):\n- NEVER prescribe, create, or modify training yourself\n\nOUTPUT: Valid LLMIntent JSON with fields: intent, response_type, message, source_cards, guardrail_triggered, guardrail_reason, replan_request, tool_call."},
            {"role": "user", "content": "Create me a training plan\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: cycling\n- Level: beginner"},
        ],
        "expected_intent": "blocked",
        "expected_response_type": "Decline",
        "check": "should decline plan creation in advisor tier",
    },
]


def sanity_check(model_path: str, max_new_tokens: int = 256):
    """Run sanity check prompts through the model and verify JSON output."""

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    passed = 0
    failed = 0

    for i, prompt in enumerate(SANITY_PROMPTS):
        print(f"\n{'='*60}")
        print(f"Sanity check {i+1}/{len(SANITY_PROMPTS)}: {prompt['check']}")
        print(f"  Tier: {prompt['tier']}, Expected: {prompt['expected_intent']}/{prompt['expected_response_type']}")

        # Format with chat template
        text = tokenizer.apply_chat_template(
            prompt["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        print(f"\n  Raw output:\n  {generated[:500]}")

        # Validate
        checks = []
        try:
            parsed = json.loads(generated)
            checks.append(("valid_json", True))
            checks.append(("has_intent", "intent" in parsed))
            checks.append(("has_response_type", "response_type" in parsed))
            checks.append(("has_message", "message" in parsed))
            checks.append(("intent_correct", parsed.get("intent") == prompt["expected_intent"]))
            checks.append(("rtype_correct", parsed.get("response_type") == prompt["expected_response_type"]))

            if prompt["expected_intent"] == "blocked":
                checks.append(("guardrail_triggered", parsed.get("guardrail_triggered") is True))
        except json.JSONDecodeError:
            checks.append(("valid_json", False))

        all_pass = all(v for _, v in checks)
        status = "PASS" if all_pass else "FAIL"
        if all_pass:
            passed += 1
        else:
            failed += 1

        print(f"\n  [{status}]")
        for name, ok in checks:
            print(f"    {'OK' if ok else 'FAIL'}: {name}")

    print(f"\n{'='*60}")
    print(f"Sanity check: {passed}/{passed+failed} passed")
    if failed > 0:
        print(f"WARNING: {failed} checks failed — review model output above")
    else:
        print("All checks passed!")

    return failed == 0


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MiValta Josi v3 — SmolLM2 Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python finetune_smollm2.py train                      # Train 360M (default)
  python finetune_smollm2.py train --model_size 1.7B    # Train 1.7B
  python finetune_smollm2.py train --no-wandb           # Train without W&B
  python finetune_smollm2.py merge --lora_path ./models/josi-v3-360M-*/lora_weights
  python finetune_smollm2.py sanity --model_path ./models/josi-v3-360M-merged
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # --- train ---
    tp = subparsers.add_parser("train", help="Run LoRA fine-tuning")
    tp.add_argument("--model_size", choices=["360M", "1.7B"], default="360M",
                    help="SmolLM2 model size (default: 360M)")
    tp.add_argument("--train_data", type=str, default=None,
                    help=f"Training data (default: {DEFAULT_TRAIN})")
    tp.add_argument("--val_data", type=str, default=None,
                    help=f"Validation data (default: {DEFAULT_VAL})")
    tp.add_argument("--output_dir", type=str, default=None,
                    help="Output directory")
    tp.add_argument("--lr", type=float, default=None,
                    help="Override learning rate")
    tp.add_argument("--epochs", type=int, default=None,
                    help="Override max epochs")
    tp.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B tracking")

    # --- merge ---
    mp = subparsers.add_parser("merge", help="Merge LoRA weights with base model")
    mp.add_argument("--lora_path", type=str, required=True,
                    help="Path to LoRA weights directory")
    mp.add_argument("--output_path", type=str, default=None,
                    help="Path to save merged model")
    mp.add_argument("--model_size", choices=["360M", "1.7B"], default="360M",
                    help="Base model size to merge against")

    # --- sanity ---
    sp = subparsers.add_parser("sanity", help="Run sanity check on merged model")
    sp.add_argument("--model_path", type=str, required=True,
                    help="Path to merged model")
    sp.add_argument("--max_tokens", type=int, default=256,
                    help="Max generation tokens")

    args = parser.parse_args()

    if args.command == "train":
        train(
            model_size=args.model_size,
            train_path=args.train_data,
            val_path=args.val_data,
            output_dir=args.output_dir,
            use_wandb=not args.no_wandb,
            lr_override=args.lr,
            epochs_override=args.epochs,
        )
    elif args.command == "merge":
        merge(
            lora_path=args.lora_path,
            output_path=args.output_path,
            model_size=args.model_size,
        )
    elif args.command == "sanity":
        ok = sanity_check(
            model_path=args.model_path,
            max_new_tokens=args.max_tokens,
        )
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

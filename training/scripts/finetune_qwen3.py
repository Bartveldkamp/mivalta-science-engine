#!/usr/bin/env python3
"""
MiValta Josi v6 — Qwen3 LoRA Fine-Tuning Script

Single-model architecture (replaces v5 dual-model):
  ONE Qwen3 model handles both modes via system prompt switching:
    - INTERPRETER mode: GATCRequest JSON output (~50 tokens, /no_think, fast)
    - COACH mode: warm coaching text (router-controlled /think for complex questions)

  On-device, the app calls the SAME model twice:
    1. Interpreter call → GATCRequest JSON (always /no_think)
    2. Router (code, not LLM) decides if coach call is needed
    3. Coach call → coaching text (/think for complex, /no_think for simple)

Model sizes (--model-size flag):
  - 8b (DEFAULT, Android): Qwen3-8B Q4_K_M ~5.0 GB — best quality, Android 12GB+
  - 4b (iPhone/low-RAM):   Qwen3-4B Q4_K_M ~2.5 GB — great quality, all phones

Why 8B for Android:
  - Outperforms Qwen2.5-14B on reasoning benchmarks
  - 5.3x more parameters than v5 per task (8B vs 1.5B)
  - Dramatically better coaching nuance, Dutch quality, injury reasoning
  - Router-controlled /think gives "fast most of the time, smart when needed"
  - Motorola Edge 60 (12GB) runs it comfortably

On-device performance (8B on Android 12GB):
  - Single 8B Q4_K_M model = ~5.0 GB GGUF
  - Interpreter call: ~400ms (/no_think, JSON output)
  - Coach call simple: ~600ms (/no_think, short answer)
  - Coach call complex: ~1200ms (/think, injury reasoning, plan tradeoffs)
  - Coach skipped for ~40% of messages (create_workout, replan, clarify)
  - 100% on-device via llama.cpp — NO network calls

Router-controlled thinking:
  - /no_think: interpreter JSON, clarify, create_workout, replan, simple answers
  - /think: "why is my readiness red?", injury patterns, plan tradeoffs, complex coaching
  - The router decides AFTER the interpreter classifies the action
  - Users never see or control this — it's invisible quality escalation

Architecture notes:
  - Qwen3 uses ChatML natively (<|im_start|>/<|im_end|>) — same as Qwen2.5
  - Thinking mode: <think>...</think> tags, controlled via /think and /no_think
  - AutoModelForCausalLM + AutoTokenizer (standard HF pipeline)
  - bf16 loading + LoRA adapters — fits on single GPU
  - LoRA targets: attention + MLP projections
  - Merged + GGUF Q4_K_M for on-device deployment

Training modes:
  - interpreter: train on interpreter data only (GATCRequest JSON)
  - coach: train on coach/explainer data only (coaching text)
  - unified: train on combined data (RECOMMENDED — model learns both tasks)

Runtime constraints (on-device):
  - Context cap: 4096 tokens (up from 2048 in v5)
  - Output cap: 200 tokens (up from 150 in v5)
  - Temperature: 0.3 interpreter / 0.5 coach
  - 100% on-device via llama.cpp on Android

Usage:
    # RECOMMENDED: Unified training (both tasks, 8B Android default)
    python finetune_qwen3.py train --mode unified

    # Train 4B variant for iPhone
    python finetune_qwen3.py train --mode unified --model-size 4b

    # Merge LoRA weights into base model
    python finetune_qwen3.py merge --lora_path ./models/josi-v6-qwen3-8b-unified-*/lora_weights

    # Sanity check both modes
    python finetune_qwen3.py sanity --model_path ./models/.../merged --mode interpreter
    python finetune_qwen3.py sanity --model_path ./models/.../merged --mode coach

    # GGUF conversion (via llama.cpp)
    python /path/to/llama.cpp/convert_hf_to_gguf.py ./models/merged --outtype q4_k_m

Requirements:
    pip install transformers torch peft datasets accelerate trl sentencepiece protobuf huggingface_hub
"""

import argparse
import json
import os
import random
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
# MODEL CONFIGURATION — Qwen3 (8B default for Android, 4B for iPhone)
# =============================================================================

# Model size configs: 8B (Android, needs >=24GB VRAM) and 4B (iPhone / low-VRAM)
# Hetzner server: RTX 4000 SFF Ada (19.5 GB VRAM) — fits 4B, not 8B
# For 8B training, use cloud GPU (A100/H100) or QLoRA (4-bit base)
MODEL_CONFIGS = {
    "8b": {
        "model_id": "Qwen/Qwen3-8B",
        "local_dir": "Qwen3-8B",
        "params": "8B",
        "gguf_size": "~5.0 GB",
        "vram_needed": "~24 GB",
        "lr": 1e-5,            # Gentler for 8B
        "batch_size": 1,       # Fits in 24GB VRAM with LoRA
        "grad_accum": 16,      # Effective batch = 16
        "epochs": 3,
        "lora_r": 32,
        "lora_alpha": 64,
    },
    "4b": {
        "model_id": "Qwen/Qwen3-4B",
        "local_dir": "Qwen3-4B",
        "params": "4B",
        "gguf_size": "~2.5 GB",
        "vram_needed": "~16 GB",
        "lr": 2e-5,            # Slightly higher for smaller model
        "batch_size": 1,       # Safe for 19.5 GB RTX 4000 SFF Ada
        "grad_accum": 16,      # Effective batch = 16
        "epochs": 3,
        "lora_r": 32,
        "lora_alpha": 64,
    },
}

# Default: 8B for Android (best quality)
# NOTE: 8B requires >=24GB VRAM. Use --model-size 4b on RTX 4000 SFF Ada (19.5GB)
MODEL_SIZE = "8b"

SCRIPT_DIR_FOR_MODEL = Path(__file__).resolve().parent


def get_config(size: str = None) -> dict:
    """Get model configuration for given size."""
    return MODEL_CONFIGS[size or MODEL_SIZE]


def resolve_model_id(size: str = None):
    """Use local download if available, otherwise pull from HuggingFace."""
    cfg = get_config(size)
    local = SCRIPT_DIR_FOR_MODEL.parent / "models" / cfg["local_dir"]
    if local.exists() and (local / "config.json").exists():
        return str(local)
    return cfg["model_id"]


LORA_DROPOUT = 0.05

# LoRA targets on the language model backbone
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]

MAX_SEQ_LENGTH = 4096       # Training context (Qwen3 supports 32K, we use 4K)
WARMUP_RATIO = 0.05

# Early stopping
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"

# Data files
INTERPRETER_TRAIN = DATA_DIR / "train_interpreter.jsonl"
INTERPRETER_VAL = DATA_DIR / "val_interpreter.jsonl"
COACH_TRAIN = DATA_DIR / "train_explainer_sequential.jsonl"
COACH_VAL = DATA_DIR / "val_explainer_sequential.jsonl"
UNIFIED_TRAIN = DATA_DIR / "train_v6_unified.jsonl"
UNIFIED_VAL = DATA_DIR / "val_v6_unified.jsonl"

# Josi runtime constraints (baked into training + enforced at inference)
INTERPRETER_TEMPERATURE = 0.3    # Low temp for deterministic JSON
COACH_TEMPERATURE = 0.5          # Higher temp for natural coaching text
INFERENCE_MAX_TOKENS = 200       # Up from 150 in v5 — 4B generates better with more room

# Actions that need the coach call (same logic as v5 router)
ACTIONS_NEEDING_COACH = {"explain", "answer_question"}


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

def load_system_prompt(filename: str) -> str:
    """Load a system prompt from the prompts directory."""
    path = PROMPTS_DIR / filename
    return path.read_text().strip()


def build_system_prompt(mode: str = "interpreter") -> str:
    """Build the system prompt for the given model mode.

    Args:
        mode: "interpreter" for GATCRequest JSON output,
              "coach" for plain coaching text output.
    """
    if mode == "coach":
        # Try v6 prompt first, fall back to v5 sequential prompt
        for name in ["josi_v6_coach.txt", "explainer_sequential_system.txt", "explainer_system.txt"]:
            path = PROMPTS_DIR / name
            if path.exists():
                return load_system_prompt(name)
        raise FileNotFoundError("No coach system prompt found in prompts/")

    # Interpreter mode
    for name in ["josi_v6_interpreter.txt", "interpreter_system.txt"]:
        path = PROMPTS_DIR / name
        if path.exists():
            return load_system_prompt(name)
    raise FileNotFoundError("No interpreter system prompt found in prompts/")


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


def validate_single_turn(messages: list[dict]) -> bool:
    """Validate that the conversation has exactly one assistant response at the end."""
    assistant_count = sum(1 for m in messages if m["role"] == "assistant")
    if assistant_count != 1:
        return False
    if messages[-1]["role"] != "assistant":
        return False
    return True


def prepare_dataset(path: str, tokenizer, max_seq_length: int = 4096) -> Dataset:
    """Load JSONL and split into prompt/completion format for completion-only loss.

    Training data is in ChatML format (system/user/assistant messages).
    Qwen3 uses ChatML natively, so no format conversion is needed.

    We split into:
      prompt:     system + user turns + generation marker
      completion: assistant response + EOS token

    TRL's SFTTrainer auto-detects prompt/completion format and creates a
    completion_mask, so the model only trains on the completion — not the
    system prompt or user message.
    """
    raw = load_jsonl(path)
    print(f"  Loaded {len(raw)} examples from {path}")

    eos_token = tokenizer.eos_token or "<|im_end|>"
    examples = []
    truncated = 0
    skipped = 0
    multi_turn = 0

    for ex in raw:
        messages = ex["messages"]

        # Validate single-turn
        if not validate_single_turn(messages):
            multi_turn += 1
            if multi_turn <= 3:
                roles = [m["role"] for m in messages]
                print(f"  WARNING: Multi-turn example skipped (roles: {roles})")
            continue

        # Split into prompt (system+user) and completion (assistant response)
        prompt_messages = messages[:-1]  # system + user
        try:
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  WARNING: Skipping example (template error): {e}")
            continue

        # Completion: assistant response + EOS
        completion = messages[-1]["content"] + eos_token

        # Truncation check
        full_text = prompt + completion
        tokens = tokenizer(full_text, truncation=True, max_length=max_seq_length)
        if len(tokens["input_ids"]) >= max_seq_length:
            truncated += 1
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)
            remaining = max_seq_length - len(prompt_tokens["input_ids"])
            if remaining <= 10:
                continue  # Prompt alone exceeds limit, skip
            completion_tokens = tokenizer(completion, truncation=True, max_length=remaining)
            completion = tokenizer.decode(completion_tokens["input_ids"], skip_special_tokens=False)

        examples.append({"prompt": prompt, "completion": completion})

    if multi_turn:
        print(f"  WARNING: {multi_turn}/{len(raw)} multi-turn examples skipped")
    if truncated:
        print(f"  WARNING: {truncated}/{len(raw)} examples truncated to {max_seq_length} tokens")
    if skipped:
        print(f"  WARNING: {skipped}/{len(raw)} examples skipped (template errors)")

    # Verify stop token
    eos_count = sum(1 for e in examples if eos_token in e["completion"])
    print(f"  Stop token ('{eos_token}') present in {eos_count}/{len(examples)} completions")
    if eos_count < len(examples):
        print(f"  WARNING: {len(examples) - eos_count} completions missing stop token!")

    # Show a sample
    if examples:
        print(f"  Sample prompt  (last 80 chars): ...{examples[0]['prompt'][-80:]}")
        print(f"  Sample completion (first 80 chars): {examples[0]['completion'][:80]}...")

    ds = Dataset.from_list(examples)
    return ds


# =============================================================================
# MODEL SETUP — bf16 + LoRA
# =============================================================================

def load_model_and_tokenizer(model_id: str = None, size: str = None):
    """Load Qwen3 model in bf16 and apply LoRA adapters."""
    cfg = get_config(size)
    if model_id is None:
        model_id = resolve_model_id(size)

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Ensure pad_token is separate from eos_token
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print(f"  Added dedicated [PAD] token (id={tokenizer.pad_token_id}) — separate from EOS (id={tokenizer.eos_token_id})")
    tokenizer.padding_side = "right"

    print(f"Loading model: {model_id} (bf16, LoRA fine-tuning)")
    print(f"  Architecture: Qwen3-{cfg['params']} (AutoModelForCausalLM)")
    print(f"  Params: {cfg['params']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Resize embeddings if we added a pad token
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Resized embeddings to {len(tokenizer)} (added [PAD] token)")

    # Freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradient checkpointing (critical for VRAM)
    model.gradient_checkpointing_enable()

    # LoRA adapters
    lora_r = cfg["lora_r"]
    lora_alpha = cfg["lora_alpha"]
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

    # Report VRAM usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  VRAM: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved, {total:.1f} GB total")

    return model, tokenizer


# =============================================================================
# TRAINING
# =============================================================================

def train(
    train_path: str = None,
    val_path: str = None,
    output_dir: str = None,
    use_wandb: bool = True,
    lr_override: float = None,
    epochs_override: int = None,
    mode: str = "unified",
    model_size: str = None,
):
    """Run LoRA fine-tuning on Qwen3.

    Args:
        mode: "interpreter" for GATCRequest JSON training,
              "coach" for coaching text training,
              "unified" for combined training (recommended).
        model_size: "8b" (Android default) or "4b" (iPhone).
    """
    cfg = get_config(model_size)
    lr = lr_override or cfg["lr"]
    epochs = epochs_override or cfg["epochs"]

    # VRAM check — warn early if GPU is too small
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        size_tag = model_size or MODEL_SIZE
        if size_tag == "8b" and total_vram_gb < 22:
            print(f"\n  WARNING: Qwen3-8B needs ~24 GB VRAM, you have {total_vram_gb:.1f} GB")
            print(f"  Options:")
            print(f"    1. Use --model-size 4b (fits in {total_vram_gb:.0f} GB, still 2.7x better than v5)")
            print(f"    2. Use a cloud GPU with >=24 GB VRAM (A100, H100)")
            print(f"  Continuing anyway — may OOM during training.\n")

    if train_path is None:
        if mode == "unified":
            if UNIFIED_TRAIN.exists():
                train_path = str(UNIFIED_TRAIN)
            else:
                print(f"Unified data not found at {UNIFIED_TRAIN}")
                print(f"Run 'python prepare_v6_data.py' first, or use --mode interpreter/coach")
                sys.exit(1)
        elif mode == "coach":
            if COACH_TRAIN.exists():
                train_path = str(COACH_TRAIN)
            else:
                print(f"Coach data not found at {COACH_TRAIN}")
                print(f"Run 'python prepare_sequential_data.py' first.")
                sys.exit(1)
        else:
            train_path = str(INTERPRETER_TRAIN)

    if val_path is None:
        if mode == "unified":
            if UNIFIED_VAL.exists():
                val_path = str(UNIFIED_VAL)
            else:
                val_path = str(INTERPRETER_VAL)
        elif mode == "coach":
            val_path = str(COACH_VAL) if COACH_VAL.exists() else str(DATA_DIR / "val_explainer.jsonl")
        else:
            val_path = str(INTERPRETER_VAL)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        size_tag = model_size or MODEL_SIZE
        output_dir = f"./models/josi-v6-qwen3-{size_tag}-{mode}-{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    batch_size = cfg["batch_size"]
    grad_accum = cfg["grad_accum"]

    # Save training config
    run_config = {
        "version": "v6",
        "model_id": cfg["model_id"],
        "model_family": f"Qwen3-{cfg['params']}",
        "model_size": model_size or MODEL_SIZE,
        "architecture": "AutoModelForCausalLM (ChatML)",
        "pipeline": "single-model, dual-mode (interpreter + coach via system prompt)",
        "thinking_mode": "router-controlled (/think for complex, /no_think for fast)",
        "mode": mode,
        "params": cfg["params"],
        "quantization": "bf16 + LoRA",
        "lora_r": cfg["lora_r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_data": train_path,
        "val_data": val_path,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "interpreter_temperature": INTERPRETER_TEMPERATURE,
        "coach_temperature": COACH_TEMPERATURE,
        "inference_max_tokens": INFERENCE_MAX_TOKENS,
        "gguf_target": f"Q4_K_M ({cfg['gguf_size']})",
        "target_platform": "Android 12GB+" if (model_size or MODEL_SIZE) == "8b" else "All phones",
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path / "training_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # W&B setup
    report_to = "none"
    if use_wandb:
        try:
            import wandb
            size_tag = model_size or MODEL_SIZE
            wandb.init(
                project="mivalta-josi-v6",
                name=f"josi-v6-qwen3-{size_tag}-{mode}-{datetime.now().strftime('%m%d_%H%M')}",
                config=run_config,
            )
            report_to = "wandb"
            print("W&B tracking enabled")
        except Exception:
            print("W&B not available, logging to console only")
            report_to = "none"

    # Load model
    model, tokenizer = load_model_and_tokenizer(size=model_size)

    # Load datasets
    print(f"\nLoading datasets (max_seq_length={MAX_SEQ_LENGTH})...")
    train_dataset = prepare_dataset(train_path, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = prepare_dataset(val_path, tokenizer, MAX_SEQ_LENGTH)

    # Verify a sample
    sample = train_dataset[0]
    print(f"\n  Sample prompt  (last 100 chars): ...{sample['prompt'][-100:]}")
    print(f"  Sample completion (first 100 chars): {sample['completion'][:100]}...")

    # Training arguments
    eval_steps = max(1, len(train_dataset) // (batch_size * grad_accum * 4))
    print(f"\n  Eval every {eval_steps} steps")

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=WARMUP_RATIO,
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
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to=report_to,
        optim="adamw_torch_fused",
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
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
    size_tag = model_size or MODEL_SIZE
    print("\n" + "=" * 60)
    print(f"Starting Josi v6 training — Qwen3-{cfg['params']} (bf16 + LoRA)")
    print(f"  Mode: {mode.upper()}")
    if mode == "unified":
        print(f"    Combined: interpreter (JSON) + coach (text) in one training run")
    elif mode == "interpreter":
        print(f"    GATCRequest JSON output only")
    else:
        print(f"    Coaching text output only")
    print(f"  Model: {cfg['model_id']}")
    print(f"  Target: {'Android 12GB+' if size_tag == '8b' else 'All phones (iPhone + Android)'}")
    print(f"  Pipeline: single-model, dual-mode + router-controlled /think")
    print(f"  Params: {cfg['params']}")
    print(f"  LoRA: r={cfg['lora_r']}, alpha={cfg['lora_alpha']}, bf16")
    print(f"  Effective batch size: {batch_size * grad_accum}")
    print(f"  Learning rate: {lr}")
    print(f"  Max epochs: {epochs} (early stopping patience={EARLY_STOPPING_PATIENCE})")
    print(f"  Context cap: {MAX_SEQ_LENGTH} tokens")
    print(f"  Output: {output_path}")
    print(f"  GGUF target: {cfg['gguf_size']} (single file, both modes)")
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
        "epochs_completed": round(trainer.state.epoch, 2) if trainer.state.epoch else epochs,
    }
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {output_path / 'final'}")
    print(f"LoRA weights: {lora_path}")
    print(f"Training config: {output_path / 'training_config.json'}")
    print(f"Metrics: {output_path / 'training_metrics.json'}")
    print(f"\nNext steps:")
    print(f"  python finetune_qwen3.py merge --lora_path {lora_path}")

    if report_to == "wandb":
        import wandb
        wandb.finish()

    return str(output_path)


# =============================================================================
# MERGE
# =============================================================================

def merge(
    lora_path: str,
    output_path: str = None,
    model_size: str = None,
):
    """Merge LoRA weights with base Qwen3 model for GGUF export.

    Loads the full-precision base model, applies the LoRA weights, and saves
    a single merged model ready for GGUF conversion.
    """
    from peft import PeftModel

    if output_path is None:
        output_path = str(Path(lora_path).parent / "merged")

    # Auto-detect model size from lora_path (e.g. "josi-v6-qwen3-4b-unified-...")
    if model_size is None:
        lora_str = str(lora_path)
        if "-4b-" in lora_str:
            model_size = "4b"
        elif "-8b-" in lora_str:
            model_size = "8b"

    model_id = resolve_model_id(model_size)
    print(f"Loading base model: {model_id} (full precision for merge)")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Loading LoRA weights: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging weights...")
    merged = model.merge_and_unload()

    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print(f"Merge complete! Ready for GGUF export:")
    print(f"  python /path/to/llama.cpp/convert_hf_to_gguf.py {output_path} --outtype q4_k_m")
    cfg = get_config(model_size)
    print(f"  Expected GGUF size: {cfg['gguf_size']} (Q4_K_M)")

    return output_path


# =============================================================================
# SANITY CHECK
# =============================================================================

SANITY_PROMPTS = [
    # --- Interpreter mode ---
    {
        "mode": "interpreter",
        "user_content": "I want to do a 45 minute run today\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
        "check": "should produce create_workout with sport=run, time=45",
        "validate": lambda parsed: parsed.get("action") == "create_workout" and parsed.get("sport") == "run",
    },
    {
        "mode": "interpreter",
        "user_content": "I want a workout",
        "check": "should clarify (no sport, no context)",
        "validate": lambda parsed: parsed.get("action") == "clarify",
    },
    {
        "mode": "interpreter",
        "user_content": "I have chest pain and I feel dizzy\n\nCONTEXT:\n- Readiness: Green",
        "check": "should output medical safety clarify",
        "validate": lambda parsed: parsed.get("action") == "clarify" and "medical" in str(parsed.get("missing", [])).lower(),
    },
    {
        "mode": "interpreter",
        "user_content": "I'm sick, can't train this week\n\nCONTEXT:\n- Sport: running\n- Readiness: Red",
        "check": "should replan with illness (not clarify)",
        "validate": lambda parsed: parsed.get("action") == "replan" and parsed.get("replan_type") == "illness",
    },
    {
        "mode": "interpreter",
        "user_content": "What is Zone 2?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
        "check": "should answer_question about Zone 2",
        "validate": lambda parsed: parsed.get("action") == "answer_question",
    },
    {
        "mode": "interpreter",
        "user_content": "Ik wil een hardlooptraining van een uur\n\nCONTEXT:\n- Sport: running\n- Readiness: Green",
        "check": "should handle Dutch: create_workout sport=run time=60",
        "validate": lambda parsed: parsed.get("action") == "create_workout" and parsed.get("sport") == "run",
    },
    # --- Coach mode ---
    {
        "mode": "coach",
        "user_content": 'What is Zone 2?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green\n\n[INTERPRETER]\n{"action": "answer_question", "question": "What is Zone 2?", "free_text": "What is Zone 2?"}',
        "check": "should give plain coaching text about Zone 2 (no JSON)",
        "validate": lambda text: "{" not in text and len(text) > 20,
    },
    {
        "mode": "coach",
        "user_content": 'Why is today an easy day?\n\nCONTEXT:\n- Sport: running\n- Session: Z2 60min Easy aerobic\n- Readiness: Green\n\n[INTERPRETER]\n{"action": "explain", "question": "Why is today an easy day?", "free_text": "Why is today an easy day?"}',
        "check": "should explain session purpose in plain language (no JSON)",
        "validate": lambda text: "{" not in text and len(text) > 20,
    },
    {
        "mode": "coach",
        "user_content": 'Wat is Zone 2?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green\n\n[INTERPRETER]\n{"action": "answer_question", "question": "Wat is Zone 2?", "free_text": "Wat is Zone 2?"}',
        "check": "should respond in Dutch (no JSON)",
        "validate": lambda text: "{" not in text and len(text) > 20,
    },
]


# =============================================================================
# OPEN CHAT — Interactive testing with full v6 pipeline
# =============================================================================

def chat(model_path: str, model_size: str = None, sport: str = None,
         readiness: str = "Green"):
    """Interactive open chat with the v6 model.

    Runs the full pipeline for every message:
      1. Interpreter call → GATCRequest JSON
      2. Router decides if coach call is needed
      3. Coach call → coaching text (if needed)

    Commands:
      /sport <sport>       Change sport context (running, cycling, strength, ...)
      /readiness <level>   Change readiness (Green, Yellow, Red)
      /mode <mode>         Force mode: auto, interpreter, coach
      /raw                 Toggle showing raw model output
      /reset               Clear conversation history
      /quit or /exit       Exit chat
    """
    import re as re_mod

    print(f"\nLoading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    interpreter_prompt = build_system_prompt("interpreter")
    coach_prompt = build_system_prompt("coach")

    show_raw = True
    force_mode = "auto"  # auto, interpreter, coach
    conversation_history = []  # Multi-turn memory: list of (user_msg, interpreter_json, coach_response)

    import time as time_mod

    def generate(system_prompt, user_content, temperature=0.3, max_tokens=200):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip thinking tags
        if "<think>" in response:
            response = re_mod.sub(r"<think>.*?</think>", "", response, flags=re_mod.DOTALL).strip()

        return response

    def build_history_block():
        """Build a conversation summary from recent turns for context."""
        if not conversation_history:
            return ""
        lines = ["\nCONVERSATION HISTORY:"]
        # Keep last 5 turns to stay within context limits
        for user_msg, interp_json, coach_resp in conversation_history[-5:]:
            lines.append(f"  Athlete: {user_msg}")
            if interp_json:
                lines.append(f"  → {interp_json}")
            if coach_resp:
                lines.append(f"  Josi: {coach_resp}")
        return "\n".join(lines)

    print("\n" + "=" * 60)
    print("  JOSI v6 — OPEN CHAT TEST")
    print("=" * 60)
    print(f"  Model:     {model_path}")
    print(f"  Sport:     {sport or '(none — model will infer or clarify)'}")
    print(f"  Readiness: {readiness}")
    print(f"  Mode:      {force_mode}")
    print(f"  Show raw:  {show_raw}")
    print()
    print("  Commands: /sport, /readiness, /mode, /raw, /reset, /quit")
    print("  NOTE: Coach is called for EVERY message (open test mode)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Bye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                print("  Bye!")
                break
            elif cmd == "/sport":
                sport = arg or None
                print(f"  Sport → {sport or '(none)'}")
                continue
            elif cmd == "/readiness":
                readiness = arg or "Green"
                print(f"  Readiness → {readiness}")
                continue
            elif cmd == "/mode":
                if arg in ("auto", "interpreter", "coach"):
                    force_mode = arg
                    print(f"  Mode → {force_mode}")
                else:
                    print(f"  Usage: /mode auto|interpreter|coach")
                continue
            elif cmd == "/raw":
                show_raw = not show_raw
                print(f"  Show raw → {show_raw}")
                continue
            elif cmd == "/reset":
                sport = None
                readiness = "Green"
                force_mode = "auto"
                conversation_history.clear()
                print("  Reset to defaults (history cleared)")
                continue
            elif cmd == "/history":
                if conversation_history:
                    print(build_history_block())
                else:
                    print("  (no history yet)")
                continue
            else:
                print(f"  Unknown command: {cmd}")
                continue

        # Build context block — only include sport if explicitly set
        context_lines = []
        if sport:
            context_lines.append(f"- Sport: {sport}")
        context_lines.append(f"- Readiness: {readiness}")
        context = "\n\nCONTEXT:\n" + "\n".join(context_lines)

        # Add conversation history for multi-turn awareness
        history_block = build_history_block()
        user_with_context = user_input + context + history_block

        # ─── Step 1: Interpreter call ────────────────────────────────────
        interpreter_response = None
        action = None
        parsed = None

        if force_mode != "coach":
            t0 = time_mod.time()
            interpreter_response = generate(
                interpreter_prompt, user_with_context,
                temperature=INTERPRETER_TEMPERATURE, max_tokens=INFERENCE_MAX_TOKENS
            )
            t_interp = time_mod.time() - t0

            if show_raw:
                print(f"\n  ┌─ INTERPRETER ({t_interp:.1f}s) ─────────────────────")
                print(f"  │ {interpreter_response}")
                print(f"  └{'─' * 50}")

            # Parse the interpreter JSON
            try:
                parsed = json.loads(interpreter_response)
                action = parsed.get("action", "unknown")
            except json.JSONDecodeError:
                action = "parse_error"
                parsed = None
                print(f"  ⚠ Interpreter returned invalid JSON")

            if force_mode == "interpreter":
                conversation_history.append((user_input, interpreter_response, None))
                continue

        # ─── Step 2: Always call coach in open chat mode ─────────────────
        if force_mode == "coach" or interpreter_response is None:
            # Coach-only mode: synthesize interpreter context
            interpreter_response = json.dumps({"action": "answer_question", "question": user_input, "free_text": user_input})

        coach_input = f"{user_input}{context}{history_block}\n\n[INTERPRETER]\n{interpreter_response}"

        t0 = time_mod.time()
        coach_response = generate(
            coach_prompt, coach_input,
            temperature=COACH_TEMPERATURE, max_tokens=INFERENCE_MAX_TOKENS
        )
        t_coach = time_mod.time() - t0

        print(f"\n  ┌─ JOSI ({t_coach:.1f}s) ──────────────────────────────")
        print(f"  │ {coach_response}")
        print(f"  └{'─' * 50}")

        # Save to conversation history
        conversation_history.append((user_input, interpreter_response, coach_response))


def sanity(model_path: str, mode: str = "interpreter"):
    """Quick sanity check on a merged or fine-tuned model."""
    print(f"Loading model for sanity check: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    system_prompt = build_system_prompt(mode)
    prompts_for_mode = [p for p in SANITY_PROMPTS if p["mode"] == mode]

    temp = INTERPRETER_TEMPERATURE if mode == "interpreter" else COACH_TEMPERATURE
    passed = 0
    failed = 0

    for i, test in enumerate(prompts_for_mode, 1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test["user_content"]},
        ]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=INFERENCE_MAX_TOKENS,
                temperature=temp,
                do_sample=True,
                top_p=0.9,
            )

        # Decode only new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip any thinking tags (Qwen3 may produce <think>...</think>)
        if "<think>" in response:
            import re
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        print(f"\n--- Test {i}: {test['check']} ---")
        print(f"  User: {test['user_content'][:80]}...")
        print(f"  Response: {response[:200]}")

        # Validate
        try:
            if mode == "interpreter":
                parsed = json.loads(response)
                ok = test["validate"](parsed)
            else:
                ok = test["validate"](response)
        except (json.JSONDecodeError, Exception):
            ok = False

        if ok:
            print(f"  PASS")
            passed += 1
        else:
            print(f"  FAIL")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("All sanity checks passed!")
    else:
        print("Some checks failed — review outputs above.")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MiValta Josi v6 — Qwen3 LoRA Fine-Tuning (Single-Model Architecture)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train
    train_parser = subparsers.add_parser("train", help="Fine-tune Qwen3 with LoRA")
    train_parser.add_argument("--mode", choices=["interpreter", "coach", "unified"], default="unified",
                             help="Training mode: interpreter (JSON), coach (text), or unified (both — recommended)")
    train_parser.add_argument("--model-size", choices=["8b", "4b"], default="8b",
                             help="Model size: 8b (Android, best quality) or 4b (iPhone, all phones)")
    train_parser.add_argument("--train_data", help="Path to training JSONL")
    train_parser.add_argument("--val_data", help="Path to validation JSONL")
    train_parser.add_argument("--output_dir", help="Output directory")
    train_parser.add_argument("--lr", type=float, help="Learning rate override")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs override")
    train_parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

    # Merge
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA weights into base model")
    merge_parser.add_argument("--lora_path", required=True, help="Path to LoRA weights")
    merge_parser.add_argument("--output_path", help="Output path for merged model")
    merge_parser.add_argument("--model-size", choices=["8b", "4b"],
                              help="Model size (auto-detected from lora_path if omitted)")

    # Sanity
    sanity_parser = subparsers.add_parser("sanity", help="Quick sanity check on model")
    sanity_parser.add_argument("--model_path", required=True, help="Path to merged model")
    sanity_parser.add_argument("--mode", choices=["interpreter", "coach"], default="interpreter",
                              help="Mode to test")

    # Chat — interactive open testing
    chat_parser = subparsers.add_parser("chat", help="Interactive open chat with v6 model")
    chat_parser.add_argument("--model_path", required=True, help="Path to merged model")
    chat_parser.add_argument("--model-size", choices=["8b", "4b"],
                             help="Model size (auto-detected from path if omitted)")
    chat_parser.add_argument("--sport", default=None, help="Initial sport context (omit to let model infer)")
    chat_parser.add_argument("--readiness", default="Green", help="Initial readiness (Green/Yellow/Red)")

    args = parser.parse_args()

    if args.command == "train":
        train(
            train_path=args.train_data,
            val_path=args.val_data,
            output_dir=args.output_dir,
            use_wandb=not args.no_wandb,
            lr_override=args.lr,
            epochs_override=args.epochs,
            mode=args.mode,
            model_size=getattr(args, "model_size", None),
        )
    elif args.command == "merge":
        merge(
            lora_path=args.lora_path,
            output_path=args.output_path,
            model_size=getattr(args, "model_size", None),
        )
    elif args.command == "sanity":
        sanity(
            model_path=args.model_path,
            mode=args.mode,
        )
    elif args.command == "chat":
        chat(
            model_path=args.model_path,
            model_size=getattr(args, "model_size", None),
            sport=args.sport,
            readiness=args.readiness,
        )
    else:
        parser.print_help()
        print("\nv6 single-model pipeline (Android-first, Qwen3-8B):")
        print("  1. python prepare_v6_data.py                                          # Prepare unified data")
        print("  2. python finetune_qwen3.py train --mode unified                      # 8B Android (default)")
        print("     python finetune_qwen3.py train --mode unified --model-size 4b      # 4B iPhone variant")
        print("  3. python finetune_qwen3.py merge --lora_path ./models/.../lora_weights")
        print("  4. python finetune_qwen3.py sanity --model_path ./models/.../merged --mode interpreter")
        print("  5. python finetune_qwen3.py sanity --model_path ./models/.../merged --mode coach")
        print("  6. python publish_models_v6.py --model ./models/.../final             # GGUF + publish")


if __name__ == "__main__":
    main()

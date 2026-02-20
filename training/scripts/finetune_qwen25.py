#!/usr/bin/env python3
"""
MiValta Josi v5 — Qwen2.5-1.5B LoRA Fine-Tuning Script

Sequential dual-model architecture:
  1. INTERPRETER: Qwen2.5-1.5B → GATCRequest JSON (~50 tokens, fast)
  2. ROUTER (code, not LLM): decides if explainer is needed
  3. EXPLAINER: Qwen2.5-1.5B → coaching text (grounded in interpreter output)

Router logic (pure code):
  - create_workout → skip explainer, return JSON (engine builds workout)
  - replan         → skip explainer, return JSON (engine handles replan)
  - clarify        → skip explainer, use interpreter's clarify_message field
  - explain        → run explainer (grounded in interpreter output)
  - answer_question→ run explainer (grounded in interpreter output)

On-device performance:
  - Two sequential calls: ~200ms + ~400ms = under 1 second on modern phone
  - Explainer skipped for ~40% of messages (free latency savings)
  - Two 1.5B Q4_K_M models = ~2 GB total VRAM
  - 100% on-device via llama.cpp on Android — NO network calls

Architecture notes:
  - Qwen2.5-1.5B-Instruct uses ChatML natively (<|im_start|>/<|im_end|>)
  - Uses AutoModelForCausalLM + AutoTokenizer (standard HF pipeline)
  - bf16 loading (~3 GB) + LoRA adapters — fits easily in any GPU
  - LoRA targets: attention + MLP projections
  - Merged + GGUF Q4_K_M for on-device deployment (~1 GB per model)

Runtime constraints (on-device):
  - Context cap: 2048 tokens (interpreter system prompt needs ~1300)
  - Output cap: 150 tokens
  - Temperature: 0.4-0.5
  - 100% on-device via llama.cpp on Android

Usage:
    # 1. Train interpreter
    python finetune_qwen25.py train --mode interpreter

    # 2. Prepare sequential explainer data (pairs with interpreter output)
    python prepare_sequential_data.py

    # 3. Train explainer (with interpreter context)
    python finetune_qwen25.py train --mode explainer

    # 4. Merge LoRA weights into base model
    python finetune_qwen25.py merge --lora_path ./models/josi-v5-qwen25-interpreter-*/lora_weights
    python finetune_qwen25.py merge --lora_path ./models/josi-v5-qwen25-explainer-*/lora_weights

    # 5. Sanity check
    python finetune_qwen25.py sanity --model_path ./models/josi-v5-qwen25-interpreter-*/merged --mode interpreter
    python finetune_qwen25.py sanity --model_path ./models/josi-v5-qwen25-explainer-*/merged --mode explainer

    # 6. GGUF conversion (via llama.cpp)
    python /path/to/llama.cpp/convert_hf_to_gguf.py ./models/merged --outtype q4_k_m

Requirements:
    pip install transformers torch peft datasets accelerate trl sentencepiece protobuf huggingface_hub
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
# MODEL CONFIGURATION — Qwen2.5-1.5B
# =============================================================================

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Local path (set by download, falls back to HF hub)
SCRIPT_DIR_FOR_MODEL = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = SCRIPT_DIR_FOR_MODEL.parent / "models" / "Qwen2.5-1.5B-Instruct"


def resolve_model_id():
    """Use local download if available, otherwise pull from HuggingFace."""
    if LOCAL_MODEL_PATH.exists() and (LOCAL_MODEL_PATH / "config.json").exists():
        return str(LOCAL_MODEL_PATH)
    return MODEL_ID


# LoRA config — higher rank for smaller model (more adaptation capacity needed)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# LoRA targets on the language model backbone
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]

# Training hyperparameters — tuned for 1.5B model
LEARNING_RATE = 5e-5       # Higher than Gemma 3n (smaller model needs stronger signal)
BATCH_SIZE = 4             # Fits easily in VRAM (~3 GB model)
GRAD_ACCUM = 4             # Effective batch = 16
MAX_SEQ_LENGTH = 2048      # Training context (interpreter system prompt is ~1300 tokens)
EPOCHS = 4                 # More epochs for smaller model
WARMUP_RATIO = 0.05

# Early stopping
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"
DEFAULT_TRAIN = DATA_DIR / "train_interpreter.jsonl"
DEFAULT_VAL = DATA_DIR / "val_interpreter.jsonl"

# Josi runtime constraints (baked into training + enforced at inference)
INFERENCE_TEMPERATURE = 0.45   # Range: 0.4-0.5
INFERENCE_MAX_TOKENS = 150     # Output cap for on-device

# Sequential architecture: actions that need the explainer
ACTIONS_NEEDING_EXPLAINER = {"explain", "answer_question"}


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
              "explainer" for plain coaching text output (sequential).
    """
    if mode == "explainer":
        # Sequential explainer prompt (with interpreter context section)
        seq_path = PROMPTS_DIR / "explainer_sequential_system.txt"
        if seq_path.exists():
            return load_system_prompt("explainer_sequential_system.txt")
        return load_system_prompt("explainer_system.txt")
    return load_system_prompt("interpreter_system.txt")


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


def prepare_dataset(path: str, tokenizer, max_seq_length: int = 1024) -> Dataset:
    """Load JSONL and split into prompt/completion format for completion-only loss.

    Training data is in ChatML format (system/user/assistant messages).
    Qwen2.5 uses ChatML natively, so no format conversion is needed.

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

def load_model_and_tokenizer(model_id: str = None):
    """Load Qwen2.5-1.5B-Instruct in bf16 and apply LoRA adapters."""
    if model_id is None:
        model_id = resolve_model_id()

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Ensure pad_token is separate from eos_token
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print(f"  Added dedicated [PAD] token (id={tokenizer.pad_token_id}) — separate from EOS (id={tokenizer.eos_token_id})")
    tokenizer.padding_side = "right"

    print(f"Loading model: {model_id} (bf16, LoRA fine-tuning)")
    print(f"  Architecture: Qwen2.5-1.5B-Instruct (AutoModelForCausalLM)")
    print(f"  Params: 1.5B")
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

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # LoRA adapters
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print(f"Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Report VRAM usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"  VRAM: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")

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
    mode: str = "interpreter",
):
    """Run LoRA fine-tuning on Qwen2.5-1.5B.

    Args:
        mode: "interpreter" for GATCRequest JSON training,
              "explainer" for sequential coaching text training.
    """
    lr = lr_override or LEARNING_RATE
    epochs = epochs_override or EPOCHS

    if train_path is None:
        if mode == "explainer":
            # Sequential explainer data (with interpreter context)
            seq_path = DATA_DIR / "train_explainer_sequential.jsonl"
            if seq_path.exists():
                train_path = str(seq_path)
            else:
                print(f"ERROR: Sequential explainer data not found at {seq_path}")
                print(f"Run 'python prepare_sequential_data.py' first.")
                sys.exit(1)
        else:
            train_path = str(DEFAULT_TRAIN)

    if val_path is None:
        if mode == "explainer":
            seq_val = DATA_DIR / "val_explainer_sequential.jsonl"
            if seq_val.exists():
                val_path = str(seq_val)
            else:
                val_path = str(DATA_DIR / "val_explainer.jsonl")
        else:
            val_path = str(DEFAULT_VAL)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./models/josi-v5-qwen25-{mode}-{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save training config
    run_config = {
        "model_id": MODEL_ID,
        "model_family": "Qwen2.5-1.5B",
        "architecture": "AutoModelForCausalLM (ChatML)",
        "pipeline": "sequential (interpreter → router → explainer)",
        "mode": mode,
        "params": "1.5B",
        "quantization": "bf16 + LoRA",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
        "lr": lr,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_data": train_path,
        "val_data": val_path,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "inference_temperature": INFERENCE_TEMPERATURE,
        "inference_max_tokens": INFERENCE_MAX_TOKENS,
        "gguf_target": "Q4_K_M (~1.0 GB per model, ~2 GB total)",
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path / "training_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # W&B setup
    report_to = "none"
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="mivalta-josi-v5",
                name=f"josi-v5-qwen25-{mode}-{datetime.now().strftime('%m%d_%H%M')}",
                config=run_config,
            )
            report_to = "wandb"
            print("W&B tracking enabled")
        except Exception:
            print("W&B not available, logging to console only")
            report_to = "none"

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Load datasets
    print(f"\nLoading datasets (max_seq_length={MAX_SEQ_LENGTH})...")
    train_dataset = prepare_dataset(train_path, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = prepare_dataset(val_path, tokenizer, MAX_SEQ_LENGTH)

    # Verify a sample
    sample = train_dataset[0]
    print(f"\n  Sample prompt  (last 100 chars): ...{sample['prompt'][-100:]}")
    print(f"  Sample completion (first 100 chars): {sample['completion'][:100]}...")

    # Training arguments
    eval_steps = max(1, len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM * 4))
    print(f"\n  Eval every {eval_steps} steps")

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
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
    print("\n" + "=" * 60)
    print(f"Starting Josi v5 training — Qwen2.5-1.5B (bf16 + LoRA)")
    print(f"  Mode: {mode.upper()} ({'GATCRequest JSON' if mode == 'interpreter' else 'coaching text (sequential)'})")
    print(f"  Model: {MODEL_ID}")
    print(f"  Pipeline: sequential (interpreter -> router -> explainer)")
    print(f"  Params: 1.5B")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, bf16")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Learning rate: {lr}")
    print(f"  Max epochs: {epochs} (early stopping patience={EARLY_STOPPING_PATIENCE})")
    print(f"  Context cap: {MAX_SEQ_LENGTH} tokens")
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
        "epochs_completed": round(trainer.state.epoch, 2) if trainer.state.epoch else epochs,
    }
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to: {output_path / 'final'}")
    print(f"LoRA weights: {lora_path}")
    print(f"Training config: {output_path / 'training_config.json'}")
    print(f"Metrics: {output_path / 'training_metrics.json'}")
    print(f"\nNext steps:")
    print(f"  python finetune_qwen25.py merge --lora_path {lora_path}")

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
):
    """Merge LoRA weights with base Qwen2.5-1.5B model for GGUF export.

    Loads the full-precision base model, applies the LoRA weights, and saves
    a single merged model ready for GGUF conversion.
    """
    from peft import PeftModel

    if output_path is None:
        output_path = str(Path(lora_path).parent / "merged")

    model_id = resolve_model_id()
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
    print(f"  Expected GGUF size: ~1.0 GB (Q4_K_M)")

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
    # --- Explainer mode (sequential — includes interpreter context) ---
    {
        "mode": "explainer",
        "user_content": 'What is Zone 2?\n\nCONTEXT:\n- Sport: running\n- Readiness: Green\n\n[INTERPRETER]\n{"action": "answer_question", "question": "What is Zone 2?", "free_text": "What is Zone 2?"}',
        "check": "should give plain coaching text about Zone 2 (no JSON)",
        "validate": lambda text: "{" not in text and len(text) > 20,
    },
    {
        "mode": "explainer",
        "user_content": 'Why is today an easy day?\n\nCONTEXT:\n- Sport: running\n- Session: Z2 60min Easy aerobic\n- Readiness: Green\n\n[INTERPRETER]\n{"action": "explain", "question": "Why is today an easy day?", "free_text": "Why is today an easy day?"}',
        "check": "should explain session purpose in plain language (no JSON)",
        "validate": lambda text: "{" not in text and len(text) > 20,
    },
]


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
                temperature=INFERENCE_TEMPERATURE,
                do_sample=True,
                top_p=0.9,
            )

        # Decode only new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

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
        description="MiValta Josi v5 — Qwen2.5-1.5B LoRA Fine-Tuning (Sequential Architecture)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train
    train_parser = subparsers.add_parser("train", help="Fine-tune Qwen2.5-1.5B with LoRA")
    train_parser.add_argument("--mode", choices=["interpreter", "explainer"], default="interpreter",
                             help="Training mode: interpreter (JSON) or explainer (coaching text)")
    train_parser.add_argument("--train_data", help="Path to training JSONL")
    train_parser.add_argument("--val_data", help="Path to validation JSONL")
    train_parser.add_argument("--output_dir", help="Output directory")
    train_parser.add_argument("--lr", type=float, help=f"Learning rate (default: {LEARNING_RATE})")
    train_parser.add_argument("--epochs", type=int, help=f"Number of epochs (default: {EPOCHS})")
    train_parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

    # Merge
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA weights into base model")
    merge_parser.add_argument("--lora_path", required=True, help="Path to LoRA weights")
    merge_parser.add_argument("--output_path", help="Output path for merged model")

    # Sanity
    sanity_parser = subparsers.add_parser("sanity", help="Quick sanity check on model")
    sanity_parser.add_argument("--model_path", required=True, help="Path to merged model")
    sanity_parser.add_argument("--mode", choices=["interpreter", "explainer"], default="interpreter",
                              help="Mode to test")

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
        )
    elif args.command == "merge":
        merge(
            lora_path=args.lora_path,
            output_path=args.output_path,
        )
    elif args.command == "sanity":
        sanity(
            model_path=args.model_path,
            mode=args.mode,
        )
    else:
        parser.print_help()
        print("\nSequential architecture pipeline:")
        print("  1. python finetune_qwen25.py train --mode interpreter")
        print("  2. python prepare_sequential_data.py")
        print("  3. python finetune_qwen25.py train --mode explainer")
        print("  4. python finetune_qwen25.py merge --lora_path ./models/.../lora_weights")
        print("  5. python finetune_qwen25.py sanity --model_path ./models/.../merged --mode interpreter")


if __name__ == "__main__":
    main()

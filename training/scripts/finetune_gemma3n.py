#!/usr/bin/env python3
"""
MiValta Josi v4 — Gemma 3n E2B LoRA Fine-Tuning Script

Fine-tunes google/gemma-3n-E2B-it (6B raw params, 2B effective via MatFormer)
on Josi coaching training data (LLMIntent JSON output format).

Architecture notes:
  - Gemma 3n E2B uses Gemma3nForConditionalGeneration (multimodal, text-only for us)
  - Uses AutoProcessor (not AutoTokenizer) for chat template
  - Chat template supports system role natively
  - Message content uses array format: [{"type": "text", "text": "..."}]
  - bf16 loading (~12 GB) + LoRA adapters + gradient checkpointing (fits ~20GB VRAM)
  - Note: 4-bit QLoRA incompatible with Gemma 3n AltUp clamp_() on quantized weights
  - LoRA targets: attention + MLP projections on the language model
  - Merged + GGUF Q4_K_M for on-device deployment (~2-3GB)

Runtime constraints (on-device):
  - Context cap: 1024 tokens
  - Output cap: 150 tokens
  - Temperature: 0.4-0.5
  - 100% on-device via llama.cpp on Android — NO network calls

Usage:
    # Train (default config)
    python finetune_gemma3n.py train

    # Train with custom params
    python finetune_gemma3n.py train --lr 3e-5 --epochs 4

    # Merge LoRA weights into base model
    python finetune_gemma3n.py merge --lora_path ./models/josi-v4-gemma3n-*/lora_weights

    # Quick sanity check on merged model
    python finetune_gemma3n.py sanity --model_path ./models/josi-v4-gemma3n-merged

Requirements:
    pip install -U transformers>=4.53.0 torch peft datasets accelerate trl timm torchvision
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
    AutoProcessor,
    Gemma3nForConditionalGeneration,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from trl import SFTTrainer, SFTConfig


# =============================================================================
# MODEL CONFIGURATION — Gemma 3n E2B
# =============================================================================

MODEL_ID = "google/gemma-3n-E2B-it"

# Local path (set by setup_hetzner.sh download, falls back to HF hub)
SCRIPT_DIR_FOR_MODEL = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = SCRIPT_DIR_FOR_MODEL.parent / "models" / "gemma-3n-E2B-it"

def resolve_model_id():
    """Use local download if available, otherwise pull from HuggingFace."""
    if LOCAL_MODEL_PATH.exists() and (LOCAL_MODEL_PATH / "config.json").exists():
        return str(LOCAL_MODEL_PATH)
    return MODEL_ID

# QLoRA config: smaller rank since effective 2B — less adaptation needed
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# LoRA targets on the language model backbone
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]

# Training hyperparameters
LEARNING_RATE = 2e-5       # Conservative for fine-tuning
BATCH_SIZE = 1             # Micro-batch=1 to fit bf16 model in 20GB VRAM
GRAD_ACCUM = 16            # Effective batch = 16
MAX_SEQ_LENGTH = 1024      # On-device context cap
EPOCHS = 3                 # Gemma converges faster than SmolLM2-360M
WARMUP_RATIO = 0.05

# Early stopping
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"
# v4 dual-model: default to interpreter data; override with --train_data for explainer
DEFAULT_TRAIN = DATA_DIR / "train_interpreter.jsonl"
DEFAULT_VAL = DATA_DIR / "val_interpreter.jsonl"

# Josi runtime constraints (baked into training + enforced at inference)
INFERENCE_TEMPERATURE = 0.45   # Range: 0.4-0.5
INFERENCE_MAX_TOKENS = 150     # Output cap for on-device


# =============================================================================
# GEMMA 3n MESSAGE FORMAT CONVERSION
# =============================================================================

def convert_to_gemma3n_messages(messages: list[dict]) -> list[dict]:
    """Convert training data messages to Gemma 3n format.

    Training data uses ChatML-style:
        {"role": "system",    "content": "plain text"}
        {"role": "user",      "content": "plain text"}
        {"role": "assistant", "content": "plain text"}

    Gemma 3n expects:
        {"role": "system", "content": [{"type": "text", "text": "..."}]}
        {"role": "user",   "content": [{"type": "text", "text": "..."}]}
        {"role": "model",  "content": [{"type": "text", "text": "..."}]}
    """
    converted = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Map "assistant" -> "model" (Gemma convention)
        if role == "assistant":
            role = "model"

        # Wrap plain string content in the array format
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        converted.append({"role": role, "content": content})

    return converted


# =============================================================================
# JOSI v4 SYSTEM PROMPTS — Dual-Model Architecture
# =============================================================================
# v4 uses separate system prompts for interpreter and explainer, loaded from
# training/prompts/. The old monolithic JOSI_SYSTEM_PROMPT (LLMIntent format)
# is no longer used.

def load_system_prompt(filename: str) -> str:
    """Load a system prompt from the prompts directory."""
    path = PROMPTS_DIR / filename
    return path.read_text().strip()


def build_system_prompt(mode: str = "interpreter") -> str:
    """Build the system prompt for the given model mode.

    Args:
        mode: "interpreter" for GATCRequest JSON output,
              "explainer" for plain coaching text output.
    """
    if mode == "explainer":
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
    """Validate that the conversation has exactly one assistant/model response at the end."""
    assistant_count = sum(1 for m in messages if m["role"] in ("assistant", "model"))
    if assistant_count != 1:
        return False
    if messages[-1]["role"] not in ("assistant", "model"):
        return False
    return True


def prepare_dataset(path: str, processor, max_seq_length: int = 1024) -> Dataset:
    """Load JSONL and split into prompt/completion format for completion-only loss.

    Training data is in ChatML format (system/user/assistant messages).
    We convert to Gemma 3n format and split into:
      prompt:     system + user turns + generation marker (everything BEFORE the response)
      completion: model response content + EOS token (what the model must learn)

    TRL's SFTTrainer auto-detects prompt/completion format and creates a
    completion_mask, so the model only trains on the completion (including the
    <end_of_turn> stop token) — not the system prompt or user message.

    Validates that all examples are single-turn (one model response) so the
    model learns to produce exactly one response and then stop with <end_of_turn>.
    """
    raw = load_jsonl(path)
    print(f"  Loaded {len(raw)} examples from {path}")

    tokenizer = processor.tokenizer
    eos_token = tokenizer.eos_token or "<end_of_turn>"
    examples = []
    truncated = 0
    skipped = 0
    multi_turn = 0

    for ex in raw:
        messages = ex["messages"]

        # Validate single-turn: model must learn to output ONE response then stop
        if not validate_single_turn(messages):
            multi_turn += 1
            if multi_turn <= 3:
                roles = [m["role"] for m in messages]
                print(f"  WARNING: Multi-turn example skipped (roles: {roles})")
            continue

        # Convert to Gemma 3n format
        gemma_messages = convert_to_gemma3n_messages(messages)

        # Split into prompt (system+user) and completion (model response)
        # Prompt: system+user turns with generation prompt marker
        prompt_messages = gemma_messages[:-1]  # system + user
        try:
            prompt = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,  # adds <start_of_turn>model\n
            )
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  WARNING: Skipping example (template error): {e}")
            continue

        # Completion: model response content + EOS stop token
        # Extract the text content from the Gemma 3n format message
        model_message = gemma_messages[-1]
        if isinstance(model_message["content"], list):
            model_content = model_message["content"][0]["text"]
        else:
            model_content = model_message["content"]
        completion = model_content + eos_token

        # Truncation check on the full sequence
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
        print(f"  WARNING: {multi_turn}/{len(raw)} multi-turn examples skipped (single-turn only)")
    if truncated:
        print(f"  WARNING: {truncated}/{len(raw)} examples truncated to {max_seq_length} tokens")
    if skipped:
        print(f"  WARNING: {skipped}/{len(raw)} examples skipped (template errors)")

    # Verify stop token is present in completions
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

def load_model_and_processor(model_id: str = None):
    """Load Gemma 3n E2B in bf16 and apply LoRA adapters.

    Uses Gemma3nForConditionalGeneration + AutoProcessor (not AutoTokenizer).

    Note: 4-bit QLoRA fails with Gemma 3n because the AltUp prediction_coefs
    module calls clamp_() during forward pass, which is incompatible with
    bitsandbytes quantized uint8 weights. We load in bf16 instead (~12 GB),
    which fits in 20 GB VRAM with LoRA + gradient checkpointing.
    """
    if model_id is None:
        model_id = resolve_model_id()

    print(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    # CRITICAL: pad_token must NOT equal eos_token.
    # When pad_token_id == eos_token_id, the DataCollator masks ALL occurrences
    # of that token ID to -100 in labels — including the real <end_of_turn> at
    # the end of the model's response. This means the model NEVER trains on the
    # stop token and won't learn to stop generating at inference time.
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print(f"  Added dedicated [PAD] token (id={tokenizer.pad_token_id}) — separate from EOS (id={tokenizer.eos_token_id})")
    tokenizer.padding_side = "right"

    print(f"Loading model: {model_id} (bf16, LoRA fine-tuning)")
    print(f"  Architecture: Gemma3nForConditionalGeneration")
    print(f"  6B raw params, 2B effective (MatFormer selective activation)")
    print(f"  Loading in bf16 (~12 GB) — 4-bit QLoRA incompatible with AltUp clamp_()")
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    # Resize embeddings if we added a pad token
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Resized embeddings to {len(tokenizer)} (added [PAD] token)")

    # Freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()

    # LoRA adapters — target the language model backbone
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

    return model, processor


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
    """Run QLoRA fine-tuning on Gemma 3n E2B.

    Args:
        mode: "interpreter" for GATCRequest JSON training,
              "explainer" for plain coaching text training.
    """

    lr = lr_override or LEARNING_RATE
    epochs = epochs_override or EPOCHS

    if train_path is None:
        if mode == "explainer":
            train_path = str(DATA_DIR / "train_explainer.jsonl")
        else:
            train_path = str(DEFAULT_TRAIN)
    if val_path is None:
        if mode == "explainer":
            val_path = str(DATA_DIR / "val_explainer.jsonl")
        else:
            val_path = str(DEFAULT_VAL)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./models/josi-v4-gemma3n-{mode}-{timestamp}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save training config for reproducibility
    run_config = {
        "model_id": MODEL_ID,
        "model_family": "gemma-3n-E2B",
        "architecture": "Gemma3nForConditionalGeneration",
        "mode": mode,  # "interpreter" or "explainer"
        "raw_params": "6B",
        "effective_params": "2B (MatFormer)",
        "quantization": "bf16 + LoRA (4-bit QLoRA incompatible with AltUp)",
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
                project="mivalta-josi-v4",
                name=f"josi-v4-gemma3n-{datetime.now().strftime('%m%d_%H%M')}",
                config=run_config,
            )
            report_to = "wandb"
            print("W&B tracking enabled")
        except Exception:
            print("W&B not available, logging to console only")
            report_to = "none"

    # Load model
    model, processor = load_model_and_processor()

    # Load datasets
    print(f"\nLoading datasets (max_seq_length={MAX_SEQ_LENGTH})...")
    train_dataset = prepare_dataset(train_path, processor, MAX_SEQ_LENGTH)
    val_dataset = prepare_dataset(val_path, processor, MAX_SEQ_LENGTH)

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
        optim="adamw_torch_fused",  # Built-in fused optimizer (no bitsandbytes needed)
        max_length=MAX_SEQ_LENGTH,
        # Completion-only loss: only train on the model's response (including
        # the <end_of_turn> stop token), not on the system prompt or user message.
        # TRL auto-detects prompt/completion format and creates a completion_mask.
        completion_only_loss=True,
    )

    # Callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
        ),
    ]

    # Trainer — use the tokenizer component from processor for SFT
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.tokenizer,
        args=training_args,
        callbacks=callbacks,
    )

    # Train
    print("\n" + "=" * 60)
    print(f"Starting Josi v4 training — Gemma 3n E2B (bf16 + LoRA)")
    print(f"  Mode: {mode.upper()} ({'GATCRequest JSON' if mode == 'interpreter' else 'plain coaching text'})")
    print(f"  Model: {MODEL_ID}")
    print(f"  Architecture: Gemma3nForConditionalGeneration")
    print(f"  Params: 6B raw, 2B effective (MatFormer)")
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
    processor.save_pretrained(str(lora_path))

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

    if report_to == "wandb":
        wandb.finish()

    return str(output_path)


# =============================================================================
# MERGE
# =============================================================================

def _ensure_spm_tokenizer(model_id: str, output_path: str):
    """Copy tokenizer.model from the base model if missing from the merge output.

    AutoProcessor.save_pretrained() may skip the SentencePiece binary when the
    tokenizer uses backend="tokenizers" (HuggingFace Rust).  llama.cpp needs
    tokenizer.model for correct GGUF conversion — without it the converter falls
    back to the BPE path which corrupts the ▁ space marker.
    """
    import shutil
    target = Path(output_path) / "tokenizer.model"
    if target.exists():
        return

    # Try local base model first
    local = LOCAL_MODEL_PATH / "tokenizer.model"
    if local.exists():
        shutil.copy2(str(local), str(target))
        print(f"  Copied tokenizer.model from local base model")
        return

    # Try HuggingFace cache
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(repo_id=model_id, filename="tokenizer.model")
        shutil.copy2(cached, str(target))
        print(f"  Copied tokenizer.model from HuggingFace cache")
    except Exception:
        print(f"  WARNING: Could not copy tokenizer.model — GGUF conversion may fail.")
        print(f"  Copy it manually from {model_id} into {output_path}/")


def merge(
    lora_path: str,
    output_path: str = None,
):
    """Merge LoRA weights with base Gemma 3n E2B model for GGUF export.

    Loads the full-precision base model (not quantized), applies the LoRA
    weights, and saves a single merged model ready for GGUF conversion.
    """
    from peft import PeftModel

    if output_path is None:
        output_path = str(Path(lora_path).parent / "merged")

    model_id = resolve_model_id()
    print(f"Loading base model: {model_id} (full precision for merge)")
    base_model = Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
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

    processor = AutoProcessor.from_pretrained(model_id)
    processor.save_pretrained(output_path)

    # Ensure tokenizer.model (SentencePiece binary) is copied.
    # AutoProcessor.save_pretrained() may only save tokenizer.json (BPE format)
    # when backend="tokenizers". llama.cpp needs tokenizer.model for correct
    # GGUF conversion via the SentencePiece path.
    _ensure_spm_tokenizer(model_id, output_path)

    print(f"Merge complete! Ready for GGUF export:")
    print(f"  python convert_gemma3n.py --model_path {output_path}")

    return output_path


# =============================================================================
# SANITY CHECK
# =============================================================================

def _build_sanity_messages(mode: str, user_content: str) -> list[dict]:
    """Build Gemma 3n formatted messages for sanity check."""
    return [
        {"role": "system", "content": [{"type": "text", "text": build_system_prompt(mode)}]},
        {"role": "user", "content": [{"type": "text", "text": user_content}]},
    ]


SANITY_PROMPTS = [
    # --- Interpreter mode sanity checks ---
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
    # --- Explainer mode sanity checks ---
    {
        "mode": "explainer",
        "user_content": "What am I doing today?\n\nCONTEXT:\n- Sport: running\n- Session: Z2 60min Easy aerobic\n- Readiness: Green",
        "check": "should produce plain coaching text explaining the session",
        "validate": lambda text: not text.strip().startswith("{") and len(text.split()) >= 10,
    },
    {
        "mode": "explainer",
        "user_content": "I'm feeling really tired today.\n\nCONTEXT:\n- Readiness: Orange\n- Sport: running",
        "check": "should produce empathetic coaching text",
        "validate": lambda text: not text.strip().startswith("{") and len(text.split()) >= 10,
    },
]


def sanity_check(model_path: str, max_new_tokens: int = 200, mode: str = "both"):
    """Run sanity check prompts through the merged model.

    Args:
        model_path: Path to merged HuggingFace model directory.
        max_new_tokens: Max tokens to generate per prompt.
        mode: "interpreter", "explainer", or "both" (default).
    """

    print(f"Loading model from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Filter prompts by mode
    prompts = [p for p in SANITY_PROMPTS
                if mode == "both" or p["mode"] == mode]

    passed = 0
    failed = 0

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Sanity check {i+1}/{len(prompts)}: [{prompt['mode']}] {prompt['check']}")

        messages = _build_sanity_messages(prompt["mode"], prompt["user_content"])

        # Format with processor's chat template
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=INFERENCE_TEMPERATURE,
                top_p=0.9,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated = processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        ).strip()

        print(f"\n  Raw output:\n  {generated[:500]}")

        # Validate based on mode
        ok = False
        if prompt["mode"] == "interpreter":
            # Strip markdown code fences
            json_text = generated
            if json_text.startswith("```"):
                lines = json_text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                json_text = "\n".join(lines).strip()
            try:
                parsed = json.loads(json_text)
                ok = prompt["validate"](parsed)
                print(f"  Parsed: action={parsed.get('action')}, sport={parsed.get('sport')}")
            except (json.JSONDecodeError, Exception) as e:
                print(f"  Parse error: {e}")
                ok = False
        else:
            # Explainer: validate raw text
            ok = prompt["validate"](generated)

        if ok:
            passed += 1
        else:
            failed += 1
        print(f"\n  [{'PASS' if ok else 'FAIL'}]")

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
        description="MiValta Josi v4 — Gemma 3n E2B QLoRA Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train interpreter (GATCRequest JSON output)
  python finetune_gemma3n.py train --mode interpreter
  python finetune_gemma3n.py train --mode interpreter --lr 3e-5 --epochs 4

  # Train explainer (plain coaching text output)
  python finetune_gemma3n.py train --mode explainer
  python finetune_gemma3n.py train --mode explainer --no-wandb

  # Merge LoRA weights
  python finetune_gemma3n.py merge --lora_path ./models/josi-v4-gemma3n-interpreter-*/lora_weights

  # Sanity check
  python finetune_gemma3n.py sanity --model_path ./models/merged --mode both
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # --- train ---
    tp = subparsers.add_parser("train", help="Run QLoRA fine-tuning")
    tp.add_argument("--mode", type=str, default="interpreter",
                    choices=["interpreter", "explainer"],
                    help="Which model to train (default: interpreter)")
    tp.add_argument("--train_data", type=str, default=None,
                    help="Training data (default: train_interpreter.jsonl or train_explainer.jsonl based on --mode)")
    tp.add_argument("--val_data", type=str, default=None,
                    help="Validation data (default: val_interpreter.jsonl or val_explainer.jsonl based on --mode)")
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

    # --- sanity ---
    sp = subparsers.add_parser("sanity", help="Run sanity check on merged model")
    sp.add_argument("--model_path", type=str, required=True,
                    help="Path to merged model")
    sp.add_argument("--mode", type=str, default="both",
                    choices=["interpreter", "explainer", "both"],
                    help="Which mode to sanity check (default: both)")
    sp.add_argument("--max_tokens", type=int, default=200,
                    help="Max generation tokens")

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
        ok = sanity_check(
            model_path=args.model_path,
            max_new_tokens=args.max_tokens,
            mode=args.mode,
        )
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

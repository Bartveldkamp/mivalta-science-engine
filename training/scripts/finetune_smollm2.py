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

    Each example is split into:
      prompt:     system + user turns + generation marker (everything BEFORE the response)
      completion: assistant response content + EOS token (what the model must learn)

    TRL's SFTTrainer auto-detects prompt/completion format and creates a
    completion_mask, so the model only trains on the completion (including the
    stop token) — not the system prompt or user message.

    Validates that all examples are single-turn (one assistant response) so the
    model learns to produce exactly one response and then stop.
    """
    raw = load_jsonl(path)
    print(f"  Loaded {len(raw)} examples from {path}")

    examples = []
    truncated = 0
    multi_turn = 0
    eos_token = tokenizer.eos_token

    for ex in raw:
        messages = ex["messages"]

        # Validate single-turn: model must learn to output ONE response then stop
        if not validate_single_turn(messages):
            multi_turn += 1
            if multi_turn <= 3:
                roles = [m["role"] for m in messages]
                print(f"  WARNING: Multi-turn example skipped (roles: {roles})")
            continue

        # Split into prompt (system+user) and completion (assistant response)
        # Prompt: apply_chat_template to system+user with generation prompt marker
        prompt_messages = messages[:-1]  # system + user
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,  # adds <|im_start|>assistant\n
        )

        # Completion: assistant content + EOS stop token
        assistant_content = messages[-1]["content"]
        completion = assistant_content + eos_token

        # Truncation check on the full sequence
        full_text = prompt + completion
        tokens = tokenizer(full_text, truncation=True, max_length=max_seq_length)
        if len(tokens["input_ids"]) >= max_seq_length:
            truncated += 1
            # Truncate completion to fit within budget
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

    # Verify EOS token is present in completions
    eos_count = sum(1 for e in examples if eos_token in e["completion"])
    print(f"  EOS token ('{eos_token}') present in {eos_count}/{len(examples)} completions")
    if eos_count < len(examples):
        print(f"  WARNING: {len(examples) - eos_count} completions missing EOS token!")

    # Show a sample
    if examples:
        print(f"  Sample prompt  (last 80 chars): ...{examples[0]['prompt'][-80:]}")
        print(f"  Sample completion (first 80 chars): {examples[0]['completion'][:80]}...")

    ds = Dataset.from_list(examples)
    return ds


# =============================================================================
# MODEL SETUP
# =============================================================================

def load_model_and_tokenizer(model_name: str, lora_r: int, lora_alpha: int):
    """Load SmolLM2 and apply LoRA adapters."""

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # CRITICAL: pad_token must NOT equal eos_token.
    # When pad_token_id == eos_token_id, the DataCollator masks ALL occurrences
    # of that token ID to -100 in labels — including the real EOS at the end of
    # the assistant's response. This means the model NEVER trains on the stop
    # token and won't learn to stop generating at inference time.
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print(f"  Added dedicated [PAD] token (id={tokenizer.pad_token_id}) — separate from EOS (id={tokenizer.eos_token_id})")
    tokenizer.padding_side = "right"

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Resize embeddings if we added a pad token
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Resized embeddings to {len(tokenizer)} (added [PAD] token)")

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
            wandb.init(
                project="mivalta-josi-v3",
                name=f"josi-v3-{model_size}-{datetime.now().strftime('%m%d_%H%M')}",
                config=run_config,
            )
            report_to = "wandb"
            print("W&B tracking enabled")
        except Exception:
            print("W&B not available, logging to console only")
            report_to = "none"

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_name, cfg["lora_r"], cfg["lora_alpha"],
    )

    # Load datasets
    max_seq = cfg["max_seq_length"]
    print(f"\nLoading datasets (max_seq_length={max_seq})...")
    train_dataset = prepare_dataset(train_path, tokenizer, max_seq)
    val_dataset = prepare_dataset(val_path, tokenizer, max_seq)

    # Verify a sample
    sample = train_dataset[0]
    print(f"\n  Sample prompt  (last 100 chars): ...{sample['prompt'][-100:]}")
    print(f"  Sample completion (first 100 chars): {sample['completion'][:100]}...")

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
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to=report_to,
        max_length=max_seq,
        # Completion-only loss: only train on the assistant's response (including
        # the <|im_end|> stop token), not on the system prompt or user message.
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
        dtype=torch.bfloat16,
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

# Full system prompt matching the training data format exactly
_I6_CONSTRAINTS = (
    "I6 CONSTRAINTS (always active):\n"
    "- NEVER prescribe, create, or modify training yourself\n"
    "- Explain decisions made by the GATC engine only\n"
    "- NEVER override readiness gates\n"
    "- NEVER invent zones, durations, paces, or power numbers\n"
    "- NEVER reference GATC internals (algorithm, viterbi, hmm, hidden markov, "
    "acwr, transition matrix, ewma, tss, ctl, atl, tsb)\n"
    "- Prescription/override requests: intent=blocked, guardrail_triggered=true\n"
    "- Medical concerns: intent=medical_red_flag, response_type=SafetyWarning\n\n"
    "OUTPUT: Valid LLMIntent JSON.\n"
    "  intent: question | blocked | replan | encouragement | general | feedback | compliance | medical_red_flag\n"
    "  response_type: QuestionAnswer | ExplainZone | ExplainWorkout | ReadinessSummary | "
    "Decline | Encouragement | SafetyWarning | WeeklyReview | DailyBrief\n"
    "  message: string\n"
    "  source_cards: array of card names\n"
    "  guardrail_triggered: true | false\n"
    "  guardrail_reason: null | \"i6_violation\" | \"tier_violation\" | \"medical_red_flag\"\n"
    "  replan_request: null | {type, reason, mode, readiness_at_request}\n"
    "  tool_call: null | {tool, args}"
)

def _sys(tier, persona_style, mode_rules):
    return (
        f"You are Josi, MiValta's AI coaching assistant. Style: {persona_style}.\n\n"
        f"{mode_rules}\n\n{_I6_CONSTRAINTS}"
    )

_COACH_SYS = _sys("coach", "warm, professional, supportive",
    "MODE: Coach\n"
    "- Reference and explain the athlete's training plan\n"
    "- Trigger replans via replan_request for valid reasons\n"
    "- Create plans via tool_call to create_plan\n"
    "- When session context present: reference planned_session data only, no new numbers\n"
    "- Replan types: skip_today, swap_days, reschedule, reduce_intensity, illness, travel, goal_change")

_ADVISOR_SYS = _sys("advisor", "no-nonsense, factual, brief",
    "MODE: Advisor\n"
    "- Explain workouts and zones, answer education questions\n"
    "- Help create today's workout via tool_call to create_today_workout\n"
    "- NEVER create training plans (Decline with tier upgrade)\n"
    "- NEVER modify or replan training (Decline with tier upgrade)\n"
    "- NEVER use prescriptive language (\"you should do\", \"I recommend\", \"try this\")")

SANITY_PROMPTS = [
    {
        "tier": "coach",
        "messages": [
            {"role": "system", "content": _COACH_SYS},
            {"role": "user", "content": "What am I doing today?\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Session: Z2 60min \"Easy aerobic\" (base phase)\n- Sport: running\n- Level: intermediate"},
        ],
        "expected_intent": "question",
        "expected_response_type": "ExplainWorkout",
        "check": "should explain the Z2 60min session",
    },
    {
        "tier": "coach",
        "messages": [
            {"role": "system", "content": _COACH_SYS},
            {"role": "user", "content": "Give me a harder workout\n\nCONTEXT:\n- Readiness: Green (Recovered)\n- Sport: running\n- Level: intermediate"},
        ],
        "expected_intent": "blocked",
        "expected_response_type": "Decline",
        "check": "should block with guardrail_triggered=true",
    },
    {
        "tier": "advisor",
        "messages": [
            {"role": "system", "content": _ADVISOR_SYS},
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
        dtype=torch.bfloat16,
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
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
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

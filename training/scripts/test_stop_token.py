#!/usr/bin/env python3
"""
Integration test: verify stop token training pipeline with TRL 0.28.

Tests the EXACT same pipeline used in finetune_smollm2.py and finetune_gemma3n.py:
1. prompt/completion dataset format (split at generation marker)
2. TRL's DataCollatorForLanguageModeling with completion_only_loss
3. Verifies completion_mask correctly marks only the completion tokens
4. Verifies labels are -100 for prompt tokens and real values for completion tokens
5. Verifies the EOS/stop token is included in the trained labels (NOT masked)

Runs entirely locally — no model download needed. Uses a minimal tokenizer built
from scratch to simulate the ChatML format used by SmolLM2.
"""

import json
import sys
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from transformers import PreTrainedTokenizerFast
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
import torch

# ─── Build a minimal ChatML tokenizer (no download needed) ──────────────
def build_chatml_tokenizer():
    """Build a minimal tokenizer with ChatML special tokens for testing."""
    # Create a simple word-level tokenizer
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # Train on a small corpus that covers our training data vocabulary
    trainer = trainers.WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "<|im_start|>", "<|im_end|>"],
        min_frequency=1,
    )
    # Training corpus — just enough words to tokenize our test examples
    corpus = [
        "system user assistant You are Josi MiValta AI coaching",
        "intent question blocked encouragement general replan",
        "response_type QuestionAnswer ExplainZone Encouragement Decline",
        "message source_cards guardrail_triggered guardrail_reason",
        "replan_request tool_call null false true",
        "That was tough but I did it Great work today",
        "What does Z2 mean Z2 is your Aerobic zone",
        "Good morning How can I assist with training",
        "readiness Green Recovered Yellow Productive Red Overreached",
        "running cycling intermediate advanced beginner",
        "josi_personas_v1 zone_physiology load_monitoring periodization",
        "CONTEXT Readiness Sport Level Session",
    ]
    tok.train_from_iterator(corpus, trainer=trainer)

    # Wrap in HuggingFace's PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )

    # Set up a ChatML-style chat template
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

    return tokenizer


# ─── Load real training data ──────────────────────────────────────────────
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_single_turn(messages):
    assistant_count = sum(1 for m in messages if m["role"] == "assistant")
    return assistant_count == 1 and messages[-1]["role"] == "assistant"


# ─── Main test ────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  STOP TOKEN PIPELINE TEST (TRL 0.28)")
    print("  Tests: prompt/completion format, completion_mask, label masking")
    print("=" * 70)

    # Build tokenizer
    print("\n[1/4] Building minimal ChatML tokenizer...")
    tokenizer = build_chatml_tokenizer()
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    print(f"  PAD token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"  PAD != EOS: {tokenizer.pad_token_id != tokenizer.eos_token_id}")

    # Verify chat template works
    test_messages = [
        {"role": "system", "content": "You are Josi"},
        {"role": "user", "content": "Hello"},
    ]
    prompt_text = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    print(f"\n  Chat template test (with generation prompt):")
    print(f"    '{prompt_text}'")
    assert "<|im_start|>assistant\n" in prompt_text, "Generation prompt missing!"

    # Load real training data
    print("\n[2/4] Loading real training data...")
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "train_v3.jsonl"
    if not train_path.exists():
        print(f"  ERROR: {train_path} not found. Skipping real data test.")
        return 1

    raw = load_jsonl(str(train_path))
    print(f"  Loaded {len(raw)} examples")

    # Process first 20 examples into prompt/completion format
    print("\n[3/4] Processing 20 examples into prompt/completion format...")
    examples = []
    eos_token = tokenizer.eos_token

    for i, ex in enumerate(raw[:20]):
        messages = ex["messages"]

        if not validate_single_turn(messages):
            print(f"  #{i+1}: SKIP (multi-turn)")
            continue

        # Split: same logic as finetune_smollm2.py
        prompt_messages = messages[:-1]  # system + user
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        completion = messages[-1]["content"] + eos_token

        examples.append({"prompt": prompt, "completion": completion})

    print(f"  Processed {len(examples)} examples")

    # Simulate TRL's tokenize_fn (prompt/completion case, non-conversational)
    print("\n  Simulating TRL tokenize_fn (prompt/completion → input_ids + completion_mask)...")
    tokenized_examples = []
    for i, ex in enumerate(examples):
        prompt_ids = tokenizer(text=ex["prompt"])["input_ids"]
        prompt_completion_ids = tokenizer(text=ex["prompt"] + ex["completion"])["input_ids"]

        # This is exactly what TRL 0.28's tokenize_fn does (line 140)
        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))

        tokenized_examples.append({
            "input_ids": prompt_completion_ids,
            "completion_mask": completion_mask,
        })

    # Apply the DataCollator
    print("\n[4/4] Applying DataCollatorForLanguageModeling (completion_only_loss=True)...")
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
    )

    # Process examples one at a time to check each individually
    all_pass = True
    eos_trained_count = 0
    prompt_masked_count = 0

    for i, tok_ex in enumerate(tokenized_examples):
        batch = collator([tok_ex])
        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        comp_mask = tok_ex["completion_mask"]

        # Find where completion starts
        comp_start = comp_mask.index(1) if 1 in comp_mask else len(comp_mask)
        prompt_len = comp_start
        comp_len = len(comp_mask) - comp_start

        # Check 1: ALL prompt tokens should have labels = -100
        prompt_labels = labels[:prompt_len].tolist()
        prompt_all_masked = all(l == -100 for l in prompt_labels)

        # Check 2: ALL completion tokens should have labels = real token IDs (NOT -100)
        comp_labels = labels[prompt_len:len(comp_mask)].tolist()
        comp_all_trained = all(l != -100 for l in comp_labels)

        # Check 3: The LAST completion token should be the EOS token
        last_comp_label = comp_labels[-1] if comp_labels else None
        last_comp_is_eos = last_comp_label == tokenizer.eos_token_id

        # Check 4: EOS token is NOT masked to -100
        eos_in_labels = tokenizer.eos_token_id in comp_labels

        # Report
        user_msg = examples[i]["prompt"].split("user\n")[-1].split("<|im_end|>")[0][:50]
        status = "PASS" if (prompt_all_masked and comp_all_trained and eos_in_labels) else "FAIL"
        if status == "FAIL":
            all_pass = False

        if prompt_all_masked:
            prompt_masked_count += 1
        if eos_in_labels:
            eos_trained_count += 1

        print(f"\n  Example {i+1}: \"{user_msg}...\"")
        print(f"    Prompt tokens:     {prompt_len} (all masked to -100: {prompt_all_masked})")
        print(f"    Completion tokens: {comp_len} (all have real labels: {comp_all_trained})")
        print(f"    Last comp token is EOS ({tokenizer.eos_token}): {last_comp_is_eos}")
        print(f"    EOS in trained labels: {eos_in_labels}")
        print(f"    [{status}]")

        # Show the actual label values for the completion (for first 3 examples)
        if i < 3:
            comp_tokens = tokenizer.decode(input_ids[prompt_len:len(comp_mask)].tolist())
            print(f"    Completion text: {comp_tokens[:100]}...")
            print(f"    Completion labels ({len(comp_labels)} tokens): [{comp_labels[0]}, ..., {comp_labels[-1]}]")

    # ─── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STOP TOKEN PIPELINE TEST RESULTS")
    print("=" * 70)
    print(f"  Examples tested:       {len(tokenized_examples)}")
    print(f"  Prompt fully masked:   {prompt_masked_count}/{len(tokenized_examples)}")
    print(f"  EOS token trained on:  {eos_trained_count}/{len(tokenized_examples)}")
    print(f"  PAD != EOS:            {tokenizer.pad_token_id != tokenizer.eos_token_id}")
    print()

    if all_pass:
        print("  VERDICT: ALL PASS")
        print("    - Prompt tokens are masked (-100) in labels → model does NOT train on prompts")
        print("    - Completion tokens have real labels → model trains on the response")
        print(f"    - EOS token ({tokenizer.eos_token}) is included in trained labels → model learns to STOP")
        print("    - The stop token fix is working correctly.")
    else:
        print("  VERDICT: SOME FAILURES — see details above")

    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

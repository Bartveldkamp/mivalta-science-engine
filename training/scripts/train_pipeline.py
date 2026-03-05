#!/usr/bin/env python3
"""
MiValta Josi — End-to-End Training Pipeline

Orchestrates the complete training flow:
    1. Parse knowledge cards (source of truth)
    2. Generate grounded training data (Claude distillation)
    3. Merge with existing gold examples
    4. Validate all data against knowledge cards
    5. Split into train/val
    6. Fine-tune Qwen3 (LoRA)
    7. Merge LoRA weights
    8. Export GGUF for on-device deployment

Usage:
    # Full pipeline (generate + validate + train + export)
    python train_pipeline.py full --count 2000

    # Generate + validate only (no training)
    python train_pipeline.py generate --count 2000

    # Validate existing data
    python train_pipeline.py validate

    # Train from existing data
    python train_pipeline.py train --model-size 8b

    # Merge and export
    python train_pipeline.py export --lora-path ./models/josi-v6-*/lora_weights

    # Status: show what data exists
    python train_pipeline.py status
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = SCRIPT_DIR.parent / "data"
GROUNDED_DIR = DATA_DIR / "grounded"
GOLD_DIR = DATA_DIR / "gold_examples"

sys.path.insert(0, str(SCRIPT_DIR))


def show_status():
    """Show current training data status."""
    from knowledge_card_parser import load_all_cards

    cards = load_all_cards()
    print(f"\n{'='*60}")
    print(f"  MIVALTA TRAINING PIPELINE STATUS")
    print(f"{'='*60}")

    # Knowledge cards
    print(f"\n  Knowledge Cards: {len(cards)}")
    for cid, card in cards.items():
        tables = len(card.alg_tables)
        josi = len(card.josi_sections)
        print(f"    {cid}: {tables} tables, {josi} josi sections ({card.axis_owner})")

    # Existing gold examples
    print(f"\n  Gold Examples:")
    total_gold = 0
    if GOLD_DIR.exists():
        for f in sorted(GOLD_DIR.glob("*.jsonl")):
            count = sum(1 for line in open(f) if line.strip())
            total_gold += count
            print(f"    {f.name}: {count}")
    print(f"    TOTAL: {total_gold}")

    # Grounded data
    print(f"\n  Grounded Data:")
    total_grounded = 0
    if GROUNDED_DIR.exists():
        for f in sorted(GROUNDED_DIR.glob("*.jsonl")):
            count = sum(1 for line in open(f) if line.strip())
            total_grounded += count
            print(f"    {f.name}: {count}")
    print(f"    TOTAL: {total_grounded}")

    # Existing training data
    print(f"\n  Training Data (ready):")
    for name in ["train_v6_unified.jsonl", "val_v6_unified.jsonl",
                  "train_interpreter.jsonl", "val_interpreter.jsonl",
                  "train_explainer_sequential.jsonl", "val_explainer_sequential.jsonl"]:
        f = DATA_DIR / name
        if f.exists():
            count = sum(1 for line in open(f) if line.strip())
            print(f"    {name}: {count}")

    print(f"\n  Grand total examples: {total_gold + total_grounded}")


def merge_datasets(output_dir: Path = DATA_DIR):
    """Merge gold examples + grounded data into unified train/val splits."""
    all_interpreter = []
    all_coach = []

    # Load gold examples
    if GOLD_DIR.exists():
        for f in sorted(GOLD_DIR.glob("*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    messages = ex.get("messages", [])
                    if len(messages) < 2:
                        continue

                    # Classify by system prompt content
                    sys_content = messages[0].get("content", "") if messages[0]["role"] == "system" else ""
                    if "interpreter" in sys_content.lower() or "gatcrequest" in sys_content.lower():
                        all_interpreter.append(ex)
                    else:
                        all_coach.append(ex)

    # Load grounded data
    if GROUNDED_DIR.exists():
        for f in sorted(GROUNDED_DIR.glob("grounded_interpreter_*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        all_interpreter.append(json.loads(line))

        for f in sorted(GROUNDED_DIR.glob("grounded_coach_*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        all_coach.append(json.loads(line))

    # Also load existing training data
    for f in [DATA_DIR / "train_interpreter.jsonl", DATA_DIR / "train_explainer_sequential.jsonl"]:
        if f.exists():
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    messages = ex.get("messages", [])
                    if len(messages) < 2:
                        continue
                    sys_content = messages[0].get("content", "") if messages[0]["role"] == "system" else ""
                    if "interpreter" in sys_content.lower():
                        all_interpreter.append(ex)
                    else:
                        all_coach.append(ex)

    print(f"  Interpreter examples: {len(all_interpreter)}")
    print(f"  Coach examples: {len(all_coach)}")

    # Dedup by assistant content hash
    def dedup(examples):
        seen = set()
        unique = []
        for ex in examples:
            messages = ex.get("messages", [])
            asst = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            h = hash(asst)
            if h not in seen:
                seen.add(h)
                unique.append(ex)
        return unique

    all_interpreter = dedup(all_interpreter)
    all_coach = dedup(all_coach)

    print(f"  After dedup — Interpreter: {len(all_interpreter)}, Coach: {len(all_coach)}")

    # Split 90/10 train/val
    rng = random.Random(42)
    rng.shuffle(all_interpreter)
    rng.shuffle(all_coach)

    split_i = int(len(all_interpreter) * 0.9)
    split_c = int(len(all_coach) * 0.9)

    train_interp = all_interpreter[:split_i]
    val_interp = all_interpreter[split_i:]
    train_coach = all_coach[:split_c]
    val_coach = all_coach[split_c:]

    # Unified = interleaved interpreter + coach
    train_unified = train_interp + train_coach
    val_unified = val_interp + val_coach
    rng.shuffle(train_unified)
    rng.shuffle(val_unified)

    # Strip metadata for training (keep only messages)
    def strip_meta(examples):
        return [{"messages": ex["messages"]} for ex in examples]

    # Write outputs
    outputs = {
        "train_v6_unified.jsonl": strip_meta(train_unified),
        "val_v6_unified.jsonl": strip_meta(val_unified),
        "train_interpreter.jsonl": strip_meta(train_interp),
        "val_interpreter.jsonl": strip_meta(val_interp),
        "train_explainer_sequential.jsonl": strip_meta(train_coach),
        "val_explainer_sequential.jsonl": strip_meta(val_coach),
    }

    for name, data in outputs.items():
        path = output_dir / name
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Written: {name} ({len(data)} examples)")

    print(f"\n  Total train: {len(train_unified)}")
    print(f"  Total val: {len(val_unified)}")

    return train_unified, val_unified


def run_validate():
    """Validate all training data against knowledge cards."""
    from knowledge_card_parser import load_all_cards
    from validate_grounding import validate_jsonl_file

    cards = load_all_cards()
    print(f"Loaded {len(cards)} knowledge cards for validation\n")

    files_to_validate = []
    if GOLD_DIR.exists():
        files_to_validate.extend(sorted(GOLD_DIR.glob("*.jsonl")))
    if GROUNDED_DIR.exists():
        files_to_validate.extend(sorted(GROUNDED_DIR.glob("*.jsonl")))
    for name in ["train_v6_unified.jsonl", "val_v6_unified.jsonl"]:
        f = DATA_DIR / name
        if f.exists():
            files_to_validate.append(f)

    total_clean = 0
    total_total = 0

    for filepath in files_to_validate:
        stats = validate_jsonl_file(filepath)
        if stats["total"] == 0:
            continue

        total_clean += stats["clean"]
        total_total += stats["total"]

        status = "PASS" if stats["with_violations"] == 0 else "WARN" if stats["errors"] == 0 else "FAIL"
        print(f"  [{status}] {filepath.name}: "
              f"{stats['clean']}/{stats['total']} clean  "
              f"score={stats['avg_grounding_score']}  "
              f"errors={stats['errors']}")

    if total_total:
        pct = 100 * total_clean / total_total
        print(f"\n  Overall: {total_clean}/{total_total} clean ({pct:.0f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="MiValta Josi — End-to-End Training Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    # Status
    subparsers.add_parser("status", help="Show pipeline status")

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate grounded training data")
    gen_parser.add_argument("--count", type=int, default=1000)
    gen_parser.add_argument("--axis", type=str, default=None)
    gen_parser.add_argument("--seed", type=int, default=42)
    gen_parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    gen_parser.add_argument("--concurrency", type=int, default=5)

    # Validate
    subparsers.add_parser("validate", help="Validate all training data")

    # Merge
    subparsers.add_parser("merge", help="Merge all data into unified train/val")

    # Train
    train_parser = subparsers.add_parser("train", help="Fine-tune Qwen3")
    train_parser.add_argument("--model-size", type=str, default="8b", choices=["8b", "4b"])
    train_parser.add_argument("--mode", type=str, default="unified",
                              choices=["unified", "interpreter", "coach"])
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--lr", type=float, default=None)

    # Export
    export_parser = subparsers.add_parser("export", help="Merge LoRA + export GGUF")
    export_parser.add_argument("--lora-path", type=str, required=True)

    # Full pipeline
    full_parser = subparsers.add_parser("full", help="Full pipeline: generate + validate + merge + train")
    full_parser.add_argument("--count", type=int, default=2000)
    full_parser.add_argument("--model-size", type=str, default="8b", choices=["8b", "4b"])
    full_parser.add_argument("--axis", type=str, default=None)
    full_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "status":
        show_status()

    elif args.command == "generate":
        import asyncio
        sys.argv = [
            "generate_grounded_dataset.py", "--run",
            "--count", str(args.count),
            "--seed", str(args.seed),
            "--model", args.model,
            "--concurrency", str(args.concurrency),
        ]
        if args.axis:
            sys.argv.extend(["--axis", args.axis])
        from generate_grounded_dataset import run_generation
        asyncio.run(run_generation(args))

    elif args.command == "validate":
        run_validate()

    elif args.command == "merge":
        merge_datasets()

    elif args.command == "train":
        # Merge first
        print("Step 1: Merging datasets...")
        merge_datasets()

        # Then train
        print(f"\nStep 2: Training Qwen3-{args.model_size.upper()}...")
        train_args = [
            "finetune_qwen3.py", "train",
            "--mode", args.mode,
            "--model-size", args.model_size,
        ]
        if args.epochs:
            train_args.extend(["--epochs", str(args.epochs)])
        if args.lr:
            train_args.extend(["--lr", str(args.lr)])
        os.execvp(sys.executable, [sys.executable, str(SCRIPT_DIR / "finetune_qwen3.py")] + train_args[1:])

    elif args.command == "export":
        print(f"Merging LoRA from {args.lora_path}...")
        os.execvp(sys.executable, [
            sys.executable, str(SCRIPT_DIR / "finetune_qwen3.py"),
            "merge", "--lora_path", args.lora_path,
        ])

    elif args.command == "full":
        print(f"\n{'='*60}")
        print(f"  MIVALTA FULL TRAINING PIPELINE")
        print(f"{'='*60}")

        # Step 1: Status
        print("\n--- Step 1: Current Status ---")
        show_status()

        # Step 2: Generate (requires API key)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            print(f"\n--- Step 2: Generate {args.count} grounded examples ---")
            import asyncio
            from generate_grounded_dataset import run_generation, generate_all_scenarios, load_all_cards

            class GenArgs:
                count = args.count
                axis = args.axis
                seed = args.seed
                model = "claude-sonnet-4-20250514"
                concurrency = 5

            asyncio.run(run_generation(GenArgs()))
        else:
            print("\n--- Step 2: SKIP generation (no ANTHROPIC_API_KEY) ---")
            print("  Using existing data only")

        # Step 3: Validate
        print("\n--- Step 3: Validate ---")
        run_validate()

        # Step 4: Merge
        print("\n--- Step 4: Merge datasets ---")
        merge_datasets()

        # Step 5: Train
        print(f"\n--- Step 5: Train Qwen3-{args.model_size.upper()} ---")
        print("  Run manually:")
        print(f"  python finetune_qwen3.py train --mode unified --model-size {args.model_size}")

    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python train_pipeline.py status")
        print("  python train_pipeline.py generate --count 2000")
        print("  python train_pipeline.py validate")
        print("  python train_pipeline.py merge")
        print("  python train_pipeline.py train --model-size 8b")
        print("  python train_pipeline.py full --count 2000")


if __name__ == "__main__":
    main()

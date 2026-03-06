#!/usr/bin/env python3
"""
MiValta Josi — Conversational Training Data Format (.conv)

Human-readable format for writing, reading, and maintaining training examples.
Converts to/from JSONL for the training pipeline.

Format spec:
    # domain: onboarding
    # tags: beginner, first-session
    # system: coach          (optional: "coach" or "interpreter", default "coach")

    USER:
    Hi, I just signed up. Not really sure how this works.

    JOSI:
    Welcome! I'm Josi, your personal training coach.

    ---

    USER:
    Another conversation starts here.

    JOSI:
    And the response goes here.

Rules:
    - Lines starting with # at the top are metadata (before the first USER:)
    - USER: and JOSI: mark speaker turns (must be alone on a line)
    - --- separates independent conversations (single-turn or multi-turn)
    - Blank lines within a turn are preserved
    - Multi-turn conversations are supported (USER/JOSI/USER/JOSI/...)
    - CONTEXT:, HISTORY:, MEMORY:, INTERPRETER: blocks can appear in USER turns

Usage:
    # Convert .conv to JSONL
    python conv_format.py to-jsonl gold_onboarding.conv -o gold_onboarding.jsonl

    # Convert JSONL to .conv
    python conv_format.py to-conv gold_onboarding.jsonl -o gold_onboarding.conv

    # Validate .conv files
    python conv_format.py validate gold_onboarding.conv

    # Convert all .conv files in a directory
    python conv_format.py to-jsonl training/data/conversations/ -o training/data/gold_examples/

    # Stats
    python conv_format.py stats training/data/conversations/
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConvMetadata:
    domain: str = ""
    tags: list[str] = field(default_factory=list)
    system: str = "coach"
    extras: dict[str, str] = field(default_factory=dict)


@dataclass
class Turn:
    role: str          # "user" or "assistant"
    content: str


@dataclass
class Conversation:
    turns: list[Turn]
    metadata: ConvMetadata
    line_number: int = 0  # where this conversation starts in the file


# ---------------------------------------------------------------------------
# Parser: .conv → Conversation objects
# ---------------------------------------------------------------------------

def parse_conv(text: str, filename: str = "<input>") -> list[Conversation]:
    """Parse a .conv file into a list of Conversations."""
    lines = text.split("\n")
    metadata = _parse_metadata(lines)

    conversations: list[Conversation] = []
    current_turns: list[Turn] = []
    current_role: Optional[str] = None
    current_lines: list[str] = []
    conv_start_line = 0

    # Find where metadata ends (first non-comment, non-blank line)
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            content_start = i
            break

    for i in range(content_start, len(lines)):
        line = lines[i]
        stripped = line.strip()

        # Separator: finalize current conversation
        if stripped == "---":
            if current_role and current_lines:
                current_turns.append(_make_turn(current_role, current_lines))
            if current_turns:
                conversations.append(Conversation(
                    turns=current_turns,
                    metadata=metadata,
                    line_number=conv_start_line + 1,
                ))
            current_turns = []
            current_role = None
            current_lines = []
            conv_start_line = i + 1
            continue

        # Speaker marker
        if stripped == "USER:":
            if current_role and current_lines:
                current_turns.append(_make_turn(current_role, current_lines))
            current_role = "user"
            current_lines = []
            if not current_turns and not conversations:
                conv_start_line = i
            elif not current_turns:
                conv_start_line = i
            continue

        if stripped == "JOSI:":
            if current_role and current_lines:
                current_turns.append(_make_turn(current_role, current_lines))
            current_role = "assistant"
            current_lines = []
            continue

        # Content line (belongs to current speaker)
        if current_role is not None:
            current_lines.append(line)

    # Finalize last conversation
    if current_role and current_lines:
        current_turns.append(_make_turn(current_role, current_lines))
    if current_turns:
        conversations.append(Conversation(
            turns=current_turns,
            metadata=metadata,
            line_number=conv_start_line + 1,
        ))

    return conversations


def _parse_metadata(lines: list[str]) -> ConvMetadata:
    """Extract metadata from # comment lines at the top of the file."""
    meta = ConvMetadata()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        # Parse "# key: value"
        m = re.match(r"^#\s*(\w+)\s*:\s*(.+)$", stripped)
        if m:
            key, value = m.group(1).lower(), m.group(2).strip()
            if key == "domain":
                meta.domain = value
            elif key == "tags":
                meta.tags = [t.strip() for t in value.split(",") if t.strip()]
            elif key == "system":
                meta.system = value
            else:
                meta.extras[key] = value
    return meta


def _make_turn(role: str, lines: list[str]) -> Turn:
    """Create a Turn from accumulated content lines."""
    # Strip leading/trailing blank lines, preserve internal ones
    content = "\n".join(lines)
    content = content.strip()
    return Turn(role=role, content=content)


# ---------------------------------------------------------------------------
# Serializer: Conversation objects → .conv text
# ---------------------------------------------------------------------------

def to_conv_text(conversations: list[Conversation]) -> str:
    """Serialize Conversations back to .conv format."""
    if not conversations:
        return ""

    parts = []
    meta = conversations[0].metadata

    # Write metadata
    if meta.domain:
        parts.append(f"# domain: {meta.domain}")
    if meta.tags:
        parts.append(f"# tags: {', '.join(meta.tags)}")
    if meta.system and meta.system != "coach":
        parts.append(f"# system: {meta.system}")
    for k, v in meta.extras.items():
        parts.append(f"# {k}: {v}")

    if parts:
        parts.append("")  # blank line after metadata

    conv_texts = []
    for conv in conversations:
        turn_texts = []
        for turn in conv.turns:
            speaker = "USER:" if turn.role == "user" else "JOSI:"
            turn_texts.append(f"{speaker}\n{turn.content}")
        conv_texts.append("\n\n".join(turn_texts))

    parts.append("\n\n---\n\n".join(conv_texts))
    parts.append("")  # trailing newline

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSONL conversion
# ---------------------------------------------------------------------------

def conv_to_jsonl_rows(conversations: list[Conversation]) -> list[dict]:
    """Convert Conversations to JSONL rows (messages format)."""
    rows = []
    for conv in conversations:
        messages = []
        for turn in conv.turns:
            messages.append({"role": turn.role, "content": turn.content})
        row = {"messages": messages}
        # Add metadata as top-level fields for filtering/sorting
        if conv.metadata.domain:
            row["domain"] = conv.metadata.domain
        if conv.metadata.tags:
            row["tags"] = conv.metadata.tags
        rows.append(row)
    return rows


def jsonl_to_conversations(rows: list[dict], domain: str = "",
                            tags: list[str] | None = None) -> list[Conversation]:
    """Convert JSONL rows back to Conversations."""
    meta = ConvMetadata(domain=domain, tags=tags or [])
    conversations = []
    for row in rows:
        messages = row.get("messages", [])
        # Skip system messages — those are injected by the pipeline
        turns = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                continue
            turns.append(Turn(role=role, content=msg["content"]))
        if turns:
            # Use row-level domain/tags if present
            row_meta = ConvMetadata(
                domain=row.get("domain", meta.domain),
                tags=row.get("tags", meta.tags),
            )
            conversations.append(Conversation(turns=turns, metadata=row_meta))
    return conversations


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def load_conv(path: Path) -> list[Conversation]:
    """Load a .conv file."""
    return parse_conv(path.read_text(encoding="utf-8"), filename=str(path))


def save_conv(conversations: list[Conversation], path: Path):
    """Save conversations to a .conv file."""
    path.write_text(to_conv_text(conversations), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path):
    """Save rows to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_conv(conversations: list[Conversation], filename: str = "") -> list[str]:
    """Validate conversations and return list of issues."""
    issues = []
    prefix = f"{filename}: " if filename else ""

    if not conversations:
        issues.append(f"{prefix}No conversations found")
        return issues

    for i, conv in enumerate(conversations, 1):
        loc = f"{prefix}conversation {i} (line {conv.line_number})"

        if not conv.turns:
            issues.append(f"{loc}: empty conversation")
            continue

        if conv.turns[0].role != "user":
            issues.append(f"{loc}: must start with USER turn")

        if conv.turns[-1].role != "assistant":
            issues.append(f"{loc}: must end with JOSI turn")

        for j, turn in enumerate(conv.turns):
            if not turn.content.strip():
                speaker = "USER" if turn.role == "user" else "JOSI"
                issues.append(f"{loc}: empty {speaker} turn at position {j + 1}")

        # Check alternating roles
        for j in range(1, len(conv.turns)):
            if conv.turns[j].role == conv.turns[j - 1].role:
                issues.append(f"{loc}: consecutive same-role turns at position {j + 1}")

    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_to_jsonl(args):
    """Convert .conv file(s) to JSONL."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if input_path.is_dir():
        conv_files = sorted(input_path.glob("*.conv"))
        if not conv_files:
            print(f"No .conv files found in {input_path}")
            return 1

        out_dir = output_path or input_path
        out_dir.mkdir(parents=True, exist_ok=True)

        total = 0
        for f in conv_files:
            convs = load_conv(f)
            issues = validate_conv(convs, f.name)
            if issues:
                for issue in issues:
                    print(f"  WARNING: {issue}", file=sys.stderr)

            rows = conv_to_jsonl_rows(convs)
            out_file = out_dir / f"{f.stem}.jsonl"
            save_jsonl(rows, out_file)
            total += len(rows)
            print(f"  {f.name} → {out_file.name} ({len(rows)} examples)")

        print(f"\nConverted {len(conv_files)} files, {total} total examples")
    else:
        convs = load_conv(input_path)
        issues = validate_conv(convs, input_path.name)
        if issues:
            for issue in issues:
                print(f"  WARNING: {issue}", file=sys.stderr)

        rows = conv_to_jsonl_rows(convs)
        out_file = output_path or input_path.with_suffix(".jsonl")
        save_jsonl(rows, out_file)
        print(f"Converted {len(rows)} conversations → {out_file}")

    return 0


def cmd_to_conv(args):
    """Convert JSONL file(s) to .conv format."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if input_path.is_dir():
        jsonl_files = sorted(input_path.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No .jsonl files found in {input_path}")
            return 1

        out_dir = output_path or input_path
        out_dir.mkdir(parents=True, exist_ok=True)

        total = 0
        for f in jsonl_files:
            rows = load_jsonl(f)
            # Derive domain from filename (gold_onboarding.jsonl → onboarding)
            domain = f.stem.replace("gold_", "")
            convs = jsonl_to_conversations(rows, domain=domain)
            out_file = out_dir / f"{f.stem.replace('gold_', '')}.conv"
            save_conv(convs, out_file)
            total += len(convs)
            print(f"  {f.name} → {out_file.name} ({len(convs)} conversations)")

        print(f"\nConverted {len(jsonl_files)} files, {total} total conversations")
    else:
        rows = load_jsonl(input_path)
        domain = input_path.stem.replace("gold_", "")
        convs = jsonl_to_conversations(rows, domain=domain)
        out_file = output_path or input_path.with_suffix(".conv")
        save_conv(convs, out_file)
        print(f"Converted {len(convs)} conversations → {out_file}")

    return 0


def cmd_validate(args):
    """Validate .conv file(s)."""
    input_path = Path(args.input)
    files = sorted(input_path.glob("*.conv")) if input_path.is_dir() else [input_path]

    all_issues = []
    total_convs = 0
    for f in files:
        convs = load_conv(f)
        total_convs += len(convs)
        issues = validate_conv(convs, f.name)
        all_issues.extend(issues)

    if all_issues:
        print(f"Found {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"  ✗ {issue}")
        return 1
    else:
        print(f"All valid: {total_convs} conversations in {len(files)} file(s)")
        return 0


def cmd_stats(args):
    """Show statistics for .conv file(s)."""
    input_path = Path(args.input)
    files = sorted(input_path.glob("*.conv")) if input_path.is_dir() else [input_path]

    total_convs = 0
    total_turns = 0
    total_words = 0

    for f in files:
        convs = load_conv(f)
        n_convs = len(convs)
        n_turns = sum(len(c.turns) for c in convs)
        n_words = sum(
            len(t.content.split())
            for c in convs
            for t in c.turns
        )
        total_convs += n_convs
        total_turns += n_turns
        total_words += n_words

        domain = convs[0].metadata.domain if convs else "?"
        print(f"  {f.name:40s}  {n_convs:4d} convs  {n_turns:5d} turns  {n_words:6d} words  [{domain}]")

    print(f"\n  {'TOTAL':40s}  {total_convs:4d} convs  {total_turns:5d} turns  {total_words:6d} words")


def main():
    parser = argparse.ArgumentParser(
        description="MiValta conversational training data format (.conv)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_to_jsonl = sub.add_parser("to-jsonl", help="Convert .conv → JSONL")
    p_to_jsonl.add_argument("input", help=".conv file or directory of .conv files")
    p_to_jsonl.add_argument("-o", "--output", help="Output file or directory")
    p_to_jsonl.set_defaults(func=cmd_to_jsonl)

    p_to_conv = sub.add_parser("to-conv", help="Convert JSONL → .conv")
    p_to_conv.add_argument("input", help="JSONL file or directory of JSONL files")
    p_to_conv.add_argument("-o", "--output", help="Output file or directory")
    p_to_conv.set_defaults(func=cmd_to_conv)

    p_val = sub.add_parser("validate", help="Validate .conv file(s)")
    p_val.add_argument("input", help=".conv file or directory")
    p_val.set_defaults(func=cmd_validate)

    p_stats = sub.add_parser("stats", help="Show statistics")
    p_stats.add_argument("input", help=".conv file or directory")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()

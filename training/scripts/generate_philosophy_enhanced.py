#!/usr/bin/env python3
"""
MiValta Philosophy Enhanced Dataset Generator

Generates Q&A pairs from the compiled JOSI coaching contexts.
Uses the actual rewritten coaching cards from knowledge/generated/context.py.

Output format (correct):
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

NOT the old format:
{"source": "...", "user": "...", "assistant": "..."}
"""

import json
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# Add knowledge package to path
SCRIPT_DIR = Path(__file__).parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from knowledge.generated.context import Context

# Output directory
OUTPUT_DIR = SCRIPT_DIR.parent / "data"


@dataclass
class PhilosophyPair:
    """Single training example for philosophy dataset."""
    user_content: str
    assistant_content: str
    context_key: str


# =============================================================================
# QUESTION TEMPLATES BY TOPIC
# =============================================================================

# Map context keys to relevant question templates
TOPIC_QUESTIONS = {
    # Cycling topics
    "cycling": [
        "How does power training work for cycling?",
        "What is FTP and why does it matter?",
        "Tell me about cycling training zones",
        "How do power zones work?",
        "Why is cycling different from running?",
        "Explain cycling power zones",
        "What's special about indoor cycling?",
        "How should I train on the bike?",
    ],
    # Recovery topics
    "recovery": [
        "Why is recovery important?",
        "How does recovery actually work?",
        "When can I train after a hard session?",
        "What's the two-lane recovery model?",
        "How do I know if I'm recovered?",
        "Why can't I just train harder?",
        "What affects my recovery needs?",
        "How often should I have recovery weeks?",
        "Is rest really part of training?",
    ],
    # Running topics
    "running": [
        "How should I structure my running?",
        "What's the right approach for running training?",
        "Why are long runs important?",
        "Tell me about strides",
        "How fast should I run?",
        "What are running pace zones?",
        "Why is most running supposed to be easy?",
    ],
    # Zone and intensity topics
    "zone": [
        "What are training zones?",
        "How do I know what zone I'm in?",
        "Explain the different training zones",
        "What does each zone do?",
        "How should I distribute my training across zones?",
        "What's the talk test?",
        "Why are easy zones important?",
    ],
    "ntiz": [
        "What is time in zone?",
        "How do you measure training dose?",
        "What are the zone limits?",
        "How much of each zone should I do?",
    ],
    # Session structure topics
    "session": [
        "How are workouts structured?",
        "What makes a good workout structure?",
        "Why do interval lengths matter?",
        "How long should intervals be?",
        "What's the right rest between intervals?",
    ],
    # Periodization topics
    "periodization": [
        "What is periodization?",
        "How do training phases work?",
        "What's the taper?",
        "How do mesocycles work?",
        "Why do we have training phases?",
    ],
    # Performance topics
    "performance": [
        "How fast can I improve?",
        "What's realistic for improvement?",
        "How does age affect training?",
        "How often should I train?",
        "What determines my improvement rate?",
    ],
    # Goal topics
    "goal": [
        "How do goals affect training?",
        "What about weight loss?",
        "How do I choose the right goal?",
    ],
    # Threshold topics
    "threshold": [
        "What is threshold?",
        "How do I find my threshold?",
        "Why should I test my threshold?",
        "How do I test my threshold?",
    ],
    # Micro training topics
    "micro": [
        "Do short workouts count?",
        "Can my commute be training?",
        "What are exercise snacks?",
        "How do micro workouts work?",
    ],
    # Persona topics
    "persona": [
        "What coaching styles are available?",
        "How does the coach communicate?",
        "What are the different personas?",
    ],
    # Energy zones
    "energy": [
        "How do energy zones work?",
        "What's the difference between zones?",
        "How should warmup and cooldown work?",
        "What does each zone feel like?",
    ],
    # Anchoring
    "anchoring": [
        "How are zones calculated?",
        "What determines my zones?",
        "How do heart rate zones work?",
        "How do power zones get set?",
    ],
}

# Generic questions that work for any context
GENERIC_QUESTIONS = [
    "Can you explain this to me?",
    "Tell me more about this",
    "How does this work?",
    "Why is this important?",
    "What should I know about this?",
    "Help me understand this",
    "What's the key insight here?",
    "Break this down for me",
]


def get_topic_from_key(context_key: str) -> str:
    """Extract topic from context key."""
    key_lower = context_key.lower()

    if "cycling" in key_lower:
        return "cycling"
    elif "recovery" in key_lower:
        return "recovery"
    elif "running" in key_lower:
        return "running"
    elif "ntiz" in key_lower:
        return "ntiz"
    elif "zone" in key_lower and "energy" not in key_lower:
        return "zone"
    elif "session" in key_lower:
        return "session"
    elif "periodization" in key_lower:
        return "periodization"
    elif "performance" in key_lower:
        return "performance"
    elif "goal" in key_lower:
        return "goal"
    elif "threshold" in key_lower:
        return "threshold"
    elif "micro" in key_lower or "commute" in key_lower:
        return "micro"
    elif "persona" in key_lower or "josi" in key_lower:
        return "persona"
    elif "energy" in key_lower:
        return "energy"
    elif "anchoring" in key_lower:
        return "anchoring"
    else:
        return "general"


def get_questions_for_context(context_key: str) -> List[str]:
    """Get relevant questions for a context."""
    topic = get_topic_from_key(context_key)

    topic_qs = TOPIC_QUESTIONS.get(topic, [])

    # Combine topic-specific questions with generic ones
    all_questions = topic_qs + GENERIC_QUESTIONS

    return all_questions


def create_contextual_question(context_key: str, content: str) -> str:
    """Create a contextual question based on content analysis."""
    content_lower = content.lower()

    # Analyze content to create specific questions
    if "here's why" in content_lower or "let me explain" in content_lower:
        return random.choice([
            "Why does this matter?",
            "Can you explain the reasoning?",
            "What's the logic behind this?",
        ])

    if "the key insight" in content_lower:
        return random.choice([
            "What's the key takeaway?",
            "What should I remember?",
            "What's most important here?",
        ])

    if "good question" in content_lower:
        return random.choice([
            "I have a question about this",
            "Can you clarify something?",
            "Help me understand",
        ])

    if "research" in content_lower or "science" in content_lower:
        return random.choice([
            "What does the research say?",
            "Is this backed by science?",
            "What's the evidence for this?",
        ])

    # Default to topic-based questions
    return random.choice(get_questions_for_context(context_key))


def generate_pairs_from_context(
    context_key: str,
    content: str,
    num_variations: int = 3
) -> List[PhilosophyPair]:
    """Generate multiple Q&A pairs from a single context."""

    pairs = []
    questions = get_questions_for_context(context_key)

    # Ensure we have enough questions
    while len(questions) < num_variations:
        questions.extend(GENERIC_QUESTIONS)

    # Sample questions without replacement
    selected_questions = random.sample(questions, min(num_variations, len(questions)))

    for question in selected_questions:
        pairs.append(PhilosophyPair(
            user_content=question,
            assistant_content=content.strip(),
            context_key=context_key,
        ))

    # Add one contextual question based on content analysis
    contextual_q = create_contextual_question(context_key, content)
    pairs.append(PhilosophyPair(
        user_content=contextual_q,
        assistant_content=content.strip(),
        context_key=context_key,
    ))

    return pairs


def generate_philosophy_dataset() -> List[PhilosophyPair]:
    """Generate philosophy training dataset from JOSI contexts."""

    pairs = []

    print("Loading JOSI contexts from knowledge/generated/context.py...")
    ctx = Context.load()
    context_keys = ctx.list_contexts()

    print(f"Found {len(context_keys)} JOSI contexts")

    # Generate Q&A pairs from each context
    for context_key in context_keys:
        content = ctx._contexts.get(context_key, "")

        if not content or len(content) < 50:
            continue

        # Generate 3-4 variations per context
        num_variations = random.randint(3, 4)
        context_pairs = generate_pairs_from_context(
            context_key, content, num_variations
        )
        pairs.extend(context_pairs)

    print(f"Generated {len(pairs)} Q&A pairs from {len(context_keys)} contexts")
    return pairs


def save_chat_format(pairs: List[PhilosophyPair], output_path: Path):
    """
    Save in correct chat format for fine-tuning.

    CORRECT FORMAT:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    NOT the old wrong format:
    {"source": "...", "user": "...", "assistant": "..."}
    """

    with open(output_path, 'w') as f:
        for pair in pairs:
            # Correct chat format with messages array
            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": pair.user_content
                    },
                    {
                        "role": "assistant",
                        "content": pair.assistant_content
                    }
                ]
            }
            f.write(json.dumps(example) + '\n')

    print(f"Saved {len(pairs)} examples to {output_path}")


def main():
    """Generate and save philosophy training dataset."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MiValta Philosophy Enhanced Dataset Generator")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")

    pairs = generate_philosophy_dataset()

    if len(pairs) < 300:
        print(f"WARNING: Only {len(pairs)} pairs generated, target is 300+")

    # Shuffle for training
    random.shuffle(pairs)

    # Split train/val (90/10)
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Save in correct chat format
    save_chat_format(train_pairs, OUTPUT_DIR / "philosophy_train.jsonl")
    save_chat_format(val_pairs, OUTPUT_DIR / "philosophy_val.jsonl")

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"  Total examples: {len(pairs)}")
    print(f"  Training: {len(train_pairs)}")
    print(f"  Validation: {len(val_pairs)}")

    # Topic breakdown
    topic_counts = {}
    for p in pairs:
        topic = get_topic_from_key(p.context_key)
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    print(f"\n  Topic breakdown:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"    {topic}: {count}")

    # Show sample output format
    print("\n" + "=" * 60)
    print("Sample output format (correct):")
    print("=" * 60)
    if pairs:
        sample = {
            "messages": [
                {"role": "user", "content": pairs[0].user_content},
                {"role": "assistant", "content": pairs[0].assistant_content[:200] + "..."}
            ]
        }
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()

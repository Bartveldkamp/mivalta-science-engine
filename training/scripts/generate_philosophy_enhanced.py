#!/usr/bin/env python3
"""
MiValta Philosophy Enhanced Dataset Generator

Generates philosophical conversation pairs for fine-tuning.
Follows the standard chat format with messages array.

Output format (correct):
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

NOT the old format:
{"source": "...", "user": "...", "assistant": "..."}
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "data"


@dataclass
class PhilosophyPair:
    """Single training example for philosophy dataset."""
    user_content: str
    assistant_content: str
    topic: str = "general"
    source: str = "philosophy_enhanced"


# =============================================================================
# PHILOSOPHY CONTENT
# =============================================================================

TRAINING_PHILOSOPHY = {
    "consistency": [
        {
            "user": "Why is consistency more important than intensity?",
            "assistant": "Consistency builds the foundation that intensity can build upon. Regular training creates adaptations that compound over time - neural pathways strengthen, mitochondria multiply, and your body learns to efficiently manage stress. Sporadic intense efforts, while feeling productive, often lead to injury or burnout. Think of it like compound interest: small, regular deposits grow exponentially over time."
        },
        {
            "user": "I missed a week of training. Should I make up for lost time?",
            "assistant": "No, don't try to 'make up' missed training. Your body doesn't work that way. Cramming extra volume creates injury risk and fatigue without proportional benefit. Instead, simply resume your normal schedule. A week of rest rarely causes significant fitness loss, and your body may actually benefit from the unexpected recovery. Focus forward, not backward."
        },
        {
            "user": "How do I stay consistent when motivation is low?",
            "assistant": "Don't rely on motivation - it's unreliable. Build systems instead: set a regular schedule, prepare your gear the night before, find a training partner, or commit to just starting (the hardest part). Remember that showing up matters more than the quality of any single session. On low days, do something - even 15 minutes maintains the habit."
        },
    ],
    "recovery": [
        {
            "user": "Is recovery really part of training?",
            "assistant": "Recovery isn't just part of training - it's when training actually works. Workouts create stress and break down tissue. Recovery is when your body rebuilds stronger. Without adequate recovery, you accumulate fatigue without adaptation. The best athletes are often the best recoverers. Sleep, nutrition, and rest days are as important as the workouts themselves."
        },
        {
            "user": "Why can't I just train harder instead of resting?",
            "assistant": "Training harder without recovery is like trying to build a wall while someone keeps taking bricks away. Your body needs time to adapt to the stress you've applied. More stress without recovery leads to overtraining, decreased performance, increased injury risk, and potential long-term health issues. Hard training earns its value through subsequent recovery."
        },
        {
            "user": "What happens if I skip rest days?",
            "assistant": "Skipping rest days occasionally isn't catastrophic, but making it a habit leads to accumulated fatigue, stagnant performance, increased injury risk, and potentially overtraining syndrome. Your nervous system needs recovery as much as your muscles. Signs you're not recovering enough: persistent fatigue, declining performance, irritability, poor sleep, and increased resting heart rate."
        },
    ],
    "progression": [
        {
            "user": "How fast should I increase my training?",
            "assistant": "The general guideline is no more than 10% increase in volume per week, but this varies by individual and training age. Beginners can often progress faster; experienced athletes need more patience. Watch your body's response: persistent fatigue, declining performance, or nagging pain means you're progressing too fast. Sustainable progress beats rapid burnout."
        },
        {
            "user": "Why does progress seem to slow down over time?",
            "assistant": "This is the law of diminishing returns in action. Early gains come easily because you're far from your genetic potential. As you improve, each additional gain requires more specific, harder work. This isn't a problem - it's physiology. Embrace the process, celebrate smaller improvements, and remember that maintenance itself is an achievement at higher levels."
        },
        {
            "user": "I feel like I've plateaued. What should I do?",
            "assistant": "Plateaus are normal and often signal your body needs something different. Consider: varying your training stimulus, addressing a weakness you've been ignoring, taking a recovery week, or focusing on a different aspect of fitness. Sometimes the answer is patience - adaptations don't follow linear timelines. Analyze your sleep, nutrition, and stress as well."
        },
    ],
    "mindset": [
        {
            "user": "How do I deal with comparing myself to others?",
            "assistant": "Comparison can motivate or destroy, depending on how you use it. Others have different genetics, backgrounds, time availability, and goals. The only fair comparison is to your past self. Use others for inspiration and learning, not measurement. Your journey is unique, and progress relative to your starting point is what matters."
        },
        {
            "user": "How do I stay patient with long-term goals?",
            "assistant": "Break long-term goals into shorter milestones you can celebrate along the way. Focus on process goals (consistent training, proper nutrition) rather than outcome goals (race times). Trust that daily deposits compound. Keep a training log to see progress that's hard to notice day-to-day. Remember: the time will pass anyway - you might as well spend it building."
        },
        {
            "user": "What's the right mindset for hard training days?",
            "assistant": "Approach hard days with purpose, not dread. Know why the session matters and what adaptation you're seeking. Accept discomfort as temporary and purposeful. Stay present - focus on the current interval, not the ones to come. Afterward, recognize the accomplishment. Hard days build physical and mental resilience that easy days can't provide."
        },
    ],
    "balance": [
        {
            "user": "How do I balance training with life?",
            "assistant": "Training should enhance life, not consume it. Be realistic about your time and energy. Quality often beats quantity - focused shorter sessions can be more effective than exhausted longer ones. Communicate with family and friends. Build training around your life priorities, and be willing to adjust when life demands it. Sustainable balance beats temporary extremes."
        },
        {
            "user": "Is it okay to take extended time off?",
            "assistant": "Absolutely. Life has seasons, and training should adapt to them. Extended breaks for travel, family, work demands, or mental health are not just okay - they're sometimes necessary. Fitness is remarkably resilient; you'll lose less than you fear and regain it faster than expected. A break taken intentionally often leads to renewed motivation and better long-term consistency."
        },
        {
            "user": "How do I know if I'm overtraining?",
            "assistant": "Warning signs include: persistent fatigue that doesn't improve with rest, declining performance despite training, elevated resting heart rate, disturbed sleep, mood changes, loss of motivation, frequent illness, and nagging injuries. If you suspect overtraining, the answer is always more rest, not more training. It's easier to prevent than to recover from."
        },
    ],
    "adaptation": [
        {
            "user": "Why do different zones train different things?",
            "assistant": "Your body has multiple energy systems and physiological capacities. Easy aerobic work builds mitochondria and capillaries. Tempo work improves lactate clearance. Threshold training enhances sustainable power. VO2max work improves oxygen delivery. Each intensity creates a specific stress that triggers specific adaptations. A complete training program addresses all systems appropriately."
        },
        {
            "user": "Why is most training supposed to be easy?",
            "assistant": "Easy training builds aerobic base without excessive fatigue. It develops fat oxidation, mitochondrial density, and capillary networks - the foundation for all other training. Hard training is limited by recovery capacity; easy training has no such limit. The polarized approach (lots of easy, some very hard, little moderate) has proven most effective for endurance athletes."
        },
        {
            "user": "How long does it take to see adaptations?",
            "assistant": "It depends on the adaptation. Neural improvements can show within days. Cardiovascular changes take 2-4 weeks. Mitochondrial adaptations take 4-8 weeks. Structural changes (tendons, bones) take months. Mental adaptations are ongoing. This is why consistency matters - you're always building multiple systems at different timescales. Trust the process."
        },
    ],
}

GENERAL_PHILOSOPHY = [
    {
        "user": "What's the most important thing in training?",
        "assistant": "Consistency over everything. No single workout matters as much as the habit of showing up. A mediocre plan executed consistently beats a perfect plan executed sporadically. Focus on making training sustainable and enjoyable enough that you'll do it for years, not weeks."
    },
    {
        "user": "How do I know if I'm doing enough?",
        "assistant": "You're doing enough if you're recovering well, gradually progressing, staying healthy, and still enjoying the process. More isn't always better. Watch for signs of under-recovery rather than trying to maximize volume. Sustainable training you can maintain is more valuable than unsustainable training that leads to burnout."
    },
    {
        "user": "What makes a good training plan?",
        "assistant": "A good plan fits your life, matches your goals, includes progressive overload, balances stress and recovery, and is sustainable long-term. It should be specific enough to drive adaptation but flexible enough to accommodate life. The best plan is one you'll actually follow consistently."
    },
    {
        "user": "Should I train when I feel tired?",
        "assistant": "It depends. Normal tiredness often improves once you start moving - trust the warmup. But persistent fatigue, illness, or unusual exhaustion deserve respect. Listen to your body: sometimes pushing through builds mental toughness; other times it digs a deeper recovery hole. When in doubt, start easy and see how you feel."
    },
    {
        "user": "How important is data and metrics?",
        "assistant": "Data is a tool, not a master. Metrics like heart rate, pace, and power provide objective feedback that can guide training. But don't become enslaved to numbers. Perceived effort, enjoyment, and life context matter too. Use data to inform decisions, not to override how your body feels."
    },
    {
        "user": "What's the role of suffering in training?",
        "assistant": "Purposeful discomfort is part of adaptation. Pushing limits occasionally teaches your body and mind to handle more. But suffering for its own sake has no value. Know why you're pushing hard and what adaptation you're seeking. Recovery from hard efforts is as important as the efforts themselves."
    },
    {
        "user": "How do I stay motivated long-term?",
        "assistant": "Connect training to deeper values - health, capability, community, or personal growth. Set goals but don't depend on them for motivation. Find training partners or groups. Vary your routine to prevent staleness. Celebrate progress, however small. Remember that motivation follows action - start moving and it often appears."
    },
    {
        "user": "Is there a perfect training approach?",
        "assistant": "No. What works varies by individual genetics, lifestyle, goals, preferences, and training history. Principles matter more than specifics. Progressive overload, specificity, recovery, and consistency work regardless of the exact method. Find what you can sustain and enjoy, then trust the process."
    },
]


def generate_philosophy_dataset() -> List[PhilosophyPair]:
    """Generate philosophy training dataset."""

    pairs = []

    print("Generating philosophy enhanced dataset...")

    # Add topic-specific conversations
    for topic, conversations in TRAINING_PHILOSOPHY.items():
        for conv in conversations:
            # Add multiple variations for training
            for _ in range(5):  # Repeat for emphasis
                pairs.append(PhilosophyPair(
                    user_content=conv["user"],
                    assistant_content=conv["assistant"],
                    topic=topic,
                ))

    # Add general philosophy
    for conv in GENERAL_PHILOSOPHY:
        for _ in range(5):  # Repeat for emphasis
            pairs.append(PhilosophyPair(
                user_content=conv["user"],
                assistant_content=conv["assistant"],
                topic="general",
            ))

    print(f"  Generated {len(pairs)} philosophy examples")
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
        topic_counts[p.topic] = topic_counts.get(p.topic, 0) + 1
    print(f"\n  Topic breakdown:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"    {topic}: {count}")

    # Show sample output format
    print("\n" + "=" * 60)
    print("Sample output format (correct):")
    print("=" * 60)
    sample = {
        "messages": [
            {"role": "user", "content": pairs[0].user_content},
            {"role": "assistant", "content": pairs[0].assistant_content}
        ]
    }
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()

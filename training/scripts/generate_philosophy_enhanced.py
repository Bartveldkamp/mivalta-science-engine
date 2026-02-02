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
from typing import List, Dict

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
# SPECIFIC QUESTIONS MAPPED TO EACH CONTEXT KEY
# =============================================================================

# Each context gets 4-6 specific questions that naturally lead to that answer
CONTEXT_QUESTIONS: Dict[str, List[str]] = {
    # Cycling contexts
    "cycling_v4__overview": [
        "What is FTP and why does it matter?",
        "How does power training work for cycling?",
        "Why do cyclists use power meters?",
        "What's the difference between cycling and running training?",
        "Explain FTP to me",
        "I'm new to cycling - what should I know about power?",
    ],
    "cycling_v4__zones": [
        "What are the cycling power zones?",
        "How do I know what power zone I'm in?",
        "Explain the power zones to me",
        "What percentage of FTP is each zone?",
        "How much time should I spend in each power zone?",
        "I don't understand power zones - can you explain?",
    ],
    "cycling_v4__recovery": [
        "Why do cyclists recover faster than runners?",
        "How does cycling recovery compare to running?",
        "Can I do more volume on the bike?",
        "Why is there no impact in cycling?",
        "Does cycling allow for more training?",
    ],
    "cycling_v4__indoor": [
        "Why is my indoor power lower than outdoor?",
        "How do I train effectively indoors?",
        "What's special about indoor cycling?",
        "My FTP drops on the trainer - is that normal?",
        "How do I deal with heat on the indoor trainer?",
        "Tips for indoor cycling training?",
    ],
    "cycling_v4__safety": [
        "How often should I test my FTP?",
        "When should I retest my cycling zones?",
        "What safety considerations for cycling training?",
        "My power zones feel wrong - what should I do?",
        "Can I do intervals in traffic?",
    ],

    # Recovery contexts
    "recovery_v4__overview": [
        "Why is recovery important?",
        "Is rest really part of training?",
        "How does recovery actually work?",
        "Why can't I just train more?",
        "What happens during recovery?",
        "I feel lazy taking rest days - is that wrong?",
    ],
    "recovery_v4__two_lane": [
        "Can I train the day after hard intervals?",
        "What's the two-lane recovery model?",
        "When can I do easy training after a hard session?",
        "How soon after intervals can I run easy?",
        "Does easy work interfere with recovery?",
        "I did hard intervals yesterday - can I run today?",
    ],
    "recovery_v4__readiness": [
        "How do I know if I'm recovered?",
        "What does green/yellow/red readiness mean?",
        "Should I train when my HRV is low?",
        "How does readiness affect my training?",
        "What if my readiness is red but I feel fine?",
        "Can I override the readiness recommendation?",
    ],
    "recovery_v4__timing": [
        "How long between hard sessions?",
        "Does age affect recovery time?",
        "How many hours between intensity work?",
        "I'm 45 - how much recovery do I need?",
        "What's the recovery timing by age?",
    ],
    "recovery_v4__modifiers": [
        "What affects my personal recovery needs?",
        "Does training experience change recovery?",
        "Why might I need more recovery than others?",
        "How does age affect my recovery?",
        "Do different sports need different recovery?",
    ],
    "recovery_v4__week": [
        "How often should I have a recovery week?",
        "What happens in a recovery week?",
        "Why do I need a deload week?",
        "How much should I reduce during recovery week?",
        "Can I skip the recovery week if I feel good?",
    ],
    "recovery_v4__cross_domain": [
        "How does recovery connect to other training?",
        "Does recovery zone count as training?",
        "What's the relationship between recovery and zones?",
    ],
    "recovery_v4__safety": [
        "What are the non-negotiable recovery rules?",
        "When must I absolutely rest?",
        "What are signs I'm not recovering enough?",
        "How do I know if I'm overtraining?",
        "What should I do if I'm getting sick?",
    ],

    # Running contexts
    "running_v4__overview": [
        "How should I structure my running training?",
        "What percentage of running should be easy?",
        "Why is running different from cycling?",
        "What makes running training unique?",
        "How much of my running should be hard?",
    ],
    "running_v4__models": [
        "What's polarized training?",
        "Which running training model should I use?",
        "What's the difference between polarized and threshold training?",
        "I'm time-crunched - how should I train?",
        "What training approach works best for runners?",
    ],
    "running_v4__long_runs": [
        "Why are long runs important?",
        "How long should my long run be?",
        "What pace should I run my long runs?",
        "Can I skip long runs and just do intervals?",
        "How do I build up my long run distance?",
        "I hate long runs - are they really necessary?",
    ],
    "running_v4__strides": [
        "What are strides?",
        "How do I do strides?",
        "Why should I add strides to easy runs?",
        "How many strides should I do?",
        "What's the point of strides?",
    ],
    "running_v4__safety": [
        "Why do runners get injured more than cyclists?",
        "What are the safety rules for running?",
        "How do I prevent running injuries?",
        "How long to prepare for a marathon?",
        "What's a safe weekly mileage increase?",
    ],
    "running_v4__pace_zones": [
        "What's my easy running pace?",
        "How slow should easy runs be?",
        "What are the running pace zones?",
        "How do I calculate my running zones?",
        "I think my easy pace is too slow - is it?",
    ],

    # NTIZ contexts
    "ntiz_dose_v4__overview": [
        "What is NTIZ?",
        "How is training dose measured?",
        "What's normalized time in zone?",
        "How do you track training volume?",
        "Why does time in zone matter?",
    ],
    "ntiz_dose_v4__beginner": [
        "Why are my intensity zones locked?",
        "When do I unlock harder training?",
        "Why can't I do intervals as a beginner?",
        "How does intensity unlock for beginners?",
        "I want to do hard workouts but they're not available - why?",
    ],
    "ntiz_dose_v4__all_zones": [
        "How much time in each zone should I do?",
        "What are the zone limits?",
        "Is there a maximum for easy training?",
        "How much Z2 is too much?",
        "What's the cap for threshold work?",
    ],
    "ntiz_dose_v4__safety": [
        "Are zone caps targets or limits?",
        "What happens if I exceed zone limits?",
        "Why are there caps on training zones?",
        "Should I try to hit maximum zone time?",
    ],

    # Session structure contexts
    "session_dose_v4__clean_workouts": [
        "What makes a good workout structure?",
        "Why shouldn't I mix zones in one workout?",
        "What's zone soup?",
        "Should each workout have one focus?",
        "Does warmup count as base training?",
    ],
    "session_dose_v4__meso_wallet": [
        "How does the training budget work?",
        "What's the meso wallet concept?",
        "How do I spend my training budget?",
        "Why can't I do threshold every day?",
    ],
    "session_dose_v4__recovery_shadow": [
        "What's the recovery shadow?",
        "How long after hard work before more intensity?",
        "Can I do easy work the day after intervals?",
        "What's hard-to-hard vs hard-to-easy timing?",
    ],
    "session_dose_v4__z3_definition": [
        "What is Zone 3?",
        "How does tempo feel?",
        "What's the difference between Z3 and Z4?",
        "Where does Z3 fit in training?",
        "Is Z3 too hard or too easy?",
    ],

    # Periodization contexts
    "periodization__overview": [
        "What is periodization?",
        "How do training phases work?",
        "Why do we have different training phases?",
        "What's the difference between base and build?",
        "How do I know what phase I'm in?",
    ],
    "periodization__mesocycles": [
        "What's a mesocycle?",
        "How long is a training block?",
        "What's the 28-day pattern?",
        "Why do masters athletes use shorter mesos?",
        "How does the weekly loading pattern work?",
    ],
    "periodization__taper": [
        "What's the taper?",
        "How long should I taper before a race?",
        "Why reduce volume before a race?",
        "Won't I lose fitness during taper?",
        "How does tapering improve performance?",
    ],
    "periodization__safety": [
        "Can I skip the recovery week?",
        "What are the periodization rules?",
        "How do I adjust if my schedule is tight?",
    ],

    # Performance contexts
    "performance_v4__overview": [
        "How fast can I expect to improve?",
        "What's realistic for improvement?",
        "Why does progress slow down over time?",
        "How does training age affect gains?",
    ],
    "performance_v4__improvement_rates": [
        "What's a realistic weekly improvement rate?",
        "How much can I improve in a month?",
        "Why is progress slower for experienced athletes?",
        "How do improvement rates change over time?",
    ],
    "performance_v4__age_factors": [
        "How does age affect my training?",
        "Can I still improve at 50?",
        "What changes as I get older?",
        "Do older athletes need different training?",
    ],
    "performance_v4__goal_status": [
        "Is my goal realistic?",
        "How do you assess if a goal is achievable?",
        "What does 'ambitious' goal mean?",
        "Can any goal be achieved?",
    ],
    "performance_v4__frequency": [
        "How often should I train?",
        "Is training 4 times per week enough?",
        "What's the optimal training frequency?",
        "Can I improve training only 3 days per week?",
    ],
    "performance_v4__meso_adaptation": [
        "Why do I feel tired during training but good after rest?",
        "How does adaptation happen in waves?",
        "What should I expect each mesocycle?",
    ],
    "performance_v4__taper_magic": [
        "Why do I run PRs after reducing training?",
        "How does taper actually work?",
        "What's happening physiologically during taper?",
    ],
    "performance_v4__backward_planning": [
        "How do you build my training plan?",
        "Why do you plan backwards from race day?",
        "How is my peak timed for race day?",
    ],

    # Zone anchoring contexts
    "zone_anchoring_v4__overview": [
        "How are my zones calculated?",
        "What determines my training zones?",
        "What's LT1 and LT2?",
        "Where do my zones come from?",
    ],
    "zone_anchoring_v4__hr_zones": [
        "How do heart rate zones work?",
        "How are my HR zones calculated?",
        "What if I only know my max heart rate?",
        "How often should I retest heart rate zones?",
    ],
    "zone_anchoring_v4__power_zones": [
        "How are power zones set?",
        "What's the best way to find my FTP?",
        "How accurate are ramp tests?",
        "When should I retest my FTP?",
    ],
    "zone_anchoring_v4__talk_test": [
        "What's the talk test?",
        "Can I find my zones without a heart rate monitor?",
        "How do I know what zone I'm in by feel?",
        "Is there a simple way to check my zone?",
    ],
    "zone_anchoring_v4__safety": [
        "Why might age-based HR formulas be wrong?",
        "What if my zones feel off?",
        "How often should I retest my zones?",
    ],

    # Energy zones contexts
    "energy_zones_v4__overview": [
        "What are the training zones?",
        "How do the 9 zones work?",
        "What's the difference between Z1 and Z5?",
        "Why are there so many zones?",
        "Explain the zone system to me",
    ],
    "energy_zones_v4__talk_test": [
        "How do I know what zone I'm in?",
        "Can I tell my zone without gadgets?",
        "What should each zone feel like?",
        "How do I use the talk test?",
    ],
    "energy_zones_v4__day_types": [
        "What are the different types of training days?",
        "What's the difference between recovery and off days?",
        "What happens on a load day?",
        "How many rest days do I need?",
    ],
    "energy_zones_v4__cross_domain": [
        "How do zones connect to recovery?",
        "What's the relationship between zones and readiness?",
        "How do phases affect zone distribution?",
    ],
    "energy_zones_v4__safety": [
        "What are the safety rules for intensity?",
        "When should I stop a workout?",
        "What are warning signs during training?",
        "How many hard sessions per week is safe?",
    ],
    "energy_zones_v4__interval_rest": [
        "How much rest between intervals?",
        "What's the right work-to-rest ratio?",
        "Why does rest matter so much in intervals?",
        "How long should I rest between VO2max efforts?",
    ],
    "energy_zones_v4__phase_structure": [
        "Why does workout structure change by phase?",
        "What's different about base vs peak workouts?",
        "Why more intervals in peak phase?",
    ],
    "energy_zones_v4__warmup_cooldown": [
        "How long should I warm up?",
        "Do I need a cooldown?",
        "Why warm up longer for hard sessions?",
        "Does warmup length change with age?",
    ],
    "energy_zones_v4__zone_feel": [
        "What does Z2 feel like?",
        "How should threshold feel?",
        "What's the difference between Z4 and Z5 by feel?",
        "How do I know if I'm going too hard?",
    ],

    # Session structure variants
    "session_structure_variants_v4__overview": [
        "Why do different people get different interval structures?",
        "How is workout structure chosen?",
        "What determines my interval length?",
    ],
    "session_structure_variants_v4__science_of_structure": [
        "Why do 30/30 intervals work?",
        "What's the science behind interval structure?",
        "How do elite coaches progress workouts?",
    ],
    "session_structure_variants_v4__continuous_zones": [
        "When should workouts be continuous vs intervals?",
        "Why are easy sessions continuous?",
        "Do I need intervals for Z1 and Z2?",
    ],
    "session_structure_variants_v4__tempo_zone": [
        "What are the tempo workout structures?",
        "How long should tempo intervals be?",
        "What's a cruise interval?",
    ],
    "session_structure_variants_v4__threshold_zone": [
        "What are good threshold workouts?",
        "How long should threshold intervals be?",
        "What's the Norwegian threshold method?",
        "What are over-unders?",
    ],
    "session_structure_variants_v4__vo2max_zone": [
        "What are good VO2max workouts?",
        "How do 30/30 intervals work?",
        "What's the Rønnestad protocol?",
        "How long should VO2max intervals be?",
    ],
    "session_structure_variants_v4__anaerobic_zones": [
        "How do anaerobic intervals work?",
        "Why so much rest for sprint intervals?",
        "What's a descending ladder workout?",
    ],
    "session_structure_variants_v4__sprint_zone": [
        "How do I train sprints?",
        "What's a flying sprint?",
        "How many sprints should I do?",
    ],
    "session_structure_variants_v4__phase_guidance": [
        "How do workouts change through phases?",
        "What structures are used in base vs peak?",
        "Why do I get different workouts in build phase?",
    ],
    "session_structure_variants_v4__cross_domain": [
        "How does structure selection work?",
        "What determines my workout structure?",
    ],
    "session_structure_variants_v4__safety": [
        "What if I can't complete the prescribed intervals?",
        "Are micro-intervals easier than they seem?",
        "When should I drop to an easier structure?",
    ],

    # Session templates
    "session_templates_v4__warmup_scaling": [
        "How is warmup time determined?",
        "Why shorter warmup for easy sessions?",
        "How does warmup scale with session length?",
    ],

    # Goal modifiers
    "goal_modifiers_v4__overview": [
        "How does my goal affect training?",
        "Why do 5K and marathon training differ?",
        "How does goal distance change zone distribution?",
    ],
    "goal_modifiers_v4__weight_loss": [
        "What's the best training for weight loss?",
        "Should I do endless cardio to lose weight?",
        "Does HIIT help with weight loss?",
        "Why not train in the 'fat burning zone'?",
    ],
    "goal_modifiers_v4__choosing": [
        "How do I choose my training goal?",
        "What goal should I pick?",
        "I'm not sure what goal to set - help?",
    ],
    "goal_modifiers_v4__safety": [
        "Why can't beginners do marathon training?",
        "What goals require experience?",
        "Are there restrictions on goal types?",
    ],

    # Micro training
    "micro_training_commute_v4__overview": [
        "Can short workouts really be effective?",
        "Do micro-workouts actually work?",
        "Is my 15-minute commute useful training?",
        "Does science support short training sessions?",
    ],
    "micro_training_commute_v4__commute_modes": [
        "How can I use my commute for training?",
        "What types of commute workouts can I do?",
        "Can I do intervals on my commute?",
    ],
    "micro_training_commute_v4__exercise_snacks": [
        "What are exercise snacks?",
        "Can I train in small chunks through the day?",
        "Do stair sprints at work count?",
    ],
    "micro_training_commute_v4__micro_hiit": [
        "What is micro-HIIT?",
        "Can I do Tabata on my commute?",
        "How do 10-minute HIIT sessions work?",
    ],
    "micro_training_commute_v4__caps": [
        "How much can commute training count?",
        "Is there a limit on commute training?",
        "Can my whole plan be commute workouts?",
    ],
    "micro_training_commute_v4__safety": [
        "Is it safe to do intervals while commuting?",
        "What are the safety rules for commute training?",
        "Should I skip intensity in traffic?",
    ],

    # Threshold estimation
    "threshold_estimation_v4__overview": [
        "How do I find my threshold?",
        "What is threshold and how do I measure it?",
        "Do I need to test my threshold?",
    ],
    "threshold_estimation_v4__why_test": [
        "Why should I test instead of using estimates?",
        "How accurate are estimated zones?",
        "What's wrong with age-based formulas?",
    ],
    "threshold_estimation_v4__test_protocols": [
        "How do I test my threshold?",
        "What's a 20-minute threshold test?",
        "How do I do an FTP test?",
    ],
    "threshold_estimation_v4__estimates": [
        "What if I can't test my threshold?",
        "How are threshold estimates calculated?",
        "What's a typical threshold for my age?",
    ],
    "threshold_estimation_v4__safety": [
        "Are there safety concerns with threshold testing?",
        "Should I test if I'm on beta blockers?",
        "When should I avoid high intensity?",
    ],

    # NTIZ master
    "ntiz_master_v4__overview": [
        "What's the difference between session time and time in zone?",
        "Why does NTIZ matter?",
        "How do you measure my actual training stimulus?",
    ],
    "ntiz_master_v4__levels": [
        "What are the training levels?",
        "Why are some zones locked for beginners?",
        "How do I advance to the next training level?",
    ],
    "ntiz_master_v4__all_zones": [
        "What percentage goes to each zone?",
        "How much easy vs hard training should I do?",
        "What's the right zone distribution?",
    ],
    "ntiz_master_v4__safety": [
        "What are the NTIZ safety rules?",
        "How fast can I increase training?",
        "What are the ramp limits?",
    ],

    # Josi personas
    "josi_personas_v1__personas": [
        "What coaching personalities are available?",
        "Can I change how Josi talks to me?",
        "What's the difference between the coach styles?",
    ],
    "josi_personas_v1__style_mapping": [
        "How do the different styles communicate?",
        "What makes each coaching style unique?",
    ],
    "josi_personas_v1__balanced": [
        "What's the balanced coaching style?",
        "How does the default coach communicate?",
    ],
    "josi_personas_v1__drill_sergeant": [
        "What's the drill sergeant style like?",
        "Can I get a more direct coach?",
    ],
    "josi_personas_v1__science_nerd": [
        "What's the science coach style?",
        "Can I get more technical explanations?",
    ],
    "josi_personas_v1__cheerleader": [
        "What's the motivator style like?",
        "Can I get more encouragement?",
    ],
    "josi_personas_v1__zen_master": [
        "What's the mindful coaching style?",
        "Can I get a calmer coach?",
    ],
    "josi_personas_v1__dutch_directness": [
        "What's the Dutch coach style?",
        "Can I get honest, direct feedback?",
    ],
    "josi_personas_v1__frame_templates_direct": [
        "How does the direct coach frame sessions?",
    ],
    "josi_personas_v1__frame_templates_technical": [
        "How does the science coach explain sessions?",
    ],
    "josi_personas_v1__frame_templates_encouraging": [
        "How does the motivator present sessions?",
    ],
    "josi_personas_v1__frame_templates_balanced": [
        "How does the balanced coach present sessions?",
    ],
    "josi_personas_v1__i6_contract": [
        "What can Josi do and not do?",
        "Can Josi change my workout?",
        "Why can't Josi prescribe training?",
    ],

    # Planner policy
    "planner_policy_v4__validation_explanation": [
        "How does the plan validation work?",
        "What checks happen before planning?",
    ],
}


def get_questions_for_key(context_key: str) -> List[str]:
    """Get specific questions for a context key."""
    # Direct lookup first
    if context_key in CONTEXT_QUESTIONS:
        return CONTEXT_QUESTIONS[context_key].copy()

    # Fallback: find partial matches
    for key, questions in CONTEXT_QUESTIONS.items():
        if key in context_key or context_key in key:
            return questions.copy()

    # Last resort: generate from key
    return generate_fallback_questions(context_key)


def generate_fallback_questions(context_key: str) -> List[str]:
    """Generate questions from context key when no mapping exists."""
    parts = context_key.replace("__", " ").replace("_v4", "").replace("_v1", "")
    parts = parts.replace("_", " ").title()

    return [
        f"Tell me about {parts}",
        f"What should I know about {parts}?",
        f"Explain {parts} to me",
        f"How does {parts} work?",
    ]


def generate_pairs_from_context(
    context_key: str,
    content: str,
) -> List[PhilosophyPair]:
    """Generate Q&A pairs from a single context using specific questions."""

    pairs = []
    questions = get_questions_for_key(context_key)

    # Use all available questions for this context (typically 4-6)
    for question in questions:
        pairs.append(PhilosophyPair(
            user_content=question,
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
    contexts_used = 0
    for context_key in context_keys:
        content = ctx._contexts.get(context_key, "")

        if not content or len(content) < 50:
            continue

        context_pairs = generate_pairs_from_context(context_key, content)
        pairs.extend(context_pairs)
        contexts_used += 1

    print(f"Generated {len(pairs)} Q&A pairs from {contexts_used} contexts")
    return pairs


def save_chat_format(pairs: List[PhilosophyPair], output_path: Path):
    """Save in correct chat format for fine-tuning."""

    with open(output_path, 'w') as f:
        for pair in pairs:
            example = {
                "messages": [
                    {"role": "user", "content": pair.user_content},
                    {"role": "assistant", "content": pair.assistant_content}
                ]
            }
            f.write(json.dumps(example) + '\n')

    print(f"Saved {len(pairs)} examples to {output_path}")


def show_samples(pairs: List[PhilosophyPair], n: int = 10):
    """Show sample Q&A pairs for review."""
    print("\n" + "=" * 70)
    print(f"SAMPLE Q&A PAIRS ({n} examples)")
    print("=" * 70)

    # Get diverse samples from different topics
    samples = random.sample(pairs, min(n, len(pairs)))

    for i, pair in enumerate(samples, 1):
        print(f"\n--- Example {i} [{pair.context_key}] ---")
        print(f"Q: {pair.user_content}")
        # Show truncated answer
        answer = pair.assistant_content
        if len(answer) > 300:
            answer = answer[:300] + "..."
        print(f"A: {answer}")


def main():
    """Generate and save philosophy training dataset."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MiValta Philosophy Enhanced Dataset Generator")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")

    pairs = generate_philosophy_dataset()

    if len(pairs) < 400:
        print(f"WARNING: Only {len(pairs)} pairs generated, target is 400+")

    # Show samples BEFORE shuffling (for review)
    show_samples(pairs, n=10)

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

    # Unique questions count
    unique_questions = set(p.user_content for p in pairs)
    print(f"  Unique questions: {len(unique_questions)}")

    # Show format sample
    print("\n" + "=" * 60)
    print("Output format (correct):")
    print("=" * 60)
    if pairs:
        sample = {
            "messages": [
                {"role": "user", "content": pairs[0].user_content},
                {"role": "assistant", "content": pairs[0].assistant_content[:150] + "..."}
            ]
        }
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()

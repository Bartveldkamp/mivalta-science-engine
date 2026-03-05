#!/usr/bin/env python3
"""
MiValta Josi v6 — Coach Response Regenerator

Rewrites the 812 coach training responses so they actually:
  1. Ground in the [KNOWLEDGE] block (not ignore it)
  2. React to CONTEXT (readiness, session, sport, level)
  3. Use plain language (no jargon the system prompt bans)
  4. Feel like a warm, direct human coach
  5. Include ~20% Dutch responses (matching Dutch athlete messages)

The model runs 100% on-device with NO internet, so responses must be
self-contained — grounded only in the knowledge cards shipped with the model.

Usage:
    # Set your API key
    export ANTHROPIC_API_KEY=sk-ant-...

    # Preview 5 examples (dry run)
    python regenerate_coach_responses.py --preview 5

    # Regenerate all coach responses
    python regenerate_coach_responses.py --run

    # Regenerate with Dutch augmentation (~20% of examples get Dutch variants)
    python regenerate_coach_responses.py --run --dutch-ratio 0.2

    # Resume from a specific example (if interrupted)
    python regenerate_coach_responses.py --run --resume-from 150
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
GOLD_DIR = DATA_DIR / "gold_examples"

INPUT_FILE = DATA_DIR / "train_v6_unified.jsonl"
OUTPUT_FILE = DATA_DIR / "train_v6_regenerated.jsonl"
CHECKPOINT_FILE = DATA_DIR / ".regeneration_checkpoint.jsonl"

# ---------------------------------------------------------------------------
# Banned jargon (from system prompt)
# ---------------------------------------------------------------------------
BANNED_JARGON = [
    "periodization", "mesocycle", "microcycle", "macrocycle",
    "supercompensation", "vo2max", "vo2 max", "VO2max",
    "lactate threshold", "ftp", "threshold power",
    "anaerobic capacity", "mitochondrial density", "capillary growth",
    "algorithm", "viterbi", "hmm", "hidden markov",
    "acwr", "gatc", "ewma", "tss", "ctl", "atl", "tsb",
]

# ---------------------------------------------------------------------------
# Gold few-shot examples (curated from gold_examples/)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {
        "athlete_text": "Why is today just an easy Zone 2 session? I feel like I could do more.",
        "context": "Readiness: Green (Recovered)\nSession: Z2 60min \"Continuous Z2 60min\" (build phase)\nSport: cycling\nLevel: intermediate",
        "knowledge": "Training zones connect to everything else in your plan. After Z4 through Z8 work, you need 48 to 72 hours before more intensity. But easy Z1 or Z2 can happen the next day. If your HRV shows red status, stick to recovery or very easy Z1 only.",
        "interpreter": '{"action": "explain", "question": "Why is today an easy Z2 session?", "free_text": "Why is today just an easy Zone 2 session? I feel like I could do more."}',
        "response": "Great question, and I love that you're feeling strong. Here's my thinking: you did threshold work two days ago, and your body needs 48-72 hours before another hard session. That's not a guess — it's how the energy systems recover. Today's easy ride isn't wasted time. It builds your aerobic base, promotes blood flow to help clear residual fatigue, and sets you up for quality in your next hard session. Think of it as the foundation that makes the intensity work actually stick."
    },
    {
        "athlete_text": "What is Zone 2 and why do I do so much of it?",
        "context": "Readiness: Green (Recovered)\nSport: running\nLevel: beginner",
        "knowledge": "Zone 2 is the aerobic training zone. At this intensity, your body gets efficient at burning fat for fuel and builds the aerobic engine that powers everything else. About 80% of elite endurance athletes train at this easy pace. The magic happens when you combine lots of easy with a little bit of hard.",
        "interpreter": '{"action": "answer_question", "question": "What is Zone 2 and why do I do so much of it?", "free_text": "What is Zone 2 and why do I do so much of it?"}',
        "response": "Zone 2 is that easy, conversational pace where you could chat with a friend without gasping. It feels almost too easy, which is exactly the point. At this intensity, your body gets really efficient at burning fat for fuel and builds the aerobic engine that powers everything else. About 80% of elite endurance athletes train at this easy pace — not because they're lazy, but because it builds the foundation that makes hard efforts actually work. The magic happens when you combine lots of easy with a little bit of hard."
    },
    {
        "athlete_text": "I've completely lost my motivation to train. Help.",
        "context": "Readiness: Amber (Fatigued)\nSport: running\nLevel: intermediate",
        "knowledge": "Motivation is temporary and unreliable. What sustains training long-term is discipline and habit. Connect exercise to what matters — your family, your independence, feeling good. Try something different — a new activity, route, or workout style. Set a small, achievable goal for the next two weeks. Remove friction — lay out your kit the night before.",
        "interpreter": '{"action": "answer_question", "question": "Lost motivation to train", "free_text": "I\'ve completely lost my motivation to train. Help."}',
        "response": "I hear you, and this happens to everyone — even professional athletes. The first thing to recognise: motivation is temporary and unreliable. What sustains training long-term is discipline and habit, not motivation. That said, let's reignite the spark. Ask yourself why you started. Connect exercise to what matters — your family, your independence, feeling good. Try something different — a new route, a different workout style. Set a small, achievable goal for the next two weeks. Remove friction — lay out your kit the night before. And go easy on yourself. You're here asking for help, which means you haven't given up."
    },
    {
        "athlete_text": "Why is my readiness amber today?",
        "context": "Readiness: Amber (Fatigued)\nSession: Z2 45min \"Continuous Z2 45min\" (build phase)\nSport: cycling\nLevel: intermediate",
        "knowledge": "Readiness reflects the balance between the training stress you've accumulated and how well your body has recovered. When recovery is keeping pace with training, you're green. When it's falling behind, you shift to amber or red. The two-lane model governs what you can do — after hard sessions, you need 48 to 72 hours before more intensity.",
        "interpreter": '{"action": "explain", "question": "Why is readiness amber?", "free_text": "Why is my readiness amber today?"}',
        "response": "Looking at your trends, your body is showing signs of accumulated fatigue from the past few days. The amber means we'll keep today's session effective but skip the highest intensity work. You can still train well — just no all-out sprints or max efforts today. This protects your progress while letting your body catch up. Think of it as your body asking for a lighter touch, not a day off."
    },
]

# Dutch few-shot examples
DUTCH_FEW_SHOT = [
    {
        "athlete_text": "Waarom is mijn readiness rood vandaag?",
        "context": "Readiness: Red (Overreached)\nSession: Z1 30min \"Recovery spin\" (recovery)\nSport: cycling\nLevel: intermediate",
        "knowledge": "Readiness reflects the balance between the training stress you've accumulated and how well your body has recovered. When it's red, your body is telling you it needs rest. Training through red readiness consistently leads to overtraining and injury.",
        "interpreter": '{"action": "explain", "question": "Why is readiness red?", "free_text": "Waarom is mijn readiness rood vandaag?"}',
        "response": "Je lichaam geeft een duidelijk signaal: de afgelopen dagen hebben meer van je gevraagd dan je hebt kunnen herstellen. Rood betekent niet dat je iets verkeerd hebt gedaan — het betekent dat je hard hebt gewerkt en nu even pas op de plaats moet maken. Vandaag staat er een rustige herstellessie van 30 minuten op het programma, en dat is precies wat je nodig hebt. Houd het echt licht — als het te zwaar voelt, mag je ook gewoon laten zitten. De training loopt niet weg. Morgen ziet de wereld er weer anders uit."
    },
]


def build_generation_prompt(
    athlete_text: str,
    context_block: str,
    knowledge_block: str,
    interpreter_block: str,
    language: str = "english",
) -> str:
    """Build the Claude prompt for generating a high-quality coach response."""

    # Pick language-appropriate few-shots
    examples = FEW_SHOT_EXAMPLES[:3]
    if language == "dutch":
        examples = DUTCH_FEW_SHOT + FEW_SHOT_EXAMPLES[:2]

    few_shot_text = ""
    for i, ex in enumerate(examples, 1):
        few_shot_text += f"""
<example_{i}>
ATHLETE: {ex['athlete_text']}
CONTEXT: {ex['context']}
KNOWLEDGE: {ex['knowledge'][:300]}
INTERPRETER: {ex['interpreter']}

JOSI RESPONSE:
{ex['response']}
</example_{i}>
"""

    lang_instruction = ""
    if language == "dutch":
        lang_instruction = """
LANGUAGE REQUIREMENT: The athlete writes in Dutch. You MUST respond entirely in Dutch.
Write natural, warm Dutch — not translated English. Use Dutch idioms and phrasing.
"""

    prompt = f"""You are rewriting training data for Josi, an on-device AI coaching assistant for endurance athletes (running, cycling, skiing, skating). Josi runs 100% on-device with NO internet — responses must be self-contained.

Your task: given an athlete message, their context, a knowledge block, and the interpreter's classification, write the PERFECT coach response that Josi should give.

QUALITY BAR — your response must:
1. GROUND in the [KNOWLEDGE] block. Rephrase the science naturally in coaching voice — don't quote verbatim, but USE the knowledge.
2. REACT to CONTEXT. If readiness is Red → acknowledge fatigue/concern. If Green → energy. If session details are present → reference them specifically.
3. Use PLAIN LANGUAGE. Absolutely no: periodization, mesocycle, microcycle, macrocycle, supercompensation, VO2max, lactate threshold, FTP, threshold power, anaerobic capacity, mitochondrial density, capillary growth, or any internal system names (GATC, ACWR, TSS, CTL, ATL, TSB, HMM, Viterbi, EWMA).
4. Be WARM and DIRECT. You are the athlete's coach. You care. No corporate speak, no filler, no fortune-cookie platitudes.
5. ANSWER FIRST. Lead with substance. Don't start with "Great question!" or similar. Get to the point warmly.
6. 60-120 words for simple topics (zones, session explanation). 120-200 words for complex topics (readiness patterns, motivation, injury concerns).
7. End with a statement or encouragement, NOT a question (unless truly necessary).
8. Never recommend other apps, coaches, doctors (unless medical red flag), or external resources — the athlete has NO internet.
9. Never mention knowledge cards, blocks, interpreters, or internal systems.
{lang_instruction}
{few_shot_text}

Now write the response for:

ATHLETE: {athlete_text}
CONTEXT: {context_block}
KNOWLEDGE: {knowledge_block}
INTERPRETER: {interpreter_block}

JOSI RESPONSE:"""

    return prompt


def parse_coach_example(messages: list[dict]) -> dict | None:
    """Parse a coach training example into its component parts."""
    system_content = ""
    user_content = ""
    assistant_content = ""

    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] == "user":
            user_content = msg["content"]
        elif msg["role"] == "assistant":
            assistant_content = msg["content"]

    # Only process coach examples
    if "Interpreter" in system_content and "GATCRequest" in system_content:
        return None  # This is an interpreter example
    if "coach" not in system_content.lower():
        return None

    # Extract athlete text (before CONTEXT:)
    athlete_text = user_content
    context_block = ""
    knowledge_block = ""
    interpreter_block = ""
    history_block = ""

    # Parse CONTEXT
    ctx_match = re.search(r'CONTEXT:\s*\n', user_content)
    if ctx_match:
        after_ctx = user_content[ctx_match.end():]
        athlete_text = user_content[:ctx_match.start()].strip()

        # Find end of CONTEXT (next block marker or end)
        next_block = re.search(r'\n\s*\n\s*(?:HISTORY:|(?:\[KNOWLEDGE\])|\[INTERPRETER\])', after_ctx)
        if next_block:
            context_block = after_ctx[:next_block.start()].strip()
        else:
            context_block = after_ctx.strip()

    # Parse HISTORY
    hist_match = re.search(r'HISTORY:\s*\n', user_content)
    if hist_match:
        after_hist = user_content[hist_match.end():]
        next_block = re.search(r'\n\s*\n\s*(?:\[KNOWLEDGE\]|\[INTERPRETER\])', after_hist)
        if next_block:
            history_block = after_hist[:next_block.start()].strip()
        else:
            history_block = after_hist.strip()

    # Parse KNOWLEDGE
    know_match = re.search(r'\[KNOWLEDGE\]\s*\n?', user_content)
    if know_match:
        after_know = user_content[know_match.end():]
        next_block = re.search(r'\n\s*\n\s*\[INTERPRETER\]', after_know)
        if next_block:
            knowledge_block = after_know[:next_block.start()].strip()
        else:
            knowledge_block = after_know.strip()

    # Parse INTERPRETER
    interp_match = re.search(r'\[INTERPRETER\]\s*\n?', user_content)
    if interp_match:
        interpreter_block = user_content[interp_match.end():].strip()

    # Detect language (simple heuristic)
    dutch_words = ["ik", "mijn", "wat", "hoe", "vandaag", "waarom", "goed", "training",
                   "gevoel", "moe", "vertel", "sessie", "week", "kan", "wil"]
    words = athlete_text.lower().split()
    dutch_hits = sum(1 for w in words if w in dutch_words)
    language = "dutch" if dutch_hits >= 2 else "english"

    return {
        "athlete_text": athlete_text,
        "context_block": context_block,
        "knowledge_block": knowledge_block,
        "interpreter_block": interpreter_block,
        "history_block": history_block,
        "original_response": assistant_content,
        "language": language,
        "system_content": system_content,
    }


def check_jargon(text: str) -> list[str]:
    """Check response for banned jargon. Returns list of violations."""
    violations = []
    text_lower = text.lower()
    for term in BANNED_JARGON:
        if term.lower() in text_lower:
            violations.append(term)
    return violations


def check_quality(response: str, parsed: dict) -> dict:
    """Validate response quality. Returns dict with pass/fail and issues."""
    issues = []

    # Check jargon
    jargon = check_jargon(response)
    if jargon:
        issues.append(f"Jargon: {', '.join(jargon)}")

    # Check length
    words = len(response.split())
    if words < 30:
        issues.append(f"Too short: {words} words")
    if words > 250:
        issues.append(f"Too long: {words} words")

    # Check for generic openers
    generic_openers = [
        "great question", "that's a great", "absolutely!",
        "of course!", "certainly!", "definitely!",
    ]
    for opener in generic_openers:
        if response.lower().startswith(opener):
            issues.append(f"Generic opener: '{opener}'")
            break

    # Check knowledge grounding (if knowledge was provided)
    if parsed["knowledge_block"] and len(parsed["knowledge_block"]) > 50:
        # Simple check: response should share some content words with knowledge
        know_words = set(parsed["knowledge_block"].lower().split())
        resp_words = set(response.lower().split())
        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "have", "has", "had", "do", "does", "did", "will", "would",
                     "could", "should", "may", "might", "can", "shall", "to",
                     "of", "in", "for", "on", "with", "at", "by", "from", "and",
                     "or", "but", "not", "this", "that", "it", "your", "you",
                     "i", "my", "me", "we", "our", "they", "their"}
        know_content = know_words - stopwords
        resp_content = resp_words - stopwords
        overlap = know_content & resp_content
        if len(overlap) < 3 and len(know_content) > 10:
            issues.append("Weak knowledge grounding")

    # Check readiness acknowledgment
    ctx = parsed["context_block"].lower()
    resp_lower = response.lower()
    if "red" in ctx and "overreached" in ctx:
        fatigue_words = ["fatigue", "tired", "rest", "recover", "easy", "light",
                         "careful", "signal", "back off", "moe", "rust", "herstel"]
        if not any(w in resp_lower for w in fatigue_words):
            issues.append("Red readiness not acknowledged")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "word_count": words,
    }


async def generate_response(
    client,
    parsed: dict,
    language: str = "english",
    max_retries: int = 2,
) -> tuple[str, dict]:
    """Generate a single coach response using Claude."""
    prompt = build_generation_prompt(
        athlete_text=parsed["athlete_text"],
        context_block=parsed["context_block"],
        knowledge_block=parsed["knowledge_block"],
        interpreter_block=parsed["interpreter_block"],
        language=language,
    )

    for attempt in range(max_retries + 1):
        try:
            message = await asyncio.to_thread(
                client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            response = message.content[0].text.strip()

            # Remove any quotes if Claude wrapped the response
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]

            # Quality check
            quality = check_quality(response, parsed)

            # Retry on jargon violations (most common fixable issue)
            if not quality["passed"] and "Jargon" in str(quality["issues"]) and attempt < max_retries:
                violations = check_jargon(response)
                prompt += f"\n\nIMPORTANT: Your previous response contained banned jargon: {', '.join(violations)}. Rewrite WITHOUT these terms. Use plain language equivalents."
                continue

            return response, quality

        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                print(f"    Retry in {wait}s: {e}")
                await asyncio.sleep(wait)
            else:
                return parsed["original_response"], {
                    "passed": False,
                    "issues": [f"API error: {e}"],
                    "word_count": len(parsed["original_response"].split()),
                }

    return parsed["original_response"], {"passed": False, "issues": ["Max retries"], "word_count": 0}


def rebuild_example(original: dict, new_response: str) -> dict:
    """Rebuild the training example with the new assistant response."""
    messages = []
    for msg in original["messages"]:
        if msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": new_response})
        else:
            messages.append(msg)
    return {"messages": messages}


def make_dutch_variant(parsed: dict) -> dict:
    """Create a Dutch variant of an English example by translating the athlete text."""
    # We'll let Claude handle the full Dutch generation
    # Just mark it as Dutch so the prompt asks for Dutch response
    return {**parsed, "language": "dutch"}


async def run_regeneration(args):
    """Main regeneration loop."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load all training data
    print(f"Loading training data from {INPUT_FILE}...")
    all_examples = []
    with open(INPUT_FILE) as f:
        for line in f:
            all_examples.append(json.loads(line.strip()))
    print(f"  Total examples: {len(all_examples)}")

    # Separate coach vs interpreter examples
    coach_indices = []
    interpreter_examples = []
    for i, ex in enumerate(all_examples):
        msgs = ex["messages"]
        sys_content = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
        if "coach" in sys_content.lower() and "Interpreter" not in sys_content:
            coach_indices.append(i)
        else:
            interpreter_examples.append(ex)

    print(f"  Coach examples to regenerate: {len(coach_indices)}")
    print(f"  Interpreter examples (kept as-is): {len(interpreter_examples)}")

    # Load checkpoint if resuming
    completed = {}
    if args.resume_from > 0 and CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            for line in f:
                entry = json.loads(line.strip())
                completed[entry["index"]] = entry["response"]
        print(f"  Resumed from checkpoint: {len(completed)} already done")

    # Preview mode
    if args.preview > 0:
        print(f"\n{'='*60}")
        print(f"  PREVIEW MODE — showing {args.preview} examples")
        print(f"{'='*60}")

        random.seed(42)
        preview_indices = random.sample(coach_indices, min(args.preview, len(coach_indices)))

        for idx in preview_indices:
            ex = all_examples[idx]
            parsed = parse_coach_example(ex["messages"])
            if not parsed:
                continue

            print(f"\n--- Example {idx} ---")
            print(f"Athlete: {parsed['athlete_text'][:80]}")
            print(f"Context: {parsed['context_block'][:100]}")
            print(f"Knowledge: {'Yes' if parsed['knowledge_block'] else 'No'} ({len(parsed['knowledge_block'])} chars)")
            print(f"Action: {parsed['interpreter_block'][:100]}")
            print(f"\nOLD response ({len(parsed['original_response'])} chars):")
            print(f"  {parsed['original_response']}")

            response, quality = await generate_response(client, parsed)
            print(f"\nNEW response ({len(response)} chars, {quality['word_count']} words):")
            print(f"  {response}")
            print(f"Quality: {'PASS' if quality['passed'] else 'FAIL'} {quality['issues']}")
        return

    # Full regeneration
    print(f"\n{'='*60}")
    print(f"  REGENERATING {len(coach_indices)} coach responses")
    print(f"  Dutch ratio: {args.dutch_ratio:.0%}")
    print(f"{'='*60}")

    # Decide which examples get Dutch variants
    random.seed(args.seed)
    dutch_count = int(len(coach_indices) * args.dutch_ratio)
    dutch_indices = set(random.sample(coach_indices, min(dutch_count, len(coach_indices))))
    print(f"  Dutch variants: {len(dutch_indices)}")

    regenerated = {}
    dutch_extras = []
    stats = {"total": 0, "passed": 0, "failed": 0, "jargon": 0, "short": 0}
    start_time = time.time()

    # Process in batches for controlled concurrency
    batch_size = args.batch_size
    for batch_start in range(0, len(coach_indices), batch_size):
        batch = coach_indices[batch_start:batch_start + batch_size]

        tasks = []
        for idx in batch:
            if idx in completed and args.resume_from > 0:
                regenerated[idx] = completed[idx]
                stats["total"] += 1
                stats["passed"] += 1
                continue

            ex = all_examples[idx]
            parsed = parse_coach_example(ex["messages"])
            if not parsed:
                regenerated[idx] = ex["messages"][-1]["content"]
                continue

            # English response
            tasks.append((idx, parsed, "english"))

            # Dutch variant (extra example, not replacing)
            if idx in dutch_indices:
                tasks.append((idx, parsed, "dutch"))

        # Run batch concurrently
        sem = asyncio.Semaphore(args.concurrency)

        async def process_one(task_idx, task_parsed, task_lang):
            async with sem:
                return await generate_response(client, task_parsed, language=task_lang)

        results = await asyncio.gather(
            *[process_one(idx, p, lang) for idx, p, lang in tasks],
            return_exceptions=True,
        )

        # Collect results
        for (idx, parsed, lang), result in zip(tasks, results):
            if isinstance(result, Exception):
                print(f"  ERROR at {idx}: {result}")
                response = parsed["original_response"]
                quality = {"passed": False, "issues": [str(result)], "word_count": 0}
            else:
                response, quality = result

            if lang == "english":
                regenerated[idx] = response
                stats["total"] += 1
                if quality["passed"]:
                    stats["passed"] += 1
                else:
                    stats["failed"] += 1
                    if any("Jargon" in i for i in quality.get("issues", [])):
                        stats["jargon"] += 1
                    if any("short" in i.lower() for i in quality.get("issues", [])):
                        stats["short"] += 1

                # Checkpoint
                with open(CHECKPOINT_FILE, "a") as f:
                    f.write(json.dumps({"index": idx, "response": response}) + "\n")
            else:
                # Dutch variant: create a new training example
                original = all_examples[idx]
                dutch_ex = rebuild_example(original, response)
                # Also translate the athlete text in the user message
                dutch_extras.append(dutch_ex)

        # Progress
        elapsed = time.time() - start_time
        done = stats["total"]
        total = len(coach_indices)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] {rate:.1f}/s  ETA: {eta/60:.0f}min  "
              f"pass={stats['passed']} fail={stats['failed']} jargon={stats['jargon']}")

    # Rebuild the full dataset
    print(f"\n{'='*60}")
    print(f"  REBUILDING DATASET")
    print(f"{'='*60}")

    output_examples = []

    # Add interpreter examples unchanged
    for ex in all_examples:
        msgs = ex["messages"]
        sys_content = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
        if "Interpreter" in sys_content and "GATCRequest" in sys_content:
            output_examples.append(ex)

    # Add regenerated coach examples
    for idx in coach_indices:
        if idx in regenerated:
            new_ex = rebuild_example(all_examples[idx], regenerated[idx])
            output_examples.append(new_ex)
        else:
            output_examples.append(all_examples[idx])

    # Add Dutch extras
    output_examples.extend(dutch_extras)

    # Shuffle
    random.shuffle(output_examples)

    # Save
    with open(OUTPUT_FILE, "w") as f:
        for ex in output_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Also save a backup of the original
    backup = DATA_DIR / "train_v6_unified_backup.jsonl"
    if not backup.exists():
        import shutil
        shutil.copy2(INPUT_FILE, backup)
        print(f"  Backed up original to: {backup.name}")

    # Stats
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Regenerated: {stats['total']} coach responses")
    print(f"  Quality pass: {stats['passed']}/{stats['total']} ({stats['passed']/max(stats['total'],1)*100:.0f}%)")
    print(f"  Jargon violations: {stats['jargon']}")
    print(f"  Dutch extras added: {len(dutch_extras)}")
    print(f"  Total examples: {len(output_examples)} (was {len(all_examples)})")
    print(f"  Output: {OUTPUT_FILE}")
    print()
    print(f"  Next steps:")
    print(f"    1. Review: python regenerate_coach_responses.py --preview 10")
    print(f"    2. Validate: python validate_training_data.py --input {OUTPUT_FILE.name}")
    print(f"    3. Replace: cp {OUTPUT_FILE.name} train_v6_unified.jsonl")
    print(f"    4. Retrain: python finetune_qwen3.py train --mode unified")

    # Clean up checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Josi coach training responses using Claude"
    )
    parser.add_argument("--preview", type=int, default=0,
                        help="Preview N random examples (dry run)")
    parser.add_argument("--run", action="store_true",
                        help="Run full regeneration")
    parser.add_argument("--dutch-ratio", type=float, default=0.15,
                        help="Fraction of examples to add Dutch variants (default: 0.15)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Examples per batch (default: 10)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Concurrent API calls per batch (default: 5)")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume from example index (uses checkpoint)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    if not args.run and args.preview == 0:
        parser.print_help()
        print("\n  Start with: python regenerate_coach_responses.py --preview 5")
        sys.exit(0)

    asyncio.run(run_regeneration(args))


if __name__ == "__main__":
    main()

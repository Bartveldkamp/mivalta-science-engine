# Josi — The Road to Perfect

**Date:** 2026-02-19
**Starting point:** All code and data exist. No model has been trained yet. No GPU on this server.

---

## Current Reality Check

| What | Status |
|------|--------|
| Training scripts (Gemma 3n + SmolLM2) | Written, never run |
| Training data (1,450 interpreter + 1,120 explainer) | Ready |
| Gold examples (678 curated) | Ready |
| Eval harness (3-layer) | Written, never run against a trained model |
| Post-processing (parser, postprocessor, governor) | Written |
| Memory extractor | Written, not integrated into Rust engine |
| System prompts (interpreter + explainer) | Written |
| JSON schemas (4) | Written |
| GBNF grammar | Does not exist |
| llama.cpp | Not installed on this server |
| GPU | None on this server (21 GB RAM, 16 cores, CPU only) |
| Trained model checkpoints | None |
| GGUF files | None |
| Rust engine (GATC) | Separate repo, not here |

**Bottom line:** You have an excellent blueprint. Zero trained models. Step one is getting a GPU.

---

## Phase 0: Infrastructure (Day 1)

### 0.1 — Get a GPU Server

You need **one** of these:

| Option | GPU | VRAM | Cost | Notes |
|--------|-----|------|------|-------|
| **Hetzner GPU (GEX44)** | A100 or similar | 40 GB | ~€1.50/hr | Same datacenter as this server, fastest transfer |
| **RunPod** | A100 40GB | 40 GB | ~$1.60/hr | Spot pricing can be cheaper |
| **Lambda Cloud** | A100 40GB | 40 GB | ~$1.10/hr | Often sold out |
| **Vast.ai** | A100/A6000 | 40-48 GB | ~$0.80/hr | Cheapest, less reliable |

**Minimum requirement:** 16 GB VRAM for Gemma 3n bf16 + LoRA.
**Recommended:** 40 GB (A100) — comfortable headroom, faster training, room for longer contexts.

Estimated GPU hours needed for the full plan: **~40-60 hours** (~€60-90).

### 0.2 — Install Missing Dependencies

On the GPU server:
```bash
# Run the existing setup script
bash training/scripts/setup_hetzner.sh

# It will install peft, bitsandbytes, accelerate, datasets, wandb
# And download google/gemma-3n-E2B-it (~10 GB)
```

### 0.3 — Build llama.cpp (on THIS server or GPU server)

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp && make -j$(nproc)
```

This CPU server (21 GB RAM, 16 cores) is fine for:
- GGUF conversion
- GGUF inference testing
- GBNF grammar testing
- Eval on quantized models (slow but works)

---

## Phase 1: First Working Models (Week 1)

**Goal:** Train Interpreter + Explainer, export to GGUF, run eval. Get a baseline.

### 1.1 — Train Interpreter (GPU server, ~1-2 hours)

```bash
python scripts/finetune_gemma3n.py train --mode interpreter
```

This trains on `train_interpreter.jsonl` (1,450 examples) with LoRA r=8.

### 1.2 — Train Explainer (GPU server, ~1-2 hours)

```bash
python scripts/finetune_gemma3n.py train --mode explainer
```

This trains on `train_explainer.jsonl` (1,120 examples).

### 1.3 — Merge + Export

```bash
# Merge LoRA weights
python scripts/finetune_gemma3n.py merge --lora_path ./models/josi-v4-gemma3n-interpreter-*/lora_weights
python scripts/finetune_gemma3n.py merge --lora_path ./models/josi-v4-gemma3n-explainer-*/lora_weights

# Convert to GGUF Q4_K_M
python scripts/convert_gemma3n.py --model_path ./models/josi-v4-gemma3n-interpreter-*/merged
python scripts/convert_gemma3n.py --model_path ./models/josi-v4-gemma3n-explainer-*/merged
```

### 1.4 — Run 3-Layer Eval

```bash
python scripts/evaluate_gemma3n_v4.py \
  --interpreter models/josi-v4-gemma3n-interpreter-*/final \
  --explainer models/josi-v4-gemma3n-explainer-*/final \
  --verbose
```

**This is your baseline.** Every subsequent change is measured against these numbers.

Target minimums:
- Interpreter: Valid JSON > 95%, action accuracy > 90%
- Explainer: Pass rate > 80%, warmth > 3.5/5, brevity > 3.5/5

### 1.5 — Test on This CPU Server

Copy the GGUFs back here. Run inference with llama.cpp CLI to sanity-check:
```bash
~/llama.cpp/llama-cli -m josi-v4-interpreter-q4_k_m.gguf -p "..." -n 150
```

**Phase 1 outcome:** Two working GGUF files. A baseline score. You can now start improving.

---

## Phase 2: GBNF Grammar for Interpreter (Week 1-2)

**Why:** Eliminates ALL JSON parsing failures at decode time. Your `gatc_postprocessor.py` currently patches broken JSON after the fact. Grammar prevents it.

### 2.1 — Write the Grammar

Create `shared/grammars/gatc_request.gbnf` matching the schema exactly:

```gbnf
root ::= "{" ws action-field "," ws free-text-field optional-fields "}" ws

action-field ::= "\"action\"" ws ":" ws action-value
action-value ::= "\"create_workout\"" | "\"replan\"" | "\"explain\"" | "\"answer_question\"" | "\"clarify\""

sport-value ::= "\"run\"" | "\"bike\"" | "\"ski\"" | "\"skate\"" | "\"strength\"" | "\"other\""
replan-value ::= "\"skip_today\"" | "\"swap_days\"" | "\"reschedule\"" | "\"reduce_intensity\"" | "\"illness\"" | "\"travel\"" | "\"goal_change\""
goal-value ::= "\"endurance\"" | "\"threshold\"" | "\"vo2\"" | "\"recovery\"" | "\"strength\"" | "\"race_prep\""
fatigue-value ::= "\"fresh\"" | "\"ok\"" | "\"tired\"" | "\"very_tired\""

free-text-field ::= "\"free_text\"" ws ":" ws string
# ... (full grammar covering all optional fields, types, constraints object)
```

### 2.2 — Test Against Eval Set

Run inference with grammar on the 1,000 test prompts. Target: **100% valid JSON** (not just 95%).

### 2.3 — Measure Quality Impact

Grammar constrains the model. Check that action accuracy doesn't drop. If it does, the grammar may be too restrictive — loosen string fields while keeping enums tight.

### 2.4 — Simplify Post-Processing

Once grammar guarantees valid JSON:
- `gatc_postprocessor.py` can drop: markdown fence stripping, truncated JSON repair, string concatenation fixes
- Keep: semantic validation (missing sport → force clarify), medical red-flag detection, time extraction from free_text

**Phase 2 outcome:** 100% valid JSON from Interpreter. Simpler post-processing. Kotlin port becomes easier.

---

## Phase 3: Expand Context to 4K (Week 2)

**Why:** 1,024 tokens is too tight. You overflow with memory + GATC grounding. 4K gives breathing room.

### 3.1 — New Token Budget

| Component | Tokens | Notes |
|-----------|--------|-------|
| System prompt | 400 | Current ~300, slight room to grow |
| Athlete context + readiness | 250 | Profile, phase, readiness state |
| Athlete memory | 400 | 15 key_facts + 5 patterns + 5 coaching_notes |
| GATC result (new, Phase 5) | 200 | Workout summary + rationale + safety caps |
| Conversation history | 500 | 3-4 turns |
| User message | 100 | Current turn |
| **Output budget** | **300** | Up from 150 |
| **Reserve** | **1,850** | Headroom within 4K |

### 3.2 — Retrain at 4K

```bash
python scripts/finetune_gemma3n.py train --mode interpreter --max_seq_length 4096
python scripts/finetune_gemma3n.py train --mode explainer --max_seq_length 4096
```

**Important:** Gemma 3n supports 8K context natively. Going to 4K is safe, no RoPE scaling needed.

### 3.3 — Generate Longer Training Examples

Augment training data with examples that use the full context budget:
- Examples with 3-4 turn conversation history
- Examples with athlete memory blocks
- Examples with GATC result blocks (placeholder for Phase 5)

Target: 500 new examples per model.

### 3.4 — Eval at 4K

Re-run the 3-layer eval. Compare against Phase 1 baseline. Context expansion should improve quality (more info available) without degrading schema compliance.

**Phase 3 outcome:** Models comfortable at 4K context. Room for memory and GATC grounding.

---

## Phase 4: Training Data Quality Push (Week 2-3)

**Why:** Your training data is the single biggest lever. Better data > bigger model > fancier techniques.

### 4.1 — Expand Interpreter Training Data

Current: 1,450 examples. Target: **3,000-4,000**.

Add:
- 500 edge cases from `augment_training_data_v4.py` (already exists, just run it)
- 300 memory-enriched examples from `augment_memory_training_data.py` (exists)
- 200 multi-turn context examples (write: user asks follow-up after getting a workout)
- 200 ambiguous intent examples (training the model to clarify correctly)
- 200 time extraction examples (diverse formats: "about an hour", "45ish min", "two hours")

### 4.2 — Expand Explainer Training Data

Current: 1,120 examples. Target: **3,000-4,000**.

Add:
- 500 examples with GATC result context (grounded explanations — Phase 5 preview)
- 300 memory-enriched examples (personalized coaching tone)
- 200 negative examples (what NOT to do: jargon, >100 words, inventing numbers)
- 200 readiness explanation examples (amber/red with care and nuance)
- 200 encouragement examples (diverse athlete scenarios)

### 4.3 — Coach-in-the-Loop Gold Data

Have a sports scientist or experienced coach write (or review/edit) **200 gold-standard Explainer responses**. These become:
- Training anchors (highest quality examples in the dataset)
- Eval anchors (the bar for "good enough")
- DPO "chosen" examples (Phase 6)

### 4.4 — Retrain and Eval

Train both models on the expanded dataset. Compare against Phase 1 and Phase 3 baselines.

**Phase 4 outcome:** 3-4x more training data. Noticeably better model quality.

---

## Phase 5: GATC Grounding for Explainer (Week 3-4)

**Why:** This is the single biggest quality leap. The Explainer currently makes up everything. With GATC grounding, it explains specific decisions with specific numbers.

### 5.1 — Add `gatc_result` to Chat Context Schema

Update `shared/schemas/chat_context.schema.json`:
```json
"gatc_result": {
  "type": "object",
  "properties": {
    "workout_summary": {
      "type": "string",
      "description": "e.g. '40-min easy run, zones 1-2'"
    },
    "rationale_bullets": {
      "type": "array",
      "items": { "type": "string" },
      "description": "e.g. ['Readiness amber — capping intensity', 'Week 3 of 4 — building load']"
    },
    "safety_caps_applied": {
      "type": "string",
      "description": "e.g. 'max Z2 due to amber readiness'"
    }
  }
}
```

### 5.2 — Update Explainer System Prompt

Add to `training/prompts/explainer_system.txt`:
```
GATC RESULT (if present in CONTEXT):
- The coaching engine has already made the training decision
- Your job is to EXPLAIN it, not to override or second-guess it
- Reference the specific workout, zones, and rationale provided
- NEVER invent numbers — only use what GATC gives you
- If rationale_bullets are provided, weave them into your explanation naturally
```

### 5.3 — Generate Grounded Training Examples

Create 500+ examples where the Explainer sees a GATC result and explains it:

```json
{
  "context": {
    "gatc_result": {
      "workout_summary": "45-min easy run, zones 1-2",
      "rationale_bullets": ["Readiness amber", "Recovery day after yesterday's threshold work"],
      "safety_caps_applied": "max Z2 due to amber readiness"
    }
  },
  "response": "Today's an easy 45-minute run, keeping things in zones 1 and 2. Your body's still recovering from yesterday's hard session, so we're staying comfortable. Think conversational pace — if you can't chat with a friend while running, you're going too hard."
}
```

Also generate **negative examples** (Explainer inventing numbers not in the GATC result → used for DPO in Phase 6).

### 5.4 — Retrain Explainer

Train on the combined dataset (original + grounded examples). Eval should show:
- Responses reference specific GATC data
- No invented numbers/zones
- Still warm and concise

### 5.5 — Kotlin Integration Point

This requires the Rust engine to pass `gatc_result` to the Explainer prompt. The Kotlin flow becomes:

```
User message → Interpreter → GATCRequest JSON
                                    ↓
                              Rust GATC Engine
                                    ↓
                              gatc_result
                                    ↓
                   gatc_result + user message → Explainer → coaching text
```

**Phase 5 outcome:** Explainer is grounded in real GATC decisions. No more making things up.

---

## Phase 6: DPO Preference Tuning for Explainer (Week 4-5)

**Why:** SFT teaches the model what to say. DPO teaches it what NOT to say. This is how you get the voice perfect.

### 6.1 — Build Preference Dataset

500-1,000 preference pairs: `(prompt, chosen_response, rejected_response)`

Categories of **rejected** responses:

| Rejection reason | Example |
|-----------------|---------|
| Invents numbers | "Your FTP is around 250 watts" (not from GATC) |
| Uses jargon | "This periodization block targets supercompensation" |
| Too long | 150+ words when 60 would do |
| Asks too many questions | "What sport? What time? How are you feeling?" |
| Deflects | "You should ask your doctor/coach about that" |
| Mentions internals | "The algorithm determined..." |
| Generic when grounded data exists | "Just take it easy today" (when GATC gave specifics) |
| Robotic memory use | "I know you like running, so here's a run" |

Sources for rejected responses:
- Generate from the untrained base model (natural "bad" outputs)
- Manually write worst-case responses
- Take SFT model outputs that fail Layer 3 quality checks

Sources for chosen responses:
- Gold examples from the coach (Phase 4.3)
- Best SFT model outputs that pass all 3 eval layers

### 6.2 — DPO Training

```python
from trl import DPOTrainer, DPOConfig

# Two-stage: SFT checkpoint → DPO
# SFT model is the starting point
# DPO adjusts preferences without catastrophic forgetting

config = DPOConfig(
    beta=0.1,                    # KL penalty — start conservative
    learning_rate=5e-7,          # Much lower than SFT
    max_length=4096,
    max_prompt_length=3072,
    num_train_epochs=1,          # 1-2 epochs max for DPO
    gradient_accumulation_steps=4,
)
```

### 6.3 — Eval and Iterate

Run all 3 eval layers on the DPO model. Key metrics to watch:
- Layer 3 quality scores should improve (warmth, brevity, no jargon)
- Layer 2 contract compliance must not degrade
- New: "grounding accuracy" — does the model only reference GATC data?

**Phase 6 outcome:** Explainer voice is polished. Knows what to say AND what not to say.

---

## Phase 7: Model Bake-Off (Week 4-5, parallel with Phase 6)

**Why:** Gemma 3n E2B may not be the best base for each role. Test alternatives.

### 7.1 — Candidates

| Role | Candidate | Params | GGUF Q4_K_M | Why test it |
|------|-----------|--------|-------------|-------------|
| **Interpreter** | Qwen2.5-3B | 3B | ~1.8 GB | Strong structured output, smaller |
| **Interpreter** | Llama 3.2-3B | 3B | ~1.8 GB | Good instruction following |
| **Explainer** | Phi-4-mini | 3.8B | ~2.2 GB | Strong reasoning, good voice |
| **Explainer** | Llama 3.2-3B | 3B | ~1.8 GB | Good conversational quality |
| **Both** | Gemma 3n E2B | 2B effective | ~2.8 GB | Current baseline |

### 7.2 — Bake-Off Process

For each candidate:
1. Fine-tune with same data (interpreter or explainer set)
2. Same LoRA config (adjust target modules per architecture)
3. Run same 3-layer eval
4. Measure: GGUF size, inference latency, RAM peak, eval scores

### 7.3 — Decision Matrix

| Metric | Weight | Notes |
|--------|--------|-------|
| Eval Layer 1 (schema) | 25% | Must be >95% for interpreter |
| Eval Layer 2 (contract) | 25% | Must pass all safety checks |
| Eval Layer 3 (quality) | 25% | Warmth, brevity, no jargon |
| GGUF size | 10% | Smaller = more devices supported |
| Inference speed | 10% | Tokens/sec on target phone |
| RAM usage | 5% | Must fit device tier constraints |

**Phase 7 outcome:** Best model per role selected. May end up with different base models for Interpreter vs Explainer.

---

## Phase 8: Athlete Memory Integration (Week 5-6)

**Why:** Persistent memory makes Josi feel like a real coach who remembers you.

### 8.1 — Rust Engine: AthleteMemory Struct

In `mivalta-ai-rust` (separate repo), add:
```rust
pub struct AthleteMemory {
    pub key_facts: Vec<MemoryFact>,    // max 15
    pub patterns: Vec<String>,          // max 5
    pub coaching_notes: Vec<String>,    // max 5
}

pub struct MemoryFact {
    pub fact: String,
    pub source: MemorySource,           // Conversation, Behavior, ProfileChange
    pub confidence: f64,                // 0.0-1.0
    pub learned_at: DateTime<Utc>,
}
```

Persistence: new `athlete_memory` table in SQLite Vault.

### 8.2 — Port memory_extractor.py to Kotlin

Your `shared/memory_extractor.py` (363 lines) is the blueprint:
- Pattern matching for sports, time preferences, injuries, lifestyle, goals
- Confidence-weighted deduplication
- Conflict resolution (newer wins)
- Hard caps: 15 facts, 5 patterns, 5 coaching notes

Port to Kotlin, trigger post-conversation.

### 8.3 — Memory in Prompt Builder

The Kotlin `PromptBuilder` formats memory for the LLM:
```
MEMORY:
- Primary sport: running
- Prefers morning sessions
- Has knee issue (old injury, manageable)
- Typical availability: 45-60 minutes
- Responds well to data-driven explanations
```

Budget: ~400 tokens max within the 4K context.

### 8.4 — Retrain with Memory Examples

Use `augment_memory_training_data.py` + new examples from Phase 4.
Both models should learn to:
- **Interpreter:** Use memory to skip clarification (knows sport, knows time preference)
- **Explainer:** Personalize tone naturally (not "I remember you said...")

**Phase 8 outcome:** Josi remembers athletes across sessions. Feels like a real coach.

---

## Phase 9: Confidence Gate for Interpreter (Week 6)

**Why:** The Interpreter sometimes guesses wrong. A confidence signal lets the system handle uncertainty gracefully.

### 9.1 — Add Confidence Field

Update `shared/schemas/gatc_request.schema.json`:
```json
"confidence": {
  "type": "number",
  "minimum": 0.0,
  "maximum": 1.0,
  "description": "Model's self-assessed confidence in the parsed action and fields."
}
```

### 9.2 — Train Confidence Awareness

Add training examples with explicit confidence:
- Clear intent ("I want a 45-minute easy run") → confidence: 0.95
- Moderate clarity ("maybe a run today?") → confidence: 0.6
- Ambiguous ("help me today") → confidence: 0.3

### 9.3 — Runtime Behavior

| Confidence | Behavior |
|-----------|----------|
| > 0.8 | Execute action directly |
| 0.5 - 0.8 | Execute, but Explainer adds "Did I understand correctly?" |
| < 0.5 | Force `clarify` action |

**Phase 9 outcome:** Graceful handling of ambiguity. Fewer wrong actions.

---

## Phase 10: Device Optimization & Shipping (Week 6-7)

### 10.1 — Progressive Device Tiers

| Tier | RAM | Config | Context | Models |
|------|-----|--------|---------|--------|
| **Tier 1** (budget) | 2-4 GB | Single model + LoRA swap, sequential | 2K | SmolLM2-1.7B fallback |
| **Tier 2** (mid) | 4-6 GB | Two models, sequential | 4K | Gemma 3n Q4_K_M or bake-off winner |
| **Tier 3** (flagship) | 6+ GB | Two models, parallel | 4K | Full quality |

Detect tier at app startup. Load appropriate config.

### 10.2 — OTA Model Updates

- Version manifest: `models.json` on Hetzner Object Storage
- App checks on launch (or weekly)
- Delta downloads if possible (LoRA adapters are small — swap adapter, not base model)
- Fallback: if download fails, keep previous version

### 10.3 — Contract Tests as CI Gate

Golden test set: 200 critical scenarios (subset of test_prompts_1000.json).
Run on every model export. Binary pass/fail. Model doesn't ship if any fail.

```bash
python scripts/evaluate_gemma3n_v4.py --golden-only --strict
# Exit code 0 = ship, non-zero = block
```

### 10.4 — Device Testing Matrix

Test on 3 physical devices:
- Flagship: Samsung S24 Ultra / Pixel 9 Pro (12 GB)
- Mid-range: Pixel 7a / Samsung A54 (6 GB)
- Budget: Pixel 4a / older Samsung (4 GB)

Measure: latency (time-to-first-token, tokens/sec), RAM peak, battery drain.

**Phase 10 outcome:** Ships to production. Works on real devices. Updates without app store releases.

---

## Expanded Eval Harness (runs throughout)

Add to the existing 3-layer eval:

| Layer | What | When added |
|-------|------|-----------|
| Layer 1: Schema | Valid JSON, correct types, required fields | Exists |
| Layer 2: Contract | I6 guardrails, tier gating, no invented numbers | Exists |
| Layer 3: Quality | Warmth, brevity, no jargon, no deflection | Exists |
| **Layer 4: Grounding** | Explainer only references GATC data, no hallucination | Phase 5 |
| **Layer 5: Memory** | Uses memory naturally, never invents, newer wins | Phase 8 |
| **Layer 6: Confidence** | Interpreter confidence correlates with actual accuracy | Phase 9 |

---

## Summary Timeline

| Week | Phase | GPU hours | Deliverable |
|------|-------|-----------|-------------|
| 1 | 0: Infra + 1: First models | 4-6 hrs | Two working GGUFs, baseline scores |
| 1-2 | 2: GBNF grammar | 2 hrs (retrain) | 100% valid JSON |
| 2 | 3: 4K context | 4 hrs (retrain) | Models at 4K, memory + GATC room |
| 2-3 | 4: Data quality push | 4-6 hrs | 3-4x training data, retrained models |
| 3-4 | 5: GATC grounding | 4 hrs | Explainer grounded in real decisions |
| 4-5 | 6: DPO | 4-6 hrs | Polished Explainer voice |
| 4-5 | 7: Model bake-off | 10-15 hrs | Best model per role identified |
| 5-6 | 8: Memory | 4 hrs | Persistent athlete memory |
| 6 | 9: Confidence gate | 2 hrs | Graceful ambiguity handling |
| 6-7 | 10: Ship | 0 (CPU) | OTA, device tiers, CI gate |
| **Total** | | **~40-55 hrs** | **~€60-85 GPU cost** |

---

## What to Do Right Now (Today)

1. **Rent a GPU** — Hetzner GEX44 or RunPod A100. You need one to start Phase 1.
2. **Run `setup_hetzner.sh`** on the GPU server — installs deps + downloads Gemma 3n.
3. **Train your first models** — `finetune_gemma3n.py train --mode interpreter` and `--mode explainer`.
4. **Get baseline scores** — `evaluate_gemma3n_v4.py`. These numbers are your starting point.

Everything else follows from there. The plan is designed so each phase ships independently — you don't need all 10 to see improvement. Phase 1 alone gives you working models. Phase 2 gives you bulletproof JSON. Phase 5 gives you the biggest quality leap.

---

## Team Split (if 2-3 people)

| Person | Focus | Phases |
|--------|-------|--------|
| **ML Engineer** | Training, eval, DPO, bake-off | 1, 3, 4, 6, 7 |
| **Systems/Mobile** | GBNF grammar, Rust memory, Kotlin ports, device tiers | 2, 8, 9, 10 |
| **Domain Expert** | Gold data, preference pairs, eval review | 4.3, 6.1 |

If solo: follow the phase order. Each builds on the previous.

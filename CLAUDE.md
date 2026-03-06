# MiValta Science Engine — Project Context for Claude

## What this project IS
MiValta is an AI-powered sports coaching app. The "science engine" is the backend
that generates personalized training plans and powers the coaching chat (Josi).
Everything runs ON-DEVICE (Android + iPhone) — no cloud, no API calls.

## Hardware
- **Training server**: has an NVIDIA GPU (RTX-class). Use it for fine-tuning.
- **Target devices**: Android (12GB+ RAM), iPhone (6GB+ RAM)
- **On-device inference**: llama.cpp, GGUF quantized models

## Architecture (critical — read this)

```
User message
    → Josi Interpreter (LLM, on-device)
        → GATCRequest JSON (action, sport, goal, constraints)
    → Router (code, not LLM)
        → GATC Engine (Rust) — generates workouts, applies zone caps
        → Viterbi Engine (Rust, HMM) — infers readiness from biometrics
    → Josi Coach (same LLM, different system prompt)
        → Coaching text grounded in engine decisions
    → Phone UI
```

### Key engines (Rust):
- **GATC Engine**: Generates workouts. Outputs: target_zone, duration, structure_label,
  load_score (ULS = d × fz × mi × Md × Mc), phase, meso_day
- **Viterbi Engine**: Hidden Markov Model for readiness. Outputs:
  - state: Recovered / Productive / Accumulated / Overreached / IllnessRisk
  - level: Green / Yellow / Orange / Red
  - confidence: 0.0-1.0
  - data_tier: None / Minimal / Basic / Standard / Good / Full / Enhanced
- **Zone capping**: Viterbi readiness gates max zone (Green→Z6, Yellow→Z4, Orange→Z3, Red→Z2)
- **Load factors**: Z1=0.8, Z2=1.0, Z3=1.5, Z4=2.0, Z5=2.5, Z6=3.0, Z7=3.2, Z8=2.8

### LLM (Josi):
- Model: Qwen3-4B (iPhone) or Qwen3-8B (Android), LoRA fine-tuned
- Two modes via system prompt: Interpreter (JSON output) + Coach (text output)
- Training: `training/scripts/finetune_qwen3.py`
- Data: `training/data/train_v6_unified.jsonl` (unified interpreter + coach)
- Grounded data: `training/data/grounded/` (examples with GATC/Viterbi context)
- Conversations: `training/data/conversations/*.conv`
- Knowledge cards: `knowledge/gatc/*.md`

### ChatContext schema: `shared/schemas/chat_context.schema.json`
Contains: readiness, planned_session, athlete_memory, history, profile_summary

## Training pipeline
- Multi-turn conversations are UNROLLED into single-output examples (not skipped)
- Coach examples include CONTEXT block with: readiness, confidence, max_zone,
  zone_cap_reason, zone_load_factor, meso_position, meso_day, weekly_load
- The model MUST understand GATC and Viterbi outputs to explain engine decisions
- `generate_grounded_convs.py` generates training data from knowledge cards

## Key directories
- `src/` — Rust engine code (GATC, Viterbi, workout generation)
- `training/scripts/` — Fine-tuning scripts, dataset generators
- `training/data/` — JSONL training data, .conv conversations
- `training/models/` — Base models + fine-tuned checkpoints
- `knowledge/` — Coaching knowledge cards (markdown)
- `shared/` — Schemas, shared Python utilities
- `docs/` — Architecture docs, integration specs

## Rules for Claude sessions
1. ALWAYS check what hardware is available before assuming (nvidia-smi, etc)
2. The LLM must be GATC+Viterbi aware — it explains engine decisions to athletes
3. Multi-turn coaching is essential — coaches have conversations, not monologues
4. Training data quality > quantity. Every example must be grounded in real context
5. On-device constraints: 4096 token context, 200 token output, single response per call
6. The app supports Dutch and English — training data includes both languages

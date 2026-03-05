# Josi v6 — Developer README

> Everything your developer needs to download, load, and run the Josi v6 on-device coaching LLM.

---

## What's in the Bundle

One download, two files:

| File | What it is | Size |
|------|-----------|------|
| `josi-v6-q4_k_m.gguf` | Fine-tuned Qwen3-4B, quantized Q4_K_M | ~2.3 GB |
| `knowledge.json` | 114 coaching knowledge cards | ~153 KB |

**Model:** Single LLM, two modes (interpreter + coach) — switched via system prompt.
**Inference:** 100% on-device via llama.cpp. No network calls during chat.
**Languages:** Dutch, English, 100+ languages natively supported.

---

## Download

### Manifest (check for updates)
```
GET http://144.76.62.249/models/josi-v6-manifest.json
```

The manifest contains version, checksums (SHA-256), and the bundle URL. Fetch this first to check if the user already has the latest version.

### Bundle (model + knowledge in one zip)
```
GET http://144.76.62.249/models/josi-v6-bundle.zip
```

Extract the zip to get both files. The GGUF is stored uncompressed in the zip (no CPU cost to extract).

### curl
```bash
curl -LO http://144.76.62.249/models/josi-v6-bundle.zip
unzip josi-v6-bundle.zip
# -> josi-v6-q4_k_m.gguf + knowledge.json
```

### App Download Flow
1. App installs (~50 MB, no models bundled)
2. First launch: fetch manifest -> check version
3. Download bundle zip (~2.3 GB) with progress bar
4. Stream-extract zip -> GGUF + knowledge.json on device
5. Verify SHA-256 checksums (from manifest)
6. Cache locally. Re-download only when manifest version changes.

---

## Architecture: One Model, Two Modes

```
User message
    |
    v
[INTERPRETER MODE] -- system prompt: interpreter -- output: GATCRequest JSON
    |
    v
[ROUTER] (code, not LLM) -- decides what happens next
    |
    +-- create_workout --> dispatch JSON to Rust engine (no coach call)
    +-- replan         --> dispatch JSON to Rust engine (no coach call)
    +-- clarify        --> show clarify_message to user (no coach call)
    +-- explain        --> run COACH MODE with /think
    +-- answer_question -> router decides /think or /no_think
                              |
                              v
                    [COACH MODE] -- system prompt: coach -- output: plain text
                              |
                              v
                    Strip <think> tags -> dialogue governor -> show to user
```

Both modes use the **same GGUF file**, loaded once. You switch modes by changing the system prompt.

---

## Chat Template: ChatML

Qwen3 uses ChatML format. Every prompt must follow this structure:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}

{context_string}<|im_end|>
<|im_start|>assistant
```

For multi-turn conversations, include prior turns between system and the current user message:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
previous user message<|im_end|>
<|im_start|>assistant
previous assistant response<|im_end|>
<|im_start|>user
current message

CONTEXT:
- Readiness: Green (Recovered)
- Session: Z2 60min "Continuous Z2 60min" (base phase)
- Sport: running

[KNOWLEDGE]

Zone 2 is your aerobic base zone...<|im_end|>
<|im_start|>assistant
```

### CRITICAL: Stop Tokens

**You MUST set stop tokens or the model will generate forever**, inventing fake user messages and replying to itself.

```
Stop tokens: ["<|im_end|>", "<|endoftext|>"]
```

In llama.cpp: `llama_sampler_add_stop()` or `antiprompt` parameter. Token ID for `<|im_end|>` is 151645.

---

## System Prompts

### Interpreter System Prompt

The interpreter translates user messages into structured JSON (GATCRequest schema). It never generates coaching text.

```
You are Josi's Interpreter — you translate athlete messages into structured GATC requests.

TASK: Read the user message and athlete state, then output ONLY valid JSON matching
the GATCRequest schema. No markdown fences. No explanation. Just JSON.

LANGUAGE: The user may write in any language (Dutch, English, etc.). Always output
JSON with English field names and enum values. The clarify_message field should
match the user's language.

ACTIONS (action field):
- create_workout: user wants a new workout -> extract sport, time, goal, constraints
- replan: user wants to change/skip/swap a planned session -> extract replan_type
- explain: user asks about THEIR specific session, readiness, week, or plan
- answer_question: general coaching or education question
- clarify: you cannot determine the action or required fields are missing
```

Full prompt: `training/prompts/josi_v6_interpreter.txt`

### Coach System Prompt

The coach generates the friendly text the user actually reads.

```
You are Josi, MiValta's AI coaching assistant.

LANGUAGE: Always respond in the same language the athlete uses.

PERSONALITY:
- Empathetic and warm
- Direct and honest — no filler
- You ARE the coach

DIALOGUE RULES:
- Answer first, always
- Maximum 1 question per response
- Keep responses under 100 words
- Use simple language

BOUNDARIES:
- NEVER prescribe, create, or modify training yourself
- NEVER invent zones, durations, paces, or power numbers
- NEVER reference internal systems

OUTPUT: Plain coaching text only. No JSON. No markdown fences.
```

Full prompt: `training/prompts/josi_v6_coach.txt`

---

## Inference Parameters

| Parameter | Interpreter | Coach (simple) | Coach (complex) |
|-----------|------------|----------------|-----------------|
| max_tokens (nPredict) | 200 | 200 | 400 |
| temperature | 0.3 | 0.5 | 0.5 |
| top_p | 0.9 | 0.9 | 0.9 |
| context (nCtx) | 4096 | 4096 | 4096 |
| stop tokens | `<\|im_end\|>`, `<\|endoftext\|>` | same | same |
| thinking mode | `/no_think` | `/no_think` | `/think` |
| GBNF grammar | yes (gatc_request.gbnf) | no | no |

### Thinking Mode

Append `/think` or `/no_think` to the user message content to control Qwen3's reasoning:

- **`/no_think`** — fast, no internal reasoning (interpreter always, simple coach questions)
- **`/think`** — deep reasoning in `<think>...</think>` tags (complex questions, injury, "why?" questions)

The **router** (your code, not the LLM) decides which mode based on the interpreter's output:

```
Always /think:  action == "explain"
/think if complex:  action == "answer_question" AND message contains
                    "why", "waarom", "should i", "moet ik", "injury",
                    "blessure", "pain", "pijn", "plan", "instead"
Otherwise:  /no_think
```

**Always strip `<think>...</think>` tags** from coach output before showing to the user.

---

## GATCRequest JSON Schema (Interpreter Output)

The interpreter outputs JSON matching this schema:

```json
{
  "action": "create_workout",       // REQUIRED: "create_workout" | "replan" | "explain" | "answer_question" | "clarify"
  "free_text": "Give me a 45min run", // REQUIRED: original user message
  "sport": "run",                   // "run" | "bike" | "ski" | "skate" | "strength" | "other"
  "time_available_min": 45,         // integer, minutes
  "goal": "endurance",              // "endurance" | "threshold" | "vo2" | "recovery" | "strength" | "race_prep"
  "constraints": {
    "fatigue_hint": "ok",           // "fresh" | "ok" | "tired" | "very_tired"
    "injury": "none",              // "none" | "knee" | "back" | "ankle" | "shoulder" | "other"
    "no_intensity": false
  },
  "replan_type": null,              // for replan: "skip_today" | "swap_days" | "reschedule" | etc.
  "question": null,                 // for explain/answer_question
  "missing": null,                  // for clarify: ["sport"]
  "clarify_message": null           // for clarify: "Which sport?" (in user's language)
}
```

### Required fields per action:
| Action | Required fields |
|--------|----------------|
| `create_workout` | action, sport, free_text |
| `replan` | action, replan_type, free_text |
| `explain` | action, question, free_text |
| `answer_question` | action, question, free_text |
| `clarify` | action, missing, clarify_message, free_text |

### GBNF Grammar (Interpreter Only)

Use `shared/schemas/gatc_request.gbnf` to constrain the interpreter's output. This guarantees valid JSON — zero parse failures. Load the grammar file and pass it to llama.cpp's `grammar` parameter on every interpreter call.

---

## Knowledge Cards (knowledge.json)

### Structure

```json
{
  "version": "v6",
  "total_entries": 114,
  "entries": [
    {
      "id": "balance_v4__busy_periods",
      "card": "balance_v4",
      "section": "busy_periods",
      "topics": ["life_balance", "time_management"],
      "keywords": ["busy", "time", "balance", "work", "life", "stress"],
      "content": "Here's something important: scaling back during busy periods..."
    }
  ]
}
```

### How to Use Knowledge Cards

1. **Load at startup** — parse `knowledge.json` once when the model loads
2. **Select per turn** — match the user's message keywords against card keywords
3. **Inject into prompt** — append a `[KNOWLEDGE]` block to the context string

### Selection Algorithm

For each user message:
1. Lowercase the message
2. Score each knowledge entry: count keyword hits (max 6 points) + sport match bonus (+3) or mismatch penalty (-5)
3. Exclude internal cards (`josi_personas_v1`, `planner_policy_v4`)
4. Sort by score descending, deduplicate by card name, take top 3
5. Format as `[KNOWLEDGE]\n\n{content1}\n\n{content2}\n\n{content3}`

### Injection Point

Append the knowledge block to the context string in BOTH interpreter and coach prompts:

```
<|im_start|>user
What is Zone 2?
/no_think

CONTEXT:
- Readiness: Green (Recovered)
- Sport: running

[KNOWLEDGE]

Zone 2 is your aerobic base zone. It should feel comfortable — you can hold a full
conversation. This is where most of your endurance development happens...<|im_end|>
```

Without knowledge injection, the model gives generic answers. With it, Josi gives grounded, sport-science-accurate coaching.

---

## Action -> What To Do (Router)

| Interpreter action | Run coach? | Dispatch to engine? | What the user sees |
|--------------------|-----------|--------------------|--------------------|
| `create_workout` | No | Yes: `createTodayWorkout(sport, time, goal, fatigue)` | Workout from engine |
| `replan` | No | Yes: `replan(type, reason)` | Rescheduled plan |
| `clarify` | No | No | `clarify_message` field from interpreter |
| `explain` | Yes (/think) | No | Coach text explaining their session/readiness |
| `answer_question` | Yes | No | Coach text answering their question |

### Fallback

If the interpreter returns unparseable output (should be impossible with GBNF grammar, but just in case): treat as `answer_question` and run the coach.

---

## Context String Builder

Build this from app state and append to the user message:

```
CONTEXT:
- Readiness: Green (Recovered)
- Session: Z2 60min "Continuous Z2 60min" (base phase)
- Sport: running
- Level: intermediate
```

Include whatever athlete state you have. The interpreter uses this to infer sport, extract context, and decide actions. The coach uses it to give personalized responses.

---

## Dialogue Governor (Post-Processing)

Apply to coach output before displaying:
1. Count question marks
2. If more than 1 question, keep only the last one
3. This enforces "answer first, ask at most one follow-up"

---

## Tier Restrictions

| Action | Monitor (free) | Advisor | Coach |
|--------|---------------|---------|-------|
| `answer_question` | no chat | yes | yes |
| `explain` | no chat | yes | yes |
| `clarify` | no chat | yes | yes |
| `create_workout` | no chat | yes | yes |
| `replan` | no chat | no | yes |

Monitor tier has no Josi chat at all — the app handles it directly.

---

## Zone Gating (Rust Engine Enforces)

| Readiness | Max Allowed Zone |
|-----------|-----------------|
| Green | Z6 |
| Yellow | Z4 |
| Orange | Z3 |
| Red | Z2 |

The Rust engine enforces these limits. The LLM never prescribes workouts — it only routes to the engine.

---

## Expected Latency (On-Device)

| Call | Latency |
|------|---------|
| Interpreter (/no_think + GBNF) | ~400ms |
| Coach simple (/no_think) | ~600ms |
| Coach complex (/think) | ~1200ms |
| Coach skipped (create_workout, replan, clarify) | 0ms |

~40% of messages skip the coach call entirely.

---

## Quick Examples

### "Give me a 45-minute run" (create_workout)

**Interpreter output:**
```json
{"action": "create_workout", "sport": "run", "time_available_min": 45, "free_text": "Give me a 45-minute run"}
```
**Router:** skip coach -> dispatch to Rust engine -> show workout

### "What is Zone 2?" (answer_question, simple)

**Interpreter output:**
```json
{"action": "answer_question", "question": "What is Zone 2?", "free_text": "What is Zone 2?"}
```
**Router:** no complex keywords -> coach with `/no_think`
**Coach output:** "Zone 2 is your aerobic base — a comfortable pace where you can hold a conversation. Most of your endurance gains happen here."

### "Waarom is mijn readiness rood?" (answer_question, complex, Dutch)

**Interpreter output:**
```json
{"action": "answer_question", "question": "Waarom is mijn readiness rood?", "free_text": "Waarom is mijn readiness rood?"}
```
**Router:** "waarom" detected -> coach with `/think`
**Coach output:** "Je readiness is rood omdat je lichaam nog herstelt van recente belasting. Neem vandaag rust — een korte wandeling is prima, maar sla intensieve training even over."

### "I want a workout" (clarify — no sport)

**Interpreter output:**
```json
{"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport are you training for today?", "free_text": "I want a workout"}
```
**Router:** show `clarify_message`, wait for response.

---

## Files Reference

| File | Location | Purpose |
|------|----------|---------|
| Interpreter prompt | `training/prompts/josi_v6_interpreter.txt` | System prompt for JSON mode |
| Coach prompt | `training/prompts/josi_v6_coach.txt` | System prompt for text mode |
| GBNF grammar | `shared/schemas/gatc_request.gbnf` | Grammar constraint for interpreter |
| GATCRequest schema | `shared/schemas/gatc_request.schema.json` | JSON schema for interpreter output |
| ChatContext schema | `shared/schemas/chat_context.schema.json` | Input context schema |
| Knowledge cards | `knowledge/generated/knowledge.json` | Ships with model |
| Full integration guide | `docs/JOSI_INTEGRATION_GUIDE.md` | Detailed Kotlin code examples |

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Model generates fake user messages | Missing stop tokens | Set `<\|im_end\|>` as stop token on EVERY call |
| All messages in one bubble | Wrong chat format | Use ChatML with `<\|im_start\|>` / `<\|im_end\|>` tags |
| Model prescribes workouts | Single-call instead of two-step | Implement interpreter -> router -> coach pipeline |
| Response cut off mid-sentence | nPredict too low | Use 200 for simple, 400 for complex |
| Garbled JSON from interpreter | No GBNF grammar | Load `gatc_request.gbnf` and pass to grammar param |
| Generic coaching answers | No knowledge injection | Select and inject knowledge cards per turn |

---

## Summary

1. **Download** the bundle zip (manifest for version check, bundle for model + knowledge)
2. **Load** the single GGUF file with llama.cpp (nCtx=4096, stop tokens set)
3. **Load** knowledge.json at startup
4. **Every user message:** interpreter call (JSON) -> router -> optional coach call (text)
5. **Inject** knowledge cards into both prompts
6. **Strip** `<think>` tags from coach output
7. **Apply** dialogue governor (max 1 question)
8. **Dispatch** create_workout/replan to Rust engine

That's it. One model, two system prompts, knowledge cards for grounding.

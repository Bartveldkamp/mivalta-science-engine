# Josi LLM Integration Guide — mivalta-ai-rust / Kotlin

> For the MiValta mobile developer integrating the Josi on-device LLM.

---

## Overview

Josi v4 uses a **dual-model architecture** — two fine-tuned **Gemma 3n E2B** models (GGUF Q4_K_M, ~2.6 GB each) that run **100% on-device** via llama.cpp / llama.android. **No network calls during chat.**

| Model | File | Size | Purpose |
|-------|------|------|---------|
| **Interpreter** | `josi-v4-interpreter-q4_k_m.gguf` | ~2.6 GB | Translates user messages into structured GATCRequest JSON for the engine |
| **Explainer** | `josi-v4-explainer-q4_k_m.gguf` | ~2.6 GB | Generates friendly coaching text the user sees |

From the user's perspective, this is **one Josi AI coach**. Behind the scenes:

```
┌──────────────────────────────────────────────────────────────┐
│                      KOTLIN APP                               │
│                                                               │
│  1. Build ChatContext (user message + athlete state)          │
│  2. Format prompts (Gemma template, same context for both)   │
│                                                               │
│  3a. INTERPRETER model → GATCRequest JSON                    │
│      (action, sport, replan_type, question, etc.)            │
│                                                               │
│  3b. EXPLAINER model → plain coaching text                   │
│      (the message the user reads)                            │
│                                                               │
│  4. Parse GATCRequest → dispatch to Rust engine if needed    │
│  5. Apply dialogue governor to explainer text                │
│  6. Render coaching text + engine results to user            │
└──────────────────────────────────────────────────────────────┘
```

Both models receive the **same user message + context**. They can run in parallel. The interpreter tells the engine what to do; the explainer tells the user what's happening.

**Base model:** `google/gemma-3n-E2B-it` — 6B params (2B effective via MatFormer)
**Quantization:** Q4_K_M (4-bit)
**Chat template:** Gemma (`<start_of_turn>` / `<end_of_turn>`, native system role supported)

### Runtime Constraints (on-device)

| Parameter | Value |
|-----------|-------|
| Context cap | 1024 tokens |
| Output cap (interpreter) | 150 tokens |
| Output cap (explainer) | 150 tokens |
| Temperature | 0.45 (range: 0.4-0.5) |
| Top-p | 0.9 |
| Total model size | ~5.2 GB (two GGUF files) |
| Effective RAM per model | ~2 GB |

### Tier Architecture

| Tier | Chat | Josi | Capabilities |
|------|------|------|-------------|
| **Monitor** (free) | No | No | App-direct tools only: `get_user_status`, `log_workout`, `get_recent_workouts` |
| **Advisor** | Yes | Yes | Explain workouts/zones, create TODAY's workout only, no plans |
| **Coach** | Yes | Yes | Full: plans, replans, multi-day coaching, all tools |

**Monitor has no Josi chat.** The app handles monitor tools directly without LLM involvement.

---

## Step 1: Download the Models

**Total size:** ~5.2 GB (two files)
**Format:** GGUF Q4_K_M (4-bit quantized, for llama.cpp / llama.android)

### Download Links

Download both files to your development machine:

| Model | File | Size |
|-------|------|------|
| Interpreter | `josi-v4-interpreter-q4_k_m.gguf` | ~2.6 GB |
| Explainer | `josi-v4-explainer-q4_k_m.gguf` | ~2.6 GB |

**Option A — Direct HTTP download from training server:**
```bash
# Replace <SERVER_IP> with the IP provided by Bart
curl -LO http://<SERVER_IP>:8079/josi-v4-interpreter-q4_k_m.gguf
curl -LO http://<SERVER_IP>:8079/josi-v4-explainer-q4_k_m.gguf
```

**Option B — Hetzner Object Storage (when available):**
```bash
curl -LO https://objects.mivalta.com/models/josi-v4-interpreter-q4_k_m.gguf
curl -LO https://objects.mivalta.com/models/josi-v4-explainer-q4_k_m.gguf
```

**In the Kotlin app:**
```kotlin
// Model URLs (use Object Storage URLs for production app)
val interpreterUrl = "https://objects.mivalta.com/models/josi-v4-interpreter-q4_k_m.gguf"
val explainerUrl   = "https://objects.mivalta.com/models/josi-v4-explainer-q4_k_m.gguf"

// Local storage on device
val interpreterFile = File(context.filesDir, "josi-v4-interpreter-q4_k_m.gguf")
val explainerFile   = File(context.filesDir, "josi-v4-explainer-q4_k_m.gguf")
```

**Manifest URL:** `https://objects.mivalta.com/models/josi-v4-manifest.json`
The manifest contains current file names, sizes, and SHA-256 checksums. Use it to check for updates.

**App download flow (end user):**
1. User installs app (~50 MB, no models bundled)
2. First launch: "Setting up your coach..." progress bar
3. Both models download from Hetzner Object Storage (~5.2 GB total)
4. Downloads can run in parallel
5. Verify SHA-256 checksums after download
6. Cached locally, never re-downloaded unless model version updates (check manifest)
7. All inference runs on-device via llama.cpp — no network calls

Load with [llama.android](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.android) or equivalent llama.cpp Kotlin/JNI bindings.

---

## Step 2: Build the ChatContext

Before each inference call, build the context from app state. Both models receive the same context.

```kotlin
data class ChatContext(
    val athleteId: String,
    val tier: String,           // "advisor" or "coach" (never "monitor")
    val mode: String,           // same as tier
    val personaId: String,      // "balanced" | "direct" | "technical" | "encouraging"
    val today: String,          // "2026-02-10"
    val readiness: Readiness,
    val hasSessionContext: Boolean,
    val plannedSession: PlannedSession?,  // required if hasSessionContext=true
    val history: List<ChatMessage>,
    val profileSummary: ProfileSummary?,
)

data class Readiness(
    val state: String,      // "Recovered" | "Productive" | "Accumulated" | "Overreached" | "IllnessRisk"
    val confidence: Double, // 0.0–1.0
    val level: String,      // "Green" | "Yellow" | "Orange" | "Red"
    val dataTier: String,   // "None" | "Minimal" | "Basic" | "Standard" | "Good" | "Full" | "Enhanced"
)

data class PlannedSession(
    val intent: String,           // "R" | "Z1" .. "Z8" | "OFF" | "REST"
    val targetZone: String,       // "R" | "Z1" .. "Z8"
    val targetDurationMin: Int,
    val structureLabel: String,   // e.g. "4 x 5min Z4 / 3min Z1"
    val phase: String?,           // "base" | "build" | "peak" | "taper" | "recovery"
    val mesoDay: Int?,            // 1–28
)
```

### Context String Builder

Format the context block appended to the user message for both models:

```kotlin
fun buildContextString(ctx: ChatContext): String {
    val sb = StringBuilder("CONTEXT:\n")
    sb.append("- Readiness: ${ctx.readiness.level} (${ctx.readiness.state})\n")

    if (ctx.hasSessionContext && ctx.plannedSession != null) {
        val s = ctx.plannedSession
        sb.append("- Session: ${s.targetZone} ${s.targetDurationMin}min ")
        sb.append("\"${s.structureLabel}\"")
        s.phase?.let { sb.append(" ($it phase)") }
        sb.append("\n")
    }

    ctx.profileSummary?.let {
        sb.append("- Sport: ${it.sport}\n")
        sb.append("- Level: ${it.level}\n")
    }

    return sb.toString()
}
```

---

## Step 3: Format Prompts (Gemma 3n Template)

Both models use the same Gemma 3n chat template but with **different system prompts**.

### Gemma 3n Chat Template

```
<start_of_turn>system
{system_prompt}<end_of_turn>
<start_of_turn>user
{user_message}

{context_string}<end_of_turn>
<start_of_turn>model
```

### Interpreter System Prompt

The interpreter classifies intent and extracts structured data for the engine:

```
You are Josi's Interpreter — you translate athlete messages into structured GATC requests.

TASK: Read the user message and athlete state, then output ONLY valid JSON matching the GATCRequest schema. No markdown fences. No explanation. Just JSON.

ACTIONS (action field):
- create_workout: user wants a new workout → extract sport, time, goal, constraints
- replan: user wants to change/skip/swap a planned session → extract replan_type
- explain: user asks about THEIR specific session, readiness, week, or plan shown in CONTEXT → extract question
- answer_question: general coaching or education question with NO specific session/readiness context → extract question
- clarify: you cannot determine the action or required fields are missing → ask ONE question

HOW TO CHOOSE explain vs answer_question:
- If CONTEXT contains Session, Readiness, or Week info AND the user asks about it → "explain"
- If the user asks a general knowledge question (zones, recovery, training concepts) → "answer_question"

ENUMS — use these exact values only:
- sport: "run", "bike", "ski", "skate", "strength", "other"
- replan_type: "skip_today", "swap_days", "reschedule", "reduce_intensity", "illness", "travel", "goal_change"
- goal: "endurance", "threshold", "vo2", "recovery", "strength", "race_prep"
- fatigue_hint: "fresh", "ok", "tired", "very_tired"

REQUIRED FIELDS BY ACTION:
- create_workout: action, sport, free_text (+ time_available_min if mentioned)
- replan: action, replan_type, free_text (+ sport if in context)
- explain: action, question, free_text
- answer_question: action, question, free_text
- clarify: action, missing, clarify_message, free_text

RULES:
- ALWAYS include free_text with the original user message
- NEVER output markdown fences — raw JSON only
- NEVER invent workouts, zones, durations, paces, or power numbers
- NEVER output coaching text — only structured JSON
- If the user mentions pain, chest pain, dizziness, or medical symptoms: action="clarify", missing=["medical_clearance"]
- Infer sport from CONTEXT block when available (e.g., "Sport: running" → sport="run")
- Infer fatigue_hint from user language (e.g., "I'm exhausted" → "very_tired")
```

### Explainer System Prompt

The explainer generates the friendly coaching text the user reads:

```
You are Josi, MiValta's AI coaching assistant.

PERSONALITY:
- Empathetic and warm — you genuinely care about the athlete
- Direct and honest — no filler, no corporate speak
- You ARE the coach — never recommend other apps, coaches, or services

DIALOGUE RULES:
- Answer first, always. Lead with the substance of your response.
- Maximum 1 question per response. Most responses need zero questions.
- Keep responses under 100 words. Be concise, not verbose.
- Use simple language. Explain like a trusted friend who happens to be a coach.
- Do NOT end with a question unless absolutely necessary.

SAFETY:
- If the athlete mentions pain, chest pain, dizziness, or medical red flags: tell them to stop and seek medical attention immediately.

BOUNDARIES:
- NEVER prescribe, create, or modify training yourself
- Explain decisions made by the coaching engine only
- NEVER invent zones, durations, paces, or power numbers
- NEVER reference internal systems (algorithm, viterbi, hmm, acwr, gatc, ewma, tss, ctl, atl, tsb)
- NEVER use technical jargon: periodization, mesocycle, microcycle, macrocycle, supercompensation, vo2max, lactate threshold, ftp

OUTPUT: Plain coaching text only. No JSON. No markdown fences.
```

### Kotlin Prompt Builder

```kotlin
fun buildGemmaPrompt(
    systemPrompt: String,
    history: List<ChatMessage>,
    userMessage: String,
    context: String,
): String {
    val sb = StringBuilder()

    // System turn (native system role in Gemma 3n)
    sb.append("<start_of_turn>system\n")
    sb.append(systemPrompt)
    sb.append("<end_of_turn>\n")

    // Multi-turn history
    for (msg in history) {
        val role = if (msg.role == "assistant") "model" else msg.role
        sb.append("<start_of_turn>$role\n")
        sb.append(msg.message)
        sb.append("<end_of_turn>\n")
    }

    // Current user message with context
    sb.append("<start_of_turn>user\n")
    sb.append(userMessage).append("\n\n")
    sb.append(context)
    sb.append("<end_of_turn>\n")

    // Generation prompt
    sb.append("<start_of_turn>model\n")
    return sb.toString()
}
```

---

## Step 4: Run Dual Inference

Both models receive the same user message + context. They can run in parallel or sequentially.

### Inference Parameters

```kotlin
val interpreterParams = LlamaParams(
    nPredict = 150,       // Output cap: 150 tokens
    temperature = 0.45f,  // Range: 0.4-0.5
    topP = 0.9f,
    nCtx = 1024,          // Context cap: 1024 tokens
)

val explainerParams = LlamaParams(
    nPredict = 150,       // Output cap: 150 tokens
    temperature = 0.45f,
    topP = 0.9f,
    nCtx = 1024,
)
```

### Parallel Inference (recommended if RAM allows)

If the device has enough RAM (~4 GB free), run both models in parallel:

```kotlin
suspend fun runJosi(userMessage: String, ctx: ChatContext): JosiResult {
    val contextStr = buildContextString(ctx)

    val interpreterPrompt = buildGemmaPrompt(
        INTERPRETER_SYSTEM_PROMPT, ctx.history, userMessage, contextStr)
    val explainerPrompt = buildGemmaPrompt(
        EXPLAINER_SYSTEM_PROMPT, ctx.history, userMessage, contextStr)

    // Run both in parallel
    val (interpreterRaw, explainerRaw) = coroutineScope {
        val i = async { interpreterModel.generate(interpreterPrompt, interpreterParams) }
        val e = async { explainerModel.generate(explainerPrompt, explainerParams) }
        i.await() to e.await()
    }

    val gatcRequest = parseGATCRequest(interpreterRaw)
    val coachingText = governDialogue(explainerRaw.trim())

    return JosiResult(gatcRequest, coachingText)
}
```

### Sequential Inference (lower RAM devices)

If RAM is tight, load one model at a time:

```kotlin
suspend fun runJosiSequential(userMessage: String, ctx: ChatContext): JosiResult {
    val contextStr = buildContextString(ctx)

    // Run interpreter first (need structured intent for engine)
    val interpreterPrompt = buildGemmaPrompt(
        INTERPRETER_SYSTEM_PROMPT, ctx.history, userMessage, contextStr)
    val interpreterRaw = withModel(interpreterFile) { model ->
        model.generate(interpreterPrompt, interpreterParams)
    }

    // Then run explainer
    val explainerPrompt = buildGemmaPrompt(
        EXPLAINER_SYSTEM_PROMPT, ctx.history, userMessage, contextStr)
    val explainerRaw = withModel(explainerFile) { model ->
        model.generate(explainerPrompt, explainerParams)
    }

    val gatcRequest = parseGATCRequest(interpreterRaw)
    val coachingText = governDialogue(explainerRaw.trim())

    return JosiResult(gatcRequest, coachingText)
}
```

---

## Step 5: Parse Interpreter Output → GATCRequest

The interpreter outputs raw JSON. Parse it into a structured GATCRequest.

### GATCRequest Schema

```kotlin
data class GATCRequest(
    val action: String,             // "create_workout" | "replan" | "explain" | "answer_question" | "clarify"
    val freeText: String,           // Original user message (always present)
    val sport: String?,             // "run" | "bike" | "ski" | "skate" | "strength" | "other"
    val timeAvailableMin: Int?,     // Duration in minutes (if user mentioned)
    val goal: String?,              // "endurance" | "threshold" | "vo2" | "recovery" | "strength" | "race_prep"
    val fatigueHint: String?,       // "fresh" | "ok" | "tired" | "very_tired"
    val replanType: String?,        // "skip_today" | "swap_days" | "reschedule" | ... (replan only)
    val question: String?,          // User's question text (explain/answer_question)
    val missing: List<String>?,     // Missing fields (clarify only)
    val clarifyMessage: String?,    // Clarification question (clarify only)
)
```

### Kotlin Parser

```kotlin
fun parseGATCRequest(raw: String): GATCRequest? {
    val text = raw.trim()
    if (text.isEmpty()) return null

    // Extract first JSON object (handles any preamble text)
    val jsonStr = extractFirstJsonObject(text) ?: return null

    return try {
        val obj = JSONObject(jsonStr)
        val action = obj.getString("action")
        val freeText = obj.getString("free_text")

        GATCRequest(
            action = action,
            freeText = freeText,
            sport = obj.optString("sport", null),
            timeAvailableMin = if (obj.has("time_available_min")) obj.getInt("time_available_min") else null,
            goal = obj.optString("goal", null),
            fatigueHint = obj.optString("fatigue_hint", null),
            replanType = obj.optString("replan_type", null),
            question = obj.optString("question", null),
            missing = obj.optJSONArray("missing")?.let { arr ->
                (0 until arr.length()).map { arr.getString(it) }
            },
            clarifyMessage = obj.optString("clarify_message", null),
        )
    } catch (e: Exception) {
        null
    }
}

private fun extractFirstJsonObject(text: String): String? {
    val start = text.indexOf('{')
    if (start == -1) return null

    var depth = 0
    var inString = false
    var escape = false

    for (i in start until text.length) {
        val ch = text[i]
        if (escape) { escape = false; continue }
        if (ch == '\\') { escape = true; continue }
        if (ch == '"') { inString = !inString; continue }
        if (inString) continue
        if (ch == '{') depth++
        if (ch == '}') { depth--; if (depth == 0) return text.substring(start, i + 1) }
    }

    // Try closing unclosed braces
    if (depth > 0) {
        val candidate = text.substring(start) + "}".repeat(depth)
        try { JSONObject(candidate); return candidate } catch (_: Exception) {}
    }
    return null
}
```

### Fallback on Parse Failure

If `parseGATCRequest()` returns `null`, treat as `answer_question`:

```kotlin
val FALLBACK_REQUEST = GATCRequest(
    action = "answer_question",
    freeText = userMessage,
    sport = null,
    timeAvailableMin = null,
    goal = null,
    fatigueHint = null,
    replanType = null,
    question = userMessage,
    missing = null,
    clarifyMessage = null,
)
```

---

## Step 6: Dialogue Governor (Explainer Output)

Apply the dialogue governor to the explainer's coaching text to enforce answer-first and max 1 question:

```kotlin
fun governDialogue(text: String): String {
    var message = text.trim()

    // Count questions
    val questionCount = message.count { it == '?' }
    if (questionCount > 1) {
        // Keep only the last question (usually the follow-up)
        val sentences = message.split(Regex("(?<=[.!?])\\s+"))
        val nonQuestions = sentences.filter { !it.trimEnd().endsWith("?") }
        val questions = sentences.filter { it.trimEnd().endsWith("?") }
        message = (nonQuestions + questions.takeLast(1)).joinToString(" ")
    }

    return message
}
```

### Explainer Fallback

If the explainer produces empty or garbled output:

```kotlin
val FALLBACK_MESSAGE = "I'm here to help with your training. What would you like to know?"
```

---

## Step 7: Dispatch GATCRequest → Engine

Map the interpreter's GATCRequest action to the appropriate engine call.

### Action → Engine Dispatch Map

| Action | Engine Call | Tier Required |
|--------|-----------|---------------|
| `create_workout` | `rustEngine.createTodayWorkout(sport, time, goal, ...)` | Advisor+ |
| `replan` | `rustEngine.replan(replanType, reason, ...)` | Coach |
| `explain` | `rustEngine.explainWorkout(question)` | Advisor+ |
| `answer_question` | No engine call — use explainer text only | Advisor+ |
| `clarify` | No engine call — use `clarifyMessage` as response | Advisor+ |

### Kotlin Dispatch

```kotlin
data class JosiResult(
    val gatcRequest: GATCRequest?,
    val coachingText: String,
)

fun handleJosiResult(result: JosiResult, tier: String) {
    val req = result.gatcRequest

    if (req == null) {
        // Interpreter failed — show explainer text only
        showMessage(result.coachingText)
        return
    }

    when (req.action) {
        "clarify" -> {
            // Use interpreter's clarify message (more specific) or explainer text
            showMessage(req.clarifyMessage ?: result.coachingText)
        }

        "answer_question" -> {
            // No engine call needed — explainer text is the response
            showMessage(result.coachingText)
        }

        "explain" -> {
            // Explainer text is the response; optionally call engine for extra data
            showMessage(result.coachingText)
        }

        "create_workout" -> {
            // Check tier
            if (tier == "monitor") {
                showMessage("Workout creation requires an Advisor or Coach subscription.")
                return
            }
            // Show coaching text while engine works
            showMessage(result.coachingText)
            // Dispatch to Rust engine
            val engineResult = rustEngine.createTodayWorkout(
                sport = req.sport ?: return,
                timeMin = req.timeAvailableMin,
                goal = req.goal,
                fatigueHint = req.fatigueHint,
            )
            showWorkoutResult(engineResult)
        }

        "replan" -> {
            // Coach only
            if (tier != "coach") {
                showMessage("Replanning requires a Coach subscription.")
                return
            }
            showMessage(result.coachingText)
            val engineResult = rustEngine.replan(
                type = req.replanType ?: "skip_today",
                reason = req.freeText,
            )
            showReplanResult(engineResult)
        }
    }
}
```

### Tier Tool Allowlists (must match Rust `tool_dispatcher.rs`)

| Action | Monitor | Advisor | Coach |
|--------|---------|---------|-------|
| `answer_question` | - | yes | yes |
| `explain` | - | yes | yes |
| `clarify` | - | yes | yes |
| `create_workout` | - | yes | yes |
| `replan` | - | - | yes |

---

## Step 8: Zone Gating (Enforced by Rust)

The Rust engine caps workout zones based on readiness. Josi is trained to respect these, but the engine enforces them as a hard guard:

| Readiness | Max Zone |
|-----------|----------|
| Green | Z6 |
| Yellow | Z4 |
| Orange | Z3 |
| Red | Z2 |

**Exception:** Coach PlanEngine is exempt — it can schedule Z7/Z8 within a periodized plan.

---

## Schemas Reference

All schemas live in `shared/schemas/` in the `mivalta-science-engine` repo:

| File | Purpose |
|------|---------|
| `chat_context.schema.json` | Input schema — what the app sends to both models |
| `tool_dispatch.json` | Maps actions to engine calls, tier allowlists, fallbacks |

These are the **contracts** between the Kotlin app, Josi, and the Rust engine.

---

## Quick Reference: Example Flows

### Example 1: "What's my workout today?" (Advisor, Green)

**User message:** "What's my workout today?"

**Context:**
```
CONTEXT:
- Readiness: Green (Recovered)
- Session: Z2 60min "Continuous Z2 60min" (base phase)
- Sport: running
- Level: intermediate
```

**Interpreter outputs:**
```json
{"action": "explain", "question": "What's my workout today?", "free_text": "What's my workout today?"}
```

**Explainer outputs:**
```
Today you have a 60-minute easy aerobic session in Zone 2. The goal is to build your base endurance at a comfortable pace — you should be able to hold a conversation throughout.
```

**App action:** Show explainer text to user. No engine call needed.

---

### Example 2: "Give me a 45-minute run" (Advisor, Green)

**User message:** "Give me a 45-minute run"

**Interpreter outputs:**
```json
{"action": "create_workout", "sport": "run", "time_available_min": 45, "free_text": "Give me a 45-minute run"}
```

**Explainer outputs:**
```
Sure! Let me set up a 45-minute run for you. Based on your green readiness, this is a great day to get some quality work in.
```

**App action:**
1. Show explainer text
2. Call `rustEngine.createTodayWorkout(sport="run", timeMin=45)`
3. Show workout result

---

### Example 3: "I'm feeling tired, can we skip today?" (Coach, Yellow)

**Interpreter outputs:**
```json
{"action": "replan", "replan_type": "skip_today", "free_text": "I'm feeling tired, can we skip today?", "fatigue_hint": "tired"}
```

**Explainer outputs:**
```
I understand — your body is telling you something. Skipping today is the smart call. I'll adjust your schedule so nothing gets lost.
```

**App action:**
1. Show explainer text
2. Call `rustEngine.replan(type="skip_today", reason="...")`
3. Show rescheduled plan

---

### Example 4: "I want a workout" (no sport in context)

**Interpreter outputs:**
```json
{"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport are you training for today?", "free_text": "I want a workout"}
```

**Explainer outputs:**
```
I'd love to help! Which sport are you training for today — running, cycling, skating, skiing, or strength?
```

**App action:** Show explainer text (or interpreter's `clarify_message`). Wait for user response.

---

### Example 5: "Create a training plan" (Advisor — blocked)

**Interpreter outputs:**
```json
{"action": "clarify", "missing": ["tier_upgrade"], "clarify_message": "Training plans require a Coach subscription.", "free_text": "Create a training plan"}
```

**Explainer outputs:**
```
Training plans are available with the Coach tier. I can help you with today's workout or answer questions about your training. What would you like to do?
```

**App action:** Show explainer text. No engine call.

---

## Memory Management

Since both models are ~2.6 GB each, memory management on mobile is critical:

```kotlin
class JosiModelManager(private val context: Context) {
    private var interpreterModel: LlamaModel? = null
    private var explainerModel: LlamaModel? = null

    // Option A: Keep both loaded (devices with 8+ GB RAM)
    fun loadBoth() {
        interpreterModel = LlamaModel(interpreterFile.absolutePath, interpreterParams)
        explainerModel = LlamaModel(explainerFile.absolutePath, explainerParams)
    }

    // Option B: Load one at a time (devices with 4-6 GB RAM)
    suspend fun runSequential(prompt: String, isInterpreter: Boolean): String {
        val file = if (isInterpreter) interpreterFile else explainerFile
        val params = if (isInterpreter) interpreterParams else explainerParams
        return withContext(Dispatchers.IO) {
            val model = LlamaModel(file.absolutePath, params)
            try {
                model.generate(prompt, params)
            } finally {
                model.close()
            }
        }
    }

    fun release() {
        interpreterModel?.close()
        explainerModel?.close()
        interpreterModel = null
        explainerModel = null
    }
}
```

---

## Migration from v3 (Single Model → Dual Model)

| Change | v3 (single model) | v4 (dual model) |
|--------|-------------------|-----------------|
| Models | 1 GGUF file | 2 GGUF files (interpreter + explainer) |
| Download size | ~2.8 GB | ~5.2 GB |
| Model output | LLMIntent JSON (combined) | Interpreter: GATCRequest JSON, Explainer: plain text |
| Parsing | Parse one JSON output | Parse interpreter JSON + use explainer text directly |
| Engine dispatch | From `tool_call` in LLMIntent | From `action` in GATCRequest |
| Chat template | Gemma (`<start_of_turn>`) | Same (both models) |
| Temperature | 0.45 | 0.45 (both models) |

**Key integration changes:**
1. Download and store **two** model files instead of one
2. Run **two** inference calls per user message (parallel or sequential)
3. Parse **GATCRequest** from interpreter (replaces LLMIntent)
4. Use **explainer text directly** as user-facing message (no more `message` field in JSON)
5. Map GATCRequest `action` to engine calls (replaces `tool_call` dispatch)

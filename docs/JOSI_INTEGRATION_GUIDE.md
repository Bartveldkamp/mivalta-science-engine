# Josi LLM Integration Guide — mivalta-ai-rust / Kotlin

> For the MiValta mobile developer integrating the Josi on-device LLM.

---

## Overview

Josi v6 uses a **single-model architecture** — one fine-tuned **Qwen3-4B** model (GGUF Q4_K_M, ~2.5 GB) that runs **100% on-device** via llama.cpp / llama.android. **No network calls during chat.**

The same model handles two modes via different system prompts:

| Mode | System Prompt | Output | Purpose |
|------|--------------|--------|---------|
| **Interpreter** | `josi_v6_interpreter.txt` | GATCRequest JSON | Translates user messages into structured engine commands |
| **Coach** | `josi_v6_coach.txt` | Plain coaching text | Generates friendly coaching text the user sees |

From the user's perspective, this is **one Josi AI coach**. Behind the scenes:

```
+-----------------------------------------------------------------+
|                      KOTLIN APP                                  |
|                                                                  |
|  1. Build ChatContext (user message + athlete state)             |
|  2. Load SINGLE model (josi-v6-q4_k_m.gguf, ~2.5 GB)          |
|                                                                  |
|  3. INTERPRETER call (same model, interpreter system prompt)     |
|     -> GATCRequest JSON (action, sport, replan_type, etc.)      |
|                                                                  |
|  4. ROUTER (code, not LLM) — decides if coach call is needed    |
|     - create_workout -> skip coach, return JSON to engine        |
|     - replan         -> skip coach, return JSON to engine        |
|     - clarify        -> skip coach, use clarify_message          |
|     - explain        -> run coach call                           |
|     - answer_question-> run coach call                           |
|                                                                  |
|  5. COACH call (same model, coach system prompt)                 |
|     -> plain coaching text (the message the user reads)          |
|                                                                  |
|  6. Parse GATCRequest -> dispatch to Rust engine if needed       |
|  7. Apply dialogue governor to coach text                        |
|  8. Render coaching text + engine results to user                |
+-----------------------------------------------------------------+
```

**Base model:** `Qwen/Qwen3-4B` — 4B parameters
**Quantization:** Q4_K_M (4-bit)
**Chat template:** ChatML (`<|im_start|>` / `<|im_end|>`, same as Qwen2.5)
**Languages:** Dutch, English, 100+ languages natively

### Runtime Constraints (on-device)

| Parameter | Interpreter | Coach |
|-----------|------------|-------|
| Context cap | 4096 tokens | 4096 tokens |
| Output cap | 200 tokens | 200 tokens |
| Temperature | 0.3 | 0.5 |
| Top-p | 0.9 | 0.9 |

| Parameter | Value |
|-----------|-------|
| Model file | `josi-v6-q4_k_m.gguf` |
| Model size | ~2.5 GB |
| Effective RAM | ~3 GB |
| Interpreter latency | ~300ms |
| Coach latency | ~500ms |
| Coach skipped | ~40% of messages |

### Tier Architecture

| Tier | Chat | Josi | Capabilities |
|------|------|------|-------------|
| **Monitor** (free) | No | No | App-direct tools only: `get_user_status`, `log_workout`, `get_recent_workouts` |
| **Advisor** | Yes | Yes | Explain workouts/zones, create TODAY's workout only, no plans |
| **Coach** | Yes | Yes | Full: plans, replans, multi-day coaching, all tools |

**Monitor has no Josi chat.** The app handles monitor tools directly without LLM involvement.

---

## Step 1: Download the Model

**Total size:** ~2.5 GB (single file)
**Format:** GGUF Q4_K_M (4-bit quantized, for llama.cpp / llama.android)

### Download

| File | Size | Purpose |
|------|------|---------|
| `josi-v6-q4_k_m.gguf` | ~2.5 GB | Both interpreter (JSON) + coach (text) modes |

**Direct HTTP download from training server:**
```bash
# Replace <SERVER_IP> with the IP provided by Bart
curl -LO http://<SERVER_IP>/models/josi-v6-q4_k_m.gguf
```

**In the Kotlin app:**
```kotlin
// Model URL
val modelUrl = "http://<SERVER_IP>/models/josi-v6-q4_k_m.gguf"

// Local storage on device
val modelFile = File(context.filesDir, "josi-v6-q4_k_m.gguf")
```

**Manifest URL:** `http://<SERVER_IP>/models/josi-v6-manifest.json`
The manifest contains current file name, size, and SHA-256 checksum. Use it to check for updates.

**App download flow (end user):**
1. User installs app (~50 MB, no models bundled)
2. First launch: "Setting up your coach..." progress bar
3. Single model downloads (~2.5 GB)
4. Verify SHA-256 checksum after download
5. Cached locally, never re-downloaded unless model version updates (check manifest)
6. All inference runs on-device via llama.cpp — no network calls

Load with [llama.android](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.android) or equivalent llama.cpp Kotlin/JNI bindings.

---

## Step 2: Build the ChatContext

Before each inference call, build the context from app state. Both modes receive the same context.

```kotlin
data class ChatContext(
    val athleteId: String,
    val tier: String,           // "advisor" or "coach" (never "monitor")
    val mode: String,           // same as tier
    val personaId: String,      // "balanced" | "direct" | "technical" | "encouraging"
    val today: String,          // "2026-02-20"
    val readiness: Readiness,
    val hasSessionContext: Boolean,
    val plannedSession: PlannedSession?,  // required if hasSessionContext=true
    val history: List<ChatMessage>,
    val profileSummary: ProfileSummary?,
)

data class Readiness(
    val state: String,      // "Recovered" | "Productive" | "Accumulated" | "Overreached" | "IllnessRisk"
    val confidence: Double, // 0.0-1.0
    val level: String,      // "Green" | "Yellow" | "Orange" | "Red"
    val dataTier: String,   // "None" | "Minimal" | "Basic" | "Standard" | "Good" | "Full" | "Enhanced"
)

data class PlannedSession(
    val intent: String,           // "R" | "Z1" .. "Z8" | "OFF" | "REST"
    val targetZone: String,       // "R" | "Z1" .. "Z8"
    val targetDurationMin: Int,
    val structureLabel: String,   // e.g. "4 x 5min Z4 / 3min Z1"
    val phase: String?,           // "base" | "build" | "peak" | "taper" | "recovery"
    val mesoDay: Int?,            // 1-28
)
```

### Context String Builder

Format the context block appended to the user message for both modes:

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

## Step 3: Format Prompts (ChatML Template)

Both modes use the same ChatML template (native to Qwen3) but with **different system prompts**.

### ChatML Template

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}

{context_string}<|im_end|>
<|im_start|>assistant
```

### System Prompts

**Interpreter** — classifies intent and extracts structured data for the engine.
See `training/prompts/josi_v6_interpreter.txt` for the full prompt.

**Coach** — generates friendly coaching text the user reads.
See `training/prompts/josi_v6_coach.txt` for the full prompt.

### Kotlin Prompt Builder

```kotlin
fun buildChatMLPrompt(
    systemPrompt: String,
    history: List<ChatMessage>,
    userMessage: String,
    context: String,
): String {
    val sb = StringBuilder()

    // System turn
    sb.append("<|im_start|>system\n")
    sb.append(systemPrompt)
    sb.append("<|im_end|>\n")

    // Multi-turn history
    for (msg in history) {
        sb.append("<|im_start|>${msg.role}\n")
        sb.append(msg.message)
        sb.append("<|im_end|>\n")
    }

    // Current user message with context
    sb.append("<|im_start|>user\n")
    sb.append(userMessage).append("\n\n")
    sb.append(context)
    sb.append("<|im_end|>\n")

    // Generation prompt
    sb.append("<|im_start|>assistant\n")
    return sb.toString()
}
```

---

## Step 4: Run Inference (Single Model, Two Modes)

The SAME model is loaded ONCE and used for both calls. Only the system prompt changes.

### Inference Parameters

```kotlin
val interpreterParams = LlamaParams(
    nPredict = 200,       // Output cap: 200 tokens
    temperature = 0.3f,   // Low temperature for deterministic JSON
    topP = 0.9f,
    nCtx = 4096,          // Context cap: 4096 tokens
)

val coachParams = LlamaParams(
    nPredict = 200,       // Output cap: 200 tokens
    temperature = 0.5f,   // Higher temperature for natural coaching text
    topP = 0.9f,
    nCtx = 4096,
)
```

### Sequential Inference (recommended)

Run interpreter first, then decide if coach call is needed:

```kotlin
suspend fun runJosi(userMessage: String, ctx: ChatContext): JosiResult {
    val contextStr = buildContextString(ctx)

    // Step 1: Interpreter call (always runs)
    val interpreterPrompt = buildChatMLPrompt(
        INTERPRETER_SYSTEM_PROMPT, ctx.history, userMessage, contextStr)
    val interpreterRaw = model.generate(interpreterPrompt, interpreterParams)
    val gatcRequest = parseGATCRequest(interpreterRaw)

    // Step 2: Router — does this action need the coach?
    val needsCoach = gatcRequest?.action in setOf("explain", "answer_question")

    val coachingText = if (needsCoach) {
        // Build coach prompt with interpreter context appended
        val coachContext = contextStr + "\n\n[INTERPRETER]\n" + interpreterRaw.trim()
        val coachPrompt = buildChatMLPrompt(
            COACH_SYSTEM_PROMPT, ctx.history, userMessage, coachContext)
        val raw = model.generate(coachPrompt, coachParams)
        governDialogue(stripThinkingTags(raw.trim()))
    } else {
        // For clarify: use interpreter's clarify_message
        // For create_workout/replan: no coaching text needed
        gatcRequest?.clarifyMessage ?: ""
    }

    return JosiResult(gatcRequest, coachingText)
}
```

### Strip Thinking Tags

Qwen3 may produce `<think>...</think>` tags. Strip them before displaying to user:

```kotlin
fun stripThinkingTags(text: String): String {
    return text.replace(Regex("<think>.*?</think>", RegexOption.DOT_MATCHES_ALL), "").trim()
}
```

---

## Step 5: Parse Interpreter Output -> GATCRequest

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
    // Strip thinking tags first
    val text = stripThinkingTags(raw).trim()
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

## Step 6: Dialogue Governor (Coach Output)

Apply the dialogue governor to the coach's text to enforce answer-first and max 1 question:

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

### Coach Fallback

If the coach produces empty or garbled output:

```kotlin
val FALLBACK_MESSAGE = "I'm here to help with your training. What would you like to know?"
```

---

## Step 7: Dispatch GATCRequest -> Engine

Map the interpreter's GATCRequest action to the appropriate engine call.

### Action -> Engine Dispatch Map

| Action | Engine Call | Needs Coach | Tier Required |
|--------|-----------|-------------|---------------|
| `create_workout` | `rustEngine.createTodayWorkout(...)` | No | Advisor+ |
| `replan` | `rustEngine.replan(...)` | No | Coach |
| `explain` | No engine call | Yes | Advisor+ |
| `answer_question` | No engine call | Yes | Advisor+ |
| `clarify` | No engine call | No (use clarify_message) | Advisor+ |

### Kotlin Dispatch

```kotlin
data class JosiResult(
    val gatcRequest: GATCRequest?,
    val coachingText: String,
)

fun handleJosiResult(result: JosiResult, tier: String) {
    val req = result.gatcRequest

    if (req == null) {
        // Interpreter failed — show fallback message
        showMessage(FALLBACK_MESSAGE)
        return
    }

    when (req.action) {
        "clarify" -> {
            // Use interpreter's clarify message (already in user's language)
            showMessage(req.clarifyMessage ?: FALLBACK_MESSAGE)
        }

        "answer_question" -> {
            // Coach text is the response
            showMessage(result.coachingText)
        }

        "explain" -> {
            // Coach text is the response
            showMessage(result.coachingText)
        }

        "create_workout" -> {
            if (tier == "monitor") {
                showMessage("Workout creation requires an Advisor or Coach subscription.")
                return
            }
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
            if (tier != "coach") {
                showMessage("Replanning requires a Coach subscription.")
                return
            }
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
| `chat_context.schema.json` | Input schema — what the app sends to both modes |
| `gatc_request.schema.json` | Interpreter output contract (unchanged from v5) |
| `tool_dispatch.json` | Maps actions to engine calls, tier allowlists, fallbacks |

These are the **contracts** between the Kotlin app, Josi, and the Rust engine.

---

## Memory Management

Since v6 uses a single ~2.5 GB model (vs two ~2.6 GB models in v4), memory management is simpler:

```kotlin
class JosiModelManager(private val context: Context) {
    private var model: LlamaModel? = null

    // Load model once — used for both interpreter and coach calls
    fun loadModel() {
        model = LlamaModel(modelFile.absolutePath, LlamaParams(nCtx = 4096))
    }

    // Run interpreter call
    suspend fun runInterpreter(prompt: String): String {
        return withContext(Dispatchers.IO) {
            model!!.generate(prompt, interpreterParams)
        }
    }

    // Run coach call (same model, different params)
    suspend fun runCoach(prompt: String): String {
        return withContext(Dispatchers.IO) {
            model!!.generate(prompt, coachParams)
        }
    }

    fun release() {
        model?.close()
        model = null
    }
}
```

**Key advantage over v5:** One model loaded in memory, not two. Uses ~3 GB RAM instead of ~4 GB.

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

**Router:** action=explain -> run coach call

**Coach outputs:**
```
Today you have a 60-minute easy aerobic session in Zone 2. The goal is to build your base endurance at a comfortable pace — you should be able to hold a conversation throughout.
```

**App action:** Show coach text to user. No engine call needed.

---

### Example 2: "Give me a 45-minute run" (Advisor, Green)

**Interpreter outputs:**
```json
{"action": "create_workout", "sport": "run", "time_available_min": 45, "free_text": "Give me a 45-minute run"}
```

**Router:** action=create_workout -> skip coach call

**App action:**
1. Call `rustEngine.createTodayWorkout(sport="run", timeMin=45)`
2. Show workout result

---

### Example 3: "Ik voel me moe, kan ik vandaag overslaan?" (Coach, Yellow)

**Interpreter outputs:**
```json
{"action": "replan", "replan_type": "skip_today", "free_text": "Ik voel me moe, kan ik vandaag overslaan?", "sport": "run", "constraints": {"fatigue_hint": "tired"}}
```

**Router:** action=replan -> skip coach call

**App action:**
1. Call `rustEngine.replan(type="skip_today", reason="...")`
2. Show rescheduled plan

---

### Example 4: "I want a workout" (no sport in context)

**Interpreter outputs:**
```json
{"action": "clarify", "missing": ["sport"], "clarify_message": "Which sport are you training for today?", "free_text": "I want a workout"}
```

**Router:** action=clarify -> skip coach call, use clarify_message

**App action:** Show clarify_message. Wait for user response.

---

### Example 5: "Wat is Zone 2?" (Dutch, Advisor)

**Interpreter outputs:**
```json
{"action": "answer_question", "question": "Wat is Zone 2?", "free_text": "Wat is Zone 2?"}
```

**Router:** action=answer_question -> run coach call

**Coach outputs (in Dutch):**
```
Zone 2 is je aerobe zone — ideaal voor het opbouwen van je uithoudingsvermogen. Je moet nog comfortabel kunnen praten tijdens het trainen. Het voelt rustig maar doelgericht.
```

---

## Migration from v5 (Dual Model -> Single Model)

| Change | v5 (dual model) | v6 (single model) |
|--------|-----------------|-------------------|
| Models | 2 GGUF files | 1 GGUF file |
| Download size | ~1.87 GB | ~2.5 GB |
| Base model | Qwen2.5-1.5B-Instruct | Qwen3-4B |
| Params per task | 1.5B | 4B (2.7x more) |
| Chat template | ChatML | ChatML (same) |
| Interpreter output | GATCRequest JSON | GATCRequest JSON (same schema) |
| Coach output | Plain text | Plain text (same) |
| Temperature | 0.45 (both) | 0.3 interpreter / 0.5 coach |
| Context cap | 2048 tokens | 4096 tokens |
| Output cap | 150 tokens | 200 tokens |
| Dutch | Basic | Native, excellent |
| Memory management | Load 2 models (~4 GB) | Load 1 model (~3 GB) |
| Inference | 2 separate models | Same model, 2 system prompts |

**Key integration changes:**
1. Download and store **one** model file instead of two
2. Load **one** model into memory (simpler, less RAM)
3. Use **same model** for both calls, just switch system prompt
4. Strip `<think>` tags from Qwen3 output (if any)
5. Same GATCRequest schema — no engine changes needed
6. Update temperatures: 0.3 for interpreter, 0.5 for coach
7. Update context cap to 4096 tokens

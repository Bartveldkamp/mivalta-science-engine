# Josi LLM Integration Guide — mivalta-ai-rust / Kotlin

> For the MiValta mobile developer integrating the Josi on-device LLM.

---

## Overview

Josi is a fine-tuned **Gemma 3n E2B** model (GGUF Q4_K_M, ~2.8 GB) that runs **100% on-device** via llama.cpp / llama.android. It produces structured JSON (LLMIntent) that the Rust engine validates and acts on. **No network calls during chat.**

**Model:** `google/gemma-3n-E2B-it` — 6B params (2B effective via MatFormer)
**Quantization:** Q4_K_M (4-bit)
**HF class:** `Gemma3nForConditionalGeneration` + `AutoProcessor` (requires `transformers>=4.53.0`)
**Chat template:** Gemma (`<start_of_turn>` / `<end_of_turn>`, native system role supported)

```
┌─────────────────────────────────────────────────────────┐
│                    KOTLIN APP                            │
│                                                          │
│  1. Build ChatContext (JSON)                              │
│  2. Format system prompt + user message (Gemma template) │
│  3. Run inference (llama.android)                         │
│  4. Parse raw output → LLMIntent (JSON post-processor)   │
│  5. Apply dialogue governor (answer-first, max 1 question)│
│  6. If tool_call: dispatch to Rust engine via FFI         │
│  7. Render message to user                               │
└─────────────────────────────────────────────────────────┘
```

### Runtime Constraints (on-device)

| Parameter | Value |
|-----------|-------|
| Context cap | 1024 tokens |
| Output cap | 150 tokens |
| Temperature | 0.45 (range: 0.4-0.5) |
| Top-p | 0.9 |
| Model size | ~2.8 GB (Q4_K_M) |
| Effective RAM | ~2 GB |

### Tier Architecture

| Tier | Chat | Josi | Capabilities |
|------|------|------|-------------|
| **Monitor** (free) | No | No | App-direct tools only: `get_user_status`, `log_workout`, `get_recent_workouts` |
| **Advisor** | Yes | Yes | Explain workouts/zones, create TODAY's workout only, no plans |
| **Coach** | Yes | Yes | Full: plans, replans, multi-day coaching, all tools |

**Monitor has no Josi chat.** The app handles monitor tools directly without LLM involvement.

---

## Step 1: Download the Model

**Hosting:** Hetzner Object Storage
**Size:** ~2.8 GB
**Format:** GGUF Q4_K_M (4-bit quantized)

```kotlin
// Download on first launch, store in app internal storage
val modelUrl = "https://objects.mivalta.com/models/josi-v4-gemma3n-q4_k_m.gguf"
val modelFile = File(context.filesDir, "josi-v4-gemma3n-q4_k_m.gguf")
```

**App download flow:**
1. User installs app (~50 MB, no model bundled)
2. First launch: "Setting up your coach..." progress bar
3. Model downloads from Hetzner Object Storage (~2.8 GB)
4. Cached locally, never re-downloaded unless model version updates
5. All inference runs on-device via llama.cpp — no network calls

Load with [llama.android](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.android) or equivalent llama.cpp Kotlin/JNI bindings.

---

## Step 2: Build the ChatContext

Before each inference call, build the context from app state. This follows `shared/schemas/chat_context.schema.json`.

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

---

## Step 3: Format the Prompt (Gemma 3n Template)

Gemma 3n **does** support a native system role. Content uses array format with `{"type": "text", "text": "..."}` objects.

### Gemma 3n Chat Template

```
<start_of_turn>system
{system_prompt}<end_of_turn>
<start_of_turn>user
{user_message}

CONTEXT:
- Readiness: {level} ({state})
- Session: {target_zone} {target_duration_min}min "{structure_label}" ({phase} phase)
- Sport: {sport}
- Level: {level}<end_of_turn>
<start_of_turn>model
```

Only include the Session line if `hasSessionContext = true`.

> **Note:** When using HuggingFace transformers, messages use array-format content:
> ```python
> messages = [
>     {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
>     {"role": "user", "content": [{"type": "text", "text": user_message}]},
> ]
> ```
> The model role is `"model"` (not `"assistant"`).

### System Prompt

```
You are Josi, MiValta's AI coaching assistant.

PERSONALITY:
- Empathetic and warm — you genuinely care about the athlete
- Direct and honest — no filler, no corporate speak
- You ARE the coach — never recommend other apps, coaches, or services

DIALOGUE RULES:
- Answer first, always. Lead with the substance of your response.
- Maximum 1 follow-up question per turn.
- Keep responses under 100 words.

MODE: {Advisor|Coach}
{mode_rules}

I6 CONSTRAINTS (always active):
- NEVER prescribe, create, or modify training yourself
- Explain decisions made by the coaching engine only
- NEVER invent zones, durations, paces, or power numbers
- NEVER reference internal systems (algorithm, viterbi, hmm, acwr, gatc, ewma, tss, ctl, atl, tsb)

OUTPUT: Valid LLMIntent JSON.
```

### Mode Rules

**Advisor:**
```
MODE: Advisor
- Explain workouts and zones, answer education questions
- Help create TODAY's workout only via tool_call to create_today_workout
- STRICTLY today only — NEVER discuss future sessions
- NEVER create training plans (Decline with tier upgrade)
- NEVER modify or replan training (Decline with tier upgrade)
```

**Coach:**
```
MODE: Coach
- Full coaching access: explain, plan, replan, review
- May suggest replans via replan_request when readiness changes
- May reference future sessions and weekly/meso structure
- Create plans via tool_call to create_plan
- Replan types: skip_today, swap_days, reschedule, reduce_intensity, illness, travel, goal_change
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

## Step 4: Inference Parameters

```kotlin
val params = LlamaParams(
    nPredict = 150,       // Output cap: 150 tokens
    temperature = 0.45f,  // Range: 0.4-0.5
    topP = 0.9f,
    nCtx = 1024,          // Context cap: 1024 tokens
)
```

---

## Step 5: Parse Model Output → LLMIntent

The model outputs JSON. Gemma 3n E2B (6B raw, 2B effective) has significantly better JSON compliance than SmolLM2-360M, but the post-processor is still needed for edge cases.

### LLMIntent Schema

```kotlin
data class LLMIntent(
    val intent: String,            // "question" | "replan" | "encouragement" | "feedback" | "compliance" | "general" | "blocked" | "medical_red_flag"
    val responseType: String,      // "DailyBrief" | "ExplainWorkout" | "ExplainZone" | "WeeklyReview" | "Encouragement" | "SafetyWarning" | "ReadinessSummary" | "QuestionAnswer" | "Decline"
    val message: String,           // The text to display to the user
    val sourceCards: List<String>, // Knowledge cards that informed the response
    val guardrailTriggered: Boolean,
    val guardrailReason: String?,  // e.g. "i6_violation", "tier_violation"
    val replanRequest: ReplanRequest?,  // Non-null only for coach replan intents
    val toolCall: ToolCall?,       // Non-null when an engine action is needed
)

data class ToolCall(
    val tool: String,   // "get_user_status" | "explain_workout" | "create_today_workout" | "create_plan" | "replan" | "log_workout" | "get_recent_workouts"
    val args: Map<String, Any>,
)

data class ReplanRequest(
    val type: String,              // "skip_today" | "swap_days" | "reschedule" | "reduce_intensity" | "illness" | "travel" | "goal_change"
    val reason: String,
    val mode: String,              // always "coach"
    val readinessAtRequest: String, // "Green" | "Yellow" | "Orange" | "Red"
    val newGoalDate: String?,      // only for goal_change
)
```

### Kotlin Parser (port of `shared/llm_intent_parser.py`)

```kotlin
fun parseLLMIntent(raw: String): LLMIntent? {
    val text = raw.trim()
    if (text.isEmpty()) return null

    // Step 1: Try direct parse
    tryParseJson(text)?.let { return it }

    // Step 2: Fix string concatenation ("a" + "b" → "ab")
    var fixed = fixStringConcatenation(text)

    // Step 3: Extract first JSON object via brace matching
    val candidate = extractFirstJsonObject(fixed) ?: return null

    tryParseJson(candidate)?.let { return it }

    // Step 4: Try fixing concatenation on extracted candidate
    fixed = fixStringConcatenation(candidate)
    return tryParseJson(fixed)
}

private fun fixStringConcatenation(text: String): String {
    val pattern = Regex(""""([^"]*?)"\s*\+\s*"([^"]*?)"""")
    var result = text
    var prev: String? = null
    while (result != prev) {
        prev = result
        result = pattern.replace(result) { "\"${it.groupValues[1]}${it.groupValues[2]}\"" }
    }
    return result
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
        tryParseJson(candidate)?.let { return candidate }
    }
    return null
}

private fun tryParseJson(text: String): LLMIntent? {
    return try {
        val obj = JSONObject(text)
        val intent = obj.getString("intent")
        val responseType = obj.getString("response_type")
        val message = obj.getString("message")
        val sourceCards = obj.getJSONArray("source_cards").let { arr ->
            (0 until arr.length()).map { arr.getString(it) }
        }
        val guardrailTriggered = obj.getBoolean("guardrail_triggered")

        // Validate required fields
        if (message.isEmpty() || sourceCards.isEmpty()) return null

        LLMIntent(
            intent = intent,
            responseType = responseType,
            message = message,
            sourceCards = sourceCards,
            guardrailTriggered = guardrailTriggered,
            guardrailReason = obj.optString("guardrail_reason", null),
            replanRequest = null,  // parse if needed
            toolCall = parseToolCall(obj),
        )
    } catch (e: Exception) {
        null
    }
}

private fun parseToolCall(obj: JSONObject): ToolCall? {
    if (obj.isNull("tool_call")) return null
    val tc = obj.getJSONObject("tool_call")
    return ToolCall(
        tool = tc.getString("tool"),
        args = tc.getJSONObject("args").toMap(),
    )
}
```

### Dialogue Governor (post-parse)

After parsing the LLMIntent, apply the dialogue governor to enforce answer-first and max 1 question:

```kotlin
fun governDialogue(intent: LLMIntent): LLMIntent {
    // Don't govern safety warnings or declines
    if (intent.responseType in listOf("SafetyWarning", "Decline")) return intent

    var message = intent.message

    // Count questions
    val questionCount = message.count { it == '?' }
    if (questionCount > 1) {
        // Keep only the last question (usually the follow-up)
        val sentences = message.split(Regex("(?<=[.!?])\\s+"))
        val nonQuestions = sentences.filter { !it.trimEnd().endsWith("?") }
        val questions = sentences.filter { it.trimEnd().endsWith("?") }
        message = (nonQuestions + questions.takeLast(1)).joinToString(" ")
    }

    return intent.copy(message = message)
}
```

### Fallback on Parse Failure

If `parseLLMIntent()` returns `null`, use a deterministic fallback:

```kotlin
val FALLBACK = LLMIntent(
    intent = "blocked",
    responseType = "Decline",
    message = "I couldn't process that response. Let me try a different approach.",
    sourceCards = listOf("josi_explanations"),
    guardrailTriggered = false,
    guardrailReason = null,
    replanRequest = null,
    toolCall = null,
)
```

---

## Step 6: Tool Dispatch → Rust FFI

When Josi returns a `tool_call`, dispatch it to the Rust engine. The Rust `ToolDispatcher` validates the tool against the tier allowlist before executing.

### Tier Tool Allowlists (must match Rust `tool_dispatcher.rs`)

| Tool | Monitor | Advisor | Coach |
|------|---------|---------|-------|
| `get_user_status` | app-direct | yes | yes |
| `log_workout` | app-direct | yes | yes |
| `get_recent_workouts` | app-direct | yes | yes |
| `explain_workout` | - | yes | yes |
| `create_today_workout` | - | yes | yes |
| `create_plan` | - | - | yes |
| `replan` | - | - | yes |

"app-direct" = the app calls this tool directly without Josi. Monitor has no chat.

### Dispatch Flow

```kotlin
fun handleLLMIntent(intent: LLMIntent, tier: String) {
    // 1. Apply dialogue governor
    val governed = governDialogue(intent)

    // 2. Display message to user
    showMessage(governed.message)

    // 3. If tool_call present, dispatch to Rust
    governed.toolCall?.let { tc ->
        // Rust ToolDispatcher validates tier access
        val result = rustEngine.dispatchTool(tier, tc.tool, tc.args)

        when (tc.tool) {
            "create_today_workout" -> showWorkoutOptions(result)
            "replan"              -> showReplanResult(result)
            "get_user_status"     -> showReadiness(result)
            "log_workout"         -> showConfirmation(result)
            "get_recent_workouts" -> showHistory(result)
            "explain_workout"     -> showExplanation(result)
            "create_plan"         -> showPlan(result)
        }
    }

    // 4. If guardrail triggered, show decline UI
    if (governed.guardrailTriggered) {
        showDeclineUI(governed.guardrailReason)
    }
}
```

### Replan Dispatch (Coach Only)

When `intent = "replan"` and `replanRequest != null`:

```kotlin
if (intent.intent == "replan" && intent.replanRequest != null) {
    // Pass to ReplanExecutor::from_josi() via FFI
    val result = rustEngine.replanFromJosi(intent.replanRequest)
    showReplanResult(result)
}
```

---

## Step 7: Zone Gating (Enforced by Rust)

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
| `llm_intent.schema.json` | Output schema — what Josi produces |
| `chat_context.schema.json` | Input schema — what the app sends to Josi |
| `tool_dispatch.json` | Maps intents to engine calls, tier allowlists, fallbacks |

These are the **contracts** between the Kotlin app, Josi, and the Rust engine.

---

## Quick Reference: Example Flow

**User (Advisor tier, Green readiness) asks:** "What's my workout today?"

**1. App builds Gemma 3n prompt:**
```
<start_of_turn>system
You are Josi, MiValta's AI coaching assistant...
MODE: Advisor...<end_of_turn>
<start_of_turn>user
What's my workout today?

CONTEXT:
- Readiness: Green (Recovered)
- Session: Z2 60min "Continuous Z2 60min" (base phase)
- Sport: running
- Level: intermediate<end_of_turn>
<start_of_turn>model
```

**2. Model outputs:**
```json
{
  "intent": "question",
  "response_type": "ExplainWorkout",
  "message": "Today you have a 60-minute easy aerobic session in Zone 2. The goal is to build your base endurance at a comfortable pace — you should be able to hold a conversation throughout. How are you feeling?",
  "source_cards": ["session_rules", "zone_physiology"],
  "guardrail_triggered": false,
  "guardrail_reason": null,
  "replan_request": null,
  "tool_call": null
}
```

**3. App displays:** the `message` field to the user.

---

**User (Advisor tier) asks:** "Can you create a training plan for next month?"

**Model outputs:**
```json
{
  "intent": "blocked",
  "response_type": "Decline",
  "message": "I can't create training plans in Advisor mode. Plans are available with the Coach tier. I can help you with today's workout — would you like me to explain your session?",
  "source_cards": ["josi_explanations"],
  "guardrail_triggered": true,
  "guardrail_reason": "tier_violation",
  "replan_request": null,
  "tool_call": null
}
```

---

**User (Coach tier, Yellow readiness) asks:** "I'm feeling tired, can we skip today?"

**Model outputs:**
```json
{
  "intent": "replan",
  "response_type": "QuestionAnswer",
  "message": "I understand — your readiness is Yellow today. I'll request a skip for today's session and the engine will adjust your upcoming schedule.",
  "source_cards": ["fatigue_policy", "josi_explanations"],
  "guardrail_triggered": false,
  "guardrail_reason": null,
  "replan_request": {
    "type": "skip_today",
    "reason": "Athlete reports fatigue, readiness Yellow",
    "mode": "coach",
    "readiness_at_request": "Yellow"
  },
  "tool_call": {
    "tool": "replan",
    "args": {"type": "skip_today"}
  }
}
```

**App dispatches** `replan` to Rust `ReplanExecutor::from_josi()`.

---

## Migration from SmolLM2 (v3 → v4)

| Change | v3 (SmolLM2) | v4 (Gemma 3n E2B) |
|--------|-------------|-------------------|
| Model | SmolLM2-360M | Gemma 3n E2B-it |
| Size | ~250 MB | ~2.8 GB |
| Chat template | ChatML (`<\|im_start\|>`) | Gemma (`<start_of_turn>`) |
| System role | Native system message | Native system role (Gemma 3n) |
| Temperature | 0.7 | 0.45 |
| Max output tokens | 120 | 150 |
| Dialogue governor | No | Yes (answer-first, max 1 question) |
| JSON reliability | Needs heavy post-processing | Much better, still use post-processor |
| Reasoning quality | Basic pattern matching | Genuine reasoning about context |

**Key integration changes:**
1. Update model download URL and file size
2. Switch from ChatML to Gemma prompt template
3. Add dialogue governor after JSON parsing
4. Update inference params (temp 0.45, max_tokens 150)

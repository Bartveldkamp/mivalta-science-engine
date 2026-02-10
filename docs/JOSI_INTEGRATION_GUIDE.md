# Josi LLM Integration Guide — mivalta-ai-rust / Kotlin

> For the MiValta mobile developer integrating the Josi on-device LLM.

---

## Overview

Josi is a fine-tuned SmolLM2-360M model (GGUF, 0.25 GB) that runs **on-device** via llama.cpp / llama.android. It produces structured JSON (LLMIntent) that the Rust engine validates and acts on.

```
┌─────────────────────────────────────────────────────────┐
│                    KOTLIN APP                            │
│                                                          │
│  1. Build ChatContext (JSON)                              │
│  2. Format system prompt + user message                  │
│  3. Run inference (llama.android)                         │
│  4. Parse raw output → LLMIntent (JSON post-processor)   │
│  5. If tool_call: dispatch to Rust engine via FFI         │
│  6. Render message to user                               │
└─────────────────────────────────────────────────────────┘
```

### Tier Architecture

| Tier | Chat | Josi | Capabilities |
|------|------|------|-------------|
| **Monitor** (free) | No | No | App-direct tools only: `get_user_status`, `log_workout`, `get_recent_workouts` |
| **Advisor** | Yes | Yes | Explain workouts/zones, create TODAY's workout only, no plans |
| **Coach** | Yes | Yes | Full: plans, replans, multi-day coaching, all tools |

**Monitor has no Josi chat.** The app handles monitor tools directly without LLM involvement.

---

## Step 1: Download the Model

**URL:** `https://cockpit.mivalta.com/models/josi-v3-360M-q4_k_m.gguf`
**Size:** ~250 MB
**Format:** GGUF q4_k_m (4-bit quantized)

```kotlin
// Download on first launch, store in app internal storage
val modelUrl = "https://cockpit.mivalta.com/models/josi-v3-360M-q4_k_m.gguf"
val modelFile = File(context.filesDir, "josi-v3-360M-q4_k_m.gguf")
```

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

## Step 3: Format the System Prompt

The model was trained with the SmolLM2 chat template (`<|im_start|>` / `<|im_end|>`). llama.android handles the template tokens — you provide the messages.

### System Prompt Template

```
You are Josi, MiValta's AI coaching assistant. Style: warm, professional, supportive.

MODE: {Advisor|Coach}
{mode_rules}

RULES (I6 — Invariant 6):
- NEVER prescribe training (\"you should do 5x400m\")
- NEVER create/modify training plans or sessions
- NEVER output zone/duration that contradicts the planned session
- ALWAYS explain what the engine produced; NEVER invent workouts
- Respond in JSON matching the LLMIntent schema
```

### Mode Rules

**Advisor:**
```
MODE: Advisor
- Explain workouts and zones, answer education questions
- Discuss the athlete's personal data (readiness, training load, history)
- Help create TODAY's workout only via tool_call to create_today_workout
- STRICTLY today only — NEVER discuss tomorrow, next week, or future sessions
- NEVER create long-term training plans (Decline with tier upgrade)
- NEVER modify or replan training (Decline with tier upgrade)
- NEVER use prescriptive language ("you should do", "I recommend", "try this")
- Future workout/plan requests: Decline with tier upgrade suggestion
```

**Coach:**
```
MODE: Coach
- Full coaching access: explain, plan, replan, review
- May suggest replans via replan_request when readiness changes
- May reference future sessions and weekly/meso structure
- NEVER prescribe training directly — always use tool_call for actions
- NEVER use prescriptive language ("you should do", "I recommend")
- Explain the training produced by the engine, suggest adjustments via replan
```

### User Message Format

Append context after the user's message:

```
{user_message}

CONTEXT:
- Readiness: {level} ({state})
- Session: {target_zone} {target_duration_min}min "{structure_label}" ({phase} phase)
- Sport: {sport}
- Level: {level}
```

Only include the Session line if `hasSessionContext = true`.

### Full Example (llama.android messages)

```kotlin
val messages = listOf(
    ChatMessage(role = "system", content = systemPrompt),
    // ... conversation history ...
    ChatMessage(role = "user", content = userMessageWithContext),
)
```

---

## Step 4: Parse Model Output → LLMIntent

The model outputs JSON, but small models sometimes produce artifacts. Parse with this logic (port of `shared/llm_intent_parser.py`):

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

## Step 5: Tool Dispatch → Rust FFI

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
    // 1. Display message to user
    showMessage(intent.message)

    // 2. If tool_call present, dispatch to Rust
    intent.toolCall?.let { tc ->
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

    // 3. If guardrail triggered, show decline UI
    if (intent.guardrailTriggered) {
        showDeclineUI(intent.guardrailReason)
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

## Step 6: Zone Gating (Enforced by Rust)

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

**1. App builds prompt:**
```
System: You are Josi... MODE: Advisor... I6 rules...
User: What's my workout today?

CONTEXT:
- Readiness: Green (Recovered)
- Session: Z2 60min "Continuous Z2 60min" (base phase)
- Sport: running
- Level: intermediate
```

**2. Model outputs:**
```json
{
  "intent": "question",
  "response_type": "ExplainWorkout",
  "message": "Today you have a 60-minute easy aerobic session in Zone 2. The goal is to build your base endurance at a comfortable pace — you should be able to hold a conversation throughout.",
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

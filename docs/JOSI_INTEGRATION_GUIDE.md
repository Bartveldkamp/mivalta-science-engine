# Josi LLM Integration Guide — mivalta-ai-rust / Kotlin

> For the MiValta mobile developer integrating the Josi on-device LLM.

---

## Overview

Josi v6 uses a **single-model architecture** — one fine-tuned **Qwen3-8B** model (GGUF Q4_K_M, ~5.0 GB) that runs **100% on-device** via llama.cpp / llama.android. **No network calls during chat.**

The same model handles two modes via different system prompts, with **router-controlled thinking** for quality escalation:

| Mode | System Prompt | Thinking | Output | Purpose |
|------|--------------|----------|--------|---------|
| **Interpreter** | `josi_v6_interpreter.txt` | `/no_think` always | GATCRequest JSON | Translates user messages into structured engine commands |
| **Coach (simple)** | `josi_v6_coach.txt` | `/no_think` | Plain coaching text | Quick answers, confirmations |
| **Coach (complex)** | `josi_v6_coach.txt` | `/think` | Plain coaching text | Injury reasoning, "why?" questions, plan tradeoffs |

**Platform strategy:** Android-first with 8B. iPhone variant (4B) planned separately.

From the user's perspective, this is **one Josi AI coach**. Behind the scenes:

```
+-----------------------------------------------------------------------+
|                        KOTLIN APP                                      |
|                                                                        |
|  1. Build ChatContext (user message + athlete state)                   |
|  2. Load SINGLE model (josi-v6-q4_k_m.gguf, ~5.0 GB)                |
|                                                                        |
|  3. INTERPRETER call (/no_think + GBNF grammar)                       |
|     -> GATCRequest JSON (action, sport, replan_type, etc.)            |
|     -> Grammar-constrained: ALWAYS valid JSON, zero parse failures    |
|                                                                        |
|  4. ROUTER (code, not LLM) — decides next step:                       |
|     - create_workout -> skip coach, dispatch JSON to engine            |
|     - replan         -> skip coach, dispatch JSON to engine            |
|     - clarify        -> skip coach, use clarify_message                |
|     - explain        -> run coach with /think (complex context)        |
|     - answer_question-> router decides /think vs /no_think:            |
|         simple ("What is Z2?")     -> /no_think (~600ms)              |
|         complex ("Why am I tired?") -> /think (~1200ms)               |
|                                                                        |
|  5. COACH call (same model, coach system prompt)                       |
|     -> plain coaching text (the message the user reads)                |
|     -> Strip <think>...</think> tags before display                    |
|                                                                        |
|  6. Parse GATCRequest -> dispatch to Rust engine if needed             |
|  7. Apply dialogue governor to coach text                              |
|  8. Render coaching text + engine results to user                      |
+-----------------------------------------------------------------------+
```

**Base model:** `Qwen/Qwen3-8B` — 8B parameters (Android), `Qwen/Qwen3-4B` (future iPhone)
**Quantization:** Q4_K_M (4-bit)
**Chat template:** ChatML (`<|im_start|>` / `<|im_end|>`, same as Qwen2.5)
**Languages:** Dutch, English, 100+ languages natively
**Thinking mode:** Router-controlled `/think` and `/no_think`

### Runtime Constraints (on-device, Android 12GB)

| Parameter | Interpreter | Coach (simple) | Coach (complex) |
|-----------|------------|----------------|-----------------|
| Context cap | 4096 tokens | 4096 tokens | 4096 tokens |
| Output cap | 200 tokens | 200 tokens | 400 tokens |
| Temperature | 0.3 | 0.5 | 0.5 |
| Top-p | 0.9 | 0.9 | 0.9 |
| Thinking | `/no_think` | `/no_think` | `/think` |
| Grammar | GBNF | none | none |

| Parameter | Value |
|-----------|-------|
| Model file | `josi-v6-q4_k_m.gguf` |
| Model size | ~5.0 GB (8B Android) / ~2.5 GB (4B iPhone) |
| Effective RAM | ~6 GB (8B) / ~3 GB (4B) |
| Interpreter latency | ~400ms (/no_think + grammar) |
| Coach simple latency | ~600ms (/no_think) |
| Coach complex latency | ~1200ms (/think) |
| Coach skipped | ~40% of messages |

### Tier Architecture

| Tier | Chat | Josi | Capabilities |
|------|------|------|-------------|
| **Monitor** (free) | No | No | App-direct tools only: `get_user_status`, `log_workout`, `get_recent_workouts` |
| **Advisor** | Yes | Yes | Explain workouts/zones, create TODAY's workout only, no plans |
| **Coach** | Yes | Yes | Full: plans, replans, multi-day coaching, all tools |

**Monitor has no Josi chat.** The app handles monitor tools directly without LLM involvement.

---

## Step 1: Download the Model + Knowledge Bundle

The app downloads **two files** as one atomic bundle. Both are required — the model generates text, the knowledge cards provide coaching context that gets injected into prompts.

| File | Size | Purpose |
|------|------|---------|
| `josi-v6-q4_k_m.gguf` | ~5.0 GB (8B) / ~2.5 GB (4B) | GGUF model for llama.cpp inference |
| `knowledge.json` | ~153 KB (114 cards) | Coaching context cards, injected into prompts at inference time |
| `josi-v6-manifest.json` | ~1 KB | Version, checksums, download URLs |

### Download

**Direct HTTP download from training server:**
```bash
# Replace <SERVER_IP> with the IP provided by Bart
curl -LO http://<SERVER_IP>/models/josi-v6-q4_k_m.gguf
curl -LO http://<SERVER_IP>/models/knowledge.json
```

**In the Kotlin app — download both files together:**
```kotlin
val BASE_URL = "http://<SERVER_IP>/models"

// Both files are required — download as one bundle
val modelUrl = "$BASE_URL/josi-v6-q4_k_m.gguf"
val knowledgeUrl = "$BASE_URL/knowledge.json"

// Local storage on device (same directory)
val modelFile = File(context.filesDir, "josi-v6-q4_k_m.gguf")
val knowledgeFile = File(context.filesDir, "knowledge.json")
```

**Manifest URL:** `http://<SERVER_IP>/models/josi-v6-manifest.json`
The manifest contains checksums for BOTH the model and knowledge.json. Use it to check for updates — when either file changes, re-download both.

**App download flow (end user):**
1. User installs app (~50 MB, no models bundled)
2. First launch: "Setting up your coach..." progress bar
3. App fetches manifest to get current version + checksums
4. App downloads model GGUF (~5.0 GB) + knowledge.json (~153 KB) together
5. Verify SHA-256 checksums for both files (from manifest)
6. Both cached locally, re-downloaded together when version updates
7. All inference runs on-device via llama.cpp — no network calls

**The knowledge cards are NOT optional.** Without them, the coach has no sport science context and gives generic answers. They must always be downloaded alongside the model.

Load model with [llama.android](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.android) or equivalent llama.cpp Kotlin/JNI bindings. Load knowledge.json with standard JSON parsing.

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

### Knowledge Card Injection

The knowledge cards from `knowledge.json` provide sport science context. The app selects relevant cards based on the user's message and injects them into the prompt. This is what makes Josi a knowledgeable coach instead of a generic chatbot.

```kotlin
/**
 * Simple keyword-based knowledge selector.
 * Picks the most relevant coaching cards for this conversation turn.
 * Mirrors shared/knowledge_selector.py logic.
 */
class KnowledgeSelector(private val entries: List<KnowledgeEntry>) {

    data class KnowledgeEntry(
        val id: String,
        val card: String,
        val section: String,
        val topics: List<String>,
        val keywords: List<String>,
        val content: String,
        val sport: String? = null,
    )

    companion object {
        fun fromJson(jsonFile: File): KnowledgeSelector {
            val data = JSONObject(jsonFile.readText())
            val arr = data.getJSONArray("entries")
            val entries = (0 until arr.length()).map { i ->
                val obj = arr.getJSONObject(i)
                KnowledgeEntry(
                    id = obj.getString("id"),
                    card = obj.getString("card"),
                    section = obj.getString("section"),
                    topics = obj.getJSONArray("topics").let { t ->
                        (0 until t.length()).map { t.getString(it) }
                    },
                    keywords = obj.getJSONArray("keywords").let { k ->
                        (0 until k.length()).map { k.getString(it) }
                    },
                    content = obj.getString("content"),
                    sport = obj.optString("sport", null),
                )
            }
            return KnowledgeSelector(entries)
        }
    }

    fun select(userMessage: String, sport: String? = null, maxCards: Int = 3): List<KnowledgeEntry> {
        val msgLower = userMessage.lowercase()

        return entries
            .map { entry -> Pair(scoreEntry(entry, msgLower, sport), entry) }
            .filter { it.first > 0 }
            .sortedByDescending { it.first }
            .distinctBy { it.second.card }  // one card per domain
            .take(maxCards)
            .map { it.second }
    }

    fun formatBlock(cards: List<KnowledgeEntry>): String {
        if (cards.isEmpty()) return ""
        return "[KNOWLEDGE]\n\n" + cards.joinToString("\n\n") { it.content }
    }

    private fun scoreEntry(entry: KnowledgeEntry, msgLower: String, sport: String?): Float {
        var score = 0f

        // Sport match
        if (sport != null && entry.sport != null) {
            if (sport == entry.sport) score += 3f else score -= 5f
        }

        // Keyword hits
        val hits = entry.keywords.count { it in msgLower }
        score += minOf(hits.toFloat(), 6f)

        // Exclude internal cards
        if (entry.card in setOf("josi_personas_v1", "planner_policy_v4")) return 0f

        return score
    }
}
```

**Load at app startup (once):**
```kotlin
// Load knowledge alongside the model — both from the same download
val knowledgeFile = File(context.filesDir, "knowledge.json")
val knowledgeSelector = KnowledgeSelector.fromJson(knowledgeFile)
```

**Inject into prompts (every turn):**
```kotlin
// Select relevant cards for this message
val cards = knowledgeSelector.select(userMessage, sport = ctx.profileSummary?.sport)
val knowledgeBlock = knowledgeSelector.formatBlock(cards)

// Append to context string (before the prompt is built)
val fullContext = buildContextString(ctx) + "\n\n" + knowledgeBlock
```

The knowledge block is injected into BOTH interpreter and coach prompts. It gives the model grounding in zone physiology, periodization, recovery science, etc. — without this, Josi can only give generic coaching answers.

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

## Step 4: Run Inference (Single Model, Two Modes, Router-Controlled Thinking)

The SAME model is loaded ONCE and used for both calls. The system prompt and thinking mode change.

### Inference Parameters

```kotlin
val interpreterParams = LlamaParams(
    nPredict = 200,       // Output cap: 200 tokens
    temperature = 0.3f,   // Low temperature for deterministic JSON
    topP = 0.9f,
    nCtx = 4096,          // Context cap: 4096 tokens
    // GBNF grammar loaded from shared/schemas/gatc_request.gbnf
    grammar = GATC_REQUEST_GRAMMAR,
)

val coachSimpleParams = LlamaParams(
    nPredict = 200,       // Output cap: 200 tokens
    temperature = 0.5f,   // Higher temperature for natural coaching text
    topP = 0.9f,
    nCtx = 4096,
)

val coachComplexParams = LlamaParams(
    nPredict = 400,       // More room for thinking + response
    temperature = 0.5f,
    topP = 0.9f,
    nCtx = 4096,
)
```

### GBNF Grammar for Interpreter

The interpreter uses a GBNF grammar to guarantee valid GATCRequest JSON.
Load the grammar from `shared/schemas/gatc_request.gbnf`:

```kotlin
// Load at app startup
val GATC_REQUEST_GRAMMAR = context.assets.open("gatc_request.gbnf")
    .bufferedReader().readText()
```

With grammar-constrained generation:
- **Zero parse failures** — the model can ONLY produce valid JSON
- **Correct enum values** — action, sport, replan_type are enforced at token level
- **No markdown fences** — grammar doesn't allow them
- **Faster** — model doesn't waste tokens on invalid paths

### Thinking Mode Router

The router decides whether the coach call needs `/think` (slow, deep reasoning) or `/no_think` (fast, simple answer):

```kotlin
// Actions that trigger /think for deeper reasoning
val COMPLEX_ACTIONS = setOf("explain")  // Explaining readiness/sessions needs context reasoning

// Keywords in the question that trigger /think for answer_question
val COMPLEX_KEYWORDS = listOf(
    "why", "waarom",           // "Why is my readiness red?"
    "how come", "hoe komt",    // "How come I'm tired?"
    "should i", "moet ik",     // "Should I train today?"
    "injury", "blessure",      // Injury-related reasoning
    "pain", "pijn",            // Pain assessment
    "plan", "schema",          // Plan tradeoff questions
    "instead", "in plaats",    // Alternative reasoning
)

fun shouldThink(gatcRequest: GATCRequest?): Boolean {
    if (gatcRequest == null) return false

    // Always think for explain (needs to reason about session context)
    if (gatcRequest.action in COMPLEX_ACTIONS) return true

    // For answer_question, check if it's complex
    if (gatcRequest.action == "answer_question") {
        val q = (gatcRequest.question ?: gatcRequest.freeText).lowercase()
        return COMPLEX_KEYWORDS.any { q.contains(it) }
    }

    return false
}
```

### Sequential Inference with Router-Controlled Thinking

```kotlin
suspend fun runJosi(userMessage: String, ctx: ChatContext): JosiResult {
    val contextStr = buildContextString(ctx)

    // Select relevant knowledge cards for this turn
    val cards = knowledgeSelector.select(userMessage, sport = ctx.profileSummary?.sport)
    val knowledgeBlock = knowledgeSelector.formatBlock(cards)
    val fullContext = contextStr + "\n\n" + knowledgeBlock

    // Step 1: Interpreter call (always /no_think + GBNF grammar)
    val interpreterPrompt = buildChatMLPrompt(
        INTERPRETER_SYSTEM_PROMPT, ctx.history,
        userMessage + "\n/no_think",  // Force no thinking for interpreter
        fullContext)
    val interpreterRaw = model.generate(interpreterPrompt, interpreterParams)
    val gatcRequest = parseGATCRequest(interpreterRaw)

    // Step 2: Router — does this action need the coach?
    val needsCoach = gatcRequest?.action in setOf("explain", "answer_question")

    val coachingText = if (needsCoach) {
        // Step 3: Router decides thinking mode
        val useThinking = shouldThink(gatcRequest)
        val thinkTag = if (useThinking) "/think" else "/no_think"
        val params = if (useThinking) coachComplexParams else coachSimpleParams

        // Build coach prompt with interpreter context + knowledge + thinking control
        val coachContext = fullContext + "\n\n[INTERPRETER]\n" + interpreterRaw.trim()
        val coachPrompt = buildChatMLPrompt(
            COACH_SYSTEM_PROMPT, ctx.history,
            userMessage + "\n$thinkTag",  // Router controls thinking
            coachContext)
        val raw = model.generate(coachPrompt, params)
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

When `/think` is used, Qwen3 produces `<think>...</think>` tags with internal reasoning.
**Always strip these before displaying to the user:**

```kotlin
fun stripThinkingTags(text: String): String {
    return text.replace(Regex("<think>.*?</think>", RegexOption.DOT_MATCHES_ALL), "").trim()
}
```

The thinking content is invisible to the user — they just see a better, more reasoned response.

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
    lateinit var knowledgeSelector: KnowledgeSelector

    // Load model + knowledge once — both from the same download bundle
    fun loadModel() {
        model = LlamaModel(modelFile.absolutePath, LlamaParams(nCtx = 4096))
        knowledgeSelector = KnowledgeSelector.fromJson(knowledgeFile)
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

| Change | v5 (dual model) | v6 (single model, Android) |
|--------|-----------------|---------------------------|
| Models | 2 GGUF files | 1 GGUF file |
| Download size | ~1.87 GB | ~5.0 GB (8B) |
| Base model | Qwen2.5-1.5B-Instruct | Qwen3-8B |
| Params per task | 1.5B | 8B (5.3x more) |
| Chat template | ChatML | ChatML (same) |
| Interpreter | Raw JSON (hope for valid) | GBNF grammar (guaranteed valid) |
| Coach output | Plain text | Plain text (same) |
| Thinking | None | Router-controlled /think for complex |
| Temperature | 0.45 (both) | 0.3 interpreter / 0.5 coach |
| Context cap | 2048 tokens | 4096 tokens |
| Output cap | 150 tokens | 200-400 tokens |
| Dutch | Basic | Native, excellent |
| Memory management | Load 2 models (~4 GB) | Load 1 model (~6 GB) |
| Inference | 2 separate models | Same model, 2 system prompts |

**Key integration changes:**
1. Download and store **one** model file instead of two
2. Load **one** model into memory (simpler)
3. Use **same model** for both calls, just switch system prompt
4. Add **GBNF grammar** for interpreter (zero parse failures)
5. Add **thinking mode router** (/think for complex, /no_think for fast)
6. Strip `<think>` tags from coach output
7. Same GATCRequest schema — no engine changes needed
8. Update temperatures: 0.3 for interpreter, 0.5 for coach
9. Update context cap to 4096 tokens

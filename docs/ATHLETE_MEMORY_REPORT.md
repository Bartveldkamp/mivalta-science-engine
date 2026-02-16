# Athlete Memory for Josi — Technical Report & Implementation Plan

**Date:** 2026-02-16
**Status:** Proposal
**Scope:** GATC Rust Engine + mivalta-science-engine (LLM training pipeline)

---

## 1. Problem Statement

Josi currently has **no persistent conversational memory**. Every app restart resets the conversation history, and no mechanism exists to carry forward learned facts about an athlete across sessions. This means:

- Josi repeatedly asks questions it already knows the answer to ("What sport do you do?")
- Coaching rapport cannot build over time — every session starts from zero
- Patterns Josi notices ("you always skip Mondays") are lost between restarts
- Athlete preferences ("I prefer morning runs", "I have a bad knee") must be re-stated

The athlete experience suffers because Josi feels like a stranger every time, rather than a coach who knows you.

---

## 2. Current State Analysis

### 2.1 What EXISTS today (GATC Rust Engine)

The VaultSyncService persists **five** types of per-athlete data, none carrying conversation memory:

| Table | Struct | What it stores |
|---|---|---|
| `profiles` | VaultProfile | Physiology + availability (age, sport, level, weekly_hours, FTP, max_hr) |
| `profile_history` | ProfileHistoryEntry | Snapshots per write_profile() call (goal changes, level upgrades) |
| `activities` | VaultActivity | Per-workout records — has a `notes: Option<String>` field (user free-text) |
| `biometrics` | VaultBiometric | Daily readiness, HRV, sleep, Viterbi state |
| `viterbi_state` | Raw JSON string | HMM posterior state per athlete (persists across app restarts) |

**Key structs and their locations:**

- **AthleteProfile** — `gatc-types/src/lib.rs:51` — Pure physiology, no memory field
- **VaultProfile** — `gatc-vault/src/models.rs:249` — SQLite-backed profile wrapper, no memory field
- **ChatContext** — `gatc-types/src/lib.rs:734` — What Josi sees per-turn, includes `recent_messages` (in-memory only, max 30 turns)
- **EnrichedContext** — `gatc-narrative/src/context.rs:13` — Full LLM prompt context, assembled from Vault data — **no memory/notes/coaching_insights field**

### 2.2 What EXISTS today (mivalta-science-engine / LLM side)

- **ChatContext schema** (`shared/schemas/chat_context.schema.json`) — has `history` (max 60 items), `profile_summary`, `readiness`, `planned_session`, `recent_auto_replan` — **no memory field**
- **Interpreter system prompt** — receives CONTEXT block with readiness + sport + session — **no memory section**
- **Explainer system prompt** — receives same CONTEXT block — **no memory section**
- **Training data** — ~13K interpreter + ~6.5K explainer examples — **none include memory in context**

### 2.3 What is MISSING

1. **Conversation history is ephemeral** — `JosiChat` (`gatc-josi/src/chat.rs:441`) holds `history: HashMap<String, Vec<HistoryEntry>>` in RAM with 30-turn / 60-day TTL. Never persisted to Vault. App restart = amnesia.

2. **No extracted insights** — There's no mechanism to distill recurring patterns, athlete preferences, or coaching notes from past conversations into a compact struct.

3. **EnrichedContext has no memory slot** — `vault_bridge.rs` assembles EnrichedContext from biometrics + activities + load metrics, but never injects conversation-derived knowledge.

4. **LLM training data has no memory examples** — Neither interpreter nor explainer have been trained with memory context, so even if we inject it, the models won't know how to use it without retraining.

---

## 3. Proposed Architecture

### 3.1 Data Flow

```
Josi conversation
    → extract facts (post-chat hook or LLM-based extraction)
        ↓
    VaultSyncService.write_memory()
        ↓
    vault.db:athlete_memory table (SQLite)
        ↓
    vault_bridge::build_enriched_context_from_vault()
        ↓
    EnrichedContext.athlete_memory
        ↓
    PromptBuilder → CONTEXT block in system message → LLM
```

### 3.2 Design Principles

1. **Compact** — Memory must fit within ~500 tokens of the 1024-token context budget. This is tight; every byte counts.
2. **Append-only with decay** — New facts are added; old facts decay by confidence and staleness. A hard cap (e.g., 15 facts) prevents unbounded growth.
3. **Source-tagged** — Each fact knows where it came from (conversation, behavior analysis, profile change) so we can prioritize and deduplicate.
4. **Read-only for Josi** — Josi reads memory but NEVER writes to it directly. Extraction happens in a separate post-processing step, consistent with Josi's read-only architecture.
5. **On-device first** — All memory lives in the local Vault SQLite DB. No cloud dependency.

---

## 4. Implementation Plan

### Phase 1: Rust Engine — Storage & Plumbing (GATC repo)

#### 1a. New struct: `AthleteMemory` in `gatc-types/src/lib.rs`

```rust
/// Persistent coaching memory — facts learned from past Josi conversations.
/// Compact enough to include in every LLM prompt (~500 tokens max).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AthleteMemory {
    /// Key facts: "prefers morning runs", "has knee issue", "works night shifts"
    pub key_facts: Vec<MemoryFact>,
    /// Recurring patterns: "skips Mondays frequently", "ramps too fast after rest"
    pub patterns: Vec<String>,
    /// Coaching notes: "responds well to data-driven framing"
    pub coaching_notes: Vec<String>,
    /// Last updated timestamp
    pub updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFact {
    pub fact: String,
    pub source: String,          // "conversation", "behavior", "profile_change"
    pub confidence: f64,         // 0.0-1.0
    pub learned_at: DateTime<Utc>,
}
```

#### 1b. New Vault table in `gatc-vault/src/sync.rs`

```sql
CREATE TABLE IF NOT EXISTS athlete_memory (
    athlete_id TEXT PRIMARY KEY,
    memory_json TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Add `write_memory()` / `read_memory()` following the existing `write_viterbi_state()` / `read_viterbi_state()` pattern at `sync.rs:292-344`.

#### 1c. Wire into `EnrichedContext` in `gatc-narrative/src/context.rs`

```rust
pub struct EnrichedContext {
    // ... existing fields ...
    pub athlete_memory: Option<AthleteMemory>,  // ← new field
}
```

#### 1d. Populate in `vault_bridge.rs`

Load memory as a new step (step 9) in `build_enriched_context_from_vault()`:

```rust
let memory = load_athlete_memory(&conn, athlete_id)?;
context.athlete_memory = memory;
```

#### 1e. Include in LLM prompt via `prompt_builder.rs`

`PromptBuilder::build_prompt()` adds a `MEMORY:` section to the CONTEXT block when athlete_memory is present.

---

### Phase 2: Schema & Contract Update (this repo — mivalta-science-engine)

#### 2a. Extend `chat_context.schema.json`

Add a new optional `athlete_memory` property:

```json
"athlete_memory": {
  "type": "object",
  "additionalProperties": false,
  "description": "Persistent coaching memory — key facts learned across sessions. Compact (~500 tokens max).",
  "properties": {
    "key_facts": {
      "type": "array",
      "maxItems": 15,
      "items": {
        "type": "object",
        "properties": {
          "fact": { "type": "string" },
          "source": { "type": "string", "enum": ["conversation", "behavior", "profile_change"] },
          "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
        },
        "required": ["fact", "source", "confidence"]
      }
    },
    "patterns": {
      "type": "array",
      "maxItems": 5,
      "items": { "type": "string" }
    },
    "coaching_notes": {
      "type": "array",
      "maxItems": 5,
      "items": { "type": "string" }
    }
  }
}
```

#### 2b. Update system prompts for both models

**Interpreter** — Add instruction to leverage memory for intent disambiguation:
```
MEMORY (if present): Use these known facts to fill in missing fields.
For example, if memory says "primary sport: running" and user says "give me a workout",
you can set sport="run" instead of clarifying.
```

**Explainer** — Add instruction to use memory for personalization:
```
MEMORY (if present): These are things you've learned about this athlete from past
conversations. Use them naturally to personalize your responses. Don't repeat facts
back mechanically — weave them in like a coach who knows their athlete.
```

---

### Phase 3: Training Data Generation (this repo)

#### 3a. Augment gold examples with memory context

Create memory-enriched variants of existing training examples:
- Take existing interpreter examples → inject a `MEMORY:` block into the CONTEXT
- Ensure the model learns to use memory facts to skip clarification
- Ensure the model learns to NOT hallucinate memory it wasn't given

**Example pair (before/after memory):**

Without memory:
```
User: "Give me a workout for tomorrow"
→ action: "clarify", missing: ["sport"]
```

With memory:
```
User: "Give me a workout for tomorrow"
MEMORY: primary sport: running, prefers morning sessions
→ action: "create_workout", sport: "run"
```

#### 3b. Create anti-hallucination examples

Train the model to NOT invent memory it wasn't given:
- Examples where memory is absent and model must NOT assume facts
- Examples where memory is present but irrelevant to the current question

#### 3c. Target: ~2K memory-enriched examples per model

- ~1K interpreter examples with memory (mix of helpful and irrelevant memory)
- ~1K explainer examples with memory (personalized responses using facts)
- ~500 negative examples (no memory → don't assume; wrong memory → ignore)

---

### Phase 4: Memory Extraction Pipeline (new component)

#### 4a. Post-conversation fact extractor

After each Josi conversation ends (or after N turns), run a lightweight extraction:

**Option A: Rule-based extraction** (simpler, more predictable)
- Pattern matching on user messages for: sport mentions, time preferences, injury reports, schedule constraints
- Behavioral analysis on activity data: skip patterns, intensity preferences, consistency metrics

**Option B: LLM-based extraction** (richer, but requires on-device inference budget)
- Small model pass that reads the conversation and outputs structured facts
- More capable at extracting nuanced preferences ("prefers data-driven explanations")

**Recommendation: Start with Option A** (rule-based) for v1, add Option B later.

#### 4b. Memory consolidation & decay

- New facts merge with existing memory (deduplicate, update confidence)
- Facts older than 90 days with low confidence decay
- Hard cap: 15 key_facts, 5 patterns, 5 coaching_notes
- Conflicting facts: newer fact wins (e.g., "prefers cycling" replaces "prefers running" if athlete switched)

---

### Phase 5: Evaluation & Safety (this repo)

#### 5a. New eval layer: Memory compliance

Add to the existing 3-layer eval harness (`evaluate_gemma3n_v4.py`):

- **Memory utilization**: When memory contains relevant facts, does the model use them?
- **Anti-hallucination**: When memory is absent, does the model invent facts?
- **Boundary respect**: Does memory ever cause the model to prescribe (violate I6)?
- **Staleness handling**: Does the model gracefully handle outdated memory?

#### 5b. Privacy safeguards

- Memory NEVER contains: real names, health conditions beyond what athlete shared, location data, PII
- Memory is stored only in local Vault — never transmitted to any server
- Athlete can view and delete memory from app settings

---

## 5. Implementation Order & Dependencies

```
Phase 1 (Rust)  ──→  Phase 2 (Schema)  ──→  Phase 3 (Training Data)
                                               ↓
                      Phase 4 (Extraction) ──→ Phase 5 (Eval)
                                               ↓
                                          Retrain models
                                               ↓
                                          GGUF export & deploy
```

**Phase 1** blocks everything — without Rust storage, there's nothing to populate.
**Phases 2-3** can start in parallel with Phase 1 (schema + training data don't need working Rust code).
**Phase 4** is independent and can be developed alongside.
**Phase 5** eval must come after training data (Phase 3) is ready.

---

## 6. Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Memory consumes too many tokens (>500) | Model gets truncated context, worse output | Hard cap on facts + token-counting serializer |
| Model hallucinates from memory | Wrong facts, broken trust | Anti-hallucination training examples + eval layer |
| Memory extraction is inaccurate | Wrong facts persisted | Start with high-confidence rule-based extraction only |
| On-device storage growth | App size bloat | Compact JSON, 15-fact cap, ~2KB max per athlete |
| Privacy concerns | User distrust | Local-only storage, visible in settings, deletable |
| Context budget too tight (1024 tokens) | Memory doesn't fit | Priority-rank facts, include only top-N by relevance |

---

## 7. Token Budget Analysis

Current context budget breakdown (1024 tokens total, 150 reserved for output = 874 input):

| Component | Estimated Tokens | Notes |
|---|---|---|
| System prompt | ~350 | Interpreter or Explainer instructions |
| CONTEXT block (readiness, session, profile) | ~150 | Current fields |
| Conversation history (last 3-5 turns) | ~200 | Compressed recent turns |
| **Available for memory** | **~174** | Tight but workable |

With 174 tokens for memory, we can fit approximately:
- 8-10 key facts (15-20 tokens each)
- 2-3 patterns
- 1-2 coaching notes

This means the **serialization format must be extremely compact**. Recommended format for the CONTEXT block:

```
MEMORY:
- Facts: prefers morning runs | has recurring knee issue | works night shifts | primary sport: running
- Patterns: tends to skip after rest days | ramps intensity too fast in week 2
- Notes: responds well to encouragement after hard sessions
```

---

## 8. Success Metrics

1. **Clarification reduction**: Interpreter should need 20-30% fewer `action=clarify` responses for returning athletes
2. **Personalization score**: Blind eval where judges rate explainer responses for "feels like it knows the athlete" (target: 4/5 vs current ~2/5)
3. **No regression on I6 safety**: Memory must not cause any increase in zone/prescription violations
4. **Schema compliance maintained**: Memory-enriched interpreter output must still pass 100% schema validation
5. **Token budget compliance**: Memory section must stay under 200 tokens in 95% of cases

---

## 9. Open Questions

1. **Memory extraction trigger**: After every conversation? After N turns? On app background? Need to balance freshness vs battery/compute cost.
2. **Memory editing UI**: Should athletes be able to edit individual facts, or just "clear all memory"?
3. **Cross-sport memory**: If an athlete does both running and cycling, should memory be per-sport or unified?
4. **Memory in interpreter vs explainer**: Should both models get the same memory, or should interpreter get a more structured subset?
5. **Bootstrapping**: For new athletes with no conversation history, should we pre-populate memory from their profile? (e.g., sport and level are already known)

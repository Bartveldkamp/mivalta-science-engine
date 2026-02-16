# Rust Engine — Athlete Memory Integration Guide

**For:** GATC Rust engine team
**Date:** 2026-02-16
**Depends on:** Schema, prompts, and training data changes in this repo (mivalta-science-engine)

---

## What's Ready (This Repo)

The following are implemented and ready for the Rust side to consume:

| Component | File | Status |
|---|---|---|
| Schema contract | `shared/schemas/chat_context.schema.json` | `athlete_memory` field added (optional) |
| Interpreter prompt | `training/prompts/interpreter_system.txt` | MEMORY instructions added |
| Explainer prompt | `training/prompts/explainer_system.txt` | MEMORY instructions added |
| Training data script | `training/scripts/augment_memory_training_data.py` | 85 memory-enriched examples |
| Fact extractor (reference) | `shared/memory_extractor.py` | Python reference implementation |

---

## What Rust Needs to Do

### Step 1: Add `AthleteMemory` struct to `gatc-types/src/lib.rs`

```rust
use chrono::{DateTime, Utc};

/// Persistent coaching memory — facts learned from past Josi conversations.
/// Compact enough to include in every LLM prompt (~174 tokens max).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AthleteMemory {
    /// Key facts: "prefers morning runs", "has knee issue", "works night shifts"
    /// Hard cap: 15 facts.
    pub key_facts: Vec<MemoryFact>,
    /// Recurring patterns: "skips Mondays frequently", "ramps too fast after rest"
    /// Hard cap: 5 patterns.
    pub patterns: Vec<String>,
    /// Coaching notes: "responds well to data-driven framing"
    /// Hard cap: 5 notes.
    pub coaching_notes: Vec<String>,
    /// Last updated timestamp
    pub updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFact {
    pub fact: String,
    /// How this fact was learned: "conversation", "behavior", "profile_change"
    pub source: String,
    /// Confidence 0.0-1.0. Higher = more reliable.
    pub confidence: f64,
    /// When this fact was first learned
    pub learned_at: DateTime<Utc>,
}
```

### Step 2: Add SQLite table in `gatc-vault/src/sync.rs`

Add to `ensure_tables()`:

```sql
CREATE TABLE IF NOT EXISTS athlete_memory (
    athlete_id TEXT PRIMARY KEY,
    memory_json TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Add `write_memory()` / `read_memory()` following the existing `write_viterbi_state()` / `read_viterbi_state()` pattern:

```rust
pub fn write_memory(&self, athlete_id: &str, memory: &AthleteMemory) -> Result<()> {
    let json = serde_json::to_string(memory)?;
    let conn = self.conn()?;
    conn.execute(
        "INSERT OR REPLACE INTO athlete_memory (athlete_id, memory_json, updated_at)
         VALUES (?1, ?2, CURRENT_TIMESTAMP)",
        params![athlete_id, json],
    )?;
    Ok(())
}

pub fn read_memory(&self, athlete_id: &str) -> Result<Option<AthleteMemory>> {
    let conn = self.conn()?;
    let mut stmt = conn.prepare(
        "SELECT memory_json FROM athlete_memory WHERE athlete_id = ?1"
    )?;
    let result = stmt.query_row(params![athlete_id], |row| {
        let json: String = row.get(0)?;
        Ok(json)
    });
    match result {
        Ok(json) => Ok(Some(serde_json::from_str(&json)?)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}
```

### Step 3: Wire into `EnrichedContext` in `gatc-narrative/src/context.rs`

```rust
pub struct EnrichedContext {
    // ... existing fields ...
    pub athlete_memory: Option<AthleteMemory>,  // ← add this
}
```

### Step 4: Populate in `vault_bridge.rs`

In `build_enriched_context_from_vault()`, add as step 9:

```rust
// Step 9: Load athlete memory
let memory = vault.read_memory(athlete_id)?;
context.athlete_memory = memory;
```

### Step 5: Include in LLM prompt via `prompt_builder.rs`

In `PromptBuilder::build_prompt()`, when `athlete_memory` is `Some`, add a `MEMORY:` section to the CONTEXT block. Use the compact pipe-separated format:

```rust
if let Some(ref mem) = context.athlete_memory {
    let mut memory_lines = Vec::new();

    // Key facts (confidence >= 0.5 only)
    let facts: Vec<&str> = mem.key_facts.iter()
        .filter(|f| f.confidence >= 0.5)
        .map(|f| f.fact.as_str())
        .collect();
    if !facts.is_empty() {
        memory_lines.push(format!("- Facts: {}", facts.join(" | ")));
    }

    // Patterns
    if !mem.patterns.is_empty() {
        memory_lines.push(format!("- Patterns: {}", mem.patterns.join(" | ")));
    }

    // Coaching notes
    if !mem.coaching_notes.is_empty() {
        memory_lines.push(format!("- Notes: {}", mem.coaching_notes.join(" | ")));
    }

    if !memory_lines.is_empty() {
        prompt.push_str("\nMEMORY:\n");
        for line in &memory_lines {
            prompt.push_str(line);
            prompt.push('\n');
        }
    }
}
```

### Step 6: Implement fact extraction (post-conversation hook)

Port the rule-based extractor from `shared/memory_extractor.py` to Rust. The Python implementation is the reference — it uses simple regex patterns for:

| Category | Example patterns | Confidence |
|---|---|---|
| Sport | "run", "running", "cycle", "bike" | 0.8 |
| Time preference | "morning", "before work", "evening" | 0.7 |
| Duration | "usually 45 minutes", "normally 30 min" | 0.6 |
| Injury | "bad knee", "back issue", "shin splints" | 0.9 |
| Lifestyle | "night shifts", "kids", "busy schedule" | 0.7 |
| Goal | "marathon", "half marathon", "10k" | 0.8 |

**Trigger:** Run extraction after each Josi conversation ends (when `JosiChat` detects the conversation is over, or on app background).

**Merge logic:**
1. Extract facts from new conversation turns
2. Load existing memory from Vault
3. Merge: boost confidence for repeated facts, replace conflicting facts (newer wins)
4. Decay: remove facts with confidence < 0.3
5. Cap: max 15 key_facts, 5 patterns, 5 coaching_notes
6. Write merged memory back to Vault

The Python `merge_memory()` function in `shared/memory_extractor.py` shows the exact algorithm.

---

## Serialization Format (Critical)

The MEMORY block in the CONTEXT must use this exact compact format:

```
MEMORY:
- Facts: primary sport: running | has recurring knee issue | prefers morning sessions | typical duration: 45 min
- Patterns: tends to skip after rest days | ramps intensity too fast in week 2
- Notes: responds well to encouragement after hard sessions
```

This format matches what the training data uses. The models are trained to parse exactly this layout.

**Token budget:** The entire MEMORY block must stay under ~174 tokens. With the pipe-separated format, this fits ~10 facts + 3 patterns + 2 notes.

---

## Hard Caps

| Field | Max items | Max total size |
|---|---|---|
| key_facts | 15 | ~2KB JSON |
| patterns | 5 | — |
| coaching_notes | 5 | — |
| Total memory per athlete | — | ~2KB |

---

## Conflict Resolution

When a new fact conflicts with an existing one in the same category, the newer fact wins:

| Category | Conflict key | Example |
|---|---|---|
| primary_sport | `"primary sport: X"` | "cycling" replaces "running" |
| goal | `"goal: X"` | "half marathon" replaces "marathon" |
| typical_duration | `"typical duration: X"` | "60 min" replaces "45 min" |
| time_preference | `"prefers X sessions"` | "evening" replaces "morning" |
| level | `"level: X"` | "advanced" replaces "intermediate" |

---

## Testing Checklist

- [ ] `write_memory()` → `read_memory()` round-trip preserves all fields
- [ ] Empty memory → MEMORY block is omitted from prompt (not empty block)
- [ ] 15+ facts → capped at 15, sorted by confidence descending
- [ ] Conflicting facts → newer overwrites older in same category
- [ ] Low-confidence facts (< 0.3) → removed on merge
- [ ] MEMORY block appears in CONTEXT between existing fields and end of context
- [ ] Total prompt with MEMORY stays within 1024-token budget
- [ ] Existing tests pass (no regression from new field)

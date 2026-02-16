"""
MiValta Athlete Memory Extractor — Rule-Based v1

Extracts persistent coaching facts from conversation history using
pattern matching and behavioral analysis. Designed to run post-conversation
as a separate step (Josi never writes memory directly).

The extractor produces an AthleteMemory-compatible dict that the Rust engine
can persist via VaultSyncService.write_memory().

Data flow:
    conversation history (list of {role, message, ts})
        → extract_facts_from_conversation()
            → list of MemoryFact dicts

    existing_memory + new_facts
        → merge_memory()
            → updated AthleteMemory dict (ready for persistence)

Usage (Python side — design reference for Rust implementation):
    from shared.memory_extractor import extract_facts_from_conversation, merge_memory

    new_facts = extract_facts_from_conversation(history)
    updated = merge_memory(existing_memory, new_facts)
    # → pass updated to Rust: vault.write_memory(athlete_id, updated)
"""

import re
from datetime import datetime, timezone
from typing import Optional


# =============================================================================
# FACT EXTRACTION PATTERNS
# =============================================================================

# Sport mentions — map natural language to GATC sport enum
_SPORT_PATTERNS = [
    (re.compile(r'\b(?:run|running|jog|jogging)\b', re.I), "primary sport: running"),
    (re.compile(r'\b(?:cycl|bike|biking|riding|ride)\w*\b', re.I), "primary sport: cycling"),
    (re.compile(r'\b(?:ski|skiing|cross[- ]country)\b', re.I), "primary sport: skiing"),
    (re.compile(r'\b(?:skat|skating|inline|rollerblad)\w*\b', re.I), "primary sport: skating"),
    (re.compile(r'\b(?:strength|weights?|gym|lifting)\b', re.I), "does strength training"),
]

# Time preference patterns
_TIME_PREF_PATTERNS = [
    (re.compile(r'\b(?:morning|before work|early|first thing|6\s*am|7\s*am)\b', re.I),
     "prefers morning sessions"),
    (re.compile(r'\b(?:evening|after work|late|night|after dinner)\b', re.I),
     "prefers evening sessions"),
    (re.compile(r'\b(?:lunch|midday|noon|middle of the day)\b', re.I),
     "prefers midday sessions"),
]

# Duration preference patterns
_DURATION_PREF_PATTERNS = [
    (re.compile(r'\busually\s+(?:about\s+)?(\d+)\s*(?:min|minutes)\b', re.I),
     "typical duration: {0} min"),
    (re.compile(r'\bnormally\s+(?:do|run|ride|train)\s+(?:for\s+)?(?:about\s+)?(\d+)\s*(?:min|minutes)\b', re.I),
     "typical duration: {0} min"),
    (re.compile(r'\bonly\s+(?:have|got)\s+(\d+)\s*(?:min|minutes)\s+(?:usually|most days|normally)\b', re.I),
     "typical duration: {0} min"),
]

# Injury / constraint patterns
_INJURY_PATTERNS = [
    (re.compile(r'\b(?:bad|injured|sore|weak|recovering)\s+(?:right\s+)?knee\b', re.I),
     "has knee issue"),
    (re.compile(r'\bknee\s+(?:issue|injury|problem|pain)\b', re.I),
     "has knee issue"),
    (re.compile(r'\b(?:bad|injured|sore|weak|recovering)\s+back\b', re.I),
     "has back issue"),
    (re.compile(r'\bback\s+(?:issue|injury|problem|pain)\b', re.I),
     "has back issue"),
    (re.compile(r'\b(?:bad|injured|sore|weak|recovering)\s+(?:right\s+|left\s+)?ankle\b', re.I),
     "has ankle issue"),
    (re.compile(r'\b(?:bad|injured|sore|weak|recovering)\s+(?:right\s+|left\s+)?shoulder\b', re.I),
     "has shoulder issue"),
    (re.compile(r'\b(?:plantar\s+fasci|shin\s+splint|achilles|it\s*band)\w*\b', re.I),
     "has running-related injury"),
]

# Schedule / lifestyle patterns
_LIFESTYLE_PATTERNS = [
    (re.compile(r'\b(?:night\s+shift|work\s+nights?|graveyard\s+shift)\b', re.I),
     "works night shifts"),
    (re.compile(r'\b(?:kids?|children|toddler|baby|parent)\b', re.I),
     "has young children"),
    (re.compile(r'\b(?:busy|hectic|crazy)\s+(?:work|schedule|week)\b', re.I),
     "has busy work schedule"),
    (re.compile(r'\btrain(?:s|ing)?\s+(\d)\s*(?:x|times)\s*(?:a|per)\s*week\b', re.I),
     "trains {0}x per week"),
]

# Goal patterns
_GOAL_PATTERNS = [
    (re.compile(r'\b(?:marathon|26\.2|42k)\b', re.I), "goal: marathon"),
    (re.compile(r'\bhalf\s*marathon\b', re.I), "goal: half marathon"),
    (re.compile(r'\b10k\b', re.I), "goal: 10k"),
    (re.compile(r'\b5k\b', re.I), "goal: 5k"),
    (re.compile(r'\btriathlon\b', re.I), "goal: triathlon"),
    (re.compile(r'\biron\s*man\b', re.I), "goal: ironman"),
]

ALL_PATTERNS = [
    (_SPORT_PATTERNS, "conversation", 0.8),
    (_TIME_PREF_PATTERNS, "conversation", 0.7),
    (_DURATION_PREF_PATTERNS, "conversation", 0.6),
    (_INJURY_PATTERNS, "conversation", 0.9),
    (_LIFESTYLE_PATTERNS, "conversation", 0.7),
    (_GOAL_PATTERNS, "conversation", 0.8),
]


def _extract_from_message(message: str) -> list[dict]:
    """Extract memory facts from a single message using pattern matching."""
    facts = []
    for pattern_group, source, base_confidence in ALL_PATTERNS:
        for regex, fact_template in pattern_group:
            match = regex.search(message)
            if match:
                # Fill in capture groups if template uses {0}, {1}, etc.
                groups = match.groups()
                fact = fact_template
                for i, g in enumerate(groups):
                    fact = fact.replace(f"{{{i}}}", g)

                facts.append({
                    "fact": fact,
                    "source": source,
                    "confidence": base_confidence,
                })
    return facts


def extract_facts_from_conversation(
    history: list[dict],
    min_confidence: float = 0.5,
) -> list[dict]:
    """Extract memory facts from a conversation history.

    Args:
        history: List of {role: str, message: str, ts: str} dicts.
        min_confidence: Minimum confidence threshold for a fact to be kept.

    Returns:
        List of MemoryFact-compatible dicts: {fact, source, confidence}.
    """
    all_facts: dict[str, dict] = {}  # fact_text → fact_dict (dedup by text)

    for turn in history:
        if turn.get("role") != "user":
            continue

        message = turn.get("message", "")
        extracted = _extract_from_message(message)

        for f in extracted:
            key = f["fact"].lower().strip()
            if key in all_facts:
                # Boost confidence on repeated mentions
                existing = all_facts[key]
                existing["confidence"] = min(1.0, existing["confidence"] + 0.1)
            else:
                all_facts[key] = f

    # Filter by confidence
    return [f for f in all_facts.values() if f["confidence"] >= min_confidence]


# =============================================================================
# PATTERN DETECTION (from activity data — design reference for Rust)
# =============================================================================

def detect_patterns(
    activities: list[dict],
    skip_days: Optional[list[int]] = None,
) -> list[str]:
    """Detect recurring behavioral patterns from activity data.

    This is a design reference — the actual implementation runs in Rust
    with access to VaultActivity data.

    Args:
        activities: List of activity dicts with at least {date, sport, completed}.
        skip_days: Pre-computed list of weekday indices (0=Mon) that are
                   frequently skipped.

    Returns:
        List of pattern strings suitable for athlete_memory.patterns.
    """
    patterns = []

    if skip_days:
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
        for d in skip_days:
            if 0 <= d <= 6:
                patterns.append(f"often skips {day_names[d]}s")

    # Detect intensity preference from recent activities
    if activities:
        high_intensity = sum(1 for a in activities
                             if a.get("intent", "").startswith(("Z4", "Z5")))
        total = len(activities)
        if total >= 5:
            ratio = high_intensity / total
            if ratio > 0.5:
                patterns.append("tends to favor high-intensity sessions")
            elif ratio < 0.15:
                patterns.append("prefers easy/moderate sessions")

    return patterns[:5]  # Hard cap at 5 patterns


# =============================================================================
# MEMORY MERGING
# =============================================================================

# Hard caps from the report
MAX_KEY_FACTS = 15
MAX_PATTERNS = 5
MAX_COACHING_NOTES = 5


def merge_memory(
    existing: Optional[dict],
    new_facts: list[dict],
    new_patterns: Optional[list[str]] = None,
    new_notes: Optional[list[str]] = None,
    max_age_days: int = 90,
) -> dict:
    """Merge new extractions into existing athlete memory.

    Handles deduplication, confidence boosting for repeated facts,
    decay for stale facts, and hard cap enforcement.

    Args:
        existing: Current athlete_memory dict (or None for new athlete).
        new_facts: New MemoryFact dicts from extract_facts_from_conversation().
        new_patterns: New pattern strings from detect_patterns().
        new_notes: New coaching notes (rarely auto-generated, usually manual).
        max_age_days: Facts older than this with low confidence are decayed.

    Returns:
        Updated athlete_memory dict ready for persistence.
    """
    now = datetime.now(timezone.utc).isoformat()

    if existing is None:
        existing = {"key_facts": [], "patterns": [], "coaching_notes": []}

    # --- Merge key_facts ---
    fact_map: dict[str, dict] = {}
    for f in existing.get("key_facts", []):
        key = f["fact"].lower().strip()
        fact_map[key] = f

    for f in new_facts:
        key = f["fact"].lower().strip()
        if key in fact_map:
            # Boost confidence on re-extraction
            old = fact_map[key]
            old["confidence"] = min(1.0, old.get("confidence", 0.5) + 0.1)
        else:
            # Handle conflicts: newer fact about same category replaces old
            # e.g., "primary sport: cycling" replaces "primary sport: running"
            category = _fact_category(f["fact"])
            if category:
                to_remove = [k for k, v in fact_map.items()
                             if _fact_category(v["fact"]) == category]
                for k in to_remove:
                    del fact_map[k]
            fact_map[key] = f

    # Decay old low-confidence facts
    if max_age_days > 0:
        cutoff = max_age_days  # Simplified: in real impl, compare learned_at
        fact_map = {k: v for k, v in fact_map.items()
                    if v.get("confidence", 0.5) > 0.3}

    # Sort by confidence (highest first) and cap
    sorted_facts = sorted(fact_map.values(),
                          key=lambda f: f.get("confidence", 0.5),
                          reverse=True)[:MAX_KEY_FACTS]

    # --- Merge patterns ---
    existing_patterns = set(existing.get("patterns", []))
    if new_patterns:
        existing_patterns.update(new_patterns)
    merged_patterns = list(existing_patterns)[:MAX_PATTERNS]

    # --- Merge coaching notes ---
    existing_notes = set(existing.get("coaching_notes", []))
    if new_notes:
        existing_notes.update(new_notes)
    merged_notes = list(existing_notes)[:MAX_COACHING_NOTES]

    return {
        "key_facts": sorted_facts,
        "patterns": merged_patterns,
        "coaching_notes": merged_notes,
    }


def _fact_category(fact: str) -> Optional[str]:
    """Extract the category from a fact for conflict resolution.

    Returns a category string like "primary_sport" or "goal", or None
    if the fact doesn't belong to a conflicting category.
    """
    lower = fact.lower()
    if lower.startswith("primary sport:"):
        return "primary_sport"
    if lower.startswith("goal:"):
        return "goal"
    if lower.startswith("typical duration:"):
        return "typical_duration"
    if lower.startswith("level:"):
        return "level"
    if lower.startswith("prefers ") and "sessions" in lower:
        return "time_preference"
    return None


# =============================================================================
# SERIALIZATION — compact format for LLM context
# =============================================================================

def serialize_memory_for_prompt(memory: dict) -> str:
    """Serialize athlete memory into compact MEMORY block for LLM prompt.

    Target: ≤174 tokens. Uses pipe-separated facts for density.

    Args:
        memory: AthleteMemory-compatible dict with key_facts, patterns,
                coaching_notes.

    Returns:
        String suitable for injection into CONTEXT block, or empty string
        if no memory exists.
    """
    parts = []

    facts = memory.get("key_facts", [])
    if facts:
        fact_strs = [f["fact"] for f in facts if f.get("confidence", 0) >= 0.5]
        if fact_strs:
            parts.append("- Facts: " + " | ".join(fact_strs))

    patterns = memory.get("patterns", [])
    if patterns:
        parts.append("- Patterns: " + " | ".join(patterns))

    notes = memory.get("coaching_notes", [])
    if notes:
        parts.append("- Notes: " + " | ".join(notes))

    if not parts:
        return ""

    return "MEMORY:\n" + "\n".join(parts)

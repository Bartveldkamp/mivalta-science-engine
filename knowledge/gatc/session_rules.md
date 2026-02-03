<!-- META -->
# Session Rules — Canonical

concept_id: session_rules
axis_owner: session
version: 1.3
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Foster (1998) | Training load scales with duration × perceived intensity | Load unit foundation |
| Banister (1975) | Fitness–fatigue response to impulse load | Planned load relevance |
| Laursen & Jenkins (2002) | Work–rest structure governs adaptation | Session shaping |
| Seiler (2010) | Easy work fills gaps between hard work | Two-lane model |
| Kenttä & Hassmén (1998) | Monotony and strain predict breakdown | Variety constraints |

---

<!-- ALG: session_contexts -->
## Session Contexts

| session_context | description |
|-----------------|-------------|
| standard | normal planned training session |
| micro | short opportunistic session |
| commute | commute-based training |
| senior | conservative constraints |

Notes:
- Context is explicit input (UI / profile)
- Context is never inferred from duration

---

<!-- ALG: zone_block_map -->
## Zone → Zone Block Map

| zone | zone_block |
|------|------------|
| Z1 | Z1_Z2 |
| Z2 | Z1_Z2 |
| Z3 | Z3 |
| Z4 | Z4_Z5 |
| Z5 | Z4_Z5 |
| Z6 | Z6_Z8 |
| Z7 | Z6_Z8 |
| Z8 | Z6_Z8 |

Notes:
- Zone blocks exist only for spacing, caps, and variety
- Load and accounting remain per-zone

---

<!-- ALG: zone_load_factor -->
## Zone Load Factors (fz)

| zone | fz |
|------|----|
| R | 0.5 |
| Z1 | 0.8 |
| Z2 | 1.0 |
| Z3 | 1.3 |
| Z4 | 1.8 |
| Z5 | 2.5 |
| Z6 | 3.0 |
| Z7 | 3.2 |
| Z8 | 2.8 |

Notes:
- Dimensionless relative stress factors
- Used only for load computation

---

<!-- ALG: duration_amplification -->
## Duration Amplification Parameters (Md)

| zone | tau_minutes | lambda | cap |
|------|-------------|--------|-----|
| Z1 | 60 | 0.10 | 1.5 |
| Z2 | 50 | 0.12 | 1.6 |
| Z3 | 40 | 0.15 | 1.7 |
| Z4 | 25 | 0.18 | 1.8 |
| Z5 | 20 | 0.20 | 1.9 |
| Z6 | 10 | 0.25 | 2.0 |
| Z7 | 8 | 0.30 | 2.0 |
| Z8 | 5 | 0.30 | 1.5 |

Duration amplification formula:

```
Md = min(cap, 1 + lambda × max(0, d − tau_minutes) / 60)
```

Definitions:
- d → NTIZ minutes in the zone
- tau_minutes → zone-specific tolerance threshold
- lambda → amplification rate
- cap → safety cap preventing runaway load

---

<!-- ALG: session_class_multiplier -->
## Session Class Multiplier (Mc)

| session_context | Mc |
|-----------------|----|
| standard | 1.0 |
| micro | 0.6 |
| commute | 0.7 |
| senior | 0.7 |

Notes:
- Adjusts load impact by context
- Does not change physiology or zone meaning

---

<!-- ALG: planned_load_formula -->
## Planned Load Formula (ULS)

```
ULS = d × fz × mi × Md × Mc
```

Definitions:
- d  → NTIZ minutes in zone
- fz → zone load factor
- mi → athlete modifier (ResolvedRuleset output)
- Md → duration amplification
- Mc → session class multiplier

---

<!-- ALG: load_signature -->
## Load Signature (Qualitative Buckets)

| zone | load_bucket |
|------|-------------|
| R | recovery |
| Z1 | volume |
| Z2 | volume |
| Z3 | threshold |
| Z4 | threshold |
| Z5 | vo2 |
| Z6 | vo2 |
| Z7 | neuromuscular |
| Z8 | neuromuscular |

Notes:
- Same ULS value ≠ same adaptation
- Signature preserved for Load Monitoring diagnostics

---

<!-- ALG: med_by_zone -->
## Minimum Effective Dose (MED) by Zone

| zone | med_minutes |
|------|-------------|
| R | 20 |
| Z1 | 30 |
| Z2 | 40 |
| Z3 | 20 |
| Z4 | 15 |
| Z5 | 8 |
| Z6 | 4 |
| Z7 | 3 |
| Z8 | 2 |

Notes:
- med_minutes = minimum NTIZ minutes per session for adaptation
- Below MED = no training credit for that zone
- Zero allocation is valid; partial is not

---

<!-- ALG: ntiz_credit -->
## NTIZ Credit

| zone | ntiz_credit |
|------|-------------|
| R | false |
| Z1 | true |
| Z2 | true |
| Z3 | true |
| Z4 | true |
| Z5 | true |
| Z6 | true |
| Z7 | true |
| Z8 | true |

---

<!-- ALG: session_duration_limits -->
## Per-Zone NTIZ Limits by Session Context

| session_context | zone | ntiz_min | ntiz_max |
|-----------------|------|----------|----------|
| standard | Z2 | 30 | 150 |
| micro | Z2 | 5 | 20 |
| commute | Z4 | 3 | 12 |
| senior | Z2 | 5 | 20 |
| standard | Z4 | 15 | 45 |
| micro | Z4 | 3 | 8 |
| senior | Z4 | 0 | 8 |

Notes:
- Values are minutes **in zone**
- Warm-up and cooldown are excluded
- Zero minimum means "not recommended by default"

---

<!-- ALG: session_structures -->
## Session Structures (Form Only)

| structure_type | zone_block | is_continuous |
|----------------|------------|---------------|
| continuous | Z1_Z2 | true |
| aerobic_blocks | Z1_Z2 | false |
| tempo_blocks | Z3 | false |
| threshold_intervals | Z4_Z5 | false |
| vo2_intervals | Z4_Z5 | false |
| anaerobic_reps | Z6_Z8 | false |
| sprint_reps | Z6_Z8 | false |

Notes:
- Structures define **form**, not workouts
- Selection must respect spacing and caps

---

<!-- JOSI: session_overview -->
## How Load Is Planned

Every session has:
- one or more zones
- NTIZ minutes per zone
- a computed load value (ULS)
- a load signature

Three Z2 sessions in a row are allowed —
but they will **not** produce identical load.

Meso-wave position, duration, and context always matter.

---

<!-- ALG: paradigm_selector -->
## Planning Paradigm Selection

| min_session_max | max_session_max | min_sessions_per_7d | max_sessions_per_7d | paradigm | max_quality_per_7d | max_micro_pct_meso | allow_long_run |
|-----------------|-----------------|---------------------|---------------------|----------|--------------------|--------------------|----------------|
| 0 | 60 | 1 | 14 | micro_structured | 2 | 0.40 | false |
| 75 | 999 | 8 | 14 | high_volume | 4 | 0.00 | true |
| 0 | 999 | 1 | 14 | standard | 3 | 0.00 | true |

Notes:
- First matching row wins (order matters)
- micro_structured: short sessions ≤60 min
- high_volume: long sessions ≥75 min with high frequency
- standard: default paradigm

---

<!-- ALG: micro_med -->
## Minimum Effective Dose for Micro Sessions

| zone | standard_med_min | micro_med_min | micro_max_per_session |
|------|------------------|---------------|-----------------------|
| Z1 | 20 | 10 | 30 |
| Z2 | 20 | 10 | 30 |
| Z3 | 10 | 5 | 15 |
| Z4 | 8 | 4 | 12 |
| Z5 | 4 | 2 | 8 |
| Z6 | 2 | 1 | 4 |
| Z8 | 0.5 | 0.2 | 2 |

Notes:
- micro_med_min = minimum NTIZ for credit in micro paradigm
- Below micro_med_min = no zone credit
- Used only for micro_structured paradigm

---

<!-- ALG: spacing_rules -->
## Spacing Rules (Context-Aware)

| session_context | zone | min_hours |
|-----------------|------|----------:|
| standard | Z3 | 24 |
| standard | Z4 | 48 |
| standard | Z5 | 60 |
| standard | Z6 | 72 |
| standard | Z7 | 72 |
| standard | Z8 | 72 |
| micro | Z3 | 12 |
| micro | Z4 | 24 |
| micro | Z5 | 36 |
| micro | Z6 | 48 |
| micro | Z7 | 48 |
| micro | Z8 | 48 |
| commute | Z3 | 18 |
| commute | Z4 | 36 |
| commute | Z5 | 48 |
| commute | Z6 | 60 |
| commute | Z7 | 60 |
| commute | Z8 | 60 |
| senior | Z3 | 36 |
| senior | Z4 | 72 |
| senior | Z5 | 96 |
| senior | Z6 | 96 |
| senior | Z7 | 96 |
| senior | Z8 | 96 |

Notes:
- Context-aware minimum hours between sessions touching same zone
- Micro allows tighter spacing (shorter sessions = faster recovery)
- Senior requires longer gaps (reduced recovery capacity)
- Research basis: ≥48h between Z4+ sessions for standard adaptation

---

<!-- ALG: zone_spacing_hours -->
## Zone Spacing Hours (Fallback)

| zone | min_hours |
|------|----------:|
| Z3 | 24 |
| Z4 | 48 |
| Z5 | 60 |
| Z6 | 72 |
| Z7 | 72 |
| Z8 | 72 |

Notes:
- Fallback when context-specific spacing not available
- Z1/Z2 have no spacing constraint (can stack)
- Used when session_context is unknown or default

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|------|----------|------|
| Zone meaning | zone_physiology.md | Physiology |
| Athlete scaling | modifiers.md | Modifiers |
| Periodization wave | periodization.md | Periodization |
| Load gating | load_monitoring.md | Load |

---

<!-- META: boundaries -->
## Hard Boundaries

This card:
- Computes **planned and executed load**
- Does NOT gate sessions
- Does NOT assess readiness
- Does NOT change plans retroactively

Load is **planned here**.
Load is **monitored elsewhere**.

Invariant:
- Engine reasoning uses **meso days only**
- Calendar weeks exist only in UI language

End of card.

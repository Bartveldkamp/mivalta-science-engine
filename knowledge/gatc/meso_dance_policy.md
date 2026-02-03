<!-- META -->
# Meso Dance Policy — Canonical

concept_id: meso_dance_policy
axis_owner: load_positioning
version: 1.0
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Issurin (2016) | Block and step-loading outperform linear loading | Slice-based meso structure |
| Bompa & Haff (2009) | Neuromuscular adaptation needs intro period | Ramp-in rules |
| Meeusen et al. (2013) | Front-loaded fatigue increases overtraining risk | Early intensity caps |
| Foster et al. (2001) | Load variation required for adaptation | Intra-meso load shaping |
| Mujika (2017) | Deload = volume down, intensity preserved | Unload slice rules |
| Ronnestad et al. (2014) | Maintain frequency during deload | No session removal |
| Seiler (2010) | Adaptation depends on contrast over time | Load positioning |

**Key conclusion:**
Correct adaptation depends not only on *how much* load is applied, but **when** it is applied inside the mesocycle.

---

<!-- CONCEPT -->
## Concept: Meso Dance

The *Meso Dance* defines **how load is positioned over time** inside a mesocycle.

It:
- shapes **temporal distribution of load**
- respects all feasibility, recovery, and monotony constraints
- never invents sessions
- never changes load math
- never overrides safety caps

> Meso Dance answers:
> **"Given this total load, where should it live in the meso?"**

---

<!-- ALG: slice_model -->
## Slice Model (Meso-Native)

Slices are **physiological windows**, not calendar weeks.

| meso_days | slice_count | slice_index | slice_start | slice_end | purpose |
|-----------|-------------|-------------|-------------|-----------|---------|
| 21 | 3 | 1 | 1 | 7 | ramp_in |
| 21 | 3 | 2 | 8 | 14 | overload |
| 21 | 3 | 3 | 15 | 21 | unload |
| 28 | 4 | 1 | 1 | 7 | ramp_in |
| 28 | 4 | 2 | 8 | 14 | accumulation |
| 28 | 4 | 3 | 15 | 21 | overload |
| 28 | 4 | 4 | 22 | 28 | unload |

Notes:
- Slices are mapped as **percentages of the meso**, not weeks
- Each slice has a distinct physiological purpose
- Unload slice is always the final slice

---

<!-- ALG: ramp_in_rules -->
## Ramp-In Rules (Planner-Phase)

Hard invariants for meso start.

| rule_id | description | days_affected | intensity_cap | volume_cap_pct |
|---------|-------------|---------------|---------------|----------------|
| RI1 | First 3 meso days intensity cap | 1-3 | Z3 | 80 |
| RI2 | First 2 training sessions must be easy | first_2_sessions | Z2 | 80 |
| RI3 | No back-to-back quality in ramp-in | 1-3 | n/a | n/a |

Notes:
- `intensity_cap`: maximum zone allowed (capped by max_zone)
- `volume_cap_pct`: max percentage of meso average daily load
- RI2 applies to first 2 actual training sessions, not days

---

<!-- ALG: ramp_in_modifiers -->
## Ramp-In Modifiers

Adjustments to ramp-in rules by athlete group.

| group | ramp_days | intensity_delta | volume_cap_delta |
|-------|-----------|-----------------|------------------|
| default | 3 | 0 | 0 |
| masters_40_plus | 5 | -1 | -10 |
| novice | 3 | -1 | -5 |
| elite | 3 | 0 | +5 |
| tier_c | 4 | 0 | -5 |
| tier_d | 5 | -1 | -10 |

Notes:
- `ramp_days`: how many days the ramp-in period lasts
- `intensity_delta`: zone adjustment (-1 means cap one zone lower)
- `volume_cap_delta`: adjustment to volume cap percentage

---

<!-- ALG: load_positioning -->
## Intra-Meso Load Positioning

Load targets are **relative to meso total**, not absolute.

| phase | meso_days | slice_index | load_min_pct | load_max_pct | max_intensity |
|-------|-----------|-------------|--------------|--------------|---------------|
| base | 28 | 1 | 70 | 80 | Z3 |
| base | 28 | 2 | 85 | 95 | Z4 |
| base | 28 | 3 | 95 | 105 | Z5 |
| base | 28 | 4 | 60 | 70 | Z3 |
| base | 21 | 1 | 75 | 85 | Z3 |
| base | 21 | 2 | 95 | 105 | Z4 |
| base | 21 | 3 | 65 | 75 | Z3 |
| build | 28 | 1 | 75 | 85 | Z4 |
| build | 28 | 2 | 90 | 100 | Z5 |
| build | 28 | 3 | 95 | 105 | Z6 |
| build | 28 | 4 | 65 | 75 | Z4 |
| build | 21 | 1 | 80 | 90 | Z4 |
| build | 21 | 2 | 95 | 105 | Z6 |
| build | 21 | 3 | 70 | 80 | Z4 |
| peak | 28 | 1 | 80 | 90 | Z5 |
| peak | 28 | 2 | 90 | 100 | Z6 |
| peak | 28 | 3 | 95 | 105 | Z6 |
| peak | 28 | 4 | 60 | 70 | Z5 |
| peak | 21 | 1 | 85 | 95 | Z5 |
| peak | 21 | 2 | 95 | 105 | Z6 |
| peak | 21 | 3 | 65 | 75 | Z5 |

Notes:
- `load_min_pct` / `load_max_pct`: target range as % of meso average slice load
- `max_intensity`: highest zone allowed in that slice (capped by max_zone)
- Overload slice (2 or 3) has highest targets
- Unload slice (3 or 4) has lowest targets

---

<!-- ALG: deload_rules -->
## Deload / Unload Rules

Hard rules for the unload slice.

| rule_id | description | value |
|---------|-------------|-------|
| DL1 | Deload slice is mandatory | true |
| DL2 | Volume reduction target | 40-60 pct |
| DL3 | Intensity preservation minimum | 80 pct |
| DL4 | Session frequency preservation | 80 pct |
| DL5 | No complete rest days added | true |
| DL6 | No intensity collapse | true |
| DL7 | No extra sessions | true |

Notes:
- Volume goes down, intensity maintained
- Session count stays similar (within 80%)
- These are constraints, not prescriptions

---

<!-- ALG: phase_caps -->
## Phase-Specific Intensity Caps

Maximum intensity by phase and slice type.

| phase | early_max | overload_max | deload_max |
|-------|-----------|--------------|------------|
| base | Z3 | Z4 | Z3 |
| build | Z4 | Z6 | Z4 |
| peak | Z5 | Z6 | Z5 |
| deload | Z3 | Z3 | Z2 |

Notes:
- `early_max`: max intensity in ramp-in slice
- `overload_max`: max intensity in overload slice
- `deload_max`: max intensity in unload slice
- All caps further constrained by athlete's max_zone

---

<!-- JOSI: explanation -->
## Josi Explanation — Why the Dance Matters

Training works best when stress **arrives at the right moment**.

If hard work comes too early:
- fatigue accumulates before adaptation
- injury risk rises
- the meso collapses

If everything feels the same:
- adaptation stalls
- motivation drops
- injury risk increases

### What MiValta's Meso Dance does

**Eases you in**: The first few days are deliberately lighter. Your body needs to remember it's training again.

**Builds pressure gradually**: The middle of your meso is where the real work happens. This is when your body is ready.

**Releases fatigue deliberately**: The final days back off just enough to let adaptation consolidate — without losing fitness.

### What you'll notice

- First sessions feel manageable
- Middle sessions feel challenging (that's the point)
- Final sessions feel refreshing
- You arrive at the next meso ready, not wrecked

This is how experienced coaches periodize — now done consistently.

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Load math | training_load_model.md | Load |
| Monotony prevention | monotony_policy.md | Load |
| Feasibility caps | feasibility_policy.md | Policy |
| Zone definitions | zone_physiology.md | Physiology |
| Phase structure | periodization.md | Periodization |
| Energy systems | energy_systems.md | Physiology |

---

<!-- META: boundaries -->
## Hard Boundaries

This policy does NOT:
- compute load (that's training_load_model)
- change minutes (availability is truth)
- override feasibility or monotony guards
- create or remove sessions
- perform monitoring (that's load_monitoring)

**Meso Dance only positions load over time.**

End of card.

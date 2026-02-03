<!-- META -->
# Pack Composition — Canonical

concept_id: pack_composition
axis_owner: periodization
version: 1.0
status: frozen

---

## Purpose

This core card enables MiValta to **add new sports / activities as modular packs** and to **activate/deactivate those packs over time** inside a single macro plan (e.g., "add strength from meso 3 onward").

This card defines **composition only**:
- what packs exist
- when packs are active (macro → meso timeline)
- how much each pack is allowed to contribute (allocation per meso)
- how new packs ramp in safely (intro ramp)
- deterministic scaling rules across different meso lengths (21/28 days)

It does **not** define how a pack builds sessions (Session axis packs do that), and it does **not** define monitoring thresholds (Load axis does that).

---

<!-- ALG: pack_registry -->
## Pack Registry (What Packs Exist)

| pack_id | pack_type | provides_axes | enabled_by_default | notes |
|--------|-----------|---------------|--------------------|------|
| running | sport | session+modifiers | 1 | endurance running |
| cycling | sport | session+modifiers | 1 | endurance cycling |
| strength | activity | session | 0 | gym lifting |
| pilates | activity | session | 0 | control + stability |
| core | activity | session | 0 | trunk/stability |

Notes:
- `pack_id` is the canonical selector used by the engine.
- Packs live under `knowledge/cards/packs/<pack_id>/...`
- `provides_axes` indicates which axes the pack contributes tables to.
- Packs are modular: removing a pack folder removes that capability without touching core cards.

---

<!-- ALG: macro_pack_schedule -->
## Macro Pack Schedule (When Packs Are Active)

| macro_id | meso_index_from | meso_index_to | active_packs |
|---------|------------------|---------------|--------------|
| default | 1 | 999 | running+cycling |
| default_8mo_strength_from_3 | 1 | 2 | running+cycling |
| default_8mo_strength_from_3 | 3 | 8 | running+cycling+strength |

Notes:
- `active_packs` is a `+`-joined list of pack_ids.
- This is the deterministic mechanism that enables "turn on strength after 2 meso blocks."
- If a pack is not active in the current meso window, the session builder must not use its templates.

---

<!-- ALG: pack_allocation_basis -->
## Allocation Basis (Deterministic Scaling)

| basis_id | meso_days_basis | rounding_rule |
|---------|------------------|---------------|
| default | 28 | round_half_up |

Notes:
- All values in `pack_meso_allocation` are defined for `meso_days_basis`.
- If the actual meso length differs (e.g., 21 days), values must be scaled deterministically:
  scaled = round_half_up(value * meso_days_actual / meso_days_basis)
- `round_half_up` means: x.5 rounds to the next integer away from zero.

---

<!-- ALG: pack_meso_allocation -->
## Pack Meso Allocation (How Much Each Pack May Contribute)

| macro_id | meso_index_from | meso_index_to | pack_id | sessions_per_meso_min | sessions_per_meso_max | hit_like_cap_per_meso |
|---------|------------------|---------------|--------|------------------------|------------------------|-----------------------|
| default | 1 | 999 | running | 12 | 24 | 8 |
| default | 1 | 999 | cycling | 0 | 24 | 8 |
| default_8mo_strength_from_3 | 1 | 2 | strength | 0 | 0 | 0 |
| default_8mo_strength_from_3 | 3 | 8 | strength | 4 | 12 | 8 |

Notes:
- Values assume 28 meso_days (see `pack_allocation_basis` for scaling rule).
- `hit_like_cap_per_meso` is a hard cap on sessions marked "HIT-like" by the pack's Session rules.
  - Endurance packs: HIT-like means touches Z4–Z8.
  - Strength pack: HIT-like means high neural/mechanical demand bands (e.g., S3/S4) as defined by that pack.
- The engine must enforce these as feasibility constraints (not coach preferences).
- Scaling for non-standard meso lengths is mandatory and must follow Allocation Basis.

---

<!-- ALG: pack_intro_ramp -->
## Pack Intro Ramp (Safe Enablement of a New Pack)

| pack_id | ramp_meso_days | allowed_intensity_bands_during_ramp | post_ramp_allowed_bands |
|--------|-----------------|-------------------------------------|--------------------------|
| strength | 14 | S1+S2 | S1+S2+S3+S4+S5 |
| pilates | 7 | P1+P2 | P1+P2+P3 |
| core | 7 | C1+C2 | C1+C2+C3 |

Notes:
- When a pack becomes active in the schedule, it must start in a conservative intro ramp.
- `ramp_meso_days` = number of meso_days before full intensity bands unlock.
- Band IDs are defined inside the pack's Session card(s). This table only constrains *which* bands are eligible.
- This prevents a "new pack spike" from breaking LoadMonitor / injury risk.

---

<!-- ALG: pack_priority -->
## Pack Priority (Tie-breaker When Feasibility Is Tight)

| macro_id | meso_index_from | meso_index_to | primary_pack | secondary_packs |
|---------|------------------|---------------|--------------|-----------------|
| default | 1 | 999 | running | cycling |
| default_8mo_strength_from_3 | 1 | 2 | running | cycling |
| default_8mo_strength_from_3 | 3 | 8 | running | cycling+strength |

Notes:
- If feasibility constraints force reductions, the engine reduces from lowest priority packs first.
- This is deterministic and prevents chaotic tradeoffs.

---

<!-- JOSI: pack_explained -->
## What Are Packs?

Packs are add-on activities you can bring into your training plan.

**Sport packs** (running, cycling) are your main endurance work.
**Activity packs** (strength, pilates, core) complement your endurance training.

When you add a new pack, it starts conservatively — we don't throw you into the deep end. The system ramps you in safely over the first couple of weeks.

---

<!-- JOSI: adding_pack -->
## Adding a New Activity

When you tell Josi "I want to add strength training," here's what happens:

1. The pack activates at the next meso boundary (or at your chosen meso index)
2. You start with lighter intensity bands only during the intro ramp
3. After the intro ramp, full intensity unlocks
4. Your plan automatically balances the new load using the allocation rules

You don't need to figure out how to fit it in — the system handles the integration.

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis | Status |
|------|----------|------|--------|
| Endurance zones meaning | zone_physiology.md | physiology | active |
| Athlete scalers | modifiers.md | modifiers | active |
| Macro/meso wave logic | periodization.md | periodization | active |
| Session legality + load units | session_rules.md | session | active |
| Load math thresholds | training_load_model.md | load | active |
| Load policy mapping | load_monitoring.md | load | active |
| Strength templates/bands | packs/strength/session_strength.md | session | planned |
| Pilates templates/bands | packs/pilates/session_pilates.md | session | planned |

---

<!-- META: boundaries -->
## Hard Boundaries

This card does NOT:
- define session templates (packs do)
- define load thresholds (training_load_model does)
- override readiness state (Viterbi does)
- compute load (LoadMonitor does)

This card defines **which packs are active and allowed** in each meso window and **how much** each pack may contribute.

**Invariant:** All allocation values are per MESO (meso_days), never per week.

End of card.

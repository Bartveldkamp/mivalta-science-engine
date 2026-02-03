<!-- META -->
# Session Variety Policy — Canonical

concept_id: session_variety_policy
axis_owner: expression
version: 1.0
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Issurin (2010) | Training monotony reduces adaptation | Variant rotation within blocks |
| Kiely (2012) | Variability prevents stagnation | No identical stimulus patterns |
| Rhea et al. (2002) | Varied programming superior for strength | Extend to endurance quality work |
| Foster et al. (2001) | Monotony index predicts overtraining | Avoid repetitive stimulus |

**Key conclusion:**
Athletes adapt better when the same physiological target is approached through varied stimulus patterns.

---

<!-- CONCEPT -->
## Concept: Session Variety

Session variety ensures **different workout structures** for the **same energy system target** within a training block (slice).

It:
- prevents repetitive stimulus patterns
- maintains athlete engagement
- supports broader adaptation
- operates at expression layer only (never changes load math)

> Session Variety answers:
> **"Given this zone intent, which structural variant should we use?"**

---

<!-- ALG: zone_band_map -->
## Zone Band Mapping

Maps training zones to physiological zone bands for variety rules.

| zone | zone_band |
|------|-----------|
| R | recovery |
| Z1 | Z1_Z2 |
| Z2 | Z1_Z2 |
| Z3 | Z3 |
| Z4 | Z4 |
| Z5 | Z5 |
| Z6 | Z6_plus |
| Z7 | Z7_Z8 |
| Z8 | Z7_Z8 |

Notes:
- Zone bands group zones by physiological similarity
- Z1/Z2 are grouped (endurance, high repeatability)
- Z7/Z8 are grouped (neuromuscular, short efforts)
- Quality zones (Z3+) each have unique bands

---

<!-- ALG: primary_systems -->
## Primary Energy Systems by Zone Band

| zone_band | primary_system |
|-----------|----------------|
| recovery | recovery |
| Z1_Z2 | endurance |
| Z3 | steady_state_threshold |
| Z4 | aerobic_power |
| Z5 | aerobic_power |
| Z6_plus | anaerobic_power |
| Z7_Z8 | neuromuscular |

Notes:
- Primary system determines variety grouping
- Same system = same variety constraint
- Z4 and Z5 share aerobic_power system

---

<!-- ALG: repeat_windows -->
## Repeat Window Rules

Defines whether variants can repeat within the same slice.

| zone_band | primary_system | no_repeat_within_slice |
|-----------|----------------|------------------------|
| recovery | recovery | false |
| Z1_Z2 | endurance | false |
| Z3 | steady_state_threshold | true |
| Z4 | aerobic_power | true |
| Z5 | aerobic_power | true |
| Z6_plus | anaerobic_power | true |
| Z7_Z8 | neuromuscular | true |

Notes:
- `no_repeat_within_slice=true`: variant_id may appear at most once per slice for that key
- `no_repeat_within_slice=false`: repeats allowed (Z1/Z2/R have high tolerance)
- If only 1 viable variant exists, allow repeat (never block planning)

---

<!-- ALG: variety_key -->
## Variety Key Structure

The key for tracking used variants:

```
(slice_id, primary_system, zone_band, session_class, phase, position_band)
```

Components:
- `slice_id`: 1-4 based on meso_day position
- `primary_system`: from zone_band lookup
- `zone_band`: from zone lookup
- `session_class`: standard, micro_hiit, etc.
- `phase`: base, build, peak, deload
- `position_band`: early, mid, late

Memory structure:
```
key -> set(variant_id)
```

Selection rule:
1. Sort valid variants by priority (existing logic)
2. If `no_repeat_within_slice=true`: pick first not in used-set
3. If all in used-set OR only 1 viable: pick first (allow repeat)
4. Update memory with chosen variant_id

---

<!-- ALG: slice_computation -->
## Slice ID Computation

```python
def get_slice_id(meso_day: int, meso_days: int) -> int:
    if meso_days == 21:
        if meso_day <= 7: return 1
        if meso_day <= 14: return 2
        return 3
    else:  # 28 days
        if meso_day <= 7: return 1
        if meso_day <= 14: return 2
        if meso_day <= 21: return 3
        return 4
```

Notes:
- Matches meso_dance_policy slice_model
- Slice boundaries: 7, 14, 21, (28)
- Deterministic, no randomness

---

<!-- JOSI: explanation -->
## Josi Explanation — Why Variety Matters

Training works best when your body encounters **familiar challenges in fresh ways**.

If every threshold session is identical:
- your body stops adapting
- training feels monotonous
- you miss opportunities for broader fitness

### What MiValta's Session Variety does

**Rotates workout structures**: Same energy system target, different execution pattern.

**Respects energy system groupings**: A Z4 pyramid and Z4 cruise intervals both target aerobic power — you won't see both in the same week.

**Allows endurance repeats**: Easy runs (Z1/Z2) can repeat — they're meant to be familiar and restorative.

### What you'll notice

- Quality sessions feel fresh each time
- Similar workouts appear in different weeks
- Easy days can be predictable (that's fine)
- No two identical hard workouts back-to-back

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Slice model | meso_dance_policy.md | Load |
| Zone definitions | zone_physiology.md | Physiology |
| Session structure | session_rules.md | Expression |
| Monotony prevention | monotony_policy.md | Load |

---

<!-- META: boundaries -->
## Hard Boundaries

This policy does NOT:
- change load math (that's training_load_model)
- change zone assignment (that's composer/feasibility)
- persist across mesos (memory is meso-local)
- block planning (always allows fallback to best variant)

**Session Variety only selects structural execution.**

End of card.

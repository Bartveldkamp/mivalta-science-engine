<!-- META -->
# Feasibility Policy — Canonical

concept_id: feasibility_policy
axis_owner: policy
version: 1.0
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Foster (1996) | Training age correlates with adaptation rate | Experience modifiers |
| Tanaka (2001) | Age-related decline in recovery capacity | Age modifiers |
| Seiler (2010) | Diminishing returns beyond polarized threshold | Volume-adaptation curves |
| Hawkins (2003) | Masters athletes maintain trainability with adjusted recovery | Confidence scaling |
| Banister (1991) | Fitness-fatigue model shows logarithmic gains | Diminishing returns basis |
| Mujika (2000) | Goal complexity affects preparation requirements | Complexity scoring |

---

<!-- ALG: experience -->
## Experience Modifiers

Training years affect confidence and risk assessment.

| years_min | years_max | confidence_mult | risk_buffer |
|-----------|-----------|-----------------|-------------|
| 0 | 0 | 0.60 | 0.0 |
| 1 | 1 | 0.75 | 0.1 |
| 2 | 2 | 0.85 | 0.2 |
| 3 | 5 | 0.95 | 0.3 |
| 6 | 10 | 1.00 | 0.4 |
| 11 | 99 | 1.00 | 0.5 |

Notes:
- `confidence_mult`: multiplier on base confidence score
- `risk_buffer`: margin added to feasibility threshold (0.0-0.5)
- New athletes (0 years) get highest risk, lowest confidence
- Veterans (10+ years) get full confidence and highest buffer

---

<!-- ALG: age -->
## Age Modifiers

Age affects recovery and achievability confidence.

| age_min | age_max | confidence_mult | recovery_factor |
|---------|---------|-----------------|-----------------|
| 13 | 25 | 1.00 | 1.00 |
| 26 | 35 | 1.00 | 0.95 |
| 36 | 45 | 0.95 | 0.85 |
| 46 | 55 | 0.90 | 0.75 |
| 56 | 65 | 0.85 | 0.65 |
| 66 | 99 | 0.75 | 0.55 |

Notes:
- `confidence_mult`: age-based scaling of goal achievability
- `recovery_factor`: recovery capacity relative to peak (1.0)
- Masters athletes remain fully capable with appropriate recovery
- These are defaults; individual variation exists

---

<!-- ALG: diminishing_returns -->
## Diminishing Returns by Level

Adaptation units gained per hour of training.
Curves differ by experience level. Meso-native (21 or 28 day totals).

| level | meso_days | h_min | h_max | units_per_hour |
|-------|-----------|-------|-------|----------------|
| beginner | 21 | 0 | 12 | 2.0 |
| beginner | 21 | 12 | 24 | 1.5 |
| beginner | 21 | 24 | 36 | 1.0 |
| beginner | 21 | 36 | 48 | 0.5 |
| beginner | 21 | 48 | 999 | 0.25 |
| beginner | 28 | 0 | 16 | 2.0 |
| beginner | 28 | 16 | 32 | 1.5 |
| beginner | 28 | 32 | 48 | 1.0 |
| beginner | 28 | 48 | 64 | 0.5 |
| beginner | 28 | 64 | 999 | 0.25 |
| novice | 21 | 0 | 12 | 1.5 |
| novice | 21 | 12 | 24 | 1.2 |
| novice | 21 | 24 | 36 | 0.9 |
| novice | 21 | 36 | 48 | 0.6 |
| novice | 21 | 48 | 999 | 0.3 |
| novice | 28 | 0 | 16 | 1.5 |
| novice | 28 | 16 | 32 | 1.2 |
| novice | 28 | 32 | 48 | 0.9 |
| novice | 28 | 48 | 64 | 0.6 |
| novice | 28 | 64 | 999 | 0.3 |
| intermediate | 21 | 0 | 12 | 1.0 |
| intermediate | 21 | 12 | 24 | 1.0 |
| intermediate | 21 | 24 | 36 | 0.8 |
| intermediate | 21 | 36 | 48 | 0.6 |
| intermediate | 21 | 48 | 999 | 0.4 |
| intermediate | 28 | 0 | 16 | 1.0 |
| intermediate | 28 | 16 | 32 | 1.0 |
| intermediate | 28 | 32 | 48 | 0.8 |
| intermediate | 28 | 48 | 64 | 0.6 |
| intermediate | 28 | 64 | 999 | 0.4 |
| advanced | 21 | 0 | 12 | 0.8 |
| advanced | 21 | 12 | 24 | 0.8 |
| advanced | 21 | 24 | 36 | 0.7 |
| advanced | 21 | 36 | 48 | 0.6 |
| advanced | 21 | 48 | 999 | 0.5 |
| advanced | 28 | 0 | 16 | 0.8 |
| advanced | 28 | 16 | 32 | 0.8 |
| advanced | 28 | 32 | 48 | 0.7 |
| advanced | 28 | 48 | 64 | 0.6 |
| advanced | 28 | 64 | 999 | 0.5 |
| elite | 21 | 0 | 12 | 0.6 |
| elite | 21 | 12 | 24 | 0.6 |
| elite | 21 | 24 | 36 | 0.6 |
| elite | 21 | 36 | 48 | 0.55 |
| elite | 21 | 48 | 999 | 0.5 |
| elite | 28 | 0 | 16 | 0.6 |
| elite | 28 | 16 | 32 | 0.6 |
| elite | 28 | 32 | 48 | 0.6 |
| elite | 28 | 48 | 64 | 0.55 |
| elite | 28 | 64 | 999 | 0.5 |

Notes:
- `meso_days`: 21 or 28 day meso cycle
- `h_min`, `h_max`: TOTAL hours across the entire meso (not per week)
- `units_per_hour`: adaptation units gained in that hour range
- Beginners gain 2x per hour at low volume vs elites
- All levels plateau at high meso volume (injury risk rises faster than adaptation)
- Used to estimate achievable adaptation, not to prescribe volume

---

<!-- ALG: complexity -->
## Goal Complexity

Minimum experience and complexity score by archetype.

| archetype | min_experience_years | complexity_score |
|-----------|---------------------|------------------|
| marathon_endurance | 2 | 0.9 |
| long_endurance | 1 | 0.6 |
| short_endurance | 0 | 0.3 |
| speed_endurance | 1 | 0.5 |
| endurance_event | 1 | 0.6 |
| threshold_event | 0 | 0.4 |
| long_tri | 3 | 1.0 |
| medium_tri | 2 | 0.7 |
| short_tri | 1 | 0.4 |
| general | 0 | 0.1 |

Notes:
- `min_experience_years`: below this triggers risk_flag
- `complexity_score`: 0.0-1.0, affects confidence penalty when under-experienced
- Ironman (long_tri) is highest complexity, general fitness is lowest
- Under-experience penalty: confidence × (1 - complexity_score × 0.2)

---

<!-- ALG: clamps -->
## Feasibility Clamps

Hard bounds for all feasibility calculations.

| parameter | min_value | max_value |
|-----------|-----------|-----------|
| confidence_score | 0.0 | 1.0 |
| achievability_score | 0.0 | 1.0 |
| experience_mult | 0.5 | 1.0 |
| age_mult | 0.5 | 1.0 |
| recovery_factor | 0.4 | 1.0 |
| complexity_penalty | 0.0 | 0.3 |

Notes:
- All multipliers and scores clamped to these bounds
- Prevents runaway calculations from bad input
- No magic constants in code; all bounds from this table

---

<!-- ALG: tier_thresholds -->
## Achievability Tier Thresholds

Maps achievability score to tier and plan mode.

| tier | score_min | score_max | plan_mode | recheck_after_meso |
|------|-----------|-----------|-----------|-------------------|
| A | 0.85 | 1.00 | performance_path | 0 |
| B | 0.70 | 0.84 | finish_path | 0 |
| C | 0.50 | 0.69 | finish_path | 1 |
| D | 0.00 | 0.49 | maintenance_path | 1 |

Notes:
- Tier A: full goal achievable, plan for performance
- Tier B: goal likely, conservative constraints
- Tier C: goal stretch, focus on base, recheck after meso
- Tier D: maintenance mode, build foundation, recheck required
- `recheck_after_meso`: 1 means re-run feasibility after next meso

---

<!-- JOSI: feasibility_explained -->
## How We Assess Your Goal

We calculate **achievability** — how realistic your goal is given your situation.

### What we look at:
1. **Time you have** vs time the goal requires
2. **Longest session you can do** vs what's needed
3. **Your experience** with this type of training
4. **Your age** (affects recovery, not capability)

### What we do with it:
- **Tier A (85%+)**: You can chase performance goals
- **Tier B (70-84%)**: Achievable with conservative approach
- **Tier C (50-69%)**: Stretch goal, focus on completion
- **Tier D (<50%)**: Build your base, reassess after one cycle

**We never stop your training.** We adjust the approach to match reality.

---

<!-- JOSI: diminishing_explained -->
## More Training Doesn't Always Mean More Gains

Your body adapts logarithmically, not linearly.

**Beginners**: Each hour of training produces big gains
**Experienced athletes**: More hours give smaller returns

This is why:
- Beginners shouldn't train like elites
- Elites need precision, not just volume
- Everyone has an optimal training dose

**The system calculates your optimal dose** based on your level and goals.

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Goal archetypes | goal_demands.md | Physiology |
| Energy systems | energy_systems.md | Physiology |
| Age/level modifiers | modifiers.md | Modifiers |
| Load monitoring | load_monitoring.md | Load |
| Periodization | periodization.md | Periodization |

---

<!-- META: boundaries -->
## Hard Boundaries

This card does NOT define:
- Goal physiology → `goal_demands.md`
- Session structure → `session_rules.md`
- Meso/macro planning → `periodization.md`
- Zone definitions → `zone_physiology.md`

**Feasibility Policy defines only:**
- Modifiers for confidence calculation
- Diminishing returns curves
- Achievability tier thresholds
- Hard clamps on all calculations

End of card.

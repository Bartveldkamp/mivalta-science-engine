<!-- META -->
# Monotony Policy — Canonical

concept_id: monotony_policy
axis_owner: load
version: 1.0
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Foster (1996) | High monotony (mean load / SD) strongly correlates with illness and injury | Core monotony metric (PMI) |
| Foster (2001) | Monotony > 2.0 dramatically increases injury risk | Upper safety bounds |
| Banister (1991) | Adaptation depends on load variation, not just magnitude | Need for day-to-day variance |
| Seiler (2010) | Polarized models rely on contrast, not uniformity | Justifies variety even in base |
| Mujika (2000) | Excessively uniform training blunts performance gains | Prevents flat adaptation |
| Meeusen (2013) | Chronic monotony precedes non-functional overreaching | Planner-phase prevention |

**Key conclusion:**
Monotony is a **causal risk factor**, not merely a monitoring signal.
It must be controlled **before training happens**, not only observed afterward.

---

<!-- CONCEPT -->
## Concept: Planner-Phase Monotony Prevention

Monotony is defined as **excessively uniform daily training load** across a mesocycle.

In MiValta:
- Monotony is prevented **during planning**
- Not deferred to post-hoc monitoring
- Not solved by random variation
- Not solved by adding sessions

Instead:
- We **quantify monotony deterministically**
- We **adjust intent sequencing** within strict constraints
- We **never invent time or violate recovery rules**

---

<!-- ALG: metric_definition -->
## Metric Definition — Planned Monotony Index (PMI)

Planner-phase adaptation of Foster's Monotony Index.

### Daily Load Proxy
For each meso day:

```
daily_load = load_factor(zone) × session_minutes
```

- `load_factor` comes from `energy_systems.zone_map`
- Rest days contribute `0`
- Minutes are taken from canonical availability (truth)

### PMI Formula

```
PMI = mean(daily_load) / standard_deviation(daily_load)
```

Special cases:
- `std = 0` → PMI = infinity (guaranteed monotony)
- No training → PMI = 0

---

<!-- ALG: load_factors -->
## Load Factors by Zone

Zone to load factor mapping for PMI calculation.
Same as energy_systems.zone_map but isolated for monotony use.

| zone | load_factor | bucket |
|------|-------------|--------|
| R | 0.5 | recovery |
| Z1 | 0.8 | volume |
| Z2 | 1.0 | volume |
| Z3 | 1.3 | threshold |
| Z4 | 1.6 | vo2max |
| Z5 | 1.8 | vo2max |
| Z6 | 2.0 | anaerobic |
| Z7 | 2.2 | neuromuscular |
| Z8 | 2.5 | neuromuscular |

Notes:
- Higher zones have higher load factors (more stress per minute)
- Z2 is reference (1.0)
- Used only for monotony calculation, not load accounting

---

<!-- ALG: pmi_caps -->
## PMI Safety Caps (Planner Guard)

Maximum allowed PMI by phase and feasibility tier.
Meso-native: applies to entire meso (21 or 28 days).

| phase | tier | max_pmi |
|-------|------|---------|
| base | A | 2.0 |
| base | B | 2.1 |
| base | C | 2.2 |
| base | D | 2.4 |
| build | A | 1.8 |
| build | B | 1.9 |
| build | C | 2.0 |
| build | D | 2.3 |
| peak | A | 1.6 |
| peak | B | 1.7 |
| peak | C | 1.9 |
| peak | D | 2.2 |
| deload | A | 2.2 |
| deload | B | 2.3 |
| deload | C | 2.4 |
| deload | D | 2.6 |

Notes:
- Peak phases require **more contrast**, hence lower caps
- Tier D allows smoother patterns for safety
- Caps are **hard planner limits**, not advice
- PMI calculated over entire meso (21 or 28 days), not weekly

---

<!-- ALG: adjustment_rules -->
## Deterministic Adjustment Rules

When PMI exceeds its cap, the planner may apply **only these changes**, in order.

| rule_id | description | from_zone | to_zone | requires_quality | max_zone_required |
|---------|-------------|-----------|---------|------------------|-------------------|
| R1 | Alternate easy aerobic work | Z1 | Z2 | 0 | Z2 |
| R2 | Inject low-quality intensity | Z2 | Z3 | 1 | Z3 |
| R3 | Inject moderate intensity | Z3 | Z4 | 1 | Z4 |

Notes:
- Rules are applied **deterministically** (no randomness)
- Rules applied **in order** (R1 then R2 then R3)
- Each rule only applied **if legal** under:
  - max_zone constraint
  - system spacing constraint
  - max sessions per system constraint
  - quality session cap
- If no legal rule reduces PMI, plan is accepted as **best achievable**

---

<!-- ALG: quality_caps -->
## Quality Session Caps

Quality sessions = zones >= Z3.
These caps interact with feasibility tier.

| tier | quality_fraction_cap |
|------|----------------------|
| A | 1.00 |
| B | 0.90 |
| C | 0.80 |
| D | 0.50 |

Effective cap calculation:
```
effective_quality_cap = min(
    feasibility.quality_sessions_mult,
    tier.quality_fraction_cap
)
```

Notes:
- Feasibility safety always wins
- Monotony prevention never forces intensity
- Cap applies to fraction of training sessions, not absolute count

---

<!-- ALG: variance_targets -->
## Variance Targets by Phase

Target standard deviation of daily load as fraction of mean.
Used to validate PMI is within healthy range.

| phase | min_cv | target_cv | max_cv |
|-------|--------|-----------|--------|
| base | 0.35 | 0.50 | 0.80 |
| build | 0.40 | 0.55 | 0.85 |
| peak | 0.45 | 0.60 | 0.90 |
| deload | 0.30 | 0.45 | 0.70 |

Notes:
- `cv` = coefficient of variation = std / mean
- `min_cv`: below this is too monotonous
- `target_cv`: ideal variance level
- `max_cv`: above this is too chaotic
- PMI = 1/cv, so lower CV = higher PMI

---

<!-- JOSI: explanation -->
## Josi Explanation — Why Variation Matters

Your body doesn't adapt best to **repetition** — it adapts to **contrast**.

Even if all workouts are easy, doing the **same load every day**:
- Raises injury risk
- Flattens performance gains
- Reduces motivation and freshness

### What MiValta does differently

- We **measure monotony before training starts**
- We **adjust the order and type of sessions**
- We **never add training you didn't plan**
- We **never break recovery rules**

If intensity is limited for safety reasons:
- We still create variety using aerobic contrast
- We avoid "every day feels the same"

### What you'll notice

- Training weeks feel more natural
- Easy days are truly easy
- Harder days stand out — when allowed
- Fatigue accumulates more predictably

This is how experienced human coaches plan — now done consistently.

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Load factors | energy_systems.md | Load |
| Feasibility caps | feasibility_policy.md | Policy |
| Goal demands | goal_demands.md | Intent |
| Session spacing | energy_systems.md | Recovery |
| Phase logic | periodization.md | Periodization |
| Zone definitions | zone_physiology.md | Physiology |

---

<!-- META: boundaries -->
## Hard Boundaries

This policy does NOT:
- Add or remove training days
- Change session duration
- Override recovery or spacing rules
- Replace macro/meso "dance" logic
- Monitor athlete response (that's load_monitoring)

**Monotony Policy does exactly one thing:**
> Prevent unsafe or unproductive uniformity **before the plan is delivered**.

End of card.

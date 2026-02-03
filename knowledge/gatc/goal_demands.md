<!-- META -->
# Goal Demands — Canonical

concept_id: goal_demands
axis_owner: physiology
version: 1.0
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Seiler (2010) | Polarized training: 80% low / 20% high optimal for endurance | Zone distribution basis |
| Billat (2001) | Marathon performance linked to time at race pace | min_long_session requirement |
| Laursen (2002) | VO2max sessions require min 3min intervals | Energy system demands |
| Foster (1996) | Goal specificity determines energy system priority | Archetype design |
| Midgley (2007) | Event duration dictates training zone emphasis | Duration-to-zone mapping |
| Stellingwerff (2019) | Elite marathon: 120-150h prep minimum | min_total_hours basis |

---

<!-- ALG: archetypes -->
## Goal Archetypes

Stable set of physiological demand profiles (~15 rows max).
Archetypes define WHAT the goal requires, not HOW to train it.

| archetype | sport | z2_pct | z3_pct | z4_pct | z5_pct | z6_plus_pct | min_prep_hours | min_meso_hours_21 | min_meso_hours_28 | min_long_session_min |
|-----------|-------|--------|--------|--------|--------|-------------|----------------|-------------------|-------------------|----------------------|
| marathon_endurance | running | 70 | 20 | 10 | 0 | 0 | 120 | 15 | 20 | 150 |
| long_endurance | running | 60 | 25 | 15 | 0 | 0 | 60 | 12 | 16 | 90 |
| short_endurance | running | 50 | 25 | 20 | 5 | 0 | 30 | 9 | 12 | 60 |
| speed_endurance | running | 40 | 25 | 20 | 15 | 0 | 20 | 9 | 12 | 45 |
| endurance_event | cycling | 75 | 15 | 10 | 0 | 0 | 100 | 18 | 24 | 180 |
| threshold_event | cycling | 50 | 25 | 20 | 5 | 0 | 60 | 12 | 16 | 60 |
| long_tri | triathlon | 75 | 15 | 10 | 0 | 0 | 200 | 21 | 28 | 240 |
| medium_tri | triathlon | 60 | 25 | 15 | 0 | 0 | 100 | 15 | 20 | 120 |
| short_tri | triathlon | 50 | 25 | 20 | 5 | 0 | 50 | 12 | 16 | 60 |
| general | any | 60 | 25 | 15 | 0 | 0 | 0 | 0 | 0 | 40 |

Notes:
- `z*_pct` columns MUST sum to 100
- `min_prep_hours`: lifetime/prep exposure requirement (compared vs training history)
- `min_meso_hours_21`: minimum hours required per 21-day meso
- `min_meso_hours_28`: minimum hours required per 28-day meso
- `min_long_session_min`: critical constraint — if athlete can't do this, reduce goal scope
- `general` archetype has no minimum hours (always feasible)
- Zone percentages express physiological demand, not training prescription

---

<!-- ALG: type_map -->
## Goal Type Mapping

Maps user-facing goal types to archetypes.
This table can grow without affecting physiology definitions.

| goal_type | sport | goal_class | archetype |
|-----------|-------|------------|-----------|
| marathon | running | finish | marathon_endurance |
| marathon | running | performance | marathon_endurance |
| half_marathon | running | finish | long_endurance |
| half_marathon | running | performance | long_endurance |
| 10k | running | finish | short_endurance |
| 10k | running | performance | speed_endurance |
| 5k | running | finish | short_endurance |
| 5k | running | performance | speed_endurance |
| century | cycling | finish | endurance_event |
| century | cycling | performance | endurance_event |
| gran_fondo | cycling | finish | endurance_event |
| metric_century | cycling | finish | threshold_event |
| ironman | triathlon | finish | long_tri |
| half_ironman | triathlon | finish | medium_tri |
| olympic_tri | triathlon | finish | short_tri |
| sprint_tri | triathlon | finish | short_tri |
| general | any | finish | general |
| maintenance | any | finish | general |

Notes:
- `goal_class` values: finish (complete safely), performance (time goal), elite (reserved)
- Same goal_type can map to same archetype for finish vs performance
- `any` sport is wildcard for cross-sport goals
- New goal_types can be added without changing archetypes

---

<!-- ALG: sport_goal_matrix -->
## Sport/Goal Matrix

Valid sport/goal combinations with event-specific requirements.

| sport | goal_type | min_weeks | peak_intensity | volume_emphasis | notes |
|-------|-----------|-----------|----------------|-----------------|-------|
| cycling | gran_fondo | 12 | Z4 | high | endurance_event |
| cycling | stage_race | 16 | Z5 | very_high | multi_day_event |
| cycling | crit | 8 | Z6 | moderate | anaerobic_power |
| cycling | time_trial | 10 | Z4 | moderate | threshold_focus |
| running | marathon | 16 | Z4 | very_high | aerobic_endurance |
| running | half_marathon | 12 | Z4 | high | tempo_endurance |
| running | 10k | 10 | Z5 | moderate | vo2max_speed |
| running | 5k | 8 | Z5 | moderate | speed_endurance |
| triathlon | ironman | 20 | Z4 | very_high | ultra_endurance |
| triathlon | half_ironman | 16 | Z4 | very_high | long_endurance |
| triathlon | olympic_tri | 12 | Z5 | high | mixed_intensity |
| triathlon | sprint_tri | 8 | Z5 | moderate | speed_power |
| swimming | distance | 12 | Z4 | high | aerobic_technique |
| swimming | sprint | 8 | Z6 | moderate | power_speed |

Notes:
- `min_weeks`: Minimum recommended preparation time
- `peak_intensity`: Target peak intensity zone for the event
- `volume_emphasis`: How volume-dependent the goal is
- Research: Event-specific periodization requirements

---

<!-- JOSI: goal_types_explained -->
## Understanding Your Goal

Every goal has a **physiological demand profile** — what energy systems need to be developed.

### Endurance Goals (Marathon, Century, Ironman)
- Dominated by aerobic (Z2) development
- Require long sessions to build endurance capacity
- Need significant preparation time

### Threshold Goals (10k, Time Trial)
- Mix of aerobic base and higher intensity
- Shorter required sessions but more intensity

### Speed Goals (5k, Sprint Tri)
- Higher proportion of threshold and VO2max work
- Still need aerobic foundation

**Your training is shaped by your goal.**
The system matches your available time to what your goal requires.

---

<!-- JOSI: achievability_explained -->
## Can You Reach Your Goal?

The system calculates **achievability** based on:

1. **Time available** vs time required
2. **Longest session you can do** vs minimum long session needed
3. **Training history** vs goal complexity

### What happens if there's a gap?

- We don't stop your training
- We adjust the goal class (performance → finish → maintenance)
- We prioritize what matters most for YOUR available time
- We tell you honestly what's achievable

**You always get a training plan.** It's shaped to match reality.

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Energy system definitions | energy_systems.md | Physiology |
| Feasibility calculation | feasibility_policy.md | Policy |
| Zone definitions | zone_physiology.md | Physiology |
| Periodization by goal | periodization.md | Periodization |
| Age/level modifiers | modifiers.md | Modifiers |

---

<!-- META: boundaries -->
## Hard Boundaries

This card does NOT define:
- HOW to train for goals → `periodization.md`, `session_rules.md`
- Energy system definitions → `energy_systems.md`
- Feasibility policy (diminishing returns, modifiers) → `feasibility_policy.md`
- Weekly/meso scheduling → `composer.md`

**Goal Demands defines only:**
- What physiological systems the goal requires (zone %)
- Minimum exposure thresholds (hours, long session)
- Mapping from user goal types to archetypes

End of card.

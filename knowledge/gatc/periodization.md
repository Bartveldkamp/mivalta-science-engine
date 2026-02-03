<!-- META -->
# Periodization — Canonical

concept_id: periodization
axis_owner: periodization
version: 1.2
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Issurin (2008, 2016) | Block periodization improves adaptation | Phase sequencing |
| Bompa (2009) | Base → Build → Peak → Taper | Macro structure |
| Seiler (2010) | Polarized model superior for endurance | Zone distribution |
| Mujika & Padilla (2003) | Optimal taper 8–21 days, −40–60% volume | Taper rules |
| Foster (1998) | Load–recovery rhythm reduces overtraining | Meso wave |
| Banister (1991) | Progressive overload drives adaptation | Ramp logic |

---

<!-- ALG: macro_templates -->
## Macro Templates

| goal_type | horizon_weeks_min | horizon_weeks_max | phase_sequence |
|-----------|-------------------|-------------------|----------------|
| general_fitness | 8 | 52 | base,maintenance |
| 5k | 8 | 12 | base,build,peak,taper |
| 10k | 10 | 16 | base,build,build,peak,taper |
| half_marathon | 12 | 18 | base,base,build,build,peak,taper |
| marathon | 16 | 24 | base,base,base,build,build,peak,taper |
| maintenance | 4 | 52 | maintenance |

Notes:
- Planning always works **backwards from goal date**
- Phase repetition is allowed

---

<!-- ALG: macro_phase_allocation -->
## Macro Phase Allocation by Program Length

| total_mesos_min | total_mesos_max | base_mesos | build_mesos | peak_mesos | taper_mesos |
|----------------|-----------------|------------|-------------|------------|-------------|
| 1 | 4 | 2 | 1 | 0 | 1 |
| 5 | 8 | 3 | 3 | 1 | 1 |
| 9 | 12 | 5 | 4 | 2 | 1 |
| 13 | 20 | 7 | 7 | 3 | 1 |

Rules:
- Allocation overrides repetition in `macro_templates`
- If total mesos < phases required, truncate in order: base → build → peak → taper
- Maintenance excluded from event macros

---

<!-- ALG: meso_templates -->
## Meso Templates

| phase | meso_length_days | load_pattern | recovery_position |
|-------|------------------|--------------|-------------------|
| base | 28 | 3:1 | 4 |
| build | 21 | 2:1 | 3 |
| peak | 21 | 2:1 | 3 |
| taper | 14 | 1:1 | 2 |
| recovery | 7 | 0:1 | 1 |
| maintenance | 28 | 3:1 | 4 |

Notes:
- Final meso length resolved by **Modifiers** (e.g. masters → shorter mesos)
- Meso = fundamental planning unit

---

<!-- ALG: age_meso_structure -->
## Age-Based Meso Structure

| age_min | age_max | meso_days | load_slices | pattern | deload_slice |
|---------|---------|-----------|-------------|---------|--------------|
| 0 | 39 | 28 | 4 | 3:1 | 4 |
| 40 | 54 | 21 | 3 | 2:1 | 3 |
| 55 | 999 | 21 | 3 | 2:1 | 3 |

Notes:
- Masters (40+) require shorter mesos for recovery (Friel, Fast After 50)
- 3:1 pattern = 3 slices load + 1 slice recovery (28 days)
- 2:1 pattern = 2 slices load + 1 slice recovery (21 days)
- Slice = 7-day display unit for user communication

---

<!-- ALG: meso_wave -->
## Meso Wave (Single Source of Truth)

| meso_phase | order | load_mult | purpose |
|------------|-------|-----------|---------|
| intro | 1 | 0.90 | re-entry, absorb |
| build | 2 | 1.00 | main stimulus |
| overreach | 3 | 1.08 | peak stress |
| unload | 4 | 0.55 | adaptation |

Rules:
- `overreach` exists **only** for 3:1 mesos (28 days)
- 21-day mesos use: intro → build → unload
- Session axis maps meso_day → meso_phase

---

<!-- ALG: phase_wave_templates -->
## Phase Wave Templates (Slice Load Percentages)

| phase | pattern | slice_1_pct | slice_2_pct | slice_3_pct | slice_4_pct |
|-------|---------|------------:|------------:|------------:|------------:|
| base | 3:1 | 0.90 | 1.00 | 1.08 | 0.55 |
| base | 2:1 | 0.90 | 1.00 | 0.55 | 0 |
| build | 3:1 | 0.90 | 1.00 | 1.08 | 0.55 |
| build | 2:1 | 0.90 | 1.00 | 0.55 | 0 |
| peak | 2:1 | 0.90 | 1.08 | 0.55 | 0 |
| taper | 1:1 | 0.70 | 0.55 | 0 | 0 |

Notes:
- Slice percentages multiply meso target load
- 3:1 = 4 slices (intro, build, overreach, unload)
- 2:1 = 3 slices (intro, build, unload)
- 1:1 = 2 slices (taper pattern)
- slice_4_pct = 0 means pattern has only 3 slices

---

<!-- ALG: energy_system_wave -->
## Energy System Emphasis (Macro × Meso Wave)

| macro_phase | meso_phase | z1_z2 | z3 | z4_z5 | z6_z8 |
|-------------|------------|-------|----|-------|-------|
| base | intro | high | low | off | off |
| base | build | high | low | off | off |
| base | overreach | high | low | off | off |
| base | unload | medium | off | off | off |
| build | intro | medium | medium | low | off |
| build | build | medium | high | low | off |
| build | overreach | low | high | medium | off |
| build | unload | medium | low | off | off |
| peak | intro | low | medium | medium | low |
| peak | build | low | medium | high | low |
| peak | overreach | off | medium | high | medium |
| peak | unload | medium | off | low | off |
| taper | intro | low | medium | medium | off |
| taper | build | low | low | medium | off |
| taper | unload | low | off | low | off |

Values:
- off / low / medium / high
- Guides **session selection priority**, not volume or duration

---

<!-- ALG: phase_distribution -->
## Phase Zone Distribution Targets (NTIZ Percentages)

| phase | r_z2_pct_min | r_z2_pct_max | z3_pct_min | z3_pct_max | z4_z8_pct_min | z4_z8_pct_max |
|-------|--------------|--------------|------------|------------|---------------|---------------|
| base | 85 | 95 | 5 | 10 | 0 | 5 |
| build | 75 | 85 | 5 | 10 | 10 | 20 |
| peak | 70 | 80 | 5 | 10 | 15 | 25 |
| taper | 80 | 90 | 5 | 10 | 5 | 15 |
| recovery | 95 | 100 | 0 | 5 | 0 | 0 |
| maintenance | 80 | 90 | 5 | 10 | 5 | 10 |

Notes:
- Percentages of total meso NTIZ
- Z3 explicitly controlled (never a filler)

---

<!-- ALG: volume_ramp -->
## Volume Ramp Limits (Meso-Based)

| level | volume_tolerance | max_slice_increase_pct | max_meso_increase_pct |
|-------|------------------|------------------------|-----------------------|
| beginner | normal | 5 | 15 |
| beginner | high | 6 | 18 |
| novice | normal | 8 | 20 |
| novice | high | 10 | 25 |
| intermediate | normal | 10 | 25 |
| intermediate | high | 12 | 30 |
| advanced | normal | 10 | 30 |
| advanced | high | 12 | 35 |
| elite | normal | 10 | 30 |
| elite | high | 15 | 40 |

Notes:
- Slice = meso_wave step, **not calendar week**
- volume_tolerance from sport modifiers

---

<!-- ALG: progression_rate -->
## Progression Rate (Planner Expectation)

| level | meso_progression_mult |
|-------|-----------------------|
| beginner | 0.50 |
| novice | 0.75 |
| intermediate | 1.00 |
| advanced | 1.00 |
| elite | 1.00 |

Notes:
- Used for feasibility and expectation shaping
- Not a fitness prediction

---

<!-- JOSI: overview -->
## How Periodization Works

Training follows a **wave**, not a calendar.

- **Macro**: the journey to your goal
- **Meso**: repeating stress–recovery waves
- **Sessions**: daily expression of that wave

The system plans **backwards from your goal** and lets adaptation emerge.

---

<!-- JOSI: the_wave -->
## The Training Wave (The Melody)

Every meso follows the same dance:

1. Introduce stress
2. Build it
3. (If allowed) Overreach
4. Unload and adapt

This rhythm exists inside every macro phase.

---

<!-- JOSI: taper -->
## The Taper

- Volume reduced 40–60%
- Intensity maintained
- Fatigue drops, performance emerges

Typical gain: **2–3%**

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Zone meaning & effects | zone_physiology.md | Physiology |
| Age/level scaling | modifiers.md | Modifiers |
| Sport tolerance | modifiers_running.md, modifiers_cycling.md | Modifiers |
| Session construction | session_rules.md | Session |
| Load safety & readiness | load_monitoring.md | Load |

---

<!-- META: boundaries -->
## Hard Boundaries

This card does NOT define:
- Session structure or spacing → session_rules.md
- NTIZ accounting or caps → session_rules.md
- Load thresholds, readiness → load_monitoring.md
- Zone physiology → zone_physiology.md

Invariant:
- Engine reasons in **MESO DAYS only**
- No calendar weeks or weekdays exist in ALG

End of card.

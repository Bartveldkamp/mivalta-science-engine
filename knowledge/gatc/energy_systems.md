<!-- META -->
# Energy Systems — Canonical

concept_id: energy_systems
axis_owner: physiology
version: 1.0
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Gastin (2001) | ATP-CP depleted in 6-10s, glycolytic peaks 45-90s | Duration bands for systems |
| Brooks (2004) | Oxidative metabolism dominates beyond 2min | Aerobic power threshold |
| Seiler (2010) | Polarized: 80% aerobic base optimal | Z2 as baseline load factor |
| Laursen (2002) | VO2max intervals: 3-5min work optimal | Aerobic power duration |
| Tabata (1996) | 20s max efforts stress both systems | Anaerobic crossover |
| Billat (2001) | Threshold work: 20-60min sustainable | Steady-state threshold band |

---

<!-- ALG: definitions -->
## Energy System Definitions

Universal biological systems stressed by training.
6 rows, forever stable. Sport-agnostic.

| system_id | duration_min_sec | duration_max_sec | primary_fuel | recovery_hours |
|-----------|------------------|------------------|--------------|----------------|
| neuromuscular | 0 | 20 | ATP-CP | 48 |
| anaerobic_power | 20 | 120 | glycolytic | 48 |
| aerobic_power | 120 | 1800 | oxidative_glyco | 24 |
| steady_state_threshold | 1800 | 3600 | oxidative | 36 |
| endurance | 3600 | 5400 | oxidative_fat | 24 |
| long_endurance | 5400 | 99999 | fat_oxidative | 48 |

Notes:
- Duration in SECONDS (not minutes) for precision
- `recovery_hours`: minimum recommended recovery before stressing SAME system again
- `primary_fuel`: dominant energy pathway (not exclusive)
- These systems are universal across all endurance sports
- Training impulse MUST be computed from energy systems, not directly from zones

---

<!-- ALG: zone_map -->
## Zone to Energy System Mapping

Maps training zones to underlying energy systems.
Enables impulse calculation from zone-based sessions.

| zone | primary_system | secondary_system | load_factor |
|------|----------------|------------------|-------------|
| R | endurance | null | 0.5 |
| Z1 | endurance | null | 0.8 |
| Z2 | endurance | steady_state_threshold | 1.0 |
| Z3 | steady_state_threshold | aerobic_power | 1.5 |
| Z4 | aerobic_power | steady_state_threshold | 2.0 |
| Z5 | aerobic_power | anaerobic_power | 3.0 |
| Z6 | anaerobic_power | neuromuscular | 4.0 |
| Z7 | neuromuscular | anaerobic_power | 5.0 |
| Z8 | neuromuscular | null | 6.0 |

Notes:
- `load_factor`: relative impulse cost per minute, normalized to Z2 = 1.0
- `secondary_system`: activated during mixed-zone work (optional)
- Every zone maps to exactly one primary system
- `null` secondary means single-system work
- This table is the foundation for training impulse calculation

---

<!-- ALG: system_recovery -->
## System Recovery Spacing

Minimum gap between sessions stressing same energy system.
Used for session spacing constraints. Explicit for 21 and 28 day mesos.

| system_id | min_gap_hours | max_per_meso_21 | max_per_meso_28 |
|-----------|---------------|-----------------|-----------------|
| neuromuscular | 72 | 6 | 8 |
| anaerobic_power | 48 | 8 | 11 |
| aerobic_power | 36 | 10 | 14 |
| steady_state_threshold | 36 | 10 | 14 |
| endurance | 24 | 21 | 28 |
| long_endurance | 48 | 6 | 8 |

Notes:
- `min_gap_hours`: minimum recovery before same system can be primary target
- `max_per_meso_21`: maximum sessions targeting this system in a 21-day meso
- `max_per_meso_28`: maximum sessions targeting this system in a 28-day meso
- Endurance (Z1-Z2) can be trained daily (max = meso_days)
- High-stress systems (neuromuscular, anaerobic) need longer recovery
- These constraints inform Composer spacing, not rigid rules

---

<!-- JOSI: energy_explained -->
## How Your Body Produces Energy

Training targets specific **energy systems** — the biological pathways your body uses to fuel movement.

### Aerobic Systems (Z1-Z3)
- Use oxygen to burn fat and carbs
- Sustainable for long durations
- Foundation of all endurance

### Threshold System (Z4)
- Right at your lactate threshold
- Maximum sustainable intensity
- Key for race performance

### High-Intensity Systems (Z5-Z8)
- Anaerobic: no oxygen, burns glycogen fast
- Neuromuscular: pure power, depletes in seconds
- Needs significant recovery

**Your training develops these systems in the right proportion for your goal.**

---

<!-- JOSI: recovery_explained -->
## Why Recovery Matters

Each energy system needs time to adapt after training.

**Low-intensity (Z1-Z2)**
- Can train daily
- Adaptations accumulate over months

**Threshold (Z4)**
- 36+ hours between hard sessions
- Adaptations in 2-4 weeks

**High-intensity (Z5-Z8)**
- 48-72 hours between sessions
- High stress, high reward, high risk

**The system automatically spaces your hard sessions.**
You don't need to calculate — just follow the plan.

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Zone definitions | zone_physiology.md | Physiology |
| Goal demands by zone | goal_demands.md | Physiology |
| Feasibility policy | feasibility_policy.md | Policy |
| Load monitoring | load_monitoring.md | Load |
| Session rules | session_rules.md | Session |

---

<!-- META: boundaries -->
## Hard Boundaries

This card does NOT define:
- Zone HR/power/pace ranges → `zone_anchors.md`
- Zone physiological adaptations → `zone_physiology.md`
- Session structure or timing → `session_rules.md`
- Goal-specific demands → `goal_demands.md`

**Energy Systems defines only:**
- Biological energy pathways and their duration bands
- Zone-to-system mapping for impulse calculation
- Recovery constraints between system stresses

End of card.

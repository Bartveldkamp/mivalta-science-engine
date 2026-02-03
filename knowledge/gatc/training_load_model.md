<!-- META -->
# Training Load Model — Canonical

concept_id: training_load_model
axis_owner: load
version: 1.0
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Foster (1998, 2001) | Monotony (mean/SD) and Strain (load×monotony) correlate with breakdown | Monotony/strain thresholds |
| Gabbett (2016) | Acute:Chronic ratio relates to injury risk | Spike ratio thresholds |
| Banister (1975), Busso (2003) | Fitness–fatigue impulse-response | Optional time constants |
| Meeusen (2013) | Overreaching vs overtraining distinction | Conservative defaults |
| Buchheit (2014) | Multi-marker improves reliability | Safe defaults under low trust |

Notes:
- This card provides **parameters only** (thresholds, windows, caps).
- Computation happens in code (`LoadMonitor`) using values from ResolvedRuleset.
- All time is expressed in **days** (no week reasoning).

---

<!-- ALG: strain_parameters -->
## Strain Parameters (Windows in Days)

| parameter | value_days |
|----------|------------|
| acute_window_days | 7 |
| chronic_window_days | 28 |
| strain_baseline_window_days | 42 |

Notes:
- Used for spike ratio (acute/chronic) and strain z-baseline
- These are rolling day windows, not calendar weeks

---

<!-- ALG: monitoring_thresholds -->
## Monitoring Thresholds (Traffic Light)

| parameter | green_max | amber_max | red_threshold |
|-----------|-----------|-----------|---------------|
| acwr | 1.30 | 1.50 | 1.50 |
| acwr_low | 0.80 | 0.70 | 0.70 |
| monotony | 1.5 | 2.0 | 2.0 |
| strain_z | 1.5 | 2.0 | 2.0 |
| weekly_ramp | 10 | 20 | 25 |

Notes:
- `acwr` = Acute:Chronic Workload Ratio (>1 = increased risk, Gabbett 2016)
- `acwr_low` = detraining detection (<0.8 = fitness loss risk)
- `strain_z` = z-score of weekly strain vs personal baseline
- `weekly_ramp` = percent change between adjacent weeks (>10% = elevated risk)

---

<!-- ALG: monotony_strain_limits_by_level -->
## Monotony/Strain Limits by Level (Overrides)

| level | monotony_amber_min | monotony_red_min | strain_amber_min | strain_red_min |
|-------|---------------------|------------------|------------------|----------------|
| beginner | 1.4 | 1.6 | 350 | 450 |
| novice | 1.6 | 1.9 | 500 | 650 |
| intermediate | 1.8 | 2.1 | 750 | 950 |
| advanced | 2.0 | 2.3 | 950 | 1150 |
| elite | 2.2 | 2.5 | 1150 | 1400 |

Notes:
- These thresholds scale by level to avoid punishing low-load beginners
- Code may choose max(global_threshold, level_threshold) deterministically

---

<!-- ALG: recovery_capacity -->
## Recovery Capacity (HIT Limits; Days/Hours)

| training_years_min | training_years_max | age_min | age_max | max_hit_per_7d | min_gap_hours | recovery_mult |
|-------------------|--------------------|---------|---------|----------------|--------------|--------------|
| 0 | 0.5 | 0 | 999 | 0 | 72 | 1.40 |
| 0.5 | 2 | 0 | 39 | 2 | 48 | 1.15 |
| 0.5 | 2 | 40 | 999 | 2 | 60 | 1.25 |
| 2 | 5 | 0 | 39 | 3 | 48 | 1.00 |
| 2 | 5 | 40 | 999 | 3 | 60 | 1.10 |
| 5 | 999 | 0 | 39 | 4 | 36 | 0.95 |
| 5 | 999 | 40 | 999 | 4 | 48 | 1.00 |

Notes:
- HIT = sessions that touch Z4–Z8 (definition enforced in Session axis)
- `max_hit_per_7d` uses a rolling 7-day window (not calendar week)
- `min_gap_hours` is minimum gap between HIT sessions
- `recovery_mult` scales conservatively for lower training years and higher age

---

<!-- ALG: safe_defaults -->
## Safe Defaults (Insufficient Data)

| parameter | default_value |
|----------|---------------|
| min_history_days_for_full_metrics | 14 |
| min_days_for_strain_baseline | 14 |
| min_days_for_chronic_load | 28 |
| acwr_max | 1.25 |
| monotony_max | 1.8 |
| strain_z_max | 1.5 |

Notes:
- Used when history is insufficient or data trust is low
- Prevents aggressive classification on sparse data
- `acwr_max`, `monotony_max`, `strain_z_max`: caps when baseline unavailable

---

<!-- ALG: banister_defaults -->
## Banister Defaults (Optional; Diagnostic)

| parameter | value |
|----------|-------|
| fitness_time_constant_days | 42 |
| fatigue_time_constant_days | 7 |

Notes:
- Optional parameters for diagnostic summaries or taper prediction
- Viterbi may learn individual τ values; these remain safe defaults

---

<!-- ALG: rest_cost_mult -->
## Rest Cost Multiplier by Zone Band (V2)

| zone_band | rest_cost_mult | rationale |
|-----------|----------------|-----------|
| Z1_Z2 | 0.30 | Active recovery, low metabolic strain |
| Z3 | 0.40 | Tempo rest still costs glycogen |
| Z4 | 0.50 | VO2/threshold recovery is taxing |
| Z5 | 0.55 | High anaerobic stress |
| Z6_plus | 0.60 | Neural + metabolic load |
| Z7_Z8 | 0.70 | Neuromuscular + CNS fatigue |

Notes:
- Used by `LoadPlanner.compute_load_v1()` for cost calculation
- rest_cost = rest_min × (zone_factor × rest_cost_mult)
- Higher zones have higher rest metabolic cost (CNS fatigue, glycogen depletion)
- Default fallback = 0.50 (safety)

Zone band mapping:
- Z1_Z2: Z1, Z2, R
- Z3: Z3
- Z4: Z4
- Z5: Z5
- Z6_plus: Z6
- Z7_Z8: Z7, Z8

---

<!-- ALG: frequency_templates -->
## Frequency Templates (Context-Aware 7-Day Load Patterns)

| session_context | sessions_per_7d | pattern |
|-----------------|----------------:|---------|
| standard | 3 | H,OFF,M,OFF,H,OFF,OFF |
| standard | 4 | H,OFF,M,OFF,H,OFF,L |
| standard | 5 | H,OFF,M,OFF,H,L,OFF |
| standard | 6 | H,M,OFF,H,M,L,OFF |
| standard | 7 | M,M,M,M,M,M,M |
| micro | 3 | M,OFF,M,OFF,M,OFF,OFF |
| micro | 4 | M,OFF,M,OFF,M,OFF,L |
| micro | 5 | M,OFF,M,OFF,M,OFF,M |
| micro | 6 | M,M,OFF,M,M,OFF,M |
| micro | 7 | M,M,M,M,M,M,M |
| commute | 3 | M,OFF,M,OFF,L,OFF,OFF |
| commute | 4 | M,OFF,M,OFF,M,OFF,OFF |
| commute | 5 | M,OFF,M,OFF,M,L,OFF |
| senior | 3 | M,OFF,M,OFF,L,OFF,OFF |
| senior | 4 | M,OFF,L,OFF,M,OFF,L |
| senior | 5 | M,OFF,L,OFF,M,L,OFF |

Notes:
- Load class patterns: H (high), M (medium), L (low), OFF (rest)
- Standard: traditional intensity distribution (H-M-H)
- Micro: no high days (shorter sessions = lower peak stress)
- Commute: moderate patterns respecting travel fatigue
- Senior: conservative, more low days, no high days
- Pattern cycles through 7 days starting from meso day 1

---

<!-- ALG: phase_load_signatures -->
## Phase Load Signatures (Bucket Distribution Targets)

| phase | volume_pct_min | volume_pct_max | threshold_pct_min | threshold_pct_max | vo2max_pct_min | vo2max_pct_max | neuro_pct_min | neuro_pct_max | total_uls_mult |
|-------|---------------:|---------------:|------------------:|------------------:|---------------:|---------------:|--------------:|--------------:|---------------:|
| base | 70 | 85 | 10 | 20 | 0 | 10 | 0 | 5 | 1.00 |
| build | 55 | 75 | 25 | 35 | 5 | 15 | 0 | 5 | 1.00 |
| peak | 45 | 65 | 15 | 30 | 15 | 30 | 0 | 10 | 1.00 |
| unload | 90 | 100 | 0 | 10 | 0 | 0 | 0 | 0 | 0.55 |

Notes:
- Percentages of total load allocated to each bucket
- volume = Z1+Z2, threshold = Z3+Z4, vo2max = Z5+Z6, neuro = Z7+Z8
- total_uls_mult scales overall load (unload = 55% of normal)

---

<!-- JOSI: explanation -->
## What This Model Does

This model quantifies training stress patterns and flags risk when load changes too quickly or becomes too repetitive.

It does not replace your coach or your body. It adds a safety layer.

MiValta uses:
- Load spikes (acute vs chronic)
- Training monotony
- Training strain
- Recovery capacity limits

These signals support long-term durability.

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Load calculation (ULS) | session_rules.md | Session |
| Readiness state (monitor-of-truth) | Viterbi Monitor V6 (code) | Load |
| Policy mapping (Viterbi → permissions) | load_monitoring.md | Load |
| Zone blocks & spacing | session_rules.md | Session |
| Modifiers (age/level) | modifiers.md | Modifiers |

---

<!-- META: boundaries -->
## Hard Boundaries

This card does NOT:
- compute readiness from physiology (Viterbi does)
- compute ULS session load (Session Rules does)
- generate sessions (GATC does)
- define zone meaning (Physiology does)

This card defines **parameters only** for the `LoadMonitor` implementation.

End of card.

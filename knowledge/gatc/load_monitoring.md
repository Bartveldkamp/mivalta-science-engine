<!-- META -->
# Load Monitoring — Canonical (Viterbi-Led Policy)

concept_id: load_monitoring
axis_owner: load
version: 1.1
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Meeusen (2013) | Overreaching/overtraining continuum | Fatigue state semantics (Viterbi) |
| Plews (2013) | HRV methodology for fatigue detection | Recovery evidence (Viterbi) |
| Buchheit (2014) | Multi-marker fatigue detection improves reliability | Guardrails + confidence (Viterbi) |
| Foster (1998) | Load variability (monotony/strain) correlates with breakdown | Load risk signals (Viterbi/LoadMonitor) |
| Gabbett (2016) | Rapid spikes increase injury risk | ACWR as risk signal (Viterbi/LoadMonitor) |
| Seiler (2010) | Recovery blocks intensity, not easy work | Zone-block gating policy (this card) |

Notes:
- Viterbi V6 is the **monitor-of-truth** (state, readiness, confidence, guardrails, signals).
- This card defines **policy mapping** only: how GATC should behave given Viterbi outputs.
- This card does not compute readiness, ACWR, monotony, strain, illness, or recovery time.

---

<!-- ALG: readiness_states -->
## Canonical Readiness States (Policy Layer)

| readiness_state | description |
|-----------------|-------------|
| green | normal training allowed |
| amber | caution: restrict highest-stress blocks |
| red | recovery priority: restrict to low-stress blocks |
| blocked | hard stop: no training allowed |

Notes:
- Exactly one readiness_state is active at any time (policy output)
- This state is derived from Viterbi outputs by deterministic mapping

---

<!-- ALG: viterbi_readiness_mapping -->
## Viterbi ReadinessLevel → Policy Readiness State

| viterbi_readiness_level | readiness_state |
|-------------------------|-----------------|
| green | green |
| yellow | amber |
| orange | red |
| red | red |

Notes:
- If no fatigue-state override applies, this mapping is used.

---

<!-- ALG: fatigue_state_overrides -->
## Viterbi FatigueState Overrides (Hard Priority)

| viterbi_fatigue_state | readiness_override |
|-----------------------|--------------------|
| Recovered | none |
| Productive | none |
| Accumulated | amber |
| Overreached | red |
| IllnessRisk | blocked |

Rules:
- If `readiness_override` is not `none`, it overrides the readiness mapping.
- Accumulated is treated as functional overreaching (productive) but still requires caution.

---

<!-- ALG: confidence_policy -->
## Confidence Policy (Data Tier / Trust)

| data_tier | trust_level |
|----------|-------------|
| NONE | low |
| MINIMAL | low |
| BASIC | medium |
| STANDARD | medium |
| GOOD | high |
| FULL | high |
| ENHANCED | high |

Rules:
- Low trust MUST NOT produce `blocked` unless Viterbi guardrail is triggered.
- Low trust MAY produce `red` when both readiness mapping and fatigue override agree.
- High trust allows full policy range (including blocked) when Viterbi indicates it.

---

<!-- ALG: guardrail_policy -->
## Guardrail Policy

| viterbi_guardrail_triggered | readiness_override |
|----------------------------|-------------------|
| true | blocked |
| false | none |

Notes:
- Guardrails are immediate safety overrides (illness flags, extreme marker patterns).
- Guardrails outrank all other mappings.

---

<!-- ALG: readiness_zone_block_gates -->
## Readiness → Zone-Block Gates (GATC Permission Matrix)

| readiness_state | zone_block | allowed |
|-----------------|------------|---------|
| green | Z1_Z2 | true |
| green | Z3 | true |
| green | Z4_Z5 | true |
| green | Z6_Z8 | true |
| amber | Z1_Z2 | true |
| amber | Z3 | true |
| amber | Z4_Z5 | true |
| amber | Z6_Z8 | false |
| red | Z1_Z2 | true |
| red | Z3 | false |
| red | Z4_Z5 | false |
| red | Z6_Z8 | false |
| blocked | Z1_Z2 | false |
| blocked | Z3 | false |
| blocked | Z4_Z5 | false |
| blocked | Z6_Z8 | false |

Notes:
- This matrix gates **zone blocks**, not "intensity" or "hard sessions".
- GATC must still enforce all Session axis spacing/caps even when allowed.

---

<!-- ALG: policy_resolution_order -->
## Policy Resolution Order (Deterministic)

| step | rule |
|------|------|
| 1 | If guardrail_triggered == true → readiness_state = blocked |
| 2 | Else if fatigue_state_override != none → readiness_state = override |
| 3 | Else readiness_state = map(viterbi_readiness_level) |
| 4 | If trust_level == low and readiness_state == blocked → readiness_state = red |
| 5 | Apply readiness_zone_block_gates to produce allowed_zone_blocks |

Notes:
- This produces one immutable policy output used by GATC.

---

<!-- JOSI: readiness_explained -->
## What MiValta Readiness Means

MiValta monitors recovery and fatigue on-device using your available data.

- **Green**: normal training is safe
- **Amber**: caution — avoid the highest-stress work
- **Red**: recovery priority — keep training low-stress
- **Blocked**: stop training and seek guidance

If your device data is limited, MiValta stays conservative and relies more on your feedback.

---

<!-- JOSI: why_gating -->
## Why Gating Exists

Gating does not punish you. It protects long-term progress.

MiValta blocks only what is most likely to worsen fatigue.
Easy movement is preserved whenever it is safe.

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Viterbi fatigue states & readiness | Viterbi Monitor V6 (code) | Load (monitor-of-truth) |
| Session caps, spacing, contexts | session_rules.md | Session |
| Periodization wave & emphasis | periodization.md | Periodization |
| Zone meanings | zone_physiology.md | Physiology |
| Scaling & zone unlock | modifiers.md | Modifiers |

---

<!-- META: boundaries -->
## Hard Boundaries

This card does NOT:
- Compute readiness from raw markers (Viterbi does)
- Compute ACWR/monotony/strain (LoadMonitor/Viterbi does)
- Compute session load (Session Rules does)
- Generate or replace sessions (GATC does)

This card defines **policy mapping only**:
Viterbi outputs → GATC permissions.

End of card.

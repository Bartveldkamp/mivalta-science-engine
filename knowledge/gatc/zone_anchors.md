<!-- META -->
# Zone Anchors — Canonical

concept_id: zone_anchors
axis_owner: physiology
version: 1.1
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Coggan (2003) | FTP-based power zones validated across athletic populations | Power anchor for cycling |
| Daniels VDOT | Pace zones based on lactate threshold correlate with ~60-min race effort | Pace anchor for running |
| Threshold-based RCTs | Threshold anchoring reduces non-responders vs %HRmax | Prefer threshold over %max |
| Robergs (2002) | HRmax prediction error ±10–12 bpm | HRmax as fallback only |
| Karvonen Method | LTHR more stable than HRmax for zone calculation | LTHR preferred over HRmax |

---

<!-- ALG: power_zones -->
## Power Zones (Cycling — % of FTP)

| zone | power_pct_min | power_pct_max |
|------|---------------|---------------|
| R | 0 | 45 |
| Z1 | 45 | 55 |
| Z2 | 55 | 75 |
| Z3 | 75 | 90 |
| Z4 | 90 | 105 |
| Z5 | 105 | 120 |
| Z6 | 120 | 150 |
| Z7 | 150 | 200 |
| Z8 | 200 | 300 |

Notes:
- FTP = Functional Threshold Power (≈60-min maximal sustainable power)
- Percentages are **anchors**, not prescriptions
- High Z6–Z8 ranges intentionally wide to accommodate individual variability

---

<!-- ALG: pace_zones -->
## Pace Zones (Running — Multiplier on Threshold Pace)

| zone | pace_factor_min | pace_factor_max |
|------|-----------------|-----------------|
| R | 1.50 | 2.00 |
| Z1 | 1.30 | 1.50 |
| Z2 | 1.15 | 1.30 |
| Z3 | 1.05 | 1.15 |
| Z4 | 0.97 | 1.05 |
| Z5 | 0.90 | 0.97 |
| Z6 | 0.82 | 0.90 |
| Z7 | 0.75 | 0.82 |
| Z8 | 0.65 | 0.75 |

Notes:
- Threshold pace ≈ pace sustainable for ~60 minutes
- Multiply threshold pace (sec/km) by factor
- Higher factor = slower pace

---

<!-- ALG: hr_zones -->
## Heart-Rate Zones (% of LTHR, HRmax Fallback)

| zone | lthr_pct_min | lthr_pct_max | hrmax_pct_min | hrmax_pct_max |
|------|--------------|--------------|---------------|---------------|
| R | 0 | 68 | 0 | 55 |
| Z1 | 68 | 83 | 55 | 72 |
| Z2 | 83 | 94 | 72 | 82 |
| Z3 | 94 | 99 | 82 | 87 |
| Z4 | 99 | 105 | 87 | 92 |
| Z5 | 105 | 110 | 92 | 100 |
| Z6 | -1 | -1 | -1 | -1 |
| Z7 | -1 | -1 | -1 | -1 |
| Z8 | -1 | -1 | -1 | -1 |

Notes:
- LTHR = Lactate Threshold Heart Rate (preferred)
- HRmax = fallback only
- `-1` = HR not reliable for this zone
- Values >100% LTHR reflect **supra-threshold drift**, not steady-state effort
- HR is a **validation anchor**, never a primary prescription when power/pace exist

---

<!-- ALG: anchor_resolution -->
## Anchor Resolution

| priority | anchor_type | source | use_case |
|----------|-------------|--------|----------|
| 1 | power | FTP test | Cycling primary |
| 2 | pace | Threshold pace | Running primary |
| 3 | hr_lthr | LTHR test | Validation or fallback |
| 4 | hr_max | HRmax test or formula | Last-resort fallback |

Rules:
- **Exactly one anchor is primary at any time**
- Lower-priority anchors may validate but **never override**
- If only HRmax exists, confidence is low
- RPE / talk test always available as behavioral validation

---

<!-- JOSI: anchor_explanation -->
## Zone Anchoring Explained

Your training zones are based on **your physiology**, not generic formulas.

**How it works:**
- If you know your **FTP (cycling)** or **threshold pace (running)**, MiValta uses that directly.
- This produces the most accurate and stable zone boundaries.

If only **heart rate** is available:
- Zones can still be estimated
- Accuracy is lower due to daily variability (sleep, heat, stress, hydration)

**Best practice (optional):**
- Perform a threshold-type effort periodically
- Zones update automatically as fitness changes

No test data? That's fine.
RPE and talk test correlate **85–92%** with lab thresholds.

---

<!-- JOSI: field_protocols -->
## Optional Field Protocols (Non-Mandatory)

These are **examples**, not requirements.
MiValta works with partial data or no tests at all.

**Recommendation:** wear a heart-rate monitor when possible so the system can learn from passive data (drift, recovery, stability).

### 1. Talk Test (Universal)
- Gradually increase effort
- Speech shifts mark zone transitions
- Used to estimate aerobic ceiling

### 2. Steady Threshold Effort
- Controlled, "comfortably hard" effort
- Used to approximate threshold pace or power

### 3. HR Drift Observation
- Steady easy effort
- Rising HR at same pace suggests intensity too high

### 4. Step Session (RPE-Anchored)
- Short steps at increasing perceived effort
- Builds a personal intensity map

### 5. RPE Anchor Session
- Short segments at known RPE values
- Aligns perception with sensor data

### 6. "180 Rule" Walk/Run
- Keep HR < (180 − age)
- Safe baseline for beginners or return-to-training

Notes:
- Durations are illustrative only
- Shorter or gentler versions are acceptable
- No exhaustive testing is required

---

<!-- JOSI: power_explained -->
## Power Zones (Cycling)

If you know your FTP:

- **R:** 0–45% — easy spin
- **Z1:** 45–55% — light aerobic
- **Z2:** 55–75% — endurance foundation
- **Z3:** 75–90% — sustained work
- **Z4:** 90–105% — threshold
- **Z5:** 105–120% — VO₂
- **Z6–Z8:** 120–300% — short maximal efforts

Power is a guide, not a goal.

---

<!-- JOSI: pace_explained -->
## Pace Zones (Running)

If you know your threshold pace:

- **R:** very easy
- **Z1:** easy conversation
- **Z2:** comfortable endurance
- **Z3:** controlled tempo
- **Z4:** threshold pace
- **Z5:** 5K-like effort
- **Z6–Z8:** short sprints

If pace feels wrong, trust RPE.

---

<!-- JOSI: hr_explained -->
## Heart Rate Zones

Heart rate reflects **response**, not effort.

It varies with:
- heat, altitude
- sleep and stress
- hydration and caffeine
- duration (cardiac drift)

**Use HR to confirm**, not to chase numbers.

LTHR is more reliable than HRmax.
Formulas are last-resort estimates.

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|------|----------|------|
| Zone meaning & effects | zone_physiology.md | Physiology |
| Age & level scaling | modifiers.md | Modifiers |
| Session construction | session_rules.md | Session |
| Load & readiness | load_monitoring.md | Load |

---

<!-- META: boundaries -->
## Hard Boundaries

This card does NOT define:
- Zone physiology → `zone_physiology.md`
- Age/level scaling → `modifiers.md`
- Session duration or spacing → `session_rules.md`
- Load accounting or readiness → `load_monitoring.md`

Zone Anchors define **calibration only**, never planning or prescription.

End of card.

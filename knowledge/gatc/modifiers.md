<!-- META -->
# Modifiers — Canonical

concept_id: modifiers
axis_owner: modifiers
version: 1.1
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Tanaka (2001) | HRmax declines ~0.7 bpm/year after age 40 | Age-related recovery scaling |
| Hawkins (2003) | Masters athletes recover slower but remain trainable | Age modifier rationale |
| Foster (1996) | Training age determines adaptation rate | Level-based zone access |
| Seiler (2010) | Novices respond best to low intensity; elites need specificity | Level-based zone access |
| Kenttä & Hassmén (1998) | Recovery capacity decreases with cumulative stress | Rest multiplier basis |

---

<!-- ALG: age_modifiers -->
## Age Modifiers

| age_band | age_min | age_max | recovery_mult | warmup_mult | rest_mult | max_zone_allowed |
|----------|---------|---------|---------------|-------------|-----------|------------------|
| junior | 0 | 17 | 0.90 | 0.90 | 0.90 | Z7 |
| young_adult | 18 | 29 | 1.00 | 1.00 | 1.00 | Z8 |
| adult | 30 | 39 | 1.00 | 1.00 | 1.00 | Z8 |
| masters_1 | 40 | 49 | 1.15 | 1.10 | 1.15 | Z7 |
| masters_2 | 50 | 59 | 1.25 | 1.20 | 1.25 | Z6 |
| masters_3 | 60 | 69 | 1.35 | 1.30 | 1.35 | Z5 |
| senior | 70 | 999 | 1.50 | 1.40 | 1.50 | Z5 |

Notes:
- `recovery_mult`: multiplier on recovery time between high-stress sessions
- `warmup_mult`: multiplier on base warmup duration
- `rest_mult`: multiplier on interval rest durations
- `max_zone_allowed`: age-based safety ceiling
- Medical or professional clearance may override this value **before** context build

---

<!-- ALG: level_modifiers -->
## Level Modifiers

| level | recovery_mult | rest_mult | max_zone_allowed | meso_unlock |
|-------|---------------|-----------|------------------|-------------|
| beginner | 1.40 | 1.50 | Z3 | 2 |
| novice | 1.25 | 1.30 | Z5 | 1 |
| intermediate | 1.10 | 1.15 | Z7 | 0 |
| advanced | 1.00 | 1.00 | Z8 | 0 |
| elite | 0.95 | 0.95 | Z8 | 0 |

Notes:
- `recovery_mult`: beginners require more recovery to maintain quality
- `rest_mult`: longer rests needed to preserve execution quality
- `max_zone_allowed`: experience-based safety ceiling
- `meso_unlock`: number of completed mesos required before higher zones are accessible
- `meso_unlock = 0` means no restriction beyond `max_zone_allowed`

---

<!-- ALG: level_zone_unlock -->
## Level Zone Unlock Schedule

| level | meso_1 | meso_2 | meso_3 | meso_4_plus |
|-------|--------|--------|--------|-------------|
| beginner | R–Z2 | R–Z3 | R–Z4 | R–Z5 |
| novice | R–Z4 | R–Z5 | R–Z6 | R–Z7 |
| intermediate | R–Z7 | R–Z8 | R–Z8 | R–Z8 |
| advanced | R–Z8 | R–Z8 | R–Z8 | R–Z8 |
| elite | R–Z8 | R–Z8 | R–Z8 | R–Z8 |

Notes:
- This table is the **concrete expansion** of `meso_unlock`
- Zone access increases progressively with training experience
- Age and sport caps may still apply
- Overrides must occur **before** the ResolvedRuleset is frozen

---

<!-- ALG: modifier_composition -->
## Modifier Composition Rules

| key | rule |
|-----|------|
| final_recovery_mult | age.recovery_mult × level.recovery_mult × sport.recovery_mult |
| final_rest_mult | age.rest_mult × level.rest_mult |
| final_warmup_mult | age.warmup_mult |
| max_zone_allowed | MIN(age.max_zone_allowed, level.max_zone_allowed, sport.max_zone_allowed) |

Notes:
- Modifiers are applied **once** at context build
- Output is a frozen `ResolvedRuleset`
- Downstream logic MUST NOT re-query raw modifier tables
- Lowest applicable cap always wins

---

<!-- JOSI: personalization_explained -->
## How Personalization Works

MiValta personalizes training using **two independent dimensions**:

### 1. Age (Physiological Capacity)
- Recovery capacity decreases with age
- Warmup needs increase to protect joints and connective tissue
- Very high-intensity zones may be capped for safety

### 2. Experience Level (Training History)
- Beginners adapt quickly but tolerate stress poorly
- Intermediate athletes handle full zone ranges
- Advanced and elite athletes recover efficiently and tolerate specificity

**What this means for you:**
Your training automatically adapts recovery timing, rest duration, warmup needs, and accessible zones — without you calculating anything.

These are **evidence-based defaults**, not limits on potential.

---

<!-- JOSI: age_explained -->
## Training and Age

Aging changes **recovery**, not **capability**.

**What changes:**
- Slower recovery between hard sessions
- Greater importance of warmup and rest
- Higher injury risk with frequent maximal efforts

**What stays the same:**
- The training zones themselves
- Ability to build fitness
- The importance of consistency

MiValta adjusts recovery and access so training remains effective at any age.

---

<!-- JOSI: level_explained -->
## Training and Experience

Training history shapes how your body responds to load.

**Beginner**
- Massive gains from easy aerobic work
- Intensity introduced gradually
- Zone access expands over early mesos

**Novice**
- Structure improves adaptation
- More zones become accessible
- Still building aerobic foundation

**Intermediate**
- Full zone access
- Ready for structured periodization

**Advanced / Elite**
- High tolerance for intensity
- Precision matters more than volume

**Key insight:**
Most athletes stall by adding intensity too early. The system prevents that.

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Zone definitions | zone_physiology.md | Physiology |
| Zone anchors | zone_anchors.md | Physiology |
| Sport modifiers | modifiers_running.md, modifiers_cycling.md | Modifiers |
| Session construction | session_rules.md | Session |
| Phase structure | periodization.md | Periodization |
| Load & readiness | load_monitoring.md | Load |

---

<!-- META: boundaries -->
## Hard Boundaries

This card does NOT define:
- Zone physiology → `zone_physiology.md`
- Sport-specific scalars → `modifiers_running.md`, `modifiers_cycling.md`
- Session timing or duration → `session_rules.md`
- Periodization structure → `periodization.md`
- Load thresholds or readiness logic → `load_monitoring.md`

**Modifiers define scalars and caps only.**
They never create sessions, volumes, or plans.

End of card.

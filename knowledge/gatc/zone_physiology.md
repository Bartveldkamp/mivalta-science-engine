<!-- META -->
# Zone Physiology — Canonical

concept_id: zone_physiology
axis_owner: physiology
version: 1.1
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Olympiatoppen I-Scale (2024) | 8 training zones (I-1 to I-8) used by Norwegian Olympic athletes | Basis for Z1–Z8 model |
| Critical Power Model | CP = maximal sustainable power without progressive fatigue | Defines Z4/Z5 boundary |
| Seiler (2010) | Elite endurance: 75–85% below LT1, 10–20% above LT2 | Polarized distribution |
| Threshold-Based RCTs | Threshold anchoring > %HRmax | Individual zone anchoring |
| Lactate Research | LT1 ≈ 2.0 mmol/L, LT2 ≈ 4.0 mmol/L | Physiological markers |
| Talk Test Validity | r = 0.85–0.92 vs lab thresholds | Field validation |
| Banister (1975), Busso (2003) | Fitness–fatigue impulse response | Adaptation signals |

---

<!-- ALG: zone_physiology -->
## Zone Physiology

| zone | label | primary_adaptation | rpe_min | rpe_max | hr_pct_min | hr_pct_max | lactate_min | lactate_max | mito | capillary | fat_ox | lactate_clear | stroke_vol | vo2max | glycolytic | neuromuscular |
|------|-------|--------------------|---------|---------|------------|------------|-------------|-------------|------|-----------|--------|---------------|------------|--------|------------|---------------|
| R | Recovery | recovery | 0 | 1 | 0 | 55 | 0.0 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Z1 | Very Light | aerobic_base | 1 | 2 | 55 | 72 | 0.5 | 1.5 | 4 | 3 | 4 | 1 | 2 | 0 | 0 | 1 |
| Z2 | Aerobic | aerobic_efficiency | 2 | 3 | 72 | 82 | 1.0 | 2.0 | 4 | 4 | 4 | 2 | 3 | 1 | 0 | 1 |
| Z3 | Tempo | tempo_tolerance | 4 | 5 | 82 | 87 | 1.5 | 3.5 | 3 | 2 | 2 | 3 | 3 | 2 | 1 | 1 |
| Z4 | Threshold | lactate_threshold | 6 | 7 | 87 | 92 | 3.5 | 5.0 | 2 | 1 | 1 | 5 | 3 | 3 | 2 | 1 |
| Z5 | VO₂max | vo2max | 8 | 10 | 92 | 100 | 5.0 | 10.0 | 1 | 0 | 0 | 3 | 2 | 5 | 3 | 2 |
| Z6 | Anaerobic Capacity | anaerobic_capacity | 9 | 10 | -1 | -1 | 10.0 | 20.0 | 0 | 0 | 0 | 1 | 1 | 2 | 5 | 3 |
| Z7 | Maximal Anaerobic | maximal_anaerobic | 10 | 10 | -1 | -1 | 10.0 | 20.0 | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 5 |
| Z8 | Neuromuscular Sprint | neuromuscular_power | 10 | 10 | 100 | 100 | 10.0 | 20.0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 5 |

Notes:
- RPE scale: 0–10
- HR% and lactate are **descriptive anchors**, not prescriptions
- `-1` = signal not reliable for this zone
- HR values for Z8 are **descriptive only**; heart-rate kinetics are too slow to validate sprint efforts
- Adaptation markers (0–5) are **biological signals**, not programming rules

---

<!-- JOSI: zone_plain_language -->
## Zones in Plain Language

MiValta uses **9 effort levels**, like gears on a bike.

**R (Recovery)** — Barely moving. Gentle walk or easy spin. Active restoration only.

**Z1 (Very Light)** — Easy cruising. Full conversation. Builds the aerobic base.

**Z2 (Aerobic)** — Comfortable but purposeful. Long sentences with brief pauses.

**Z3 (Tempo)** — Working. Short sentences only. Useful but easy to overuse.

**Z4 (Threshold)** — Hard, controlled discomfort. Few words only.

**Z5 (VO₂max)** — Very hard. Cannot talk. Builds high-end aerobic power.

**Z6 (Anaerobic Capacity)** — Short, powerful efforts. Gasping recovery.

**Z7 (Maximal Anaerobic)** — Near-maximal efforts. Seconds to ~1 minute.

**Z8 (Neuromuscular Sprint)** — Absolute maximum speed. Very short, explosive.

**Key insight:** Most endurance fitness is built in **Z1–Z2**, sharpened with limited **Z4–Z5**, and complemented by carefully placed high-intensity work.

---

<!-- JOSI: zone_breathing -->
## Zone Breathing & Effort

| zone | rpe | breathing | self_check |
|------|-----|-----------|------------|
| R | 0–1 | Effortless, natural | If breathing rises, slow down |
| Z1 | 1–2 | Relaxed, nose breathing possible | Fully conversational |
| Z2 | 2–3 | Controlled, slightly deeper | Audible but comfortable |
| Z3 | 4–5 | Deliberate, mouth preferred | Speech requires pauses |
| Z4 | 6–7 | Deep, rhythmic | Mouth breathing mandatory |
| Z5 | 8–10 | Heavy, rapid | Loud breathing |
| Z6 | 9–10 | Gasping | Breathing takes over |
| Z7 | 10 | Max ventilation | Full recovery needed |
| Z8 | 10 | Brief breath-hold | Effort too short for breathing |

---

<!-- JOSI: zone_talk_test -->
## Zone Talk Test

| zone | talk_test | self_check |
|------|-----------|------------|
| R | Can sing | If talking is hard, slow down |
| Z1 | Full paragraphs | Continuous conversation |
| Z2 | Long sentences | Short breathing pauses |
| Z3 | Short sentences | Replies must be brief |
| Z4 | Few words | Yes/no responses only |
| Z5 | Single word | Acknowledge only |
| Z6 | No speech | Hand signals only |
| Z7 | No speech | Focus entirely on effort |
| Z8 | No speech | Effort too brief |

---

<!-- JOSI: zone_feel -->
## Zone Feel

| zone | feel | cue_start | cue_during | cue_end |
|------|------|-----------|------------|---------|
| R | Effortless | Move gently | Stay relaxed | Recovery complete |
| Z1 | Very easy | Find rhythm | Full sentences | Relaxed |
| Z2 | Comfortable | Settle in | Controlled breathing | Maintain |
| Z3 | Purposeful | Ease in | Stay controlled | Smooth exit |
| Z4 | Hard | Build gradually | Hold steady | Earn the rest |
| Z5 | Very hard | Commit | Sustain effort | Full recovery |
| Z6 | Near-max | Explode | Give all | Stop fully |
| Z7 | Maximal | Full power | Absolute focus | Complete stop |
| Z8 | Explosive | Accelerate | Pure speed | Walk it off |

---

<!-- JOSI: zone_safety -->
## Zone Safety

**General principles:**
- Easy zones must feel easy. If Z1 feels hard, something is wrong.
- Z3 is effective but should not dominate training.
- Higher zones require adequate recovery regardless of motivation.
- RPE and talk test are primary; HR/lactate are optional validation.

**Stop immediately if you experience:**
- Chest pain or pressure
- Severe disproportionate breathlessness
- Dizziness or lightheadedness
- Irregular heart rhythm
- Sharp joint pain (not normal muscle fatigue)

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Age & level scaling | modifiers.md | Modifiers |
| Power/pace anchors | zone_anchors.md | Physiology (L2) |
| Session construction | session_rules.md | Session |
| Phase distribution | periodization.md | Periodization |
| Recovery & safety | load_monitoring.md | Load |

---

<!-- META: boundaries -->
## Hard Boundaries

This card does NOT define:
- Session duration or frequency → `session_rules.md`
- Session spacing or legality → `session_rules.md`
- Phase distribution targets → `periodization.md`
- Load accounting or readiness → `load_monitoring.md`
- Age, level, or sport scaling → `modifiers.md`
- Power or pace anchors → `zone_anchors.md`

**Zone Physiology defines meaning, not planning.**

End of card.

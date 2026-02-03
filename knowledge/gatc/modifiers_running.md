<!-- META -->
# Modifiers Running — Canonical

concept_id: modifiers_running
axis_owner: modifiers
sport: running
version: 1.1
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Hreljac (2004) | Running injury rate 37–56% annually | High impact classification |
| Daniels (2014) | Threshold pace is time-efficient for runners | Threshold emphasis |
| Hoogkamer (2016) | Ground reaction forces 2–3× body weight per stride | Increased recovery need |

---

<!-- ALG: sport_modifiers -->
## Running Sport Modifiers

| key | value |
|-----|-------|
| sport | running |
| recovery_mult | 1.10 |
| max_zone_allowed | Z8 |
| volume_tolerance | normal |

Notes:
- `recovery_mult`: +10% recovery due to impact stress
- `max_zone_allowed`: no sport-imposed cap (age/level may cap)
- `volume_tolerance`: normal tolerance for total meso NTIZ

---

<!-- JOSI: running_explained -->
## Running-Specific Training

Running is a **high-impact** sport. Every step sends force through muscles, tendons, and joints.

**What this means:**
- Recovery between hard sessions matters more
- Volume must build gradually
- Spacing protects durability

**Zone emphasis (descriptive):**
- **Z2** builds the aerobic base and resilience
- **Z4** (threshold) is highly effective for performance
- Higher zones are used sparingly and deliberately

**Key insight:**
Most running injuries come from **too much, too soon**. Sustainable progress comes from patience and controlled loading.

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Age & level modifiers | modifiers.md | Modifiers |
| Zone definitions | zone_physiology.md | Physiology |
| Pace anchors | zone_anchors.md | Physiology |
| Session construction | session_rules.md | Session |
| Progression & waves | periodization.md | Periodization |

---

<!-- META: boundaries -->
## Hard Boundaries

This card does NOT define:
- Age or level scaling → `modifiers.md`
- Zone definitions → `zone_physiology.md`
- Session structure or spacing → `session_rules.md`
- Volume progression logic → `periodization.md`
- Load safety rules → `load_monitoring.md`

This card provides **sport-specific scalars and caps only**.

End of card.

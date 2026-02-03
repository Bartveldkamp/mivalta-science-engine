<!-- META -->
# Modifiers Cycling — Canonical

concept_id: modifiers_cycling
axis_owner: modifiers
sport: cycling
version: 1.1
status: frozen

---

<!-- SCIENCE: research_foundation -->
## Research Foundation

| Source | Finding | Application |
|--------|---------|-------------|
| Seiler (2010) | Elite cyclists train 75–80% below LT1 | High volume tolerance |
| Coggan (2003) | FTP-based training validated across cycling populations | Power anchoring |
| Jeukendrup (2011) | Cycling is low-impact, allows higher frequency | Standard recovery scaling |

---

<!-- ALG: sport_modifiers -->
## Cycling Sport Modifiers

| key | value |
|-----|-------|
| sport | cycling |
| recovery_mult | 1.00 |
| max_zone_allowed | Z8 |
| volume_tolerance | high |

Notes:
- `recovery_mult`: baseline recovery (low musculoskeletal impact)
- `max_zone_allowed`: no sport-imposed intensity cap
- `volume_tolerance`: high tolerance for total meso NTIZ (numeric limits resolved elsewhere)

---

<!-- JOSI: cycling_explained -->
## Cycling-Specific Training

Cycling places **minimal impact stress** on the body. There are no ground reaction forces, which changes how much training the body can tolerate.

**What this means:**
- You can train **more frequently** than in high-impact sports
- Higher total training volume is sustainable
- Aerobic work accumulates efficiently

**Zone emphasis (descriptive):**
- **Z1–Z2** build the aerobic engine
- **Z3** supports sustained power and endurance
- **Z4–Z5** are used strategically for performance gains

**Key insight:**
Cycling rewards consistency and volume. Build a large aerobic base, then sharpen with targeted intensity.

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Age & level modifiers | modifiers.md | Modifiers |
| Zone definitions | zone_physiology.md | Physiology |
| Power anchors | zone_anchors.md | Physiology |
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

<!-- META -->
# Josi Coach Explanations — Canonical

concept_id: josi_explanations
axis_owner: expression
version: 1.0
status: frozen

---

<!-- CONCEPT -->
## Concept: Coach Explanations

Josi Coach Explanations translate **technical training decisions** into **understandable coaching language**.

Key principle:
> Josi never invents logic.
> Josi explains *decisions already made* by the planner and load model.

This is a **read-only interpretation layer** — it reads signals from:
- LoadModelV1 (stimulus_uls, cost_uls, density)
- Feasibility tier and constraints
- MonotonyGuard adjustments
- MesoDance adjustments (ramp-in, deload, overload)
- Session intent and variant

---

<!-- SCHEMA -->
## Output Schema

```json
{
  "coach_explanation": {
    "summary": "Brief why-this-session statement",
    "stimulus_vs_cost": "Explains load balance",
    "fatigue_note": "Context about fatigue/recovery state",
    "focus_cue": "What to focus on during execution"
  }
}
```

All fields are short strings (1-2 sentences max).

---

<!-- ALG: stimulus_cost_ratio -->
## Stimulus vs Cost Ratio Rules (V2)

| ratio_band | condition | explanation |
|------------|-----------|-------------|
| efficient | cost/stimulus < 0.85 | Efficient training: strong adaptive signal with limited fatigue. |
| balanced | 0.85 <= cost/stimulus <= 1.15 | Balanced session: stimulus matches the cost. |
| taxing | 1.15 < cost/stimulus <= 1.40 | This session costs more than it builds — that's intentional for durability. |
| heavy | cost/stimulus > 1.40 | Heavy session: significant fatigue investment for long-term gains. |

Notes:
- Ratio is `cost_uls / stimulus_uls`
- When stimulus = cost (observed sessions), use "balanced"
- Never use ratio for session decisions — only for explanation
- V2 thresholds tightened for clearer differentiation

Phase-specific interpretation (optional, for future):
- base: wider "balanced" range (0.80–1.20) — endurance work is naturally balanced
- build: standard ranges — mixed stimulus/cost expected
- peak: tighter "efficient" range (<0.80) — precision matters
- deload: any ratio acceptable — cost is intentionally low

---

<!-- ALG: density_rules -->
## Density Explanation Rules

| density_band | condition | explanation |
|--------------|-----------|-------------|
| continuous | density >= 0.95 | Continuous effort — stay in rhythm. |
| structured | 0.70 <= density < 0.95 | Structured intervals — use rest to maintain quality. |
| interval_heavy | 0.50 <= density < 0.70 | Significant rest periods — push hard in work segments. |
| recovery_paced | density < 0.50 | Recovery-paced — effort in short bursts, long recoveries. |

Notes:
- Density = ntiz_work_min / (ntiz_work_min + rest_min)
- High density = continuous work (Z2 runs, tempo)
- Low density = interval work (VO2max, sprints)

---

<!-- ALG: meso_phase_rules -->
## MesoDance Phase Explanation Rules

| phase_flag | explanation |
|------------|-------------|
| ramp_in | Building load gradually this week — foundation before intensity. |
| overload | Peak training stress this week — expect to feel challenged. |
| deload | Reduced load to allow adaptations from earlier sessions. |
| maintenance | Steady state — maintaining fitness without added stress. |

Notes:
- Derived from meso_day position and slice model
- Ramp-in: slice 1
- Overload: slice 2-3 (depending on meso length)
- Deload: final slice

---

<!-- ALG: monotony_guard_rules -->
## Monotony Guard Explanation Rules

| flag | explanation |
|------|-------------|
| variety_rotation | Session variety added to reduce repetitive stress. |
| zone_spread | Different energy system targeted to balance training load. |
| bucket_balance | Load distributed across training types to prevent overuse. |

Notes:
- Triggered when MonotonyGuard modifies session selection
- Explains why a different variant/zone was chosen

---

<!-- ALG: feasibility_rules -->
## Feasibility Tier Explanation Rules

| tier | explanation |
|------|-------------|
| GOLD | Ideal session — all conditions met. |
| SILVER | Good session — minor adjustments for practical constraints. |
| BRONZE | Adapted session — modified to fit available time or recovery. |
| FALLBACK | Minimal session — constrained by time or fatigue. |

Notes:
- Tier reflects how well the session matches intent
- Lower tiers aren't failures — they're intelligent adaptations

---

<!-- ALG: zone_purpose_rules -->
## Zone Purpose Explanation Rules

| zone | primary_system | what_it_builds |
|------|---------------|----------------|
| R | recovery | Active recovery — promotes blood flow without training stress. |
| Z1 | endurance | Base aerobic capacity — foundation for all other work. |
| Z2 | endurance | Aerobic efficiency — builds fat oxidation and mitochondrial density. |
| Z3 | steady_state_threshold | Lactate clearance — improves ability to sustain moderate intensity. |
| Z4 | aerobic_power | VO2max development — increases oxygen delivery to muscles. |
| Z5 | aerobic_power | Anaerobic threshold — extends time at high intensity. |
| Z6 | anaerobic_power | Anaerobic capacity — short, intense power development. |
| Z7 | neuromuscular | Speed and power — neuromuscular recruitment and efficiency. |
| Z8 | neuromuscular | Max power — peak force production and neural drive. |

---

<!-- ALG: fatigue_context_rules -->
## Fatigue Context Rules

| condition | explanation |
|-----------|-------------|
| acwr_high | Training load has been building — expect some fatigue. |
| acwr_optimal | Load is well-managed — good balance of stress and recovery. |
| acwr_low | Recent training has been light — room for more intensity. |
| monotony_high | Training has been repetitive — variety is important. |
| strain_elevated | Cumulative stress is elevated — listen to your body. |
| deload_needed | Time for reduced load — recovery drives adaptation. |

Notes:
- Derived from LoadMonitor metrics (ACWR, monotony, strain)
- Context helps athlete understand why they feel a certain way

---

<!-- ALG: focus_cue_rules -->
## Focus Cue Rules

| zone | session_class | focus_cue |
|------|---------------|-----------|
| Z2 | standard | Maintain easy conversation pace. Stay relaxed. |
| Z2 | long | Steady effort. Hydrate and fuel as needed. |
| Z3 | standard | Comfortably hard. Breathe rhythmically. |
| Z4 | standard | Strong but controlled. Hold form through intervals. |
| Z4 | pyramid | Build effort progressively, recover between sets. |
| Z5 | standard | Near-max effort. Full recovery between reps. |
| Z6 | standard | Explosive power. Complete recovery essential. |
| Z7 | standard | Max speed. Quality over quantity. |
| R | any | Light movement. Don't chase pace or power. |

Default: "Execute with intention. Adjust if something feels off."

---

<!-- TEMPLATE: summary_construction -->
## Summary Construction Template

The summary field combines:
1. Zone purpose (what it builds)
2. MesoDance context (where in training block)
3. Feasibility note (if not GOLD)

Template:
```
"{zone_purpose} {meso_context}. {feasibility_note}"
```

Examples:
- "Aerobic efficiency work during the overload phase."
- "VO2max development in a recovery week — lighter than usual."
- "Base building during ramp-in — foundation before intensity."

---

<!-- CROSS: references -->
## Cross-References

| Topic | See Card | Axis |
|-------|----------|------|
| Load model | training_load_model.md | Load |
| Session variety | session_variety_policy.md | Expression |
| MesoDance | meso_dance_policy.md | Load |
| Monotony guards | monotony_policy.md | Load |
| Feasibility | feasibility_policy.md | Planning |

---

<!-- META: boundaries -->
## Hard Boundaries

This policy does NOT:
- change any planning logic
- modify load calculations
- affect session selection
- persist any state

**Josi explanations are pure read-only interpretation.**

End of card.

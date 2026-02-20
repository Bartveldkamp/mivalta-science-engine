"""
MiValta Context Assembler — Cross-Domain Payload Enrichment

The glue between the rules engine and the explainer. Takes a base
ENGINE_PAYLOAD and enriches it with cross-domain insights before
the explainer model verbalizes it.

Pipeline position:
    Interpreter → Post-processor → Router → [GATC Engine]
        → context_assembler.enrich_payload()      ← THIS
            → Explainer → Dialogue Governor → User

What it does:
    1. Fires cross-domain rules against the current athlete state
    2. Injects fired rule messages into the ENGINE_PAYLOAD
    3. Adds relevant athlete context (preferences, injury state)
    4. Updates the athlete state store with new data from this turn

What it does NOT do:
    - Make policy decisions (that's the rules engine)
    - Generate text (that's the explainer)
    - Modify training plans (that's the GATC engine)

Usage:
    from shared.context_assembler import enrich_payload, update_state_from_turn

    # Before explainer runs:
    enriched = enrich_payload(engine_payload, store, interp_parsed, state)

    # After conversation turn completes:
    update_state_from_turn(store, interp_parsed, triage_data, user_message)
    store.save("/data/state/")
"""

from typing import Optional

from athlete_state_store import AthleteStateStore
from cross_domain_rules import evaluate_rules


def enrich_payload(
    engine_payload: dict,
    store: AthleteStateStore,
    interp_parsed: Optional[dict] = None,
    conversation_state: Optional[object] = None,
) -> dict:
    """Enrich an ENGINE_PAYLOAD with cross-domain intelligence.

    Args:
        engine_payload: Base payload from the router (action, message, etc.)
        store: AthleteStateStore with current athlete state.
        interp_parsed: Parsed interpreter output (GATCRequest fields).
        conversation_state: ConversationState from simulate.py (has last_topic).

    Returns:
        New dict with cross_domain and athlete_context fields added.
        Original payload is not mutated.
    """
    enriched = dict(engine_payload)
    action = enriched.get("action", "")

    # Extract context from interpreter output
    sport = None
    workout_type = None
    duration_min = None
    fatigue_hint = None

    if interp_parsed:
        sport = interp_parsed.get("sport")
        duration_min = interp_parsed.get("time_available_min")
        constraints = interp_parsed.get("constraints") or {}
        fatigue_hint = constraints.get("fatigue_hint")
        goal = interp_parsed.get("goal", "")

        # Infer workout type from goal/duration
        if goal == "recovery":
            workout_type = "recovery"
        elif goal == "endurance" and duration_min and duration_min > 75:
            workout_type = "long_run" if sport == "run" else "long_ride"
        elif goal in ("threshold", "vo2"):
            workout_type = "interval"
        elif goal == "strength":
            workout_type = "strength"

    # Get topic from conversation state
    topic = None
    if conversation_state and hasattr(conversation_state, "last_topic"):
        topic = conversation_state.last_topic

    # --- Fire cross-domain rules ---
    fired_rules = evaluate_rules(
        action=action,
        store=store,
        sport=sport,
        workout_type=workout_type,
        duration_min=duration_min,
        fatigue_hint=fatigue_hint,
        topic=topic,
    )

    if fired_rules:
        enriched["cross_domain"] = [r["message"] for r in fired_rules]
        enriched["cross_domain_actions"] = [r["action"] for r in fired_rules]
        enriched["cross_domain_rules"] = [r["rule_id"] for r in fired_rules]

    # --- Add relevant athlete context ---
    athlete_context = {}

    # Preferences
    if store.preferences:
        athlete_context["preferences"] = store.preferences

    # Active injuries summary
    active_injuries = store.active_injuries()
    if active_injuries:
        athlete_context["active_injuries"] = [
            {
                "area": inj["area"],
                "severity": inj["severity"],
                "occurrences": inj.get("occurrences", 1),
            }
            for inj in active_injuries
        ]

    # Load context
    load = store.training_load
    if load.get("this_week_sessions", 0) > 0:
        athlete_context["training_load"] = {
            "sessions_this_week": load["this_week_sessions"],
            "trend": load["trend"],
        }

    # Readiness context
    trend = store.readiness_trend()
    current = store.current_readiness()
    if current:
        athlete_context["readiness"] = {
            "current": current,
            "trend": trend,
        }

    if athlete_context:
        enriched["athlete_context"] = athlete_context

    return enriched


def update_state_from_turn(
    store: AthleteStateStore,
    interp_parsed: Optional[dict] = None,
    triage_data: Optional[dict] = None,
    user_message: str = "",
    readiness_level: Optional[str] = None,
    readiness_state: Optional[str] = None,
) -> None:
    """Update the athlete state store after a conversation turn.

    Call this after the pipeline completes for each turn. It extracts
    cross-domain-relevant data and persists it in the state store.

    Args:
        store: AthleteStateStore to update.
        interp_parsed: Parsed interpreter output.
        triage_data: Injury triage collected data (if triage was active).
        user_message: Raw user message.
        readiness_level: Current readiness level from ChatContext.
        readiness_state: Current readiness state from ChatContext.
    """
    # 1. Update from triage data
    if triage_data:
        store.update_from_triage(triage_data)

    # 2. Extract preferences from user message
    if user_message:
        store.update_preferences_from_message(user_message)

    # 3. Update readiness if provided
    if readiness_level:
        store.update_from_readiness(readiness_level, readiness_state)

    # 4. If a workout was created, record it
    if interp_parsed and interp_parsed.get("action") == "create_workout":
        sport = interp_parsed.get("sport", "")
        duration = interp_parsed.get("time_available_min", 0)
        goal = interp_parsed.get("goal", "")

        # Infer zone from goal
        goal_zone_map = {
            "recovery": "Z1",
            "endurance": "Z2",
            "threshold": "Z4",
            "vo2": "Z5",
            "strength": "Z3",
            "race_prep": "Z4",
        }
        zone = goal_zone_map.get(goal, "Z2")

        if sport and duration:
            store.update_from_workout(
                sport=sport,
                zone=zone,
                duration_min=int(duration) if duration else 0,
                workout_type=goal or "session",
            )

    # 5. Detect fatigue from user message
    if user_message:
        lower = user_message.lower()
        fatigue_keywords = {
            "very_tired": ["exhausted", "wiped out", "can barely", "no energy",
                           "completely drained", "dead tired"],
            "tired": ["tired", "fatigued", "worn out", "drained", "heavy legs"],
        }
        for level, keywords in fatigue_keywords.items():
            if any(kw in lower for kw in keywords):
                store.update_preference("last_fatigue_hint", level)
                break


def serialize_cross_domain_for_prompt(enriched_payload: dict) -> str:
    """Serialize cross-domain insights from enriched payload into prompt text.

    This produces a compact block that gets injected into the explainer's
    context. The explainer uses it to weave cross-domain knowledge into
    its response naturally.

    Returns empty string if no cross-domain data exists.
    """
    parts = []

    cross_domain = enriched_payload.get("cross_domain")
    if cross_domain:
        for msg in cross_domain:
            parts.append(f"  - {msg}")

    athlete_ctx = enriched_payload.get("athlete_context")
    if athlete_ctx:
        injuries = athlete_ctx.get("active_injuries")
        if injuries:
            injury_strs = [
                f"{i['area']} ({i['severity']}/10)"
                for i in injuries
            ]
            parts.append(f"  - Active injuries: {', '.join(injury_strs)}")

        load = athlete_ctx.get("training_load")
        if load:
            parts.append(
                f"  - This week: {load['sessions_this_week']} sessions, "
                f"trend {load['trend']}"
            )

        readiness = athlete_ctx.get("readiness")
        if readiness:
            parts.append(
                f"  - Readiness: {readiness['current']}, trend {readiness['trend']}"
            )

    if not parts:
        return ""

    return "CROSS_DOMAIN_CONTEXT:\n" + "\n".join(parts)

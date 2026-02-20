"""
MiValta Cross-Domain Rules Engine — Deterministic Knowledge Fusion

Evaluates rules that connect information across coaching domains:
injury + training load, fatigue + readiness, workout + injury history, etc.

The LLM stays dumb. This code gets smart.

Each rule:
  - Has a condition that checks multiple domains simultaneously
  - Produces an action (modify_workout, suggest_deload, proactive_warning, etc.)
  - Generates a plain-English message for the explainer to verbalize
  - Has a priority (higher = more important, evaluated first)

Design principles:
  - 100% deterministic — no LLM, no randomness
  - Each rule is self-contained and testable
  - Rules fire independently; multiple can fire for the same request
  - Actions are advisory — the explainer verbalizes them, doesn't execute them
  - New rules can be added without touching other code

Usage:
    from shared.cross_domain_rules import evaluate_rules
    from shared.athlete_state_store import AthleteStateStore

    store = AthleteStateStore.load("athlete_123", "/data/state/")
    fired = evaluate_rules(
        action="create_workout",
        sport="run",
        workout_type="long_run",
        duration_min=90,
        fatigue_hint="tired",
        store=store,
    )
    for rule in fired:
        print(rule["action"], rule["message"])
"""

from typing import Optional

from athlete_state_store import AthleteStateStore


# =============================================================================
# RULE RESULTS
# =============================================================================

class RuleResult:
    """Result of a fired cross-domain rule."""

    __slots__ = ("rule_id", "action", "message", "priority", "domains")

    def __init__(
        self,
        rule_id: str,
        action: str,
        message: str,
        priority: int = 50,
        domains: Optional[list[str]] = None,
    ):
        self.rule_id = rule_id
        self.action = action
        self.message = message
        self.priority = priority
        self.domains = domains or []

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "action": self.action,
            "message": self.message,
            "priority": self.priority,
            "domains": self.domains,
        }


# =============================================================================
# INDIVIDUAL RULES — each is a pure function
# =============================================================================

def _rule_injury_plus_increasing_load(
    action: str,
    store: AthleteStateStore,
    **kwargs,
) -> Optional[RuleResult]:
    """INJURY + TRAINING LOAD: active injury while load is climbing."""
    if not store.has_active_injury():
        return None
    if store.training_load.get("trend") != "increasing":
        return None

    injury = store.active_injuries()[0]
    area = injury["area"]
    return RuleResult(
        rule_id="injury_increasing_load",
        action="reduce_load",
        message=(
            f"Your {area} is still active and your training load has been climbing. "
            f"Let's ease off this week to give it time to settle."
        ),
        priority=90,
        domains=["injury", "training_load"],
    )


def _rule_injury_plus_matching_sport(
    action: str,
    sport: Optional[str] = None,
    store: AthleteStateStore = None,
    **kwargs,
) -> Optional[RuleResult]:
    """INJURY + WORKOUT: active injury in area affected by requested sport."""
    if action != "create_workout" or not sport:
        return None
    if not store or not store.has_active_injury():
        return None

    # Map injury areas to sports that aggravate them
    aggravation_map = {
        "knee": ["run"],
        "shin": ["run"],
        "ankle": ["run"],
        "foot": ["run"],
        "back": ["run", "strength"],
        "shoulder": ["strength", "ski"],
        "wrist": ["strength", "bike"],
        "elbow": ["bike", "strength"],
        "hip": ["run", "bike"],
        "calf": ["run"],
    }

    # Alternative sport suggestions
    alternatives = {
        "run": "cycling or swimming",
        "bike": "a light walk or swimming",
        "strength": "light cardio",
        "ski": "cycling or light strength work",
    }

    for injury in store.active_injuries():
        area = injury["area"]
        aggravating_sports = aggravation_map.get(area, [])
        if sport.lower() in aggravating_sports:
            severity = injury.get("severity", 5)
            alt = alternatives.get(sport.lower(), "a lower-impact option")

            if severity >= 6:
                return RuleResult(
                    rule_id="injury_matching_sport",
                    action="modify_workout",
                    message=(
                        f"With your {area} at {severity}/10, "
                        f"{sport} will likely aggravate it. "
                        f"Let's swap to {alt} — same fitness benefit, less stress on the {area}."
                    ),
                    priority=95,
                    domains=["injury", "workout"],
                )
            else:
                return RuleResult(
                    rule_id="injury_matching_sport_mild",
                    action="proactive_warning",
                    message=(
                        f"Your {area} is at {severity}/10. "
                        f"You can {sport} but stay below the pain threshold — "
                        f"if it worsens during the session, switch to {alt}."
                    ),
                    priority=70,
                    domains=["injury", "workout"],
                )

    return None


def _rule_fatigue_plus_declining_readiness(
    fatigue_hint: Optional[str] = None,
    store: AthleteStateStore = None,
    **kwargs,
) -> Optional[RuleResult]:
    """FATIGUE + READINESS: self-reported fatigue + declining readiness trend."""
    if not store or not fatigue_hint:
        return None
    if fatigue_hint not in ("tired", "very_tired"):
        return None

    trend = store.readiness_trend()
    if trend != "declining":
        return None

    current = store.current_readiness()
    days_declining = sum(
        1 for r in store.readiness_history[-5:]
        if r in ("Yellow", "Orange", "Red")
    )

    if fatigue_hint == "very_tired" or days_declining >= 3:
        return RuleResult(
            rule_id="fatigue_declining_readiness",
            action="suggest_deload",
            message=(
                f"{days_declining} days of declining readiness plus how you feel "
                f"means your body needs a lighter week. Your plan will adjust."
            ),
            priority=85,
            domains=["fatigue", "readiness"],
        )
    else:
        return RuleResult(
            rule_id="fatigue_mild_declining",
            action="reduce_intensity",
            message=(
                f"You're tired and readiness has been trending down. "
                f"Let's keep today easy — your body is telling you something."
            ),
            priority=60,
            domains=["fatigue", "readiness"],
        )


def _rule_correlation_warning(
    action: str,
    sport: Optional[str] = None,
    workout_type: Optional[str] = None,
    duration_min: Optional[int] = None,
    store: AthleteStateStore = None,
    **kwargs,
) -> Optional[RuleResult]:
    """PATTERN: historical correlation between workout type and injury.

    Example: "last 3 times you ran over 90min, your knee flared up."
    """
    if action != "create_workout" or not store:
        return None

    correlations = store.get_relevant_correlations(
        sport=sport,
        workout_type=workout_type,
        min_confidence=0.5,
    )

    if not correlations:
        return None

    # Take the highest-confidence correlation
    best = max(correlations, key=lambda c: c["confidence"])

    # Build a human-readable warning
    trigger = best["trigger"].replace("_", " ")
    result = best["result"].replace("_", " ")
    count = best["occurrences"]

    msg = f"Pattern detected: the last {count} times after {trigger}, you had {result}."
    if duration_min and duration_min > 60:
        msg += f" Want to cap the duration or add extra warm-up?"
    else:
        msg += f" Keep an eye on it during this session."

    return RuleResult(
        rule_id="correlation_warning",
        action="proactive_warning",
        message=msg,
        priority=75,
        domains=["history", "workout"],
    )


def _rule_nutrition_plus_recovery(
    action: str,
    topic: Optional[str] = None,
    store: AthleteStateStore = None,
    **kwargs,
) -> Optional[RuleResult]:
    """NUTRITION + RECOVERY: nutrition question while in poor readiness."""
    if not store:
        return None
    if action not in ("answer_question", "explain"):
        return None
    if topic not in ("nutrition", "recovery"):
        return None

    current = store.current_readiness()
    if current not in ("Orange", "Red"):
        return None

    return RuleResult(
        rule_id="nutrition_recovery",
        action="recovery_nutrition",
        message=(
            "When you're this fatigued, recovery nutrition matters more. "
            "Focus on protein within 30 minutes after training and extra "
            "carbs tonight to help your body bounce back."
        ),
        priority=50,
        domains=["nutrition", "readiness"],
    )


def _rule_repeated_injury(
    store: AthleteStateStore = None,
    **kwargs,
) -> Optional[RuleResult]:
    """INJURY HISTORY: same area injured multiple times → escalate advice."""
    if not store:
        return None

    for injury in store.active_injuries():
        occurrences = injury.get("occurrences", 1)
        if occurrences >= 3:
            area = injury["area"]
            if not store.was_advised(f"see_physio_{area}"):
                store.record_advice(f"see_physio_{area}")
                return RuleResult(
                    rule_id="repeated_injury",
                    action="escalate_medical",
                    message=(
                        f"This is the {_ordinal(occurrences)} time your {area} has "
                        f"flared up. That kind of pattern usually means something "
                        f"structural needs attention. It's worth seeing a physio."
                    ),
                    priority=88,
                    domains=["injury", "history"],
                )

    return None


def _rule_overreaching_detection(
    store: AthleteStateStore = None,
    **kwargs,
) -> Optional[RuleResult]:
    """READINESS + LOAD: load increasing while readiness declining = overreaching."""
    if not store:
        return None

    load_trend = store.training_load.get("trend")
    readiness_trend = store.readiness_trend()

    if load_trend != "increasing" or readiness_trend != "declining":
        return None

    current = store.current_readiness()
    if current not in ("Orange", "Red"):
        return None

    return RuleResult(
        rule_id="overreaching_detection",
        action="suggest_deload",
        message=(
            "Your training load is climbing but your body's response is heading "
            "the other way. That's a classic overreaching signal. "
            "A recovery week now will make you stronger next week."
        ),
        priority=92,
        domains=["training_load", "readiness"],
    )


def _rule_sleep_topic_plus_load(
    action: str,
    topic: Optional[str] = None,
    store: AthleteStateStore = None,
    **kwargs,
) -> Optional[RuleResult]:
    """SLEEP + TRAINING LOAD: sleep issues while load is high."""
    if not store:
        return None
    if topic != "sleep":
        return None
    if action not in ("answer_question", "explain"):
        return None

    load = store.training_load.get("this_week_load", 0)
    sessions = store.training_load.get("this_week_sessions", 0)

    # Only fire if there's meaningful training happening
    if sessions < 3:
        return None

    trend = store.training_load.get("trend", "stable")
    if trend == "increasing":
        return RuleResult(
            rule_id="sleep_high_load",
            action="contextual_advice",
            message=(
                "Poor sleep and increasing training load don't mix well. "
                "High training stress can actually disrupt sleep. "
                "Consider keeping today easy and prioritizing sleep tonight."
            ),
            priority=65,
            domains=["sleep", "training_load"],
        )

    return None


# =============================================================================
# RULE REGISTRY — all rules evaluated in order
# =============================================================================

_ALL_RULES = [
    _rule_overreaching_detection,
    _rule_injury_plus_matching_sport,
    _rule_injury_plus_increasing_load,
    _rule_repeated_injury,
    _rule_fatigue_plus_declining_readiness,
    _rule_correlation_warning,
    _rule_nutrition_plus_recovery,
    _rule_sleep_topic_plus_load,
]


# =============================================================================
# PUBLIC API
# =============================================================================

def evaluate_rules(
    action: str,
    store: AthleteStateStore,
    sport: Optional[str] = None,
    workout_type: Optional[str] = None,
    duration_min: Optional[int] = None,
    fatigue_hint: Optional[str] = None,
    topic: Optional[str] = None,
) -> list[dict]:
    """Evaluate all cross-domain rules against current context.

    Args:
        action: Current GATC action (create_workout, explain, etc.)
        store: AthleteStateStore with current athlete state.
        sport: Requested sport (if applicable).
        workout_type: Type of workout (long_run, interval, easy, etc.)
        duration_min: Requested duration (if applicable).
        fatigue_hint: Self-reported fatigue (fresh, ok, tired, very_tired).
        topic: Conversation topic (injury, sleep, nutrition, etc.)

    Returns:
        List of fired rule dicts, sorted by priority (highest first).
        Each dict has: rule_id, action, message, priority, domains.
    """
    kwargs = {
        "action": action,
        "store": store,
        "sport": sport,
        "workout_type": workout_type,
        "duration_min": duration_min,
        "fatigue_hint": fatigue_hint,
        "topic": topic,
    }

    fired = []
    for rule_fn in _ALL_RULES:
        result = rule_fn(**kwargs)
        if result is not None:
            fired.append(result.to_dict())

    # Sort by priority (highest first)
    fired.sort(key=lambda r: r["priority"], reverse=True)
    return fired


# =============================================================================
# HELPERS
# =============================================================================

def _ordinal(n: int) -> str:
    """Convert integer to ordinal string (1st, 2nd, 3rd, etc.)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

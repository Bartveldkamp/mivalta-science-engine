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


def _rule_readiness_vs_intensity(
    action: str,
    store: AthleteStateStore = None,
    **kwargs,
) -> Optional[RuleResult]:
    """READINESS + WORKOUT: athlete requests high intensity but body isn't ready.

    A real coach sees the whole picture: "You want to do intervals, but
    you're Orange and declining. That's how injuries happen."
    """
    if action != "create_workout" or not store:
        return None

    # Extract the requested goal from interpreter output
    goal = kwargs.get("goal") or ""
    if not goal:
        # Try to infer from workout_type
        wt = kwargs.get("workout_type") or ""
        if wt in ("interval", "tempo"):
            goal = "threshold"

    high_intensity_goals = ("threshold", "vo2", "race_prep")
    if goal not in high_intensity_goals:
        return None

    current = store.current_readiness()
    if not current or current == "Green":
        return None

    trend = store.readiness_trend()

    # Orange/Red + declining = danger zone
    if current in ("Orange", "Red"):
        return RuleResult(
            rule_id="readiness_vs_intensity",
            action="modify_workout",
            message=(
                f"Your body is at {current} readiness"
                + (f" and trending {trend}" if trend != "unknown" else "")
                + f". High-intensity work right now risks pushing you into "
                f"overtraining. Let's swap this to an easy session — you'll "
                f"be able to hit those intervals harder when you're recovered."
            ),
            priority=88,
            domains=["readiness", "workout"],
        )

    # Yellow + declining = caution
    if current == "Yellow" and trend == "declining":
        return RuleResult(
            rule_id="readiness_vs_intensity_caution",
            action="reduce_intensity",
            message=(
                "Readiness has been dropping and you're asking for intensity. "
                "You can still train, but let's dial it back — a moderate "
                "session will keep the fitness coming without digging a hole."
            ),
            priority=72,
            domains=["readiness", "workout"],
        )

    return None


def _rule_high_load_workout_request(
    action: str,
    store: AthleteStateStore = None,
    duration_min: Optional[int] = None,
    **kwargs,
) -> Optional[RuleResult]:
    """LOAD + WORKOUT: athlete requests a long/hard session when load is already high.

    Catches: "Give me a 2-hour ride" when they've already done 6 sessions this week.
    """
    if action != "create_workout" or not store:
        return None

    sessions = store.training_load.get("this_week_sessions", 0)
    trend = store.training_load.get("trend", "stable")

    if sessions < 4:
        return None  # Not enough volume to worry

    # 5+ sessions AND either long duration or increasing trend
    long_session = duration_min and duration_min > 75
    high_load_week = sessions >= 5 and trend == "increasing"

    if not (long_session or high_load_week):
        return None

    # Don't fire if goal is already recovery
    goal = kwargs.get("goal") or ""
    if goal == "recovery":
        return None

    if sessions >= 6:
        return RuleResult(
            rule_id="high_load_workout",
            action="suggest_deload",
            message=(
                f"That's session #{sessions + 1} this week and your load "
                f"has been climbing. Adding more volume now risks diminishing "
                f"returns. A recovery day today means better performance tomorrow."
            ),
            priority=80,
            domains=["training_load", "workout"],
        )
    else:
        msg = (
            f"You've already logged {sessions} sessions this week"
        )
        if long_session:
            msg += f" and this would be {duration_min} minutes"
        msg += (
            ". Keep this one easier than usual — accumulated fatigue "
            "is real even when you feel fine."
        )
        return RuleResult(
            rule_id="high_load_moderate",
            action="reduce_intensity",
            message=msg,
            priority=62,
            domains=["training_load", "workout"],
        )


def _rule_fresh_injury_high_intensity(
    action: str,
    store: AthleteStateStore = None,
    **kwargs,
) -> Optional[RuleResult]:
    """INJURY + INTENSITY: recently injured athlete requesting hard workout.

    Not sport-specific (that's _rule_injury_plus_matching_sport).
    This catches: athlete has ANY active injury and wants intervals/threshold.
    """
    if action != "create_workout" or not store:
        return None
    if not store.has_active_injury():
        return None

    goal = kwargs.get("goal") or ""
    if goal not in ("threshold", "vo2", "race_prep", "strength"):
        return None

    injury = store.active_injuries()[0]
    area = injury["area"]
    severity = injury.get("severity", 5)

    if severity >= 5:
        return RuleResult(
            rule_id="fresh_injury_intensity",
            action="modify_workout",
            message=(
                f"You still have an active {area} issue at {severity}/10. "
                f"Hard efforts create more inflammation and can turn a "
                f"manageable issue into a serious one. Easy work only until "
                f"it's below a 3."
            ),
            priority=86,
            domains=["injury", "workout"],
        )

    return None


def _rule_comeback_after_break(
    action: str,
    store: AthleteStateStore = None,
    **kwargs,
) -> Optional[RuleResult]:
    """LOAD + HISTORY: athlete requesting workout after extended break.

    Catches: zero sessions this week, zero last week, and athlete wants
    to jump back in at full intensity.
    """
    if action != "create_workout" or not store:
        return None

    this_week = store.training_load.get("this_week_sessions", 0)
    last_week = store.training_load.get("last_week_load", 0)

    if this_week > 0 or last_week > 0:
        return None  # Not a comeback situation

    # Check if there are any recent workouts at all
    if store.recent_workouts:
        return None  # Has some recent activity

    goal = kwargs.get("goal") or ""
    duration = kwargs.get("duration_min") or 0

    # Only fire for ambitious requests
    if goal in ("threshold", "vo2") or (duration and duration > 60):
        return RuleResult(
            rule_id="comeback_after_break",
            action="modify_workout",
            message=(
                "Welcome back! Since you've been away, let's start with "
                "something moderate. Your cardiovascular system bounces back "
                "fast, but tendons and joints need a gentler ramp. "
                "Easy to moderate for the first week."
            ),
            priority=78,
            domains=["training_load", "history"],
        )

    return None


def _rule_consistent_progress(
    action: str,
    store: AthleteStateStore = None,
    **kwargs,
) -> Optional[RuleResult]:
    """READINESS + LOAD: everything is going well — encourage the athlete.

    A good coach doesn't just warn. When load is stable, readiness is
    Green/improving, and no injuries — acknowledge the good work.
    """
    if action != "create_workout" or not store:
        return None

    current = store.current_readiness()
    if current != "Green":
        return None

    trend = store.readiness_trend()
    if trend not in ("improving", "stable"):
        return None

    if store.has_active_injury():
        return None

    sessions = store.training_load.get("this_week_sessions", 0)
    if sessions < 2:
        return None  # Not enough data

    load_trend = store.training_load.get("trend", "stable")
    if load_trend == "increasing":
        # Green + improving readiness + increasing load = adapting well
        return RuleResult(
            rule_id="consistent_progress",
            action="encouragement",
            message=(
                "Your body is responding well — load is building and "
                "readiness is holding. You're in a good spot to push a bit."
            ),
            priority=30,  # Low priority — warnings always come first
            domains=["readiness", "training_load"],
        )

    return None


def _rule_morning_athlete_evening_intensity(
    action: str,
    store: AthleteStateStore = None,
    **kwargs,
) -> Optional[RuleResult]:
    """PREFERENCES + WORKOUT: leveraging known preferences for better coaching.

    If we know athlete prefers mornings and they're asking for a high-intensity
    session, that's actually fine — affirm the timing choice.
    If they have an avoids_intensity preference and ask for VO2 work, gently
    check if they really want that.
    """
    if action != "create_workout" or not store:
        return None

    goal = kwargs.get("goal") or ""

    # Athlete who avoids intensity but asks for it
    if store.preferences.get("avoids_intensity") and goal in ("threshold", "vo2"):
        return RuleResult(
            rule_id="preference_intensity_check",
            action="proactive_warning",
            message=(
                "You usually prefer easier sessions. If you're feeling "
                "motivated for something harder today, that's great — just "
                "make sure you warm up well and listen to your body."
            ),
            priority=35,
            domains=["preferences", "workout"],
        )

    return None


# =============================================================================
# RULE REGISTRY — all rules evaluated in order
# =============================================================================

_ALL_RULES = [
    # --- Critical safety rules (priority 85-95) ---
    _rule_overreaching_detection,           # load↑ + readiness↓ = overreaching
    _rule_injury_plus_matching_sport,       # injury + aggravating sport
    _rule_injury_plus_increasing_load,      # injury + climbing load
    _rule_readiness_vs_intensity,           # orange/red + wants hard workout
    _rule_fresh_injury_high_intensity,      # any injury + wants intervals
    _rule_repeated_injury,                  # same area hurt 3+ times → see physio
    # --- Load management rules (priority 60-85) ---
    _rule_fatigue_plus_declining_readiness, # tired + readiness dropping
    _rule_high_load_workout_request,        # 5+ sessions + wants more
    _rule_comeback_after_break,             # zero recent load + wants intensity
    # --- Contextual intelligence (priority 30-65) ---
    _rule_correlation_warning,              # historical pattern match
    _rule_sleep_topic_plus_load,            # sleep issues + high training
    _rule_nutrition_plus_recovery,          # nutrition Q + poor readiness
    _rule_morning_athlete_evening_intensity,# preference-aware coaching
    _rule_consistent_progress,              # positive reinforcement
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
    goal: Optional[str] = None,
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
        goal: Requested training goal (endurance, threshold, vo2, etc.)

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
        "goal": goal,
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

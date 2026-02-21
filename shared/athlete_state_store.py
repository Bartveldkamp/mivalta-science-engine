"""
MiValta Athlete State Store — Cross-Domain Knowledge Persistence

Maintains a unified athlete state that spans all coaching domains:
injury history, training load, readiness trends, learned preferences,
and cross-domain correlations. This is the foundation for intelligent
cross-interaction between knowledge areas.

Design principles:
  - JSON-serializable for on-device persistence (SQLite or file)
  - Compact enough for mobile (< 4 KB per athlete)
  - Deterministic updates — no LLM involvement in state mutations
  - Append-only with decay — facts accumulate, old ones fade

Data flow:
    Post-conversation / post-workout hooks
        → update_from_triage(), update_from_workout(), update_from_readiness()
            → AthleteStateStore (in-memory + persisted JSON)
                → cross_domain_rules.evaluate() reads this
                    → context_assembler enriches ENGINE_PAYLOAD

Usage (Python reference — port to Rust/Kotlin for production):
    from shared.athlete_state_store import AthleteStateStore

    store = AthleteStateStore.load("athlete_123", "/data/state/")
    store.update_from_triage(triage_data)
    store.update_from_readiness("Yellow", "Accumulated")
    store.save()
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional


# Hard caps to keep state compact
MAX_INJURIES = 5
MAX_READINESS_HISTORY = 14  # 2 weeks of daily readings
MAX_WORKOUT_HISTORY = 10
MAX_CORRELATIONS = 10
MAX_PREFERENCES = 10


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class AthleteStateStore:
    """Persistent cross-domain athlete state.

    Tracks injuries, training load, readiness trends, preferences,
    and cross-domain correlations across sessions.
    """

    def __init__(self, athlete_id: str):
        self.athlete_id = athlete_id
        self.updated_at: str = _now_iso()

        # --- Injury domain ---
        # List of {area, side, severity, trigger, first_seen, last_seen, status, occurrences}
        self.injuries: list[dict] = []

        # --- Training load domain ---
        self.training_load: dict = {
            "this_week_sessions": 0,
            "this_week_load": 0,     # Simplified TSS-like load score
            "last_week_load": 0,
            "trend": "stable",       # "increasing", "stable", "declining"
        }

        # --- Recent workouts (for correlation detection) ---
        # List of {date, sport, zone, duration_min, type}
        self.recent_workouts: list[dict] = []

        # --- Readiness domain ---
        # List of level strings, most recent last
        self.readiness_history: list[str] = []

        # --- Preferences (learned from conversations) ---
        # Dict of {key: value} — e.g., {"morning_athlete": True, "primary_sport": "run"}
        self.preferences: dict = {}

        # --- Cross-domain correlations ---
        # List of {trigger, result, occurrences, confidence, last_seen}
        # e.g., {"trigger": "long_run > 90min", "result": "knee_flare",
        #        "occurrences": 3, "confidence": 0.8}
        self.correlations: list[dict] = []

        # --- Recent advice given (avoid repeating) ---
        self.recent_advice: list[str] = []

    # =========================================================================
    # INJURY UPDATES
    # =========================================================================

    def update_from_triage(self, triage_data: dict) -> None:
        """Update injury state from a completed triage session.

        Args:
            triage_data: Dict with pain_area, side, severity_0_10,
                         trigger, duration_days, swelling.
        """
        area = triage_data.get("pain_area")
        if not area:
            return

        side = triage_data.get("side", "unknown")
        severity = triage_data.get("severity_0_10", 5)
        trigger = triage_data.get("trigger", "unknown")
        now = _now_iso()

        # Check if we already track this injury
        existing = self._find_injury(area, side)
        if existing:
            existing["severity"] = severity
            existing["trigger"] = trigger
            existing["last_seen"] = now
            existing["occurrences"] = existing.get("occurrences", 1) + 1
            existing["status"] = "active"
        else:
            self.injuries.append({
                "area": area,
                "side": side,
                "severity": severity,
                "trigger": trigger,
                "first_seen": now,
                "last_seen": now,
                "status": "active",
                "occurrences": 1,
            })

        # Enforce cap — keep most recent
        if len(self.injuries) > MAX_INJURIES:
            self.injuries.sort(key=lambda i: i.get("last_seen", ""), reverse=True)
            self.injuries = self.injuries[:MAX_INJURIES]

        # Record correlation if we have recent workout context
        if self.recent_workouts and trigger != "rest":
            last_workout = self.recent_workouts[-1]
            self._record_correlation(
                trigger=f"{last_workout.get('sport', 'workout')}_{last_workout.get('type', 'session')}",
                result=f"{area}_pain",
            )

        self.updated_at = now

    def resolve_injury(self, area: str, side: str = "unknown") -> None:
        """Mark an injury as resolved (athlete reports it's better)."""
        existing = self._find_injury(area, side)
        if existing:
            existing["status"] = "resolved"
            existing["last_seen"] = _now_iso()

    def active_injuries(self) -> list[dict]:
        """Return currently active injuries."""
        return [i for i in self.injuries if i.get("status") == "active"]

    def has_active_injury(self, area: Optional[str] = None) -> bool:
        """Check if athlete has any (or specific) active injury."""
        active = self.active_injuries()
        if area:
            return any(i["area"] == area for i in active)
        return len(active) > 0

    def _find_injury(self, area: str, side: str = "unknown") -> Optional[dict]:
        for injury in self.injuries:
            if injury["area"] == area and injury.get("side", "unknown") == side:
                return injury
        return None

    # =========================================================================
    # TRAINING LOAD UPDATES
    # =========================================================================

    def update_from_workout(
        self,
        sport: str,
        zone: str = "Z2",
        duration_min: int = 0,
        workout_type: str = "session",
    ) -> None:
        """Record a completed workout.

        Args:
            sport: Sport enum (run, bike, ski, etc.)
            zone: Primary zone (Z1-Z8, R)
            duration_min: Total duration in minutes
            workout_type: "easy", "interval", "long_run", "recovery", "session"
        """
        today = _today_iso()
        self.recent_workouts.append({
            "date": today,
            "sport": sport,
            "zone": zone,
            "duration_min": duration_min,
            "type": workout_type,
        })

        # Cap recent workouts
        if len(self.recent_workouts) > MAX_WORKOUT_HISTORY:
            self.recent_workouts = self.recent_workouts[-MAX_WORKOUT_HISTORY:]

        # Update weekly load (simplified: zone-weighted duration)
        zone_weight = _zone_load_weight(zone)
        session_load = int(duration_min * zone_weight)
        self.training_load["this_week_sessions"] += 1
        self.training_load["this_week_load"] += session_load

        self._update_load_trend()
        self.updated_at = _now_iso()

    def start_new_week(self) -> None:
        """Roll over weekly load tracking. Call at start of new training week."""
        self.training_load["last_week_load"] = self.training_load["this_week_load"]
        self.training_load["this_week_sessions"] = 0
        self.training_load["this_week_load"] = 0
        self._update_load_trend()

    def _update_load_trend(self) -> None:
        this_week = self.training_load["this_week_load"]
        last_week = self.training_load["last_week_load"]
        if last_week == 0:
            self.training_load["trend"] = "stable"
        elif this_week > last_week * 1.15:
            self.training_load["trend"] = "increasing"
        elif this_week < last_week * 0.85:
            self.training_load["trend"] = "declining"
        else:
            self.training_load["trend"] = "stable"

    # =========================================================================
    # READINESS UPDATES
    # =========================================================================

    def update_from_readiness(self, level: str, state: Optional[str] = None) -> None:
        """Record today's readiness level.

        Args:
            level: "Green", "Yellow", "Orange", "Red"
            state: Optional state name ("Recovered", "Productive", etc.)
        """
        self.readiness_history.append(level)
        if len(self.readiness_history) > MAX_READINESS_HISTORY:
            self.readiness_history = self.readiness_history[-MAX_READINESS_HISTORY:]
        self.updated_at = _now_iso()

    def readiness_trend(self) -> str:
        """Compute readiness trend from recent history.

        Returns: "improving", "stable", "declining", "unknown"
        """
        history = self.readiness_history
        if len(history) < 3:
            return "unknown"

        _LEVEL_SCORE = {"Green": 3, "Yellow": 2, "Orange": 1, "Red": 0}
        recent = [_LEVEL_SCORE.get(h, 2) for h in history[-3:]]
        older = [_LEVEL_SCORE.get(h, 2) for h in history[-6:-3]] if len(history) >= 6 else recent

        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)

        if avg_recent > avg_older + 0.3:
            return "improving"
        elif avg_recent < avg_older - 0.3:
            return "declining"
        return "stable"

    def current_readiness(self) -> Optional[str]:
        """Return most recent readiness level."""
        return self.readiness_history[-1] if self.readiness_history else None

    # =========================================================================
    # PREFERENCE UPDATES
    # =========================================================================

    def update_preference(self, key: str, value) -> None:
        """Store or update a learned preference."""
        self.preferences[key] = value
        # Enforce cap by removing oldest if needed
        if len(self.preferences) > MAX_PREFERENCES:
            keys = list(self.preferences.keys())
            del self.preferences[keys[0]]
        self.updated_at = _now_iso()

    def update_preferences_from_message(self, message: str) -> None:
        """Extract and store preferences from a user message.

        Lightweight extraction — heavier extraction stays in memory_extractor.
        This only catches cross-domain-relevant preferences.
        """
        lower = message.lower()

        # Sport preference
        sport_map = {
            "run": r"\b(?:run|running|jog)\b",
            "bike": r"\b(?:cycl|bike|biking|ride)\w*\b",
            "ski": r"\b(?:ski|skiing)\b",
            "strength": r"\b(?:strength|gym|weights?|lifting)\b",
        }
        for sport, pattern in sport_map.items():
            if re.search(pattern, lower):
                self.preferences["primary_sport"] = sport

        # Time preference
        if re.search(r"\b(?:morning|before work|early|first thing)\b", lower):
            self.preferences["morning_athlete"] = True
        elif re.search(r"\b(?:evening|after work|night)\b", lower):
            self.preferences["morning_athlete"] = False

        # Intensity avoidance
        if re.search(r"\b(?:no|hate|avoid)\s+(?:hard|intense|interval)", lower):
            self.preferences["avoids_intensity"] = True

    # =========================================================================
    # CROSS-DOMAIN CORRELATIONS
    # =========================================================================

    def _record_correlation(self, trigger: str, result: str) -> None:
        """Record a potential cross-domain correlation.

        Called automatically when triage data + recent workout align.
        After N occurrences, confidence increases and the correlation
        can trigger proactive warnings.
        """
        # Check if we already track this correlation
        for corr in self.correlations:
            if corr["trigger"] == trigger and corr["result"] == result:
                corr["occurrences"] += 1
                corr["confidence"] = min(1.0, 0.3 + corr["occurrences"] * 0.2)
                corr["last_seen"] = _now_iso()
                return

        self.correlations.append({
            "trigger": trigger,
            "result": result,
            "occurrences": 1,
            "confidence": 0.3,
            "last_seen": _now_iso(),
        })

        # Cap correlations
        if len(self.correlations) > MAX_CORRELATIONS:
            self.correlations.sort(key=lambda c: c["confidence"], reverse=True)
            self.correlations = self.correlations[:MAX_CORRELATIONS]

    def get_relevant_correlations(
        self,
        sport: Optional[str] = None,
        workout_type: Optional[str] = None,
        min_confidence: float = 0.5,
    ) -> list[dict]:
        """Find correlations relevant to a given workout context.

        Returns correlations where the trigger matches the sport/workout
        and confidence meets the threshold.
        """
        relevant = []
        for corr in self.correlations:
            if corr["confidence"] < min_confidence:
                continue
            trigger = corr["trigger"].lower()
            if sport and sport.lower() in trigger:
                relevant.append(corr)
            elif workout_type and workout_type.lower() in trigger:
                relevant.append(corr)
        return relevant

    # =========================================================================
    # ADVICE TRACKING
    # =========================================================================

    def record_advice(self, advice_key: str) -> None:
        """Record that a piece of advice was given (avoid repeating)."""
        if advice_key not in self.recent_advice:
            self.recent_advice.append(advice_key)
        # Keep last 10
        if len(self.recent_advice) > 10:
            self.recent_advice = self.recent_advice[-10:]

    def was_advised(self, advice_key: str) -> bool:
        """Check if this advice was recently given."""
        return advice_key in self.recent_advice

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> dict:
        """Serialize entire state to a JSON-compatible dict."""
        return {
            "athlete_id": self.athlete_id,
            "updated_at": self.updated_at,
            "injuries": self.injuries,
            "training_load": self.training_load,
            "recent_workouts": self.recent_workouts,
            "readiness_history": self.readiness_history,
            "preferences": self.preferences,
            "correlations": self.correlations,
            "recent_advice": self.recent_advice,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AthleteStateStore":
        """Reconstruct state from a serialized dict."""
        store = cls(data.get("athlete_id", "unknown"))
        store.updated_at = data.get("updated_at", _now_iso())
        store.injuries = data.get("injuries", [])
        store.training_load = data.get("training_load", store.training_load)
        store.recent_workouts = data.get("recent_workouts", [])
        store.readiness_history = data.get("readiness_history", [])
        store.preferences = data.get("preferences", {})
        store.correlations = data.get("correlations", [])
        store.recent_advice = data.get("recent_advice", [])
        return store

    def save(self, directory: str) -> str:
        """Persist state to a JSON file.

        Args:
            directory: Directory to write the state file.

        Returns:
            Path to the written file.
        """
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.athlete_id}.state.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, athlete_id: str, directory: str) -> "AthleteStateStore":
        """Load state from disk, or create a fresh store if not found.

        Args:
            athlete_id: Unique athlete identifier.
            directory: Directory to read from.

        Returns:
            AthleteStateStore instance (loaded or fresh).
        """
        path = os.path.join(directory, f"{athlete_id}.state.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        return cls(athlete_id)

    # =========================================================================
    # COMPACT SERIALIZATION FOR LLM CONTEXT
    # =========================================================================

    def serialize_for_prompt(self) -> str:
        """Serialize cross-domain state into a compact block for LLM context.

        Target: <= 100 tokens. Includes only active, relevant cross-domain facts.
        Returns empty string if no cross-domain state exists.
        """
        parts = []

        # Active injuries
        active = self.active_injuries()
        if active:
            injury_strs = []
            for inj in active:
                s = f"{inj['area']}"
                if inj.get("side") and inj["side"] != "unknown":
                    s = f"{inj['side']} {s}"
                s += f" ({inj['severity']}/10)"
                if inj.get("occurrences", 1) > 1:
                    s += f" x{inj['occurrences']}"
                injury_strs.append(s)
            parts.append("- Active injuries: " + " | ".join(injury_strs))

        # Load trend
        load = self.training_load
        if load["this_week_load"] > 0 or load["last_week_load"] > 0:
            parts.append(
                f"- Load: {load['this_week_sessions']} sessions this week, "
                f"trend {load['trend']}"
            )

        # Readiness trend
        trend = self.readiness_trend()
        if trend != "unknown":
            current = self.current_readiness() or "?"
            parts.append(f"- Readiness: {current}, trend {trend}")

        # High-confidence correlations (warnings)
        warnings = [c for c in self.correlations if c["confidence"] >= 0.6]
        if warnings:
            warn_strs = [
                f"{c['trigger']} -> {c['result']} (x{c['occurrences']})"
                for c in warnings[:3]
            ]
            parts.append("- Known patterns: " + " | ".join(warn_strs))

        if not parts:
            return ""

        return "CROSS_DOMAIN:\n" + "\n".join(parts)


# =============================================================================
# HELPERS
# =============================================================================

def _zone_load_weight(zone: str) -> float:
    """Simplified zone-based load weighting for weekly load tracking.

    Higher zones = more stress per minute of training.
    """
    weights = {
        "R": 0.3, "Z1": 0.5, "Z2": 0.7, "Z3": 1.0,
        "Z4": 1.4, "Z5": 1.8, "Z6": 2.2, "Z7": 2.5, "Z8": 2.8,
        "REST": 0.0, "OFF": 0.0,
    }
    return weights.get(zone.upper(), 0.7)

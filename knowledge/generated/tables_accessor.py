"""
Tables Accessor — Bridges cards and engine defaults.

Priority:
1) Card data (if exists)
2) Engine defaults (stable science)
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from gatc.defaults import (
    ALPHA_DEFAULT,
    BETA_DEFAULT,
    get_age_modifier,
    get_level_modifier,
    get_tau_values,
    get_d_max,
)


# Keep a single authoritative zone ordering.
ZONES = ("R", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8")


class TablesAccessor:
    """
    Provides GATC parameters sourced from knowledge tables with fallback to defaults.

    Note:
    - defaults.py expresses recovery/decay via τ multipliers (time constants), not decay-rate multipliers.
    - This accessor therefore returns τ values (days) for fitness/fatigue.
    """

    def __init__(self, tables):
        self.tables = tables

    def get_zone_parameters(self, sport: str, level: str) -> Dict[str, Any]:
        """
        Get α and β vectors.

        - Start from engine defaults
        - Optionally override from tables
        - Apply level modifiers: alpha_mult, beta_mult
        """
        alpha = np.array(ALPHA_DEFAULT, dtype=float).copy()
        beta = np.array(BETA_DEFAULT, dtype=float).copy()

        # Card override (if present)
        zone_effects = self.tables.get_raw_table("energy_zones_v4", "zone_effects")
        if zone_effects:
            for i, row in enumerate(zone_effects):
                idx = i
                if "zone" in row:
                    if row["zone"] in ZONES:
                        idx = ZONES.index(row["zone"])
                    else:
                        continue

                if "alpha" in row and row["alpha"] is not None:
                    alpha[idx] = float(row["alpha"])
                if "beta" in row and row["beta"] is not None:
                    beta[idx] = float(row["beta"])

        # Apply level modifiers from defaults (stable science)
        level_mod = get_level_modifier(level)
        alpha *= float(level_mod.get("alpha_mult", 1.0))
        beta *= float(level_mod.get("beta_mult", 1.0))

        return {"alpha": alpha, "beta": beta}

    def get_decay_parameters(self, age: int, level: str) -> Dict[str, float]:
        """
        Get personalized decay parameters.

        Source of truth: defaults.get_tau_values(age, level)
        Returns:
          - tau_fitness_days
          - tau_fatigue_days
          - lambda (decay rate = 1/tau_fitness)
          - mu (decay rate = 1/tau_fatigue)
          - ratio (tau_fitness / tau_fatigue)
        """
        tau = get_tau_values(age=age, level=level)
        return {
            "tau_fitness_days": float(tau["tau_fitness"]),
            "tau_fatigue_days": float(tau["tau_fatigue"]),
            "lambda": float(tau["lambda"]),
            "mu": float(tau["mu"]),
            "ratio": float(tau["ratio"]),
        }

    def get_d_max(self, level: str, age: int) -> float:
        """Get maximum fatigue threshold (defaults-based)."""
        return float(get_d_max(level=level, age=age))

    def get_modifiers(self, age: int, level: str) -> Dict[str, float]:
        """
        Convenience accessor for modifier dicts if needed elsewhere.
        """
        age_mod = get_age_modifier(age)
        level_mod = get_level_modifier(level)
        out: Dict[str, float] = {}
        out.update({k: float(v) for k, v in age_mod.items()})
        out.update({k: float(v) for k, v in level_mod.items()})
        return out

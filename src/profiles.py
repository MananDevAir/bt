"""Profile presets — conservative / balanced / aggressive.

Each profile is a dict of overrides that gets deep-merged onto the base
config.yaml.  Anything not specified in the profile keeps its config.yaml
value, so the `balanced` profile is an empty dict (zero behavioural change).

Usage in config.yaml:
    profile: aggressive       # or conservative | balanced

The merge happens inside `config.load()` after YAML parsing and before
validation, so all downstream code sees a single coherent Config object.
"""
from __future__ import annotations

from typing import Any

__all__ = ["PROFILES", "apply_profile"]

# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
PROFILES: dict[str, dict[str, Any]] = {
    # ── Conservative ──────────────────────────────────────────────────────
    # Fewer trades, wider stops, higher R:R bar.
    # Best for: swing/positional traders who want only high-conviction setups.
    "conservative": {
        "thresholds": {
            "watch":  25,
            "signal": 50,
            "strong": 75,
        },
        "gates": {
            "min_rr":        2.0,
            "min_adx":       25,
            "max_stop_atr":  3.5,
            "atr_pct_min":   20,
            "atr_pct_max":   90,
            "cooldown_hours": 6,
            "cooldown_score_override": 20,
        },
        "risk": {
            "atr_stop_mult":            2.0,
            "structure_stop_buffer_atr": 0.30,
            "tp_r_multiples":           [1.5, 3.0, 4.5],
            "tp_allocation":            [40, 30, 30],
            "snap_tolerance_atr":       0.40,
        },
        "weights": {
            "trend":     35,
            "structure": 35,
            "momentum":  15,
            "zones":     10,
            "volume":     5,
        },
        "tf_multiplier": {
            "1w": 5.0, "1d": 4.0, "4h": 2.0, "1h": 1.5, "15m": 1.0,
        },
    },

    # ── Balanced (default) ────────────────────────────────────────────────
    # Empty: uses config.yaml values as-is.
    "balanced": {},

    # ── Aggressive ────────────────────────────────────────────────────────
    # More trades, tighter stops, lower thresholds.
    # Best for: intraday traders who want early entries and accept more noise.
    "aggressive": {
        "thresholds": {
            "watch":  12,
            "signal": 30,
            "strong": 55,
        },
        "gates": {
            "min_rr":        1.0,
            "min_adx":       15,
            "max_stop_atr":  2.5,
            "atr_pct_min":   10,
            "atr_pct_max":   97,
            "cooldown_hours": 2,
            "cooldown_score_override": 10,
        },
        "risk": {
            "atr_stop_mult":            1.2,
            "structure_stop_buffer_atr": 0.15,
            "tp_r_multiples":           [0.8, 1.5, 2.5],
            "tp_allocation":            [50, 30, 20],
            "snap_tolerance_atr":       0.25,
        },
        "weights": {
            "trend":     25,
            "structure": 25,
            "momentum":  25,
            "zones":     15,
            "volume":    10,
        },
        "tf_multiplier": {
            "1w": 3.0, "1d": 2.5, "4h": 2.0, "1h": 2.0, "15m": 1.5,
        },
    },
}


# --------------------------------------------------------------------------- #
# Merge logic
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge `overlay` into a *copy* of `base`.

    - Dict values are merged recursively (overlay keys win on conflict).
    - Non-dict values in the overlay replace the base value entirely.
    - Keys in the base that are not in the overlay are preserved.
    """
    out = dict(base)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def apply_profile(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply the selected profile onto the raw YAML dict.

    Reads ``raw["profile"]`` (default: ``"balanced"``), looks up the preset,
    and deep-merges it.  Returns a new dict — the original is not mutated.

    Raises ValueError for unknown profile names.
    """
    name = str(raw.get("profile", "balanced")).lower().strip()
    if name not in PROFILES:
        valid = ", ".join(sorted(PROFILES))
        raise ValueError(
            f"unknown profile {name!r}; choose from: {valid}"
        )
    overlay = PROFILES[name]
    if not overlay:
        return dict(raw)       # balanced = no-op copy
    return _deep_merge(raw, overlay)

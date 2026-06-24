"""Miner-Regel zur linearen Schadensakkumulation."""

import numpy as np


class MinerRule:
    """
    Lineare Schadensakkumulation nach Palmgren-Miner:

        D = Σ (n_i / N_i)

    Versagen tritt ein bei D = 1 (modifiziert: 0.7 – 2.0 je nach Werkstoff).

    Referenz: IEC 60068-2-14, DIN 50100
    """

    def __init__(self, failure_threshold: float = 1.0):
        """
        Args:
            failure_threshold: Schadenssumme bei Versagen (default 1.0)
        """
        self.failure_threshold = failure_threshold

    def total_damage(self, damage_per_cycle_array: np.ndarray) -> float:
        """Summiert alle Einzel-Schädigungen."""
        return float(np.sum(damage_per_cycle_array))

    def remaining_life_fraction(self, accumulated_damage: float) -> float:
        """Gibt den verbleibenden Lebensdaueranteil (0–1) zurück."""
        return max(0.0, 1.0 - accumulated_damage / self.failure_threshold)

    def is_failed(self, accumulated_damage: float) -> bool:
        """True wenn Versagen eingetreten."""
        return accumulated_damage >= self.failure_threshold

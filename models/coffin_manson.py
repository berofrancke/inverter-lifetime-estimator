"""Coffin-Manson-Modell für thermomechanische Ermüdung."""

import numpy as np


class CoffinMansonModel:
    """
    Coffin-Manson-Gleichung für Temperaturwechselermüdung:

        N_f(ΔT) = N_ref * (ΔT_ref / ΔT)^n

    Häufig erweitert durch Norris-Landzberg zur Berücksichtigung
    der Temperaturabhängigkeit.

    Referenz: IEC 60749-34, JEDEC JEP122
    """

    def __init__(self, n: float = 2.0, delta_T_ref: float = 40.0, N_ref: float = 50000.0):
        """
        Args:
            n: Coffin-Manson-Exponent (typisch 1–3 für Leistungselektronik)
            delta_T_ref: Referenz-Temperaturschwingbreite [K]
            N_ref: Zyklen bis Versagen bei delta_T_ref
        """
        self.n = n
        self.delta_T_ref = delta_T_ref
        self.N_ref = N_ref

    def cycles_to_failure(self, delta_T: float) -> float:
        """
        Berechnet die Anzahl der Zyklen bis Versagen.

        Args:
            delta_T: Temperaturschwingbreite [K]
        Returns:
            N_f: Zyklen bis Versagen (inf wenn delta_T = 0)
        """
        if delta_T <= 0:
            return float("inf")
        return self.N_ref * (self.delta_T_ref / delta_T) ** self.n

    def damage_per_cycle(self, delta_T: float) -> float:
        """Schädigung pro Zyklus (1/N_f)."""
        N_f = self.cycles_to_failure(delta_T)
        return 0.0 if N_f == float("inf") else 1.0 / N_f

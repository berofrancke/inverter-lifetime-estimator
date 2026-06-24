"""Arrhenius-Lebensdauermodell für thermische Alterung."""

import numpy as np

k_B = 8.617333e-5  # Boltzmann-Konstante [eV/K]


class ArrheniusModel:
    """
    Arrhenius-Gleichung für temperaturabhängige Lebensdauer:

        L(T) = L_ref * exp(Ea/k_B * (1/(T+273.15) - 1/(T_ref+273.15)))

    Referenz: MIL-HDBK-217, IEC 62380
    """

    def __init__(self, Ea: float = 0.7, T_ref: float = 25.0, L_ref: float = 100000.0):
        """
        Args:
            Ea: Aktivierungsenergie [eV] (typisch 0.5–1.0 eV für Halbleiter)
            T_ref: Referenztemperatur [°C]
            L_ref: Referenzlebensdauer bei T_ref [Stunden]
        """
        self.Ea = Ea
        self.T_ref = T_ref
        self.L_ref = L_ref

    def acceleration_factor(self, T: float | np.ndarray) -> float | np.ndarray:
        """Berechnet den Beschleunigungsfaktor AF(T) relativ zur Referenztemperatur."""
        T_K = np.asarray(T) + 273.15
        T_ref_K = self.T_ref + 273.15
        return np.exp(self.Ea / k_B * (1.0 / T_K - 1.0 / T_ref_K))

    def lifetime(self, T: float | np.ndarray) -> float | np.ndarray:
        """Berechnet die Lebensdauer [h] bei Temperatur T [°C]."""
        return self.L_ref * self.acceleration_factor(T)

    def temperature_for_lifetime(self, target_hours: float) -> float:
        """Berechnet die Temperatur [°C], bei der eine Ziel-Lebensdauer erreicht wird."""
        # Numerische Umkehrfunktion
        from scipy.optimize import brentq
        f = lambda T: self.lifetime(T) - target_hours
        try:
            return brentq(f, -50, 250)
        except ValueError:
            return float("nan")

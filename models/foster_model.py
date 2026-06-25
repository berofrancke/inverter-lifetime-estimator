"""
Foster-Thermalmodell – Z_th(t) und T_j(t) Simulation.

Das Foster-Modell approximiert die thermische Impedanz Z_th(t) als Summe
von RC-Gliedern (Foster-Kette):

    Z_th(t) = Σ_i  r_i * (1 - exp(-t / τ_i))

Daraus folgt die zeitabhängige Sperrschichttemperatur:

    T_j(t) = T_h + P_tot * Z_th(t)

Stationärer Endwert (t → ∞):

    T_j,∞ = T_h + P_tot * Σ_i r_i  = T_h + P_tot * R_th,jh

Modellgrenzen (bitte im UI anzeigen):
  - T_h ist ein externer Eingabeparameter (z. B. Kühlkörpertemperatur).
  - Foster-Parameter r_i / τ_i stammen aus dem Datenblatt-Z_th.
  - P_tot wird als zeitlich konstant angenommen (keine transienten Lasten).
  - Modell gilt für Chip→Gehäuse (junction-to-case oder junction-to-heatsink).
"""

import numpy as np
from typing import Sequence, Union


# ---------------------------------------------------------------------------
# Default Foster-Parameter (4 Glieder, aus Datenblatt-Z_th)
# ---------------------------------------------------------------------------

FOSTER_DEFAULT_R = [0.0098, 0.0302, 0.107, 0.107]   # Wärmewiderstand je Glied [K/W]
FOSTER_DEFAULT_TAU = [4.72e-4, 0.0132, 0.152, 0.152]  # Zeitkonstante je Glied [s]
# Summe ≈ 0.254 K/W (Rth,jh gesamt)


# ---------------------------------------------------------------------------
# Z_th(t) – skalarer und vektorieller Aufruf
# ---------------------------------------------------------------------------

def foster_zth(
    t: Union[float, np.ndarray],
    r: Sequence[float],
    tau: Sequence[float],
) -> Union[float, np.ndarray]:
    """
    Berechnet die thermische Impedanz Z_th(t) nach dem Foster-Modell.

    Z_th(t) = Σ_i  r_i * (1 - exp(-t / τ_i))

    Args:
        t:    Zeitpunkt(e) [s] – float oder numpy-Array
        r:    Liste/Array der Partialwiderstände r_i [K/W]
        tau:  Liste/Array der Zeitkonstanten τ_i [s]

    Returns:
        Z_th [K/W] – gleicher Typ wie t

    Hinweis:
        Anzahl der Foster-Glieder ist flexibel (len(r) == len(tau)).
        Mindestens 1 Glied, empfohlen 4 für gute Kurvenanpassung.
    """
    r   = np.asarray(r,   dtype=float)
    tau = np.asarray(tau, dtype=float)

    if np.isscalar(t):
        zth = float(np.sum(r * (1.0 - np.exp(-t / tau))))
    else:
        t = np.asarray(t, dtype=float)
        # Broadcasting: t shape (N,), r/tau shape (M,) → outer sum
        zth = np.sum(r[np.newaxis, :] * (1.0 - np.exp(-t[:, np.newaxis] / tau[np.newaxis, :])), axis=1)
    return zth


# ---------------------------------------------------------------------------
# Th(t) – zeitabhängige Kühlkörpertemperatur (transiente Aufheizung)
# ---------------------------------------------------------------------------

def heatsink_temperature(
    P_tot: float,
    Rth: float,
    m: float,
    c: float,
    Tamb: float,
    t: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Zeitabhängige Kühlkörper-/Modultemperatur T_h(t) (transiente Aufheizung
    einer thermischen Masse über einen Wärmewiderstand R_th).

        T_h(t) = (P_tot * R_th * t) / (m * c * R_th + t) + T_amb

    Args:
        P_tot: Gesamtverlustleistung [W]
        Rth:   Wärmewiderstand [K/W]
        m:     Masse des Halbleitermoduls [kg]
        c:     Spezifische Wärmekapazität [J/(kg·K)]
        Tamb:  Umgebungstemperatur [°C]
        t:     Zeitpunkt(e) [s] – float oder numpy-Array

    Returns:
        T_h [°C] – gleicher Typ wie t

    Hinweis:
        Für t → ∞ strebt T_h gegen P_tot * R_th + T_amb (stationärer Endwert).
    """
    denom = m * c * Rth + t
    return (P_tot * Rth * t) / denom + Tamb


# ---------------------------------------------------------------------------
# T_j(t) – Simulation der Sperrschichttemperatur
# ---------------------------------------------------------------------------

def simulate_tj(
    P_tot: float,
    T_h: Union[float, np.ndarray],
    r: Sequence[float],
    tau: Sequence[float],
    t_end: float,
    dt: float,
) -> dict:
    """
    Simuliert den zeitlichen Verlauf der Sperrschichttemperatur T_j(t).

    T_j(t) = T_h(t) + P_tot * Z_th(t)

    Args:
        P_tot:  Gesamtverlustleistung [W] (konstant, aus Verlustmodell)
        T_h:    Kühlkörper-/Gehäusetemperatur [°C]. Skalar (konstanter Modus)
                oder Array gleicher Länge wie der Zeitvektor (dynamischer Modus).
        r:      Foster-Partialwiderstände r_i [K/W]
        tau:    Foster-Zeitkonstanten τ_i [s]
        t_end:  Simulationsendzeit [s]
        dt:     Zeitschritt [s]

    Returns:
        dict mit:
            t      – Zeitvektor [s]
            Zth    – Z_th(t) [K/W]
            Th     – T_h(t) [°C] (Array, auch im konstanten Modus)
            Tj     – T_j(t) [°C]
            Tj_inf – stationärer Endwert T_j,∞ [°C]
            Th_end – T_h am Ende der Simulation [°C]
            Rth_total – Σ r_i (Gesamt-Wärmewiderstand) [K/W]
    """
    t = np.arange(0.0, t_end + dt, dt)
    zth = foster_zth(t, r, tau)

    th_arr = np.asarray(T_h, dtype=float)
    if th_arr.ndim == 0:
        th_arr = np.full_like(t, float(T_h))

    tj = th_arr + P_tot * zth

    r_arr = np.asarray(r, dtype=float)
    rth_total = float(np.sum(r_arr))
    th_end = float(th_arr[-1])
    tj_inf = th_end + P_tot * rth_total

    return {
        "t":          t,
        "Zth":        zth,
        "Th":         th_arr,
        "Tj":         tj,
        "Tj_inf":     tj_inf,
        "Th_end":     th_end,
        "Rth_total":  rth_total,
    }

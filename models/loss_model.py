"""
Halbleiter-Verlustmodell – IGBT/MOSFET.

Implementiert:
  P_cond = V_CE,sat(T) * I_avg          (Leitverluste, linear interpoliert)
  P_sw   = (E_on + E_off) * f_sw        (Schaltverluste)
  P_tot  = P_cond + P_sw                (Gesamtverluste)

Hinweise / Modellgrenzen:
  - I_avg ist ein manueller Eingabeparameter (kein automatisches Lastmodell).
  - V_CE,sat wird linear zwischen drei Datenblattpunkten interpoliert.
  - Noch keine Duty-Cycle- oder Topologie-Modellierung.
  - I_avg kann später durch ein Lastmodell (z. B. aus Modulationsindex und Strom)
    ersetzt werden – dazu einfach `I_avg` in `compute_p_losses()` durch den
    berechneten Wert ersetzen.
"""

import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# Hilfsfunktion: V_CE,sat-Interpolation
# ---------------------------------------------------------------------------

def interpolate_vce_sat(
    T_j: float,
    vce_25: float,
    vce_T2: float,
    T2: float,
    vce_125: float,
    T1: float = 25.0,
    T3: float = 125.0,
) -> float:
    """
    Lineare Interpolation von V_CE,sat zwischen drei Temperaturstufen.

    Stützstellen: (T1=25°C, vce_25), (T2, vce_T2), (T3=125°C, vce_125)

    Args:
        T_j:      Aktuelle Sperrschichttemperatur [°C]
        vce_25:   V_CE,sat bei 25 °C [V]
        vce_T2:   V_CE,sat bei T2 [V]
        T2:       Mittlere Temperatur-Stützstelle [°C]
        vce_125:  V_CE,sat bei 125 °C [V]
        T1:       Untere Stützstelle (default 25 °C)
        T3:       Obere Stützstelle (default 125 °C)

    Returns:
        V_CE,sat bei T_j [V]
    """
    T_j = float(np.clip(T_j, T1, T3))
    if T_j <= T2:
        t = (T_j - T1) / max(T2 - T1, 1e-9)
        return vce_25 + t * (vce_T2 - vce_25)
    else:
        t = (T_j - T2) / max(T3 - T2, 1e-9)
        return vce_T2 + t * (vce_125 - vce_T2)


# ---------------------------------------------------------------------------
# Hauptfunktion: Verlustberechnung
# ---------------------------------------------------------------------------

def compute_p_losses(
    I_avg: float,
    vce_sat_25: float,
    vce_sat_T2: float,
    T2: float,
    vce_sat_125: float,
    E_on: float,
    E_off: float,
    f_sw: float,
    T_j: float = 25.0,
    T1: float = 25.0,
    T3: float = 125.0,
) -> dict:
    """
    Berechnet Leit-, Schalt- und Gesamtverluste.

    P_cond = V_CE,sat(T_j) * I_avg
    P_sw   = (E_on + E_off) * f_sw
    P_tot  = P_cond + P_sw

    Args:
        I_avg:        Mittlerer Kollektorstrom (manuelle Eingabe) [A]
        vce_sat_25:   V_CE,sat bei 25 °C [V]
        vce_sat_T2:   V_CE,sat bei mittlerer Temperatur T2 [V]
        T2:           Mittlere Temperatur-Stützstelle [°C]
        vce_sat_125:  V_CE,sat bei 125 °C [V]
        E_on:         Einschaltverlustenergie [J]
        E_off:        Ausschaltverlustenergie [J]
        f_sw:         Schaltfrequenz [Hz]
        T_j:          Sperrschichttemperatur für V_CE,sat-Interpolation [°C]
        T1:           Untere Temperatur-Stützstelle (default 25 °C)
        T3:           Obere Temperatur-Stützstelle (default 125 °C)

    Returns:
        dict mit: P_cond, P_sw, P_tot, V_CE_sat_eff
    """
    vce_eff = interpolate_vce_sat(
        T_j, vce_sat_25, vce_sat_T2, T2, vce_sat_125, T1, T3
    )
    p_cond = vce_eff * I_avg
    p_sw   = (E_on + E_off) * f_sw
    p_tot  = p_cond + p_sw

    return {
        "P_cond":       float(p_cond),
        "P_sw":         float(p_sw),
        "P_tot":        float(p_tot),
        "V_CE_sat_eff": float(vce_eff),
    }

"""
Halbleiter-Verlustmodell – IGBT/MOSFET.

Implementiert:
  P_cond = D * V_CE,sat(T_j) * I_avg           (Leitverluste, mit Duty-Cycle)
  P_sw   = (E_on(T_j) + E_off(T_j)) * f_sw     (Schaltverluste, T_j-abhängig)
  P_tot  = P_cond + P_sw                         (Gesamtverluste)

Hinweise / Modellgrenzen:
  - I_avg ist ein manueller Eingabeparameter (kein automatisches Lastmodell).
  - V_CE,sat und E_sw werden linear zwischen 25 °C und 125 °C interpoliert.
  - Duty-Cycle D = t_on / T_sw (0…1); beeinflusst P_cond.
  - E_on und E_off skalieren nicht mit D (Schaltenergie pro Puls, unabhängig).
  - I_avg kann später durch ein Lastmodell ersetzt werden.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Hilfsfunktion: lineare Interpolation zwischen 25 °C und 125 °C
# ---------------------------------------------------------------------------

def _interp_25_125(T_j: float, val_25: float, val_125: float) -> float:
    """Lineare Interpolation zwischen 25 °C und 125 °C."""
    T_j = float(np.clip(T_j, 25.0, 125.0))
    t = (T_j - 25.0) / 100.0          # 0 bei 25 °C, 1 bei 125 °C
    return val_25 + t * (val_125 - val_25)


def select_vce_sat(
    vce_sat_25: float,
    vce_sat_mid: float,
    vce_sat_125: float,
    mode: str = "interp",
    T_mid: float = 75.0,
    T_op: float = 75.0,
) -> float:
    """
    Wählt bzw. interpoliert den V_CE,sat-Wert aus drei Stützstellen
    (25 °C, Mitteltemperatur, 125 °C).

    Args:
        vce_sat_25:  V_CE,sat bei 25 °C [V]
        vce_sat_mid: V_CE,sat bei der Mitteltemperatur T_mid [V]
        vce_sat_125: V_CE,sat bei 125 °C [V]
        mode:        "25" | "mid" | "125" | "interp"
        T_mid:       Mitteltemperatur der mittleren Stützstelle [°C] (z. B. 75 °C)
        T_op:        Betriebstemperatur für die Interpolation [°C] (nur bei mode="interp")

    Returns:
        V_CE,sat [V]

    Hinweis:
        Bei mode="interp" wird stückweise linear über die drei Stützstellen
        (25 → T_mid → 125) interpoliert; ausserhalb wird auf die Randwerte begrenzt.
    """
    if mode == "25":
        return float(vce_sat_25)
    if mode == "mid":
        return float(vce_sat_mid)
    if mode == "125":
        return float(vce_sat_125)
    # mode == "interp": stückweise lineare Interpolation über 3 Stützstellen
    return float(np.interp(
        T_op,
        [25.0, float(T_mid), 125.0],
        [vce_sat_25, vce_sat_mid, vce_sat_125],
    ))


# ---------------------------------------------------------------------------
# Hauptfunktion: Verlustberechnung
# ---------------------------------------------------------------------------

def compute_p_losses(
    I_avg: float,
    vce_sat_25: float,
    vce_sat_125: float,
    E_on_25: float,
    E_on_125: float,
    E_off_25: float,
    E_off_125: float,
    f_sw: float,
    duty_cycle: float = 0.5,
    T_j: float = 25.0,
    vce_sat_mid: float = None,
    vce_mode: str = "interp",
    T_mid: float = 75.0,
    T_op: float = None,
) -> dict:
    """
    Berechnet Leit-, Schalt- und Gesamtverluste.

    P_cond = D * V_CE,sat(T_j) * I_avg
    P_sw   = (E_on(T_j) + E_off(T_j)) * f_sw
    P_tot  = P_cond + P_sw

    Args:
        I_avg:        Mittlerer Kollektorstrom (manuelle Eingabe) [A]
        vce_sat_25:   V_CE,sat bei 25 °C [V]
        vce_sat_125:  V_CE,sat bei 125 °C [V]
        E_on_25:      Einschaltverlustenergie bei 25 °C [J]
        E_on_125:     Einschaltverlustenergie bei 125 °C [J]
        E_off_25:     Ausschaltverlustenergie bei 25 °C [J]
        E_off_125:    Ausschaltverlustenergie bei 125 °C [J]
        f_sw:         Schaltfrequenz [Hz]
        duty_cycle:   Einschaltverhältnis D = t_on / T_sw [-] (0…1)
        T_j:          Sperrschichttemperatur für Interpolation [°C]

    Returns:
        dict mit: P_cond, P_sw, P_tot, V_CE_sat_eff, E_on_eff, E_off_eff
    """
    if vce_sat_mid is None:
        vce_sat_mid = 0.5 * (vce_sat_25 + vce_sat_125)
    if T_op is None:
        T_op = T_j
    vce_eff = select_vce_sat(
        vce_sat_25, vce_sat_mid, vce_sat_125,
        mode=vce_mode, T_mid=T_mid, T_op=T_op,
    )
    e_on_eff = _interp_25_125(T_j, E_on_25,    E_on_125)
    e_off_eff = _interp_25_125(T_j, E_off_25,  E_off_125)

    p_cond = duty_cycle * vce_eff * I_avg
    p_sw   = (e_on_eff + e_off_eff) * f_sw
    p_tot  = p_cond + p_sw

    return {
        "P_cond":        float(p_cond),
        "P_sw":          float(p_sw),
        "P_tot":         float(p_tot),
        "V_CE_sat_eff":  float(vce_eff),
        "E_on_eff":      float(e_on_eff),
        "E_off_eff":     float(e_off_eff),
    }

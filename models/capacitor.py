"""
Kondensator-Lebensdauermodelle.

Implementierte Modelle:
  1. Würth-Modell (Elektrolyt & Hybrid-Polymer): Arrheniusbasis mit Basis-2-Exponent
     - Elektrolyt/Hybrid: Lx = L_nom * 2^((T_max - T_A) / 10)
     - Polymer (THT/V-Chip): Lx = L_nom * 10^((T_max - T_A) / 20)

  2. Chemicon-Modell (Arrhenius, allgemein):
     - Lx = L0 * exp[(Ea / k_B) * (1/T_op_K - 1/T_ref_K)]
     - Jede 10 K Temperaturerhöhung halbiert näherungsweise die Lebensdauer.

Literatur:
  - Würth Elektronik: Application Note „Lifetime Calculation for Aluminum Electrolytic Capacitors"
  - Nichicon/Chemicon: General Specification for Aluminum Electrolytic Capacitors
  - IEC 62830
"""

import numpy as np

# Boltzmann-Konstante in eV/K (Naturkonstante – nicht ändern)
k_B: float = 8.617333e-5


# ---------------------------------------------------------------------------
# Würth-Modell
# ---------------------------------------------------------------------------

def wuerth_electrolyt(L_nom: float, T_max: float, T_A: float) -> float:
    """
    Würth-Modell für Aluminium-Elektrolyt- und Hybrid-Polymer-Kondensatoren (SMD H-Chip).

    Formel: Lx = L_nom * 2^((T_max - T_A) / 10)

    Args:
        L_nom: Nennlebensdauer aus Datenblatt [h]
        T_max: Maximal erlaubte Bauteiltemperatur aus Datenblatt [°C]
        T_A:   Tatsächliche Umgebungstemperatur im Gerät [°C]

    Returns:
        Lx: Erwartete Lebensdauer unter Betriebsbedingungen [h]

    Hinweis:
        Anwendbar auf: Alum. Elektrolyt, Alum. Hybrid-Polymer, Alum. Polymer (SMD H-Chip).
        Jede 10 K Temperaturdifferenz (T_max - T_A) verdoppelt die Lebensdauer.
    """
    return L_nom * (2.0 ** ((T_max - T_A) / 10.0))


def wuerth_polymer_tht(L_nom: float, T_max: float, T_A: float) -> float:
    """
    Würth-Modell für Aluminium-Polymer-Kondensatoren (THT und V-Chip SMD).

    Formel: Lx = L_nom * 10^((T_max - T_A) / 20)

    Args:
        L_nom: Nennlebensdauer aus Datenblatt [h]
        T_max: Maximal erlaubte Bauteiltemperatur aus Datenblatt [°C]
        T_A:   Tatsächliche Umgebungstemperatur im Gerät [°C]

    Returns:
        Lx: Erwartete Lebensdauer unter Betriebsbedingungen [h]

    Hinweis:
        Anwendbar auf: Alum. Polymer Kondensatoren (THT und V-Chip SMD).
        Basis 10 statt 2 — flachere Temperaturabhängigkeit als Elektrolyt-Typ.
    """
    return L_nom * (10.0 ** ((T_max - T_A) / 20.0))


# ---------------------------------------------------------------------------
# Chemicon / allgemeines Arrhenius-Modell
# ---------------------------------------------------------------------------

def arrhenius_capacitor(L0: float, T_ref: float, T_op: float, Ea: float = 0.94) -> float:
    """
    Allgemeines Arrhenius-Modell für Kondensatoren (Chemicon-Methodik).

    Formel: Lx = L0 * exp[(Ea / k_B) * (1/T_op_K - 1/T_ref_K)]

    Args:
        L0:    Nennlebensdauer bei Referenztemperatur T_ref [h]
        T_ref: Referenztemperatur aus Datenblatt [°C] (z. B. 105 °C)
        T_op:  Tatsächliche Betriebstemperatur des Kondensators [°C]
               (Umgebung + Eigenerwärmung durch Ripple-Strom)
        Ea:    Aktivierungsenergie [eV] (typisch 0,6–1,0 eV für Elektrolyt-C,
               default 0,94 eV nach Chemicon)

    Returns:
        Lx: Erwartete Lebensdauer [h]

    Hinweis:
        Faustformel: Jede 10 K Temperaturerhöhung halbiert näherungsweise die
        Lebensdauer (gilt bei Ea ≈ 0,7–0,9 eV im mittleren Temperaturbereich).
    """
    T_op_K = T_op + 273.15
    T_ref_K = T_ref + 273.15
    return L0 * np.exp((Ea / k_B) * (1.0 / T_op_K - 1.0 / T_ref_K))


def ripple_temperature_rise(I_ripple: float, ESR: float) -> float:
    """
    Eigenerwärmung des Kondensators durch Ripple-Strom.

    Formel: ΔT = ESR * I_ripple²   (Näherung ohne Wärmewiderstand)

    Args:
        I_ripple: Betriebsmäßiger Ripple-Strom [A]
        ESR:      Äquivalenter Serienwiderstand bei Betriebsfrequenz [Ω]

    Returns:
        ΔT: Eigenerwärmung [K] (addieren zu Umgebungstemperatur → T_op)

    Hinweis:
        Genaue Rechnung: ΔT = R_th * ESR * I_ripple².
        Ohne Wärmewiderstand R_th ist dies eine vereinfachte Schätzung.
    """
    return ESR * (I_ripple ** 2)

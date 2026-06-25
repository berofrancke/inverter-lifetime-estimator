"""
Kondensator-Lebensdauermodelle.

Implementierte Modelle:
  Würth-Modell (Elektrolyt & Hybrid-Polymer): Arrheniusbasis mit Basis-2-Exponent
     - Elektrolyt/Hybrid: Lx = L_nom * 2^((T_max - T_A) / 10) * k
     - Polymer (THT/V-Chip): Lx = L_nom * 10^((T_max - T_A) / 20) * k

  Der Faktor k berücksichtigt das Ripple-Derating: Liegt der tatsächliche
  Ripple-Strom unter dem Nennripple, verlängert sich die Lebensdauer (k > 1).

Literatur:
  - Würth Elektronik: Application Note „Lifetime Calculation for Aluminum Electrolytic Capacitors"
  - IEC 62830
"""


# ---------------------------------------------------------------------------
# Würth-Modell
# ---------------------------------------------------------------------------

def wuerth_electrolyt(L_nom: float, T_max: float, T_A: float, k: float = 1.0) -> float:
    """
    Würth-Modell für Aluminium-Elektrolyt- und Hybrid-Polymer-Kondensatoren (SMD H-Chip).

    Formel: Lx = L_nom * 2^((T_max - T_A) / 10) * k

    Args:
        L_nom: Nennlebensdauer aus Datenblatt [h]
        T_max: Maximal erlaubte Bauteiltemperatur aus Datenblatt [°C]
        T_A:   Tatsächliche Umgebungstemperatur im Gerät [°C]
        k:     Lebensdauerfaktor (Ripple-Derating); k > 1 verlängert die
               Lebensdauer, wenn der Betriebs-Ripple < Nennripple ist.

    Returns:
        Lx: Erwartete Lebensdauer unter Betriebsbedingungen [h]

    Hinweis:
        Anwendbar auf: Alum. Elektrolyt, Alum. Hybrid-Polymer, Alum. Polymer (SMD H-Chip).
        Jede 10 K Temperaturdifferenz (T_max - T_A) verdoppelt die Lebensdauer.
    """
    return L_nom * (2.0 ** ((T_max - T_A) / 10.0)) * k


def wuerth_polymer_tht(L_nom: float, T_max: float, T_A: float, k: float = 1.0) -> float:
    """
    Würth-Modell für Aluminium-Polymer-Kondensatoren (THT und V-Chip SMD).

    Formel: Lx = L_nom * 10^((T_max - T_A) / 20) * k

    Args:
        L_nom: Nennlebensdauer aus Datenblatt [h]
        T_max: Maximal erlaubte Bauteiltemperatur aus Datenblatt [°C]
        T_A:   Tatsächliche Umgebungstemperatur im Gerät [°C]
        k:     Lebensdauerfaktor (Ripple-Derating); k > 1 verlängert die
               Lebensdauer, wenn der Betriebs-Ripple < Nennripple ist.

    Returns:
        Lx: Erwartete Lebensdauer unter Betriebsbedingungen [h]

    Hinweis:
        Anwendbar auf: Alum. Polymer Kondensatoren (THT und V-Chip SMD).
        Basis 10 statt 2 — flachere Temperaturabhängigkeit als Elektrolyt-Typ.
    """
    return L_nom * (10.0 ** ((T_max - T_A) / 20.0)) * k


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

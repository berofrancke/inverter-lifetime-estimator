"""
Transistor-Lebensdauermodelle – Power Cycling (IGBT/MOSFET).

Implementierte Modelle:
  1. Norris-Landzberg-Modell (Gl. 4.1)
     N_f = α · f^α2 · ΔT_j^n1 · exp(Ea / (k_B · T_jm))

  2. Bayerer-Modell / CIPS-08 (Gl. 4.2)
     N_f = K · ΔT_j^β1 · exp(β2 / (T_jm + 273)) · I_on^β3 · V^β4 · D^β5
     (EasyPACK-Parameter für Infineon-IGBT mitgeliefert)

  3. SKiM63-Modell / erweitertes Modell (Gl. 4.3)
     N_f = A · (ΔT_j)^α · exp(Ea/(T_jm·k_B)) · exp(β1·ΔT_j + β0)
           · ((C + I_on^λ)/(C+1)) · f_diode

Alle Gleichungen aus: Held et al. CIPS 2008 / Scheuermann & Hecht 2011.

Miner-Regel:
  D_total = Σ (n_i / N_f,i)  → Versagen bei D ≥ 1,0
"""

import numpy as np
from typing import Optional

# Boltzmann-Konstante in eV/K (Naturkonstante – nicht ändern)
k_B: float = 8.617333e-5


# ---------------------------------------------------------------------------
# Modell 1: Norris-Landzberg
# ---------------------------------------------------------------------------

def norris_landzberg(
    delta_T_j: float,
    T_jm: float,
    alpha: float,
    alpha2: float,
    n1: float,
    Ea: float,
    f: float = 1.0,
) -> float:
    """
    Norris-Landzberg-Modell (Gl. 4.1).

    N_f = α · f^α2 · ΔT_j^n1 · exp(Ea / (k_B · T_jm_K))

    Args:
        delta_T_j: Temperaturhub pro Lastzyklus [K]
        T_jm:      Mittlere Sperrschichttemperatur pro Zyklus [°C]
        alpha:     Modell-Vorfaktor (komponentenspezifisch aus LESIT/CIPS)
        alpha2:    Frequenz-Exponent (aus LESIT-Fitting)
        n1:        Coffin-Manson-Exponent auf ΔT_j (typisch negativ, z. B. -1,9)
        Ea:        Aktivierungsenergie [eV] (typisch 0,3–0,7 eV für Al-Bonddrähte)
        f:         Zyklusfrequenz [Hz] (default 1 Hz wenn nicht bekannt)

    Returns:
        N_f: Berechnete Zyklenzahl bis Versagen [-]

    Literatur:
        Norris & Landzberg (1969); weiterentwickelt in: Held et al., CIPS 2008.
    """
    if delta_T_j <= 0:
        return float("inf")
    T_jm_K = T_jm + 273.15
    N_f = (alpha
           * (f ** alpha2)
           * (delta_T_j ** n1)
           * np.exp(Ea / (k_B * T_jm_K)))
    return float(N_f)


# ---------------------------------------------------------------------------
# Modell 2: Bayerer (CIPS-08)
# ---------------------------------------------------------------------------

# Standardparameter für Infineon EasyPACK-IGBT-Modul (aus Held et al. 2008)
CIPS08_INFINEON_DEFAULT = {
    "K":    2.03e14,   # Skalierungs-/Technologiefaktor [-]
    "beta1": 4.416,    # Coffin-Manson-Exponent auf ΔT_j [-]
    "beta2": 1285.0,   # Arrhenius-Faktor (Einheit: K, also β2/T_K)
    "beta3": -0.463,   # Strom-Exponent auf I_on [-]
    "beta4": 0.716,    # Spannungsexponent auf V [-]
    "beta5": -0.761,   # Bondraht-Exponent auf D [-]
    # beta6 (Duty-Cycle) = 0.5 beim Originalfit
}


def bayerer_cips08(
    delta_T_j: float,
    T_jm: float,
    I_on: float,
    V: float,
    D: float,
    K: float = CIPS08_INFINEON_DEFAULT["K"],
    beta1: float = CIPS08_INFINEON_DEFAULT["beta1"],
    beta2: float = CIPS08_INFINEON_DEFAULT["beta2"],
    beta3: float = CIPS08_INFINEON_DEFAULT["beta3"],
    beta4: float = CIPS08_INFINEON_DEFAULT["beta4"],
    beta5: float = CIPS08_INFINEON_DEFAULT["beta5"],
) -> float:
    """
    Bayerer-Modell / CIPS-08 (Gl. 4.2).

    N_f = K · ΔT_j^β1 · exp(β2 / (T_jm + 273)) · I_on^β3 · V^β4 · D^β5

    Args:
        delta_T_j: Temperaturhub pro Lastzyklus [K]
        T_jm:      Mittlere Sperrschichttemperatur [°C]
        I_on:      Kollektorstrom während des Pulses [A]
        V:         Bonddraht-Spannung / Blockierspannung [V]
        D:         Bonddraht-Durchmesser [µm]
        K:         Skalierungsfaktor (technologieabhängig, default Infineon EasyPACK)
        beta1–5:   Modellkoeffizienten (default Infineon EasyPACK aus Held 2008)

    Returns:
        N_f: Zyklenzahl bis Versagen [-]

    Hinweis:
        Die Default-Parameter gelten für das Infineon EasyPACK IGBT-Modul.
        Für andere Module Koeffizienten aus Hersteller-Datenblatt oder
        LESIT/CIPS-Fitting verwenden.

    Literatur:
        Bayerer et al., CIPS 2008, Paper 3.1.
    """
    if delta_T_j <= 0:
        return float("inf")
    T_jm_K = T_jm + 273.15
    N_f = (K
           * (delta_T_j ** beta1)
           * np.exp(beta2 / T_jm_K)
           * (I_on ** beta3)
           * (V ** beta4)
           * (D ** beta5))
    return float(N_f)


# ---------------------------------------------------------------------------
# Modell 3: SKiM63 / erweitertes Modell (Gl. 4.3)
# ---------------------------------------------------------------------------

def skim63_model(
    delta_T_j: float,
    T_jm: float,
    A: float,
    alpha: float,
    Ea: float,
    beta1: float,
    beta0: float,
    C: float,
    I_on: float,
    lam: float,
    f_diode: float = 1.0,
) -> float:
    """
    Erweitertes Modell für SKiM63-Module (Gl. 4.3) –
    berücksichtigt Bonddrahtspannung, Strom und Diodenkorrekturfaktor.

    N_f = A · (ΔT_j)^α · exp(Ea/(T_jm·k_B)) · exp(β1·ΔT_j + β0)
          · ((C + I_on^λ) / (C + 1)) · f_diode

    Args:
        delta_T_j: Temperaturhub pro Lastzyklus [K]
        T_jm:      Mittlere Sperrschichttemperatur [°C]
        A:         Modell-Vorfaktor (aus LESIT/CIPS-Fitting)
        alpha:     Exponent auf ΔT_j (Coffin-Manson-Anteil)
        Ea:        Aktivierungsenergie [eV]
        beta1:     Linearer Koeffizient auf ΔT_j im Exponentialterm
        beta0:     Offset-Koeffizient im Exponentialterm
        C:         Stromnormierungskonstante (Erweiterung SKiM63)
        I_on:      Kollektorstrom [A]
        lam:       Strom-Exponent λ
        f_diode:   Dioden-Korrekturfaktor (1.0 wenn keine Freilaufdiode, sonst < 1)

    Returns:
        N_f: Zyklenzahl bis Versagen [-]

    Literatur:
        Scheuermann & Hecht, PCIM 2011.
        Gleichung 4.3 aus der bereitgestellten Methodik-Dokumentation.
    """
    if delta_T_j <= 0:
        return float("inf")
    T_jm_K = T_jm + 273.15
    N_f = (A
           * (delta_T_j ** alpha)
           * np.exp(Ea / (T_jm_K * k_B))
           * np.exp(beta1 * delta_T_j + beta0)
           * ((C + I_on ** lam) / (C + 1))
           * f_diode)
    return float(N_f)


# ---------------------------------------------------------------------------
# Arrhenius-Modell für Gate-Oxid-Degradation (SiC-MOSFETs)
# ---------------------------------------------------------------------------

def arrhenius_gate_oxide(
    L0_gate: float,
    T_ref: float,
    T_j_op: float,
    Ea_gate: float = 1.0,
) -> float:
    """
    Arrhenius-basierte Lebensdauerabschätzung für Gate-Oxid-Degradation.

    Relevant insbesondere für SiC-MOSFETs (TDDB, Threshold-Voltage-Shift).

    Formel: L_gate = L0_gate · exp[(Ea_gate/k_B) · (1/T_j,op_K - 1/T_ref_K)]

    Args:
        L0_gate:  Lebensdauer bei Referenztemperatur [h]
        T_ref:    Referenztemperatur [°C] (typisch 150 °C für SiC-MOSFETs)
        T_j_op:   Mittlere Sperrschichttemperatur im Betrieb [°C]
        Ea_gate:  Aktivierungsenergie [eV] (typisch 0,8–1,2 eV für SiC Gate-Oxid)

    Returns:
        L_gate: Geschätzte Gate-Oxid-Lebensdauer [h]

    Literatur:
        IEC 60749 / SiC-MOSFET Reliability Roadmap (PowerAmerica 2021).
    """
    T_j_K = T_j_op + 273.15
    T_ref_K = T_ref + 273.15
    return L0_gate * np.exp((Ea_gate / k_B) * (1.0 / T_j_K - 1.0 / T_ref_K))


# ---------------------------------------------------------------------------
# Hilfsfunktion: Miner-Regel für Transistoren
# ---------------------------------------------------------------------------

def miner_damage(n_i: np.ndarray, N_f_i: np.ndarray) -> float:
    """
    Lineare Schadensakkumulation nach Palmgren-Miner.

    D_total = Σ (n_i / N_f_i)

    Versagenskriterium: D_total ≥ 1,0
    Bewertungsskala (aus Excel-Methodik):
        D < 0,5   → unkritisch
        0,5–0,8   → Überwachung empfohlen
        0,8–1,0   → kritisch
        D ≥ 1,0   → Lebensdauerende erreicht

    Args:
        n_i:   Array mit tatsächlichen Zyklenzahlen je Klasse [-]
        N_f_i: Array mit Ausfallzyklenzahlen je Klasse (aus CM-Modell) [-]

    Returns:
        D_total: Gesamtschadenssumme [-]
    """
    n_i = np.asarray(n_i, dtype=float)
    N_f_i = np.asarray(N_f_i, dtype=float)

    # Divisionen durch 0 oder inf vermeiden
    with np.errstate(divide="ignore", invalid="ignore"):
        d_i = np.where(N_f_i > 0, n_i / N_f_i, 0.0)

    return float(np.sum(d_i))

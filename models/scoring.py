"""
Punktebewertungssystem für nicht-quantifizierbare Parameter.

Jede Komponente des Wechselrichters wird über drei Bewertungsarten erfasst:
  1. PUNKTEBEWERTUNG (0–5 je Parameter) – qualitative Merkmale
  2. BINÄR (0 oder 1) – Verfügbarkeit / Vorhanden-Prüfung
  3. DERATING-RATIO – Verhältnis Betrieb zu Grenzwert

Ampel-Schwellen (normiert auf 0–100 %):
  ≥ 80 %  → GUT     (grün)
  ≥ 50 %  → MITTEL  (gelb)
  < 50 %  → KRITISCH (rot)

Maximale Punkte je Komponente (aus Excel-Methodik):
  Kondensator : 35
  Spule       : 25
  Relais      : 15
  Leiterplatte: 55
  System      : 30
  GESAMT      : 160
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

# Maximale Punktzahl je Komponente (aus Gesamtbewertungs-Sheet)
MAX_SCORES: Dict[str, int] = {
    "Kondensator": 35,
    "Spule":       25,
    "Relais":      15,
    "Leiterplatte": 55,
    "System":      30,
}
TOTAL_MAX: int = sum(MAX_SCORES.values())  # 160


@dataclass
class ComponentScore:
    """Sammelt Punkte einer Komponente und berechnet normierten Score."""

    name: str
    max_points: int
    points: Dict[str, float] = field(default_factory=dict)

    def add(self, label: str, value: float) -> None:
        """Fügt einen Einzelpunkt hinzu."""
        self.points[label] = float(value)

    @property
    def total(self) -> float:
        """Summe aller Einzelpunkte."""
        return sum(self.points.values())

    @property
    def normalized(self) -> float:
        """Normierter Score 0–1 (0 % bis 100 %)."""
        if self.max_points <= 0:
            return 0.0
        return min(1.0, self.total / self.max_points)

    @property
    def status(self) -> Tuple[str, str]:
        """
        Gibt (Ampelfarbe, Statustext) zurück.
        Farben als Plotly-CSS-Farbnamen.
        """
        n = self.normalized
        if n >= 0.80:
            return "green", "GUT (≥ 80 %)"
        elif n >= 0.50:
            return "orange", "MITTEL (50–79 %)"
        else:
            return "red", "KRITISCH (< 50 %)"


# ---------------------------------------------------------------------------
# Kondensator-Scores
# ---------------------------------------------------------------------------

def score_kapazitaetstoleranz(tol_pct: float) -> float:
    """
    ≤ ±10 % → 5 | ≤ ±20 % → 3 | > ±20 % → 1
    """
    if tol_pct <= 10:
        return 5.0
    elif tol_pct <= 20:
        return 3.0
    else:
        return 1.0


# ---------------------------------------------------------------------------
# PCB-Scores
# ---------------------------------------------------------------------------

def score_pcb_material(material: str) -> float:
    """
    Polyimid → 5 | FR4-High-Tg → 4 | FR4-Standard → 2 | CEM → 1
    """
    mapping = {"Polyimid (Rogers)": 5, "FR4-High-Tg (≥170°C)": 4, "FR4-Standard": 2, "CEM": 1}
    return float(mapping.get(material, 1))


def score_coating_typ(typ: str) -> float:
    """
    Parylene → 5 | Silikon → 4 | Urethan → 3 | Acryl → 2 | kein → 0
    """
    mapping = {"Parylene": 5, "Silikon": 4, "Urethan": 3, "Acryl": 2, "kein": 0}
    return float(mapping.get(typ, 0))


# ---------------------------------------------------------------------------
# System-Scores
# ---------------------------------------------------------------------------

def score_wirkungsgrad_25(eta_pct: float) -> float:
    """
    ≥ 98 % → 5 | 96–97,9 % → 4 | 94–95,9 % → 2 | < 94 % → 0
    """
    if eta_pct >= 98:
        return 5.0
    elif eta_pct >= 96:
        return 4.0
    elif eta_pct >= 94:
        return 2.0
    else:
        return 0.0


def score_kühlung(typ: str) -> float:
    """
    Flüssigkühlung → 5 | Forcierte Luft → 3 | Naturkonvektion → 1
    """
    mapping = {"Flüssigkühlung": 5, "Forcierte Luft (Lüfter)": 3, "Naturkonvektion": 1}
    return float(mapping.get(typ, 1))


# ---------------------------------------------------------------------------
# Derating-Bewertung (Verhältnis Betrieb / Grenzwert)
# ---------------------------------------------------------------------------

def derating_status(ratio: float, target: float = 0.8) -> Tuple[str, str]:
    """
    Bewertet einen Derating-Faktor (Betrieb/Grenzwert) gegenüber Zielwert.

    Args:
        ratio:  Verhältnis Betrieb/Grenzwert (z. B. 0,75)
        target: Maximaler empfohlener Wert (default 0,8 = 80 %)

    Returns:
        (Farbe, Beschreibung) als Tuple
    """
    if ratio <= target * 0.9:
        return "green", f"OK ({ratio*100:.1f} % ≤ {target*100:.0f} % Ziel)"
    elif ratio <= target:
        return "orange", f"Grenzwertig ({ratio*100:.1f} % ≈ {target*100:.0f} % Ziel)"
    else:
        return "red", f"Überschritten ({ratio*100:.1f} % > {target*100:.0f} % Ziel)"

"""Vereinfachter Rainflow-Zählalgorithmus für Temperaturprofile."""

import numpy as np
from typing import List, Tuple


def find_turning_points(signal: np.ndarray) -> np.ndarray:
    """Extrahiert lokale Extrema (Umkehrpunkte) aus dem Signal."""
    n = len(signal)
    if n < 3:
        return signal
    diff = np.diff(signal)
    turning = [0]
    for i in range(1, n - 1):
        if (diff[i - 1] > 0 and diff[i] <= 0) or (diff[i - 1] < 0 and diff[i] >= 0):
            turning.append(i)
    turning.append(n - 1)
    return signal[turning]


def rainflow_count(signal: np.ndarray, decimals: int = 1) -> List[Tuple[float, float, float]]:
    """
    Rainflow-Zählung nach ASTM E1049.

    Args:
        signal: Zeitsignal (z.B. Temperaturverlauf)
        decimals: Rundung für delta_T-Gruppierung

    Returns:
        Liste von (delta_T, T_mean, count) Tupeln
    """
    tp = find_turning_points(np.asarray(signal, dtype=float))

    residue = []
    cycles = []

    for point in tp:
        residue.append(point)
        while len(residue) >= 4:
            R0 = abs(residue[-4] - residue[-3])
            R1 = abs(residue[-3] - residue[-2])
            R2 = abs(residue[-2] - residue[-1])
            if R1 <= R0 and R1 <= R2:
                delta_T = R1
                T_mean = (residue[-3] + residue[-2]) / 2.0
                cycles.append((delta_T, T_mean))
                residue.pop(-2)
                residue.pop(-2)
            else:
                break

    # Restzyklen als halbe Zyklen zählen
    for i in range(len(residue) - 1):
        delta_T = abs(residue[i + 1] - residue[i])
        T_mean = (residue[i] + residue[i + 1]) / 2.0
        cycles.append((delta_T, T_mean))

    # Gruppieren
    if not cycles:
        return []

    cycle_array = np.array(cycles)
    delta_T_rounded = np.round(cycle_array[:, 0], decimals)
    T_mean_rounded = np.round(cycle_array[:, 1], decimals)

    result = {}
    for dt, tm in zip(delta_T_rounded, T_mean_rounded):
        key = (dt, tm)
        result[key] = result.get(key, 0) + 1.0

    return [(k[0], k[1], v) for k, v in sorted(result.items(), key=lambda x: -x[0])]

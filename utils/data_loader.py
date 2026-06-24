"""Hilfsfunktionen zum Laden und Generieren von Temperaturprofilen."""

import numpy as np
import pandas as pd
import io


def load_temperature_profile(file_obj) -> pd.DataFrame:
    """
    Lädt ein Temperaturprofil aus einer CSV-Datei.

    Erwartetes Format:
        Zeit_h, Temperatur_C
        0.0,    25.0
        0.1,    27.3
        ...
    """
    df = pd.read_csv(file_obj)
    df.columns = [c.strip() for c in df.columns]

    # Automatische Spaltenzuordnung
    if "Zeit_h" not in df.columns or "Temperatur_C" not in df.columns:
        col_t = [c for c in df.columns if "zeit" in c.lower() or "time" in c.lower()]
        col_temp = [c for c in df.columns if "temp" in c.lower() or "°" in c.lower()]
        if col_t and col_temp:
            df = df.rename(columns={col_t[0]: "Zeit_h", col_temp[0]: "Temperatur_C"})
        else:
            # Erste zwei Spalten verwenden
            df.columns = ["Zeit_h", "Temperatur_C"] + list(df.columns[2:])

    return df[["Zeit_h", "Temperatur_C"]].dropna()


def generate_example_profile(
    duration_h: float = 24.0,
    T_min: float = 25.0,
    T_max: float = 85.0,
    n_cycles: int = 3,
    noise_std: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generiert ein synthetisches Temperaturprofil (typisch für Wechselrichter-Tagesprofil).

    Args:
        duration_h: Gesamtdauer [h]
        T_min: Minimale Temperatur [°C]
        T_max: Maximale Temperatur [°C]
        n_cycles: Anzahl Tageszyklus-Perioden
        noise_std: Standardabweichung des Rauschens [K]
        seed: Zufalls-Seed für Reproduzierbarkeit
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_h, int(duration_h * 60))  # 1-Minuten-Auflösung
    T_mean = (T_min + T_max) / 2
    T_amp = (T_max - T_min) / 2

    # Basis-Tagesprofil
    temp = T_mean + T_amp * np.sin(2 * np.pi * n_cycles * t / duration_h - np.pi / 2)

    # Überlagerte hochfrequente Last-Schwankungen
    temp += 0.3 * T_amp * np.sin(2 * np.pi * 12 * n_cycles * t / duration_h)

    # Rauschen
    temp += rng.normal(0, noise_std, len(t))
    temp = np.clip(temp, T_min - 5, T_max + 5)

    return pd.DataFrame({"Zeit_h": t, "Temperatur_C": temp})


def generate_example_csv() -> str:
    """Gibt ein Beispiel-CSV als String zurück."""
    df = generate_example_profile(duration_h=8.0, T_min=30.0, T_max=75.0, n_cycles=1)
    return df.to_csv(index=False)

# Wechselrichter-Lebensdauerabschätzung

Python-Tool zur Lebensdauerabschätzung von Wechselrichtern basierend auf thermischen Belastungsmodellen (Coffin-Manson, Arrhenius) mit interaktiver Streamlit-Visualisierung.

## Features
- Thermische Lebensdauermodelle (Arrhenius, Coffin-Manson)
- Rainflow-Zählalgorithmus für Lastwechselanalyse
- Miner-Regel zur Schadensakkumulation
- Interaktive Streamlit-Weboberfläche
- CSV-Import für Temperaturprofile

## Installation
```bash
pip install -r requirements.txt
```

## Starten
```bash
streamlit run app.py
```

## Projektstruktur
```
├── app.py               # Streamlit-Hauptapp
├── models/
│   ├── arrhenius.py     # Arrhenius-Lebensdauermodell
│   ├── coffin_manson.py # Coffin-Manson-Modell
│   └── miner.py         # Miner-Schadensakkumulation
├── utils/
│   ├── rainflow.py      # Rainflow-Zählalgorithmus
│   └── data_loader.py   # CSV/Datenlader
├── data/
│   └── example_profile.csv
└── requirements.txt
```

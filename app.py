"""Streamlit-App: Lebensdauerabschätzung für Wechselrichter"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

from models.arrhenius import ArrheniusModel
from models.coffin_manson import CoffinMansonModel
from models.miner import MinerRule
from utils.rainflow import rainflow_count
from utils.data_loader import load_temperature_profile, generate_example_profile

st.set_page_config(
    page_title="Wechselrichter Lebensdauer",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Wechselrichter-Lebensdauerabschätzung")
st.markdown("_Thermische Belastungsanalyse nach Arrhenius & Coffin-Manson_")

# --- Sidebar: Parameter ---
st.sidebar.header("🔧 Modellparameter")

st.sidebar.subheader("Arrhenius-Modell")
Ea = st.sidebar.number_input("Aktivierungsenergie Ea [eV]", value=0.7, min_value=0.1, max_value=2.0, step=0.05,
    help="Typisch 0.5–1.0 eV für Halbleiter")
T_ref = st.sidebar.number_input("Referenztemperatur T_ref [°C]", value=25, min_value=-20, max_value=100)
L_ref = st.sidebar.number_input("Referenzlebensdauer L_ref [Stunden]", value=100000, step=1000)

st.sidebar.subheader("Coffin-Manson-Modell")
cm_exp = st.sidebar.number_input("Coffin-Manson Exponent n", value=2.0, min_value=0.5, max_value=6.0, step=0.1,
    help="Typisch 1–3 für Leistungselektronik")
delta_T_ref = st.sidebar.number_input("ΔT_ref [K]", value=40.0, min_value=5.0, max_value=150.0, step=5.0)
N_ref = st.sidebar.number_input("N_ref Zyklen bei ΔT_ref", value=50000, step=1000)

st.sidebar.subheader("Betriebsprofil")
mission_years = st.sidebar.slider("Missionsdauer [Jahre]", 1, 30, 15)

# --- Hauptbereich: Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📂 Temperaturprofil", "📊 Lebensdaueranalyse", "🔍 Modellvergleich", "📋 Bericht"])

with tab1:
    st.header("Temperaturprofil")
    col1, col2 = st.columns([1, 2])

    with col1:
        data_source = st.radio("Datenquelle", ["Beispielprofil", "CSV hochladen", "Manuell eingeben"])

        if data_source == "CSV hochladen":
            uploaded = st.file_uploader("CSV-Datei (Zeit [h], Temperatur [°C])", type="csv")
            if uploaded:
                df = load_temperature_profile(uploaded)
            else:
                df = generate_example_profile()
                st.info("Kein Upload – Beispielprofil wird verwendet.")

        elif data_source == "Manuell eingeben":
            t_max = st.number_input("Profildauer [h]", value=24, min_value=1)
            t_min_t = st.number_input("Min. Temperatur [°C]", value=20)
            t_max_t = st.number_input("Max. Temperatur [°C]", value=80)
            noise = st.slider("Rauschen [K]", 0, 20, 5)
            t = np.linspace(0, t_max, t_max * 10)
            temp = t_min_t + (t_max_t - t_min_t) * (0.5 + 0.5 * np.sin(2 * np.pi * t / t_max))
            temp += np.random.normal(0, noise, len(t))
            df = pd.DataFrame({"Zeit_h": t, "Temperatur_C": temp})

        else:
            df = generate_example_profile()

        st.metric("Datenpunkte", len(df))
        st.metric("T_min", f"{df['Temperatur_C'].min():.1f} °C")
        st.metric("T_max", f"{df['Temperatur_C'].max():.1f} °C")
        st.metric("T_mean", f"{df['Temperatur_C'].mean():.1f} °C")

    with col2:
        fig = px.line(df, x="Zeit_h", y="Temperatur_C",
                      labels={"Zeit_h": "Zeit [h]", "Temperatur_C": "Temperatur [°C]"},
                      title="Temperaturverlauf")
        fig.update_traces(line_color="#e07b39")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        csv_dl = df.to_csv(index=False).encode()
        st.download_button("⬇️ Profil als CSV", csv_dl, "temperaturprofil.csv", "text/csv")

with tab2:
    st.header("Lebensdaueranalyse")

    # Arrhenius
    arr_model = ArrheniusModel(Ea=Ea, T_ref=T_ref, L_ref=L_ref)
    T_values = np.linspace(20, 120, 200)
    lifetimes = arr_model.lifetime(T_values)

    # Rainflow
    cycles = rainflow_count(df["Temperatur_C"].values)
    df_cycles = pd.DataFrame(cycles, columns=["delta_T", "T_mean", "count"])

    # Coffin-Manson
    cm_model = CoffinMansonModel(n=cm_exp, delta_T_ref=delta_T_ref, N_ref=N_ref)
    df_cycles["N_f"] = df_cycles["delta_T"].apply(lambda dt: cm_model.cycles_to_failure(dt) if dt > 0 else np.inf)
    df_cycles["damage"] = df_cycles["count"] / df_cycles["N_f"]

    # Miner
    miner = MinerRule()
    total_damage_per_profile = miner.total_damage(df_cycles["damage"].values)
    profile_duration_h = df["Zeit_h"].max()
    profiles_per_year = 8760 / profile_duration_h
    annual_damage = total_damage_per_profile * profiles_per_year
    estimated_life_years = 1.0 / annual_damage if annual_damage > 0 else float("inf")

    # Arrhenius Lebensdauer für mittlere Temperatur
    T_mean = df["Temperatur_C"].mean()
    arr_life_h = arr_model.lifetime(T_mean)
    arr_life_years = arr_life_h / 8760

    col1, col2, col3 = st.columns(3)
    col1.metric("🔥 Geschätzte Lebensdauer (Coffin-Manson + Miner)",
                f"{estimated_life_years:.1f} Jahre" if estimated_life_years < 1000 else "> 1000 Jahre")
    col2.metric("🌡️ Geschätzte Lebensdauer (Arrhenius @ T_mean)",
                f"{arr_life_years:.1f} Jahre")
    col3.metric("⚠️ Jährliche Schädigung (Miner)", f"{annual_damage*100:.3f} %/Jahr")

    # Plots
    col_l, col_r = st.columns(2)
    with col_l:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=T_values, y=lifetimes / 8760,
                                  mode="lines", name="Lebensdauer [Jahre]",
                                  line=dict(color="#01696f", width=2)))
        fig2.add_vline(x=T_mean, line_dash="dash", line_color="red",
                       annotation_text=f"T_mean={T_mean:.1f}°C")
        fig2.update_layout(title="Arrhenius-Lebensdauer vs. Temperatur",
                           xaxis_title="Temperatur [°C]", yaxis_title="Lebensdauer [Jahre]",
                           template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        if len(df_cycles) > 0 and df_cycles["delta_T"].max() > 0:
            fig3 = px.bar(df_cycles[df_cycles["delta_T"] > 1].head(20),
                          x="delta_T", y="count", color="damage",
                          color_continuous_scale="RdYlGn_r",
                          labels={"delta_T": "ΔT [K]", "count": "Anzahl Zyklen", "damage": "Schädigung"},
                          title="Rainflow-Zählung & Schädigung")
            fig3.update_layout(template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)

    # Schadensakkumulation über Zeit
    years = np.arange(0, mission_years + 1)
    cumulative_damage = years * annual_damage
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=years, y=cumulative_damage, mode="lines+markers",
                              fill="tozeroy", line=dict(color="#a12c7b")))
    fig4.add_hline(y=1.0, line_dash="dash", line_color="red",
                   annotation_text="Versagen (D=1)")
    fig4.update_layout(title="Schadensakkumulation (Miner-Regel)",
                       xaxis_title="Zeit [Jahre]", yaxis_title="Kumulativer Schaden D",
                       template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.header("Modellvergleich")
    st.markdown("Sensitivitätsanalyse: Lebensdauer bei variierenden Parametern")

    T_range = np.linspace(30, 100, 50)
    fig5 = make_subplots(rows=1, cols=2,
                          subplot_titles=("Ea-Sensitivität (Arrhenius)", "n-Sensitivität (Coffin-Manson)"))

    for ea_val in [0.5, 0.7, 0.9, 1.1]:
        m = ArrheniusModel(Ea=ea_val, T_ref=T_ref, L_ref=L_ref)
        fig5.add_trace(go.Scatter(x=T_range, y=m.lifetime(T_range) / 8760,
                                   mode="lines", name=f"Ea={ea_val} eV"), row=1, col=1)

    dt_range = np.linspace(5, 100, 50)
    for n_val in [1.5, 2.0, 3.0, 4.0]:
        m2 = CoffinMansonModel(n=n_val, delta_T_ref=delta_T_ref, N_ref=N_ref)
        fig5.add_trace(go.Scatter(x=dt_range,
                                   y=[m2.cycles_to_failure(dt) for dt in dt_range],
                                   mode="lines", name=f"n={n_val}",
                                   showlegend=True), row=1, col=2)

    fig5.update_xaxes(title_text="Temperatur [°C]", row=1, col=1)
    fig5.update_xaxes(title_text="ΔT [K]", row=1, col=2)
    fig5.update_yaxes(title_text="Lebensdauer [Jahre]", row=1, col=1)
    fig5.update_yaxes(title_text="Zyklen bis Versagen", row=1, col=2)
    fig5.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig5, use_container_width=True)

with tab4:
    st.header("Zusammenfassung / Bericht")
    st.markdown(f"""
    | Parameter | Wert |
    |---|---|
    | Aktivierungsenergie Ea | {Ea} eV |
    | Referenztemperatur T_ref | {T_ref} °C |
    | Referenzlebensdauer L_ref | {L_ref:,} h |
    | Coffin-Manson Exponent n | {cm_exp} |
    | ΔT_ref | {delta_T_ref} K |
    | N_ref | {N_ref:,} Zyklen |
    | Mittlere Betriebstemperatur | {df['Temperatur_C'].mean():.1f} °C |
    | Rainflow-Zyklen (gesamt) | {df_cycles['count'].sum():.0f} |
    | Jährl. Schädigung (Miner) | {annual_damage*100:.4f} % |
    | **Geschätzte Lebensdauer (CM+Miner)** | **{estimated_life_years:.1f} Jahre** |
    | **Geschätzte Lebensdauer (Arrhenius)** | **{arr_life_years:.1f} Jahre** |
    """)

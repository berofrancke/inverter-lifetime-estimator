"""
Streamlit-App: Wechselrichter-Lebensdauerabschätzung
=====================================================

Struktur:
  Tab 1 – Kondensatoren  : Würth-Modell (Elektrolyt / Polymer-THT) mit Ripple-Faktor k
  Tab 2 – Transistoren   : Foster-Thermalsimulation (Verluste + Th(t) + Zth + Tj)
  Tab 3 – Punktebewertung: Qualitative / nicht-quantifizierbare Parameter
  Tab 4 – Gesamtbewertung: Ampel-Score für alle Komponenten

Modellgrenzen:
  - I_avg ist ein manueller Input (kein automatisches Lastmodell).
  - T_h ist konstant (extern) ODER zeitabhängig aus thermischer Masse modellierbar.
  - P_tot wird als zeitlich konstant angenommen.
  - Foster-Modell basiert auf Datenblatt-Z_th.
  - Duty-Cycle D beeinflusst P_cond; Schaltenergien sind pro Schaltvorgang.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Modell-Module
from models.capacitor import (
    wuerth_electrolyt,
    wuerth_polymer_tht,
)
from models.loss_model import compute_p_losses
from models.foster_model import (
    simulate_tj,
    heatsink_temperature,
    FOSTER_DEFAULT_R,
    FOSTER_DEFAULT_TAU,
)
from models.scoring import (
    ComponentScore,
    MAX_SCORES,
    TOTAL_MAX,
    score_halbleitermaterial,
    score_chip_attach,
    score_bond_tech,
    score_substrat,
    score_baseplate,
    score_aec,
    score_msl,
    score_gate_charge,
    score_kapazitaetstoleranz,
    score_pcb_material,
    score_coating_typ,
    score_wirkungsgrad_25,
    score_kühlung,
    derating_status,
)

# ---------------------------------------------------------------------------
# Seiten-Konfiguration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WR Lebensdauer-Estimator",
    page_icon="⚡",
    layout="wide",
)

FARBE_GUT      = "#27ae60"
FARBE_MITTEL   = "#f39c12"
FARBE_KRITISCH = "#e74c3c"


def ampel_farbe(normalized: float) -> str:
    if normalized >= 0.80:
        return FARBE_GUT
    elif normalized >= 0.50:
        return FARBE_MITTEL
    return FARBE_KRITISCH


# ---------------------------------------------------------------------------
# Titel
# ---------------------------------------------------------------------------
st.title("⚡ Wechselrichter – Lebensdauerabschätzung")
st.markdown(
    "Gemischt quantitativ-qualitativ | "
    "**Arrhenius · Foster-Thermalsimulation · Punktesystem**"
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_c, tab_t, tab_score, tab_gesamt = st.tabs([
    "🔋 Kondensatoren",
    "💡 Transistoren",
    "📋 Punktebewertung",
    "🏁 Gesamtbewertung",
])


# ===========================================================================
# TAB 1 – KONDENSATOREN
# ===========================================================================
with tab_c:
    st.header("Kondensator-Lebensdauer")
    st.markdown(
        "Berechnung nach dem **Würth-Modell** (Basis-2/10-Regel) "
        "inkl. Ripple-Lebensdauerfaktor **k**."
    )

    col_params, col_results = st.columns([1, 1])

    with col_params:
        st.subheader("Bauteilparameter")
        L_nom_c = st.number_input(
            "Nennlebensdauer L_nom [h]",
            value=5000, min_value=100, step=500,
            key="c_lnom",
            help="Herstellerangabe, z. B. 5000 h bei 105 °C",
        )
        T_max_c = st.number_input(
            "Max. Bauteiltemperatur T_max [°C]",
            value=105, min_value=60, max_value=160,
            key="c_tmax",
        )
        T_A_c = st.number_input(
            "Betriebstemperatur T_A [°C]",
            value=60, min_value=-20, max_value=125,
            key="c_ta",
        )

        st.divider()
        st.subheader("Ripple-Derating")
        k_factor = st.number_input(
            "Lebensdauerfaktor k [-]",
            value=1.0, min_value=0.0, step=0.1,
            key="c_k",
            help="Lebensdauerfaktor k: erhöht die Lebensdauer, wenn der "
                 "tatsächliche Ripple-Strom < Nennripple-Strom ist.",
        )

    with col_results:
        st.subheader("Ergebnisse")

        lx_wuerth_elyt = wuerth_electrolyt(L_nom_c, T_max_c, T_A_c, k_factor)
        lx_wuerth_poly = wuerth_polymer_tht(L_nom_c, T_max_c, T_A_c, k_factor)

        col_r1, col_r2 = st.columns(2)
        col_r1.metric("Würth Elektrolyt",
                       f"{lx_wuerth_elyt/8760:.1f} J" if lx_wuerth_elyt < 1e9 else "> Mio. J")
        col_r2.metric("Würth Polymer THT",
                       f"{lx_wuerth_poly/8760:.1f} J" if lx_wuerth_poly < 1e9 else "> Mio. J")

        T_sweep            = np.linspace(20, T_max_c, 200)
        lx_wuerth_e_sweep  = np.array([wuerth_electrolyt(L_nom_c, T_max_c, T, k_factor) for T in T_sweep])
        lx_wuerth_p_sweep  = np.array([wuerth_polymer_tht(L_nom_c, T_max_c, T, k_factor) for T in T_sweep])

        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=T_sweep, y=lx_wuerth_e_sweep / 8760,
                                    name="Würth Elektrolyt", line=dict(color="#2980b9", width=2)))
        fig_c.add_trace(go.Scatter(x=T_sweep, y=lx_wuerth_p_sweep / 8760,
                                    name="Würth Polymer THT", line=dict(color="#8e44ad", width=2)))
        fig_c.add_vline(x=T_A_c, line_dash="dash", line_color="red",
                         annotation_text=f"T_A={T_A_c:.1f} °C")
        fig_c.update_layout(
            title="Kondensator-Lebensdauer vs. Betriebstemperatur",
            xaxis_title="Betriebstemperatur T_A [°C]",
            yaxis_title="Lebensdauer [Jahre]",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_c, use_container_width=True)

        with st.expander("ℹ️ Formeln"):
            st.markdown("""
**Würth Elektrolyt:** $L_x = L_{nom} \\cdot 2^{(T_{max}-T_A)/10} \\cdot k$

**Würth Polymer THT:** $L_x = L_{nom} \\cdot 10^{(T_{max}-T_A)/20} \\cdot k$

**Ripple-Faktor k:** erhöht die Lebensdauer, wenn der tatsächliche
Ripple-Strom unter dem Nennripple-Strom liegt ($k > 1$). Standard $k = 1{,}0$.
            """)


# ===========================================================================
# TAB 2 – TRANSISTOREN / FOSTER-THERMALSIMULATION
# ===========================================================================
with tab_t:
    st.header("Transistor – Foster-Thermalsimulation")
    st.info(
        "**Modellgrenzen:** "
        "I_avg = manueller Input · "
        "T_h = konstant ODER dynamisch T_h(t) aus thermischer Masse · "
        "P_tot zeitlich konstant · "
        "Foster-Parameter aus Datenblatt-Z_th · "
        "Duty-Cycle D beeinflusst P_cond (nicht die Schaltenergien) · "
        "Modell für erste temperaturbasierte Lebensdauerabschätzung."
    )

    st.latex(
        r"Z_{th}(t) = \sum_i r_i \left(1 - e^{-t/\tau_i}\right)"
        r"\qquad"
        r"T_j(t) = T_h(t) + P_{tot} \cdot Z_{th}(t)"
    )

    # -----------------------------------------------------------------------
    # Eingabespalten
    # -----------------------------------------------------------------------
    col_loss, col_foster = st.columns([1, 1])

    # --- Verlustmodell ---
    with col_loss:
        st.subheader("Verlustmodell")

        # ---- Leitverluste ----
        st.markdown("**Leitverluste – V_CE,sat (3 Stützstellen)**")
        lv1, lv2, lv3 = st.columns(3)
        vce_25  = lv1.number_input("V_CE,sat @ 25 °C [V]",  value=1.70, min_value=0.01,
                                    step=0.01, format="%.3f", key="vce25")
        vce_mid = lv2.number_input("V_CE,sat @ mid [V]", value=1.85, min_value=0.01,
                                    step=0.01, format="%.3f", key="vcemid")
        vce_125 = lv3.number_input("V_CE,sat @ 125 °C [V]", value=2.05, min_value=0.01,
                                    step=0.01, format="%.3f", key="vce125")

        T_mid = st.number_input(
            "Mitteltemperatur T_mid [°C]", value=75.0, min_value=26.0, max_value=124.0,
            step=1.0, key="vce_tmid",
            help="Temperatur der mittleren V_CE,sat-Stützstelle (z. B. 75 °C).",
        )
        vce_mode_label = st.selectbox(
            "V_CE,sat-Auswahl",
            ["Interpolation (T_op)", "Fester Wert @ 25 °C",
             "Fester Wert @ mid", "Fester Wert @ 125 °C"],
            key="vce_mode",
            help="Fester Stützpunkt oder lineare Interpolation über die "
                 "drei Stützstellen anhand der Betriebstemperatur T_op.",
        )
        _vce_mode_map = {
            "Interpolation (T_op)": "interp",
            "Fester Wert @ 25 °C": "25",
            "Fester Wert @ mid": "mid",
            "Fester Wert @ 125 °C": "125",
        }
        vce_mode = _vce_mode_map[vce_mode_label]
        if vce_mode == "interp":
            T_op = st.number_input(
                "Betriebstemperatur T_op [°C]", value=75.0, min_value=25.0, max_value=125.0,
                step=1.0, key="vce_top",
                help="Betriebstemperatur für die V_CE,sat-Interpolation.",
            )
        else:
            T_op = None

        I_avg = st.number_input(
            "I_avg – Mittlerer Kollektorstrom [A]",
            value=400.0, min_value=0.1, step=10.0, key="i_avg",
            help="Manuelle Eingabe. Später durch Lastmodell ersetzbar.",
        )
        duty_cycle = st.slider(
            "Duty-Cycle D = t_on / T_sw [-]",
            min_value=0.01, max_value=1.0, value=0.5, step=0.01, key="duty",
            help="Einschaltverhältnis des IGBT/MOSFET. D=0,5 bei symmetrischem Betrieb.",
        )

        st.divider()

        # ---- Schaltverluste ----
        st.markdown("**Schaltverluste – E_on / E_off (linear 25 → 125 °C)**")
        sw1, sw2 = st.columns(2)
        E_on_25  = sw1.number_input("E_on @ 25 °C [mJ]",  value=20.0, min_value=0.0,
                                     step=0.5, key="eon25")
        E_on_125 = sw2.number_input("E_on @ 125 °C [mJ]", value=28.0, min_value=0.0,
                                     step=0.5, key="eon125")
        sw3, sw4 = st.columns(2)
        E_off_25  = sw3.number_input("E_off @ 25 °C [mJ]",  value=12.0, min_value=0.0,
                                      step=0.5, key="eoff25")
        E_off_125 = sw4.number_input("E_off @ 125 °C [mJ]", value=18.0, min_value=0.0,
                                      step=0.5, key="eoff125")

        f_sw = st.number_input(
            "f_sw – Schaltfrequenz [Hz]",
            value=5000.0, min_value=1.0, step=100.0, key="fsw",
        )

    # --- Foster-Parameter ---
    with col_foster:
        st.subheader("Foster-Thermalmodell")
        st.markdown(
            "Die Z_th(t)-Kurve wird als **Summe von RC-Gliedern** (Foster-Kette) "
            "aus dem Datenblatt approximiert. Jedes Glied $r_i / \\tau_i$ "
            "entspricht einer Stufe in der Zth-Kurve (logarithmisch aufgetragen). "
            "Die Kurve ist nicht-linear, weil jede Zeitkonstante $\\tau_i$ "
            "eine andere thermische Schicht modelliert (Chip → Bond → Substrat → Baseplate)."
        )

        n_foster = st.slider("Anzahl RC-Glieder", min_value=1, max_value=8, value=4, key="n_foster")

        r_vals, tau_vals = [], []
        for i in range(n_foster):
            r_def   = FOSTER_DEFAULT_R[i]   if i < len(FOSTER_DEFAULT_R)   else 0.01
            tau_def = FOSTER_DEFAULT_TAU[i] if i < len(FOSTER_DEFAULT_TAU) else 0.01
            c_r, c_tau = st.columns(2)
            r_i   = c_r.number_input(
                f"r_{i+1} [K/W]", value=float(r_def),
                min_value=0.0, step=0.001, format="%.5f", key=f"foster_r_{i}"
            )
            tau_i = c_tau.number_input(
                f"τ_{i+1} [s]", value=float(tau_def),
                min_value=1e-6, step=0.001, format="%.5f", key=f"foster_tau_{i}"
            )
            r_vals.append(r_i)
            tau_vals.append(tau_i)

        rth_sum = sum(r_vals)
        st.info(f"Σ r_i = **{rth_sum:.4f} K/W** (R_th,jh gesamt)")

        st.divider()
        st.subheader("Simulation")

        th_mode = st.radio(
            "Kühlkörpertemperatur T_h",
            ["Konstantes T_h", "Dynamisches T_h(t)"],
            key="th_mode", horizontal=True,
            help="Konstant: fester T_h-Wert. Dynamisch: transiente Aufheizung "
                 "der thermischen Masse T_h(t) = (P_tot·R_th·t)/(m·c·R_th + t) + T_amb.",
        )

        sim1, sim2 = st.columns(2)
        t_end = sim1.number_input("t_end [s]", value=2.0, min_value=0.01, step=0.1, key="tend")
        dt    = sim2.number_input("dt [s]", value=0.002, min_value=1e-5, step=0.001,
                                   format="%.4f", key="dt")

        if th_mode == "Konstantes T_h":
            T_h = st.number_input("T_h [°C]", value=70.0, min_value=0.0, max_value=150.0, key="th")
            m_mod = c_mod = T_amb = None
        else:
            dyn1, dyn2 = st.columns(2)
            m_mod = dyn1.number_input("m – Modulmasse [kg]", value=0.150,
                                       min_value=1e-4, step=0.01, format="%.4f", key="th_m")
            c_mod = dyn2.number_input("c – spez. Wärmekap. [J/(kg·K)]", value=700.0,
                                       min_value=1.0, step=10.0, key="th_c")
            T_amb = st.number_input("T_amb – Umgebungstemperatur [°C]", value=40.0,
                                     min_value=-40.0, max_value=125.0, key="th_tamb")
            T_h = None

    # -----------------------------------------------------------------------
    # Berechnung
    # -----------------------------------------------------------------------
    # T_j-Näherung für die Verlust-Interpolation: festes T_h bzw. T_amb (dyn. Modus)
    T_j_approx = T_h if th_mode == "Konstantes T_h" else T_amb

    loss_result = compute_p_losses(
        I_avg=I_avg,
        vce_sat_25=vce_25,
        vce_sat_125=vce_125,
        vce_sat_mid=vce_mid,
        vce_mode=vce_mode,
        T_mid=T_mid,
        T_op=T_op,
        E_on_25=E_on_25 / 1000.0,    # mJ → J
        E_on_125=E_on_125 / 1000.0,
        E_off_25=E_off_25 / 1000.0,
        E_off_125=E_off_125 / 1000.0,
        f_sw=f_sw,
        duty_cycle=duty_cycle,
        T_j=T_j_approx,
    )
    P_cond = loss_result["P_cond"]
    P_sw   = loss_result["P_sw"]
    P_tot  = loss_result["P_tot"]

    # T_h: konstant (Skalar) oder dynamisch T_h(t) aus thermischer Masse
    t_arr_pre = np.arange(0.0, t_end + dt, dt)
    if th_mode == "Konstantes T_h":
        T_h_input = T_h
    else:
        T_h_input = heatsink_temperature(
            P_tot=P_tot, Rth=rth_sum, m=m_mod, c=c_mod, Tamb=T_amb, t=t_arr_pre,
        )

    sim_result = simulate_tj(
        P_tot=P_tot,
        T_h=T_h_input,
        r=r_vals,
        tau=tau_vals,
        t_end=t_end,
        dt=dt,
    )

    # -----------------------------------------------------------------------
    # Kennzahlen
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Ergebnisse")

    Th_end = sim_result["Th_end"]

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("P_cond",
               f"{P_cond:.1f} W",
               help=f"D={duty_cycle:.2f} × V_CE,sat={loss_result['V_CE_sat_eff']:.3f}V × I={I_avg:.0f}A")
    mc2.metric("P_sw",
               f"{P_sw:.1f} W",
               help=f"(E_on={loss_result['E_on_eff']*1000:.1f}mJ + E_off={loss_result['E_off_eff']*1000:.1f}mJ) × {f_sw:.0f}Hz")
    mc3.metric("P_tot",  f"{P_tot:.1f} W")
    mc4.metric("T_h,end", f"{Th_end:.1f} °C",
               help="Konstant: eingegebenes T_h · Dynamisch: T_h(t_end)")
    mc5.metric("T_j,∞",  f"{sim_result['Tj_inf']:.1f} °C",
               help=f"T_h,end + P_tot × R_th = {Th_end:.1f} + {P_tot:.1f}×{rth_sum:.4f}")
    mc6.metric("R_th,jh", f"{rth_sum:.4f} K/W")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    t_arr   = sim_result["t"]
    zth_arr = sim_result["Zth"]
    tj_arr  = sim_result["Tj"]
    th_arr  = sim_result["Th"]

    th_title = ("T_h(t) – Kühlkörpertemperatur (dynamisch)"
                if th_mode == "Dynamisches T_h(t)"
                else "T_h – Kühlkörpertemperatur (konstant)")

    fig_th = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            th_title,
            "Z_th(t) – Thermische Impedanz (Foster)",
            "T_j(t) – Sperrschichttemperatur",
        ],
        vertical_spacing=0.10,
    )

    # T_h(t) (oder konstante Linie)
    fig_th.add_trace(go.Scatter(
        x=t_arr, y=th_arr, name="T_h(t)",
        line=dict(color="#16a085", width=2),
    ), row=1, col=1)
    fig_th.add_hline(
        y=Th_end,
        line_dash="dash", line_color="#0e6655",
        annotation_text=f"T_h,end = {Th_end:.1f} °C",
        row=1, col=1,
    )

    # Z_th(t) – logarithmische Zeitachse macht den stufenförmigen Verlauf sichtbar
    fig_th.add_trace(go.Scatter(
        x=t_arr, y=zth_arr, name="Z_th(t)",
        line=dict(color="#2980b9", width=2),
    ), row=2, col=1)
    fig_th.add_hline(
        y=rth_sum,
        line_dash="dash", line_color="#1a5276",
        annotation_text=f"R_th = {rth_sum:.4f} K/W",
        row=2, col=1,
    )

    # T_j(t)
    fig_th.add_trace(go.Scatter(
        x=t_arr, y=tj_arr, name="T_j(t)",
        line=dict(color="#e74c3c", width=2),
    ), row=3, col=1)
    fig_th.add_hline(
        y=sim_result["Tj_inf"],
        line_dash="dash", line_color="#c0392b",
        annotation_text=f"T_j,∞ = {sim_result['Tj_inf']:.1f} °C",
        row=3, col=1,
    )

    fig_th.update_xaxes(title_text="Zeit t [s]", type="log", row=1, col=1)
    fig_th.update_xaxes(title_text="Zeit t [s]", type="log", row=2, col=1)
    fig_th.update_xaxes(title_text="Zeit t [s]", type="log", row=3, col=1)
    fig_th.update_yaxes(title_text="T_h [°C]",   row=1, col=1)
    fig_th.update_yaxes(title_text="Z_th [K/W]", row=2, col=1)
    fig_th.update_yaxes(title_text="T_j [°C]",   row=3, col=1)
    fig_th.update_layout(template="plotly_white", height=820, showlegend=False)
    st.plotly_chart(fig_th, use_container_width=True)

    # Sensitivität: T_j,∞ vs. Duty-Cycle
    duty_sweep = np.linspace(0.01, 1.0, 200)
    tj_inf_sweep = []
    for d in duty_sweep:
        lr = compute_p_losses(
            I_avg=I_avg, vce_sat_25=vce_25, vce_sat_125=vce_125,
            vce_sat_mid=vce_mid, vce_mode=vce_mode, T_mid=T_mid, T_op=T_op,
            E_on_25=E_on_25/1000, E_on_125=E_on_125/1000,
            E_off_25=E_off_25/1000, E_off_125=E_off_125/1000,
            f_sw=f_sw, duty_cycle=d, T_j=T_j_approx,
        )
        tj_inf_sweep.append(Th_end + lr["P_tot"] * rth_sum)

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=duty_sweep, y=tj_inf_sweep,
        line=dict(color="#8e44ad", width=2), name="T_j,∞",
    ))
    fig_sens.add_vline(x=duty_cycle, line_dash="dash", line_color="red",
                        annotation_text=f"D={duty_cycle:.2f}")
    fig_sens.update_layout(
        title="Sensitivität: T_j,∞ vs. Duty-Cycle",
        xaxis_title="Duty-Cycle D [-]",
        yaxis_title="T_j,∞ [°C]",
        template="plotly_white",
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    with st.expander("ℹ️ Formeln & Modellgrenzen"):
        st.markdown("""
**Leitverluste (mit Duty-Cycle):**
$$P_{cond} = D \\cdot V_{CE,sat} \\cdot I_{avg}$$

*V_CE,sat: fester Stützpunkt (25 °C / mid / 125 °C) oder lineare Interpolation
über die drei Stützstellen anhand T_op.*

**Schaltverluste (temperaturabhängig):**
$$P_{sw} = (E_{on}(T_j) + E_{off}(T_j)) \\cdot f_{sw}$$

*E_on und E_off werden linear zwischen 25 °C und 125 °C interpoliert.*

**Gesamtverluste:**
$$P_{tot} = P_{cond} + P_{sw}$$

**Kühlkörpertemperatur T_h:**
- *Konstant:* fester Eingabewert $T_h$
- *Dynamisch:* $T_h(t) = \\dfrac{P_{tot} \\cdot R_{th} \\cdot t}{m \\cdot c \\cdot R_{th} + t} + T_{amb}$

**Foster Z_th(t):**
$$Z_{th}(t) = \\sum_i r_i \\left(1 - e^{-t/\\tau_i}\\right)
\\qquad T_j(t) = T_h(t) + P_{tot} \\cdot Z_{th}(t)$$

Die Kurve ist **nicht-linear** weil jedes RC-Glied eine andere thermische Schicht
repräsentiert (Chip-Bond → Substrat → Baseplate → Kühlkörper). Auf logarithmischer
Zeitachse sieht man den stufenförmigen Anstieg je Schicht.

**Stationärer Endwert:**
$$T_{j,\\infty} = T_{h,end} + P_{tot} \\cdot \\sum_i r_i$$

mit $T_{h,end} = T_h$ (konstant) bzw. $T_h(t_{end})$ (dynamisch).

**Modellgrenzen:**
- I_avg: manueller Input, keine automatische Topologie-/Duty-Cycle-Modellierung des Stroms
- T_h: konstant (extern) oder dynamisch aus thermischer Masse (m, c, T_amb, R_th)
- Duty-Cycle D beeinflusst P_cond; Schaltenergien sind pro Schaltimpuls (nicht D-gewichtet)
- P_tot als zeitlich konstant angenommen
- Foster-Modell aus Datenblatt-Z_th (junction-to-heatsink oder junction-to-case)
        """)


# ===========================================================================
# TAB 3 – PUNKTEBEWERTUNG
# ===========================================================================
with tab_score:
    st.header("Punktebewertung – qualitative Parameter")
    st.markdown(
        "Bewertung nicht-quantifizierbarer Merkmale mit **0–5 Punkten** je Parameter. "
        "Binäre Kriterien: **0 = fehlt / 1 = vorhanden**."
    )

    scores: dict[str, ComponentScore] = {}

    # --- Transistor ---
    with st.expander("💡 Transistor (max. 55 Punkte)", expanded=True):
        sc_t = ComponentScore("Transistor", MAX_SCORES["Transistor"])

        c1, c2 = st.columns(2)
        with c1:
            mat = st.selectbox("Halbleitermaterial", ["SiC", "GaN", "Si (modern)", "Si (klassisch)"], key="sc_mat")
            sc_t.add("Halbleitermaterial", score_halbleitermaterial(mat))

            attach = st.selectbox("Chip-Attach", ["Sintern", "Press-fit", "Weichlot (SnAg)"], key="sc_attach")
            sc_t.add("Chip-Attach", score_chip_attach(attach))

            bond = st.selectbox("Bond-Technologie", ["Ribbon / Cu (bondlos)", "Cu-Draht", "Al-Draht"], key="sc_bond")
            sc_t.add("Bond-Technologie", score_bond_tech(bond))

            sub = st.selectbox("Substratmaterial", ["Si₃N₄", "AlN", "Al₂O₃", "andere"], key="sc_sub")
            sc_t.add("Substrat", score_substrat(sub))

        with c2:
            bp = st.selectbox("Baseplatematerial", ["AlSiC", "Cu", "ohne Baseplate"], key="sc_bp")
            sc_t.add("Baseplate", score_baseplate(bp))

            aec = st.selectbox("AEC-Qualifikation", ["AEC-Q101 / Q102", "Industrie-Qualifikation", "keine"], key="sc_aec")
            sc_t.add("AEC-Qual.", score_aec(aec))

            msl = st.selectbox("MSL-Level", [1, 2, 3, 4], index=1, key="sc_msl")
            sc_t.add("MSL", score_msl(int(msl)))

            qg = st.number_input("Gate-Ladung Q_G [nC]", value=50.0, min_value=0.0, key="sc_qg")
            sc_t.add("Q_G", score_gate_charge(qg))

        st.markdown("**Binär (0/1)**")
        b1, b2, b3, b4 = st.columns(4)
        sc_t.add("Datenblatt",      float(b1.checkbox("Datenblatt vorhanden", value=True,  key="sc_db")))
        sc_t.add("LESIT/CIPS",      float(b2.checkbox("LESIT/CIPS-Daten",    value=False, key="sc_lesit")))
        sc_t.add("CM-Parameter",    float(b3.checkbox("CM-Parameter",         value=False, key="sc_cm")))
        sc_t.add("Therm. RC-Kette", float(b4.checkbox("Therm. RC-Kette",      value=False, key="sc_rc")))

        scores["Transistor"] = sc_t
        color_t, status_t = sc_t.status
        st.markdown(
            f"**Transistor-Score: {sc_t.total:.0f} / {sc_t.max_points} "
            f"({sc_t.normalized*100:.0f} %) — "
            f"<span style='color:{color_t}'>{status_t}</span>**",
            unsafe_allow_html=True,
        )

    # --- Kondensator ---
    with st.expander("🔋 Kondensator (max. 35 Punkte)"):
        sc_c = ComponentScore("Kondensator", MAX_SCORES["Kondensator"])

        c1, c2 = st.columns(2)
        with c1:
            tol = st.number_input("Kapazitätstoleranz [%]", value=20.0, min_value=0.0, key="sc_ctol")
            sc_c.add("Toleranz", score_kapazitaetstoleranz(tol))

            esr_score = st.slider("ESR-Bewertung (0=hoch / 5=niedrig)", 0, 5, 3, key="sc_esr")
            sc_c.add("ESR", float(esr_score))

            thb_h = st.number_input("THB-Testdauer [h]", value=1000, min_value=0, key="sc_thb")
            thb_score = 5 if thb_h >= 2000 else (4 if thb_h >= 1000 else (2 if thb_h >= 500 else 0))
            sc_c.add("THB-Dauer", float(thb_score))

        with c2:
            kk_c = st.selectbox("Klimaklasse", ["55/125/21", "40/85/21", "25/85/21", "unbekannt"], key="sc_kk")
            kk_s = 5 if kk_c == "55/125/21" else (4 if "85" in kk_c else 2)
            sc_c.add("Klimaklasse", float(kk_s))

            n_parallel = st.number_input("Anzahl Parallel-C", value=1, min_value=1, key="sc_npar")
            sc_c.add("Redundanz", min(5.0, float(n_parallel)))

        st.markdown("**Binär (0/1)**")
        b1c, b2c, b3c = st.columns(3)
        sc_c.add("Datenblatt-C", float(b1c.checkbox("Datenblatt C",       value=True, key="sc_dbc")))
        sc_c.add("Typbez.-C",    float(b2c.checkbox("Typbezeichnung C",    value=True, key="sc_idc")))
        sc_c.add("Hersteller-C", float(b3c.checkbox("Hersteller bekannt",  value=True, key="sc_mfrc")))

        scores["Kondensator"] = sc_c
        color_c, status_c = sc_c.status
        st.markdown(
            f"**Kondensator-Score: {sc_c.total:.0f} / {sc_c.max_points} "
            f"({sc_c.normalized*100:.0f} %) — "
            f"<span style='color:{color_c}'>{status_c}</span>**",
            unsafe_allow_html=True,
        )

    # --- PCB ---
    with st.expander("🖥️ Leiterplatte PCB (max. 55 Punkte)"):
        sc_p = ComponentScore("Leiterplatte", MAX_SCORES["Leiterplatte"])

        c1, c2 = st.columns(2)
        with c1:
            pcb_mat = st.selectbox("PCB-Material",
                                    ["FR4-Standard", "FR4-High-Tg (≥170°C)", "Polyimid (Rogers)", "CEM"],
                                    key="sc_pcbmat")
            sc_p.add("Material", score_pcb_material(pcb_mat))

            coating = st.selectbox("Coating-Typ",
                                    ["kein", "Acryl", "Urethan", "Silikon", "Parylene"],
                                    key="sc_coat")
            sc_p.add("Coating", score_coating_typ(coating))

            coat_proc = st.selectbox("Coating-Verfahren", ["Spray", "Dip", "Selektiv"], key="sc_cproc")
            sc_p.add("Coat.-Verfahren", {"Spray": 2, "Dip": 3, "Selektiv": 5}.get(coat_proc, 2))

            coat_thick = st.number_input("Coating-Dicke [µm]", value=50, min_value=0, key="sc_cthick")
            sc_p.add("Coat.-Dicke",
                      5 if coat_thick >= 100 else (4 if coat_thick >= 50 else (2 if coat_thick >= 25 else 1)))

        with c2:
            n_layer = st.number_input("Lagenzahl", value=4, min_value=1, key="sc_nlayer")
            sc_p.add("Lagen", 5 if n_layer >= 6 else (3 if n_layer >= 4 else 1))

            cu_thick = st.selectbox("Kupferdicke [µm]", [35, 70, 105], index=1, key="sc_cu")
            sc_p.add("Cu-Dicke", {35: 2, 70: 4, 105: 5}.get(cu_thick, 2))

            caf = st.selectbox("CAF-Risiko",
                                ["unbekannt", "hoch", "gering", "kein Risiko (getestet)"],
                                key="sc_caf")
            sc_p.add("CAF", {"unbekannt": 0, "hoch": 1, "gering": 3, "kein Risiko (getestet)": 5}.get(caf, 0))

            sm_type = st.selectbox("Soldermask-Typ",
                                    ["keine", "Epoxid", "LPI (flüssig fotosensitiv)"],
                                    key="sc_sm")
            sc_p.add("Soldermask", {"keine": 0, "Epoxid": 3, "LPI (flüssig fotosensitiv)": 5}.get(sm_type, 0))

        st.markdown("**Binär (0/1)**")
        b1p, b2p = st.columns(2)
        sc_p.add("Datenblatt-PCB", float(b1p.checkbox("Fertigungsspezifikation", value=True,  key="sc_dsp")))
        sc_p.add("IPC-Klasse",     float(b2p.checkbox("IPC-A-610 Nachweis",      value=False, key="sc_ipc")))

        scores["Leiterplatte"] = sc_p
        color_p, status_p = sc_p.status
        st.markdown(
            f"**PCB-Score: {sc_p.total:.0f} / {sc_p.max_points} "
            f"({sc_p.normalized*100:.0f} %) — "
            f"<span style='color:{color_p}'>{status_p}</span>**",
            unsafe_allow_html=True,
        )

    # --- System ---
    with st.expander("⚙️ System (max. 30 Punkte)"):
        sc_s = ComponentScore("System", MAX_SCORES["System"])

        c1, c2 = st.columns(2)
        with c1:
            eta_25 = st.number_input("η bei 25 °C / Nennlast [%]", value=97.5,
                                      min_value=80.0, max_value=100.0, step=0.1, key="sc_eta25")
            sc_s.add("η (25°C)", score_wirkungsgrad_25(eta_25))

            eta_tmax = st.number_input("η bei T_max [%]", value=96.5,
                                        min_value=80.0, max_value=100.0, step=0.1, key="sc_etatmax")
            sc_s.add("η (T_max)", 5 if eta_tmax >= 97 else (4 if eta_tmax >= 95 else (2 if eta_tmax >= 93 else 0)))

            kuehl = st.selectbox("Kühlung",
                                  ["Naturkonvektion", "Forcierte Luft (Lüfter)", "Flüssigkühlung"],
                                  key="sc_kuehl")
            sc_s.add("Kühlung", score_kühlung(kuehl))

        with c2:
            i_leak = st.number_input("Leakage-Strom gegen PE [mA]", value=3.0, min_value=0.0, key="sc_leak")
            sc_s.add("Leakage",
                      5 if i_leak < 1 else (3 if i_leak <= 5 else (1 if i_leak <= 30 else 0)))

            rh_case = st.number_input("Rel. Feuchte im Gehäuse [%]", value=45.0,
                                       min_value=0.0, max_value=100.0, key="sc_rh")
            sc_s.add("RH Gehäuse",
                      5 if rh_case < 40 else (3 if rh_case <= 60 else (1 if rh_case <= 80 else 0)))

            l_loop = st.number_input("Schleifeninduktivität L_loop [nH]", value=30.0, min_value=0.0, key="sc_lloop")
            sc_s.add("L_loop",
                      5 if l_loop < 10 else (3 if l_loop <= 50 else (1 if l_loop <= 200 else 0)))

        st.subheader("Derating-Faktoren (informativ)")
        d1, d2, d3 = st.columns(3)
        V_derate = d1.number_input("k_V = V_op/V_max", value=0.75, min_value=0.0, max_value=1.5, step=0.01, key="sc_kv")
        T_derate = d2.number_input("k_T = T_op/T_max", value=0.80, min_value=0.0, max_value=1.5, step=0.01, key="sc_kt")
        I_derate = d3.number_input("k_I = I_op/I_cont", value=0.70, min_value=0.0, max_value=1.5, step=0.01, key="sc_ki")

        for label, ratio, target in [("Spannungs-Derating", V_derate, 0.80),
                                       ("Temperatur-Derating", T_derate, 0.90),
                                       ("Strom-Derating", I_derate, 0.75)]:
            col_d, txt = derating_status(ratio, target)
            st.markdown(f"**{label}:** <span style='color:{col_d}'>{txt}</span>",
                         unsafe_allow_html=True)

        scores["System"] = sc_s
        color_s, status_s = sc_s.status
        st.markdown(
            f"**System-Score: {sc_s.total:.0f} / {sc_s.max_points} "
            f"({sc_s.normalized*100:.0f} %) — "
            f"<span style='color:{color_s}'>{status_s}</span>**",
            unsafe_allow_html=True,
        )

    sc_sp = ComponentScore("Spule",  MAX_SCORES["Spule"])
    sc_sp.add("Platzhalter", 0)
    scores["Spule"]  = sc_sp

    sc_r  = ComponentScore("Relais", MAX_SCORES["Relais"])
    sc_r.add("Platzhalter", 0)
    scores["Relais"] = sc_r


# ===========================================================================
# TAB 4 – GESAMTBEWERTUNG
# ===========================================================================
with tab_gesamt:
    st.header("Gesamtbewertung – Lebensdauerindex")

    for comp in ["Transistor", "Kondensator", "Leiterplatte", "System", "Spule", "Relais"]:
        if comp not in scores:
            sc_dummy = ComponentScore(comp, MAX_SCORES[comp])
            sc_dummy.add("Platzhalter", 0)
            scores[comp] = sc_dummy

    total_punkte = sum(sc.total for sc in scores.values())
    total_norm   = total_punkte / TOTAL_MAX

    total_color = ampel_farbe(total_norm)
    total_label = ("GUT – Lebensdauer OK" if total_norm >= 0.80
                   else "MITTEL – Maßnahmen prüfen" if total_norm >= 0.50
                   else "KRITISCH – Sofort handeln")

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Gesamtpunkte", f"{total_punkte:.0f} / {TOTAL_MAX}")
    kpi2.metric("Normierter Score", f"{total_norm*100:.1f} %")
    kpi3.markdown(
        f"<div style='background:{total_color};padding:16px;border-radius:8px;"
        f"color:white;font-weight:bold;font-size:1.1em;text-align:center;'>"
        f"{total_label}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    rows_g = []
    for comp, sc in scores.items():
        c_color, c_status = sc.status
        rows_g.append({
            "Komponente": comp,
            "Punkte":     f"{sc.total:.0f} / {sc.max_points}",
            "Score [%]":  f"{sc.normalized*100:.0f} %",
            "Status":     c_status,
        })
    st.dataframe(pd.DataFrame(rows_g), use_container_width=True, hide_index=True)

    komponnenten = [sc.name for sc in scores.values()]
    norm_vals    = [sc.normalized * 100 for sc in scores.values()]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]],
        theta=komponnenten + [komponnenten[0]],
        fill="toself",
        name="Score [%]",
        line_color="#2980b9",
        fillcolor="rgba(41,128,185,0.25)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[80] * (len(komponnenten) + 1),
        theta=komponnenten + [komponnenten[0]],
        mode="lines",
        name="Ziel 80 %",
        line=dict(color=FARBE_GUT, dash="dash", width=1),
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Komponenten-Score (Radar)",
        template="plotly_white",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    colors = [ampel_farbe(sc.normalized) for sc in scores.values()]
    fig_bar = go.Figure(go.Bar(
        x=komponnenten, y=norm_vals,
        marker_color=colors,
        text=[f"{v:.0f} %" for v in norm_vals],
        textposition="outside",
    ))
    fig_bar.add_hline(y=80, line_dash="dash", line_color=FARBE_GUT,    annotation_text="GUT (80%)")
    fig_bar.add_hline(y=50, line_dash="dot",  line_color=FARBE_MITTEL, annotation_text="MITTEL (50%)")
    fig_bar.update_layout(
        title="Score je Komponente",
        yaxis=dict(range=[0, 110], title="Normierter Score [%]"),
        template="plotly_white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Handlungsempfehlungen")
    for comp, sc in scores.items():
        c_col, c_stat = sc.status
        if sc.normalized < 0.80:
            st.markdown(
                f"- **{comp}** (<span style='color:{c_col}'>{c_stat}</span>): "
                f"Score {sc.normalized*100:.0f} % – Parameter überprüfen.",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown(
        "**Bewertungsskala:** "
        f"<span style='color:{FARBE_GUT}'>■</span> ≥ 80 % GUT · "
        f"<span style='color:{FARBE_MITTEL}'>■</span> 50–79 % MITTEL · "
        f"<span style='color:{FARBE_KRITISCH}'>■</span> < 50 % KRITISCH",
        unsafe_allow_html=True,
    )

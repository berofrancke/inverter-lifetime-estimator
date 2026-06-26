"""
Streamlit-App: Wechselrichter-Lebensdauerabschätzung
=====================================================

Struktur:
  Tab 1 – Kondensatoren  : Würth-Modell (Elektrolyt / Polymer-THT) mit Ripple-Faktor k
  Tab 2 – Punktebewertung: Qualitative / nicht-quantifizierbare Parameter
  Tab 3 – Gesamtbewertung: Ampel-Score für alle Komponenten
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Modell-Module
from models.capacitor import (
    wuerth_electrolyt,
    wuerth_polymer_tht,
)
from models.scoring import (
    ComponentScore,
    MAX_SCORES,
    TOTAL_MAX,
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
    "**Arrhenius · Punktesystem**"
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_c, tab_score, tab_gesamt = st.tabs([
    "🔋 Kondensatoren",
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
# TAB 2 – PUNKTEBEWERTUNG
# ===========================================================================
with tab_score:
    st.header("Punktebewertung – qualitative Parameter")
    st.markdown(
        "Bewertung nicht-quantifizierbarer Merkmale mit **0–5 Punkten** je Parameter. "
        "Binäre Kriterien: **0 = fehlt / 1 = vorhanden**."
    )

    scores: dict[str, ComponentScore] = {}

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
# TAB 3 – GESAMTBEWERTUNG
# ===========================================================================
with tab_gesamt:
    st.header("Gesamtbewertung – Lebensdauerindex")

    for comp in ["Kondensator", "Leiterplatte", "System", "Spule", "Relais"]:
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

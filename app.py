"""
Streamlit-App: Wechselrichter-Lebensdauerabschätzung
=====================================================

Struktur:
  Tab 1 – Kondensatoren  : Würth-Modell (Elektrolyt / Polymer-THT) + Chemicon-Arrhenius
  Tab 2 – Transistoren   : Norris-Landzberg, CIPS-08 (Bayerer), SKiM63
           + Miner-Akkumulation über Rainflow-Klassen
  Tab 3 – Punktebewertung: Qualitative / nicht-quantifizierbare Parameter
           (Kondensator, Transistor, PCB, System)
  Tab 4 – Gesamtbewertung: Ampel-Score für alle Komponenten

Abhängigkeiten: streamlit, numpy, pandas, plotly, scipy
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Eigene Modell-Module
from models.capacitor import (
    wuerth_electrolyt,
    wuerth_polymer_tht,
    arrhenius_capacitor,
    ripple_temperature_rise,
)
from models.transistor import (
    norris_landzberg,
    bayerer_cips08,
    skim63_model,
    arrhenius_gate_oxide,
    miner_damage,
    CIPS08_INFINEON_DEFAULT,
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

# Farben für Ampel
FARBE_GUT      = "#27ae60"
FARBE_MITTEL   = "#f39c12"
FARBE_KRITISCH = "#e74c3c"


def ampel_farbe(normalized: float) -> str:
    """Gibt die CSS-Farbe für den normierten Score zurück."""
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
    "**Arrhenius · Coffin-Manson (Norris-Landzberg / CIPS-08 / SKiM63) · Miner · Punktesystem**"
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
        "Berechnung nach zwei Methodiken: "
        "**Würth-Modell** (Basis-2/10-Regel) und **Arrhenius-Modell** (Chemicon)."
    )

    col_params, col_results = st.columns([1, 1])

    with col_params:
        st.subheader("Bauteilparameter")
        L_nom_c = st.number_input(
            "Nennlebensdauer L_nom [h] (Datenblatt)",
            value=5000, min_value=100, step=500,
            help="Herstellerangabe, z. B. 5000 h bei 105 °C",
        )
        T_max_c = st.number_input(
            "Max. zulässige Bauteiltemperatur T_max [°C]",
            value=105, min_value=60, max_value=160,
            help="Maximaltemperatur aus Datenblatt (typisch 85 / 105 / 125 °C)",
        )
        T_A_c = st.number_input(
            "Umgebungstemperatur im Gerät T_A [°C]",
            value=60, min_value=-20, max_value=125,
            help="Effektive Temperatur am Einbauort inkl. Eigenerwärmung",
        )

        st.divider()
        st.subheader("Arrhenius-Parameter (Chemicon)")
        T_ref_c = st.number_input(
            "Referenztemperatur T_ref [°C]",
            value=105, min_value=60, max_value=160,
            help="Gleich T_max falls nicht separat angegeben",
        )
        Ea_c = st.number_input(
            "Aktivierungsenergie Ea [eV]",
            value=0.94, min_value=0.3, max_value=1.5, step=0.05,
            help="Typisch 0,6–1,0 eV für Elektrolyt-Kondensatoren",
        )

        st.divider()
        st.subheader("Ripple-Strom-Eigenerwärmung")
        I_ripple_c = st.number_input(
            "Betriebsmäßiger Ripple-Strom I_ripple [A]",
            value=1.0, min_value=0.0, step=0.1,
        )
        ESR_c = st.number_input(
            "ESR bei Betriebsfrequenz [Ω]",
            value=0.05, min_value=0.0, step=0.005, format="%.4f",
        )

    with col_results:
        st.subheader("Ergebnisse")

        # Eigenerwärmung durch Ripple
        delta_T_ripple = ripple_temperature_rise(I_ripple_c, ESR_c)
        T_op_c = T_A_c + delta_T_ripple

        st.info(
            f"Eigenerwärmung durch Ripple: **ΔT = {delta_T_ripple:.2f} K**  \n"
            f"→ Effektive Betriebstemperatur: **T_op = {T_op_c:.1f} °C**"
        )

        # --- Würth Elektrolyt ---
        lx_wuerth_elyt = wuerth_electrolyt(L_nom_c, T_max_c, T_op_c)
        # --- Würth Polymer THT ---
        lx_wuerth_poly = wuerth_polymer_tht(L_nom_c, T_max_c, T_op_c)
        # --- Chemicon Arrhenius ---
        lx_arr = arrhenius_capacitor(L_nom_c, T_ref_c, T_op_c, Ea_c)

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric(
            "Würth Elektrolyt / Hybrid-Polymer",
            f"{lx_wuerth_elyt/8760:.1f} Jahre" if lx_wuerth_elyt < 1e9 else "> Mio. Jahre",
            help="Lx = L_nom · 2^((T_max − T_A) / 10)",
        )
        col_r2.metric(
            "Würth Polymer THT / V-Chip",
            f"{lx_wuerth_poly/8760:.1f} Jahre" if lx_wuerth_poly < 1e9 else "> Mio. Jahre",
            help="Lx = L_nom · 10^((T_max − T_A) / 20)",
        )
        col_r3.metric(
            "Arrhenius (Chemicon)",
            f"{lx_arr/8760:.1f} Jahre" if lx_arr < 1e9 else "> Mio. Jahre",
            help="Lx = L0 · exp[(Ea/k_B)·(1/T_op − 1/T_ref)]",
        )

        # --- Sensitivitätskurve: Lebensdauer vs. Betriebstemperatur ---
        T_sweep = np.linspace(20, T_max_c, 200)
        lx_wuerth_e_sweep = np.array([wuerth_electrolyt(L_nom_c, T_max_c, T) for T in T_sweep])
        lx_wuerth_p_sweep = np.array([wuerth_polymer_tht(L_nom_c, T_max_c, T) for T in T_sweep])
        lx_arr_sweep      = np.array([arrhenius_capacitor(L_nom_c, T_ref_c, T, Ea_c) for T in T_sweep])

        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=T_sweep, y=lx_wuerth_e_sweep / 8760,
            name="Würth Elektrolyt / Hybrid",
            line=dict(color="#2980b9", width=2),
        ))
        fig_c.add_trace(go.Scatter(
            x=T_sweep, y=lx_wuerth_p_sweep / 8760,
            name="Würth Polymer THT",
            line=dict(color="#8e44ad", width=2),
        ))
        fig_c.add_trace(go.Scatter(
            x=T_sweep, y=lx_arr_sweep / 8760,
            name="Arrhenius (Chemicon)",
            line=dict(color="#e67e22", width=2),
        ))
        # Marker für aktuellen Betriebspunkt
        fig_c.add_vline(
            x=T_op_c,
            line_dash="dash", line_color="red",
            annotation_text=f"T_op={T_op_c:.1f} °C",
        )
        fig_c.update_layout(
            title="Kondensator-Lebensdauer vs. Betriebstemperatur",
            xaxis_title="Betriebstemperatur T_op [°C]",
            yaxis_title="Lebensdauer [Jahre]",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_c, use_container_width=True)

        # Formelübersicht
        with st.expander("ℹ️ Formeln & Hinweise"):
            st.markdown("""
**Würth Elektrolyt / Hybrid-Polymer (SMD H-Chip):**
$$L_x = L_{\\mathrm{nom}} \\cdot 2^{\\frac{T_{\\mathrm{max}} - T_A}{10}}$$

**Würth Polymer THT / V-Chip SMD:**
$$L_x = L_{\\mathrm{nom}} \\cdot 10^{\\frac{T_{\\mathrm{max}} - T_A}{20}}$$

**Arrhenius (Chemicon):**
$$L_x = L_0 \\cdot \\exp\\!\\left[\\frac{E_a}{k_B}\\cdot\\left(\\frac{1}{T_{\\mathrm{op,K}}} - \\frac{1}{T_{\\mathrm{ref,K}}}\\right)\\right]$$

Eigenerwärmung: $\\Delta T_{\\mathrm{ripple}} = \\mathrm{ESR} \\cdot I_{\\mathrm{ripple}}^2$
            """)


# ===========================================================================
# TAB 2 – TRANSISTOREN
# ===========================================================================
with tab_t:
    st.header("Transistor-Lebensdauer (Power Cycling)")
    st.markdown(
        "Drei implementierte Modelle: **Norris-Landzberg** (Gl. 4.1), "
        "**Bayerer / CIPS-08** (Gl. 4.2), **SKiM63** (Gl. 4.3). "
        "Zusätzlich: **Miner-Akkumulation** über bis zu 8 Rainflow-Klassen."
    )

    sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs([
        "Norris-Landzberg", "CIPS-08 (Bayerer)", "SKiM63", "Miner + Rainflow"
    ])

    # -----------------------------------------------------------------------
    # Norris-Landzberg
    # -----------------------------------------------------------------------
    with sub_t1:
        st.subheader("Norris-Landzberg-Modell (Gl. 4.1)")
        st.latex(
            r"N_f = \alpha \cdot f^{\alpha_2} \cdot \Delta T_j^{n_1} "
            r"\cdot \exp\!\left(\frac{E_a}{k_B \cdot T_{j,m}}\right)"
        )

        c1, c2 = st.columns(2)
        with c1:
            nl_alpha  = st.number_input("Vorfaktor α",        value=3.5e14, format="%e",
                                         help="Aus LESIT/CIPS-Fitting (komponentenspezifisch)")
            nl_alpha2 = st.number_input("Frequenz-Exponent α₂", value=-0.33, step=0.01,
                                         help="Einfluss der Zyklusfrequenz auf N_f")
            nl_n1     = st.number_input("Coffin-Manson Exponent n₁", value=-1.9, step=0.1,
                                         help="Typisch −1,9 bis −3 für Al-Bonddrähte")
        with c2:
            nl_Ea     = st.number_input("Aktivierungsenergie Ea [eV]", value=0.40, step=0.05,
                                         help="Arrhenius-Anteil; 0,3–0,7 eV für Bondermüdung")
            nl_dTj    = st.number_input("Temperaturhub ΔT_j [K]",     value=60.0, min_value=1.0,
                                         help="Amplitude aus Rainflow-Auswertung")
            nl_Tjm    = st.number_input("Mittl. Sperrschichttemp T_jm [°C]", value=90.0,
                                         help="Mittlerer Wert des Zyklus")
            nl_f      = st.number_input("Zyklusfrequenz f [Hz]", value=1.0, min_value=0.001,
                                         step=0.1, format="%.3f",
                                         help="1 Hz = 1 Zyklus/Sekunde; bei 50 Hz Netz typisch 1/3600")

        nl_Nf = norris_landzberg(nl_dTj, nl_Tjm, nl_alpha, nl_alpha2, nl_n1, nl_Ea, nl_f)
        st.metric("N_f (Norris-Landzberg)", f"{nl_Nf:,.0f} Zyklen" if nl_Nf < 1e12 else "> 10¹² Zyklen")

        # Sensitivität über ΔT_j
        dT_range = np.linspace(5, 150, 200)
        Nf_nl_sweep = np.array([
            norris_landzberg(dt, nl_Tjm, nl_alpha, nl_alpha2, nl_n1, nl_Ea, nl_f)
            for dt in dT_range
        ])
        fig_nl = go.Figure()
        fig_nl.add_trace(go.Scatter(
            x=dT_range, y=Nf_nl_sweep,
            line=dict(color="#2980b9", width=2), name="N_f (Norris-Landzberg)",
        ))
        fig_nl.add_vline(x=nl_dTj, line_dash="dash", line_color="red",
                          annotation_text=f"ΔT_j={nl_dTj} K")
        fig_nl.update_layout(
            title="Norris-Landzberg: N_f vs. ΔT_j",
            xaxis_title="ΔT_j [K]", yaxis_title="N_f (Zyklen bis Versagen)",
            yaxis_type="log", template="plotly_white",
        )
        st.plotly_chart(fig_nl, use_container_width=True)

    # -----------------------------------------------------------------------
    # CIPS-08 / Bayerer
    # -----------------------------------------------------------------------
    with sub_t2:
        st.subheader("Bayerer / CIPS-08-Modell (Gl. 4.2)")
        st.latex(
            r"N_f = K \cdot \Delta T_j^{\beta_1} \cdot \exp\!\left(\frac{\beta_2}{T_{j,m}+273}\right)"
            r"\cdot I_{on}^{\beta_3} \cdot V^{\beta_4} \cdot D^{\beta_5}"
        )
        st.caption(
            "Defaultparameter: Infineon EasyPACK-IGBT-Modul "
            "(K = 2,03 × 10¹⁴; β₁ = 4,416; β₂ = 1285; β₃ = −0,463; β₄ = 0,716; β₅ = −0,761)"
        )

        c1, c2 = st.columns(2)
        with c1:
            ci_K     = st.number_input("K (Technologiefaktor)", value=float(CIPS08_INFINEON_DEFAULT["K"]),
                                        format="%e")
            ci_b1    = st.number_input("β₁ (ΔT_j-Exponent)",   value=CIPS08_INFINEON_DEFAULT["beta1"], step=0.1)
            ci_b2    = st.number_input("β₂ (Arrhenius-Faktor [K])", value=float(CIPS08_INFINEON_DEFAULT["beta2"]))
            ci_b3    = st.number_input("β₃ (Strom-Exponent)",   value=CIPS08_INFINEON_DEFAULT["beta3"], step=0.01)
            ci_b4    = st.number_input("β₄ (Spannungs-Exponent)", value=CIPS08_INFINEON_DEFAULT["beta4"], step=0.01)
            ci_b5    = st.number_input("β₅ (Bonddraht-Exponent)", value=CIPS08_INFINEON_DEFAULT["beta5"], step=0.01)
        with c2:
            ci_dTj   = st.number_input("ΔT_j [K]",         value=60.0,  min_value=1.0)
            ci_Tjm   = st.number_input("T_jm [°C]",        value=90.0)
            ci_Ion   = st.number_input("I_on [A]",          value=200.0, min_value=0.1)
            ci_V     = st.number_input("V (Bondraht-Spg.) [V]", value=600.0, min_value=1.0)
            ci_D     = st.number_input("D (Bonddraht-Ø) [µm]", value=400.0, min_value=1.0)

        ci_Nf = bayerer_cips08(ci_dTj, ci_Tjm, ci_Ion, ci_V, ci_D,
                                ci_K, ci_b1, ci_b2, ci_b3, ci_b4, ci_b5)
        st.metric("N_f (CIPS-08)", f"{ci_Nf:,.0f} Zyklen" if ci_Nf < 1e14 else "> 10¹⁴ Zyklen")

        # Sensitivitätsplot: ΔT_j Sweep
        Nf_ci_sweep = np.array([
            bayerer_cips08(dt, ci_Tjm, ci_Ion, ci_V, ci_D,
                           ci_K, ci_b1, ci_b2, ci_b3, ci_b4, ci_b5)
            for dt in dT_range
        ])
        fig_ci = go.Figure()
        fig_ci.add_trace(go.Scatter(
            x=dT_range, y=Nf_ci_sweep,
            line=dict(color="#27ae60", width=2), name="N_f (CIPS-08)",
        ))
        fig_ci.add_vline(x=ci_dTj, line_dash="dash", line_color="red",
                          annotation_text=f"ΔT_j={ci_dTj} K")
        fig_ci.update_layout(
            title="CIPS-08: N_f vs. ΔT_j",
            xaxis_title="ΔT_j [K]", yaxis_title="N_f (Zyklen bis Versagen)",
            yaxis_type="log", template="plotly_white",
        )
        st.plotly_chart(fig_ci, use_container_width=True)

    # -----------------------------------------------------------------------
    # SKiM63
    # -----------------------------------------------------------------------
    with sub_t3:
        st.subheader("Erweitertes Modell – SKiM63 (Gl. 4.3)")
        st.latex(
            r"N_f = A \cdot (\Delta T_j)^\alpha \cdot "
            r"\exp\!\left(\frac{E_a}{T_{j,m} \cdot k_B}\right) \cdot "
            r"\exp(\beta_1 \Delta T_j + \beta_0) \cdot "
            r"\frac{C + I_{on}^\lambda}{C+1} \cdot f_{\mathrm{diode}}"
        )

        c1, c2 = st.columns(2)
        with c1:
            sk_A      = st.number_input("A (Vorfaktor)",       value=9.3e14, format="%e")
            sk_alpha  = st.number_input("α (ΔT_j-Exponent)",   value=-4.416, step=0.1)
            sk_Ea     = st.number_input("Ea [eV]",              value=0.40,   step=0.05)
            sk_beta1  = st.number_input("β₁ (lin. ΔT-Koeff.)", value=-9.012e-3, format="%.5f", step=1e-4)
            sk_beta0  = st.number_input("β₀ (Offset)",         value=1.942,  step=0.01)
        with c2:
            sk_C      = st.number_input("C (Strom-Norm.)",      value=600.0)
            sk_Ion    = st.number_input("I_on [A]",              value=200.0, min_value=0.1)
            sk_lam    = st.number_input("λ (Strom-Exponent)",    value=0.761, step=0.01)
            sk_fdiode = st.number_input("f_diode",               value=1.0,   min_value=0.0, max_value=2.0, step=0.05,
                                         help="1,0 wenn keine Freilaufdiode berücksichtigt")
            sk_dTj    = st.number_input("ΔT_j [K]   (SKiM63)",  value=60.0,  min_value=1.0)
            sk_Tjm    = st.number_input("T_jm [°C]  (SKiM63)",  value=90.0)

        sk_Nf = skim63_model(sk_dTj, sk_Tjm, sk_A, sk_alpha, sk_Ea,
                              sk_beta1, sk_beta0, sk_C, sk_Ion, sk_lam, sk_fdiode)
        st.metric("N_f (SKiM63)", f"{sk_Nf:,.0f} Zyklen" if 0 < sk_Nf < 1e14 else str(sk_Nf))

        # Gate-Oxid (Arrhenius)
        st.divider()
        st.subheader("Gate-Oxid-Degradation (SiC-MOSFET, Arrhenius)")
        c3, c4 = st.columns(2)
        with c3:
            go_L0   = st.number_input("L0_gate [h]", value=100000, step=5000)
            go_Tref = st.number_input("T_ref,gate [°C]", value=150, min_value=100, max_value=200)
        with c4:
            go_Ea   = st.number_input("Ea,gate [eV]", value=1.0, min_value=0.5, max_value=2.0, step=0.05)
            go_Tjop = st.number_input("T_j,op [°C]", value=120, min_value=25, max_value=200)

        go_life = arrhenius_gate_oxide(go_L0, go_Tref, go_Tjop, go_Ea)
        st.metric("Gate-Oxid Lebensdauer", f"{go_life/8760:.1f} Jahre")

    # -----------------------------------------------------------------------
    # Miner + Rainflow-Klassen
    # -----------------------------------------------------------------------
    with sub_t4:
        st.subheader("Miner-Akkumulation über Rainflow-Klassen")
        st.markdown(
            "Bis zu **8 ΔT_j-Klassen** aus Rainflow-Zählung. "
            "Für jede Klasse wird N_f mit dem **CIPS-08-Modell** berechnet "
            "(Parameter aus Tab 'CIPS-08' werden übernommen)."
        )
        st.info(
            "💡 CIPS-08-Parameter (K, β₁–β₅, I_on, V, D) werden aus dem vorherigen Sub-Tab übernommen. "
            "T_jm wird pro Klasse separat eingegeben."
        )

        n_classes = st.slider("Anzahl Rainflow-Klassen", min_value=1, max_value=8, value=3)

        rows = []
        for i in range(n_classes):
            c1, c2, c3, c4 = st.columns(4)
            dT_i  = c1.number_input(f"ΔT_j,{i+1} [K]",         value=float(20 + i * 20), min_value=0.1, key=f"dT_{i}")
            Tjm_i = c2.number_input(f"T_jm,{i+1} [°C]",         value=80.0 + i * 5,       key=f"Tjm_{i}")
            n_i   = c3.number_input(f"n_{i+1} (Ist-Zyklen)",    value=float(1000 * (3 - i if i < 3 else 1)),
                                     min_value=0.0, key=f"n_{i}")
            Nf_i  = bayerer_cips08(dT_i, Tjm_i, ci_Ion, ci_V, ci_D,
                                    ci_K, ci_b1, ci_b2, ci_b3, ci_b4, ci_b5)
            c4.metric(f"N_f,{i+1}", f"{Nf_i:,.0f}")
            rows.append({"Klasse": i + 1, "ΔT_j [K]": dT_i, "T_jm [°C]": Tjm_i,
                          "n_i (Ist)": n_i, "N_f,i (CM)": Nf_i,
                          "D_i = n_i/N_f,i": n_i / Nf_i if Nf_i > 0 else 0.0})

        df_miner = pd.DataFrame(rows)
        n_arr  = df_miner["n_i (Ist)"].values
        Nf_arr = df_miner["N_f,i (CM)"].values
        D_total = miner_damage(n_arr, Nf_arr)

        # Farbe für D_total
        if D_total < 0.5:
            d_color = FARBE_GUT
            d_label = "unkritisch"
        elif D_total < 0.8:
            d_color = FARBE_MITTEL
            d_label = "Überwachung empfohlen"
        elif D_total < 1.0:
            d_color = FARBE_KRITISCH
            d_label = "kritisch"
        else:
            d_color = FARBE_KRITISCH
            d_label = "⚠️ Lebensdauerende erreicht!"

        st.markdown(
            f"<h3 style='color:{d_color}'>D_total = {D_total:.4f} — {d_label}</h3>",
            unsafe_allow_html=True,
        )

        # Tabelle
        df_display = df_miner.copy()
        df_display["D_i = n_i/N_f,i"] = df_display["D_i = n_i/N_f,i"].map("{:.6f}".format)
        df_display["N_f,i (CM)"] = df_display["N_f,i (CM)"].map("{:,.0f}".format)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Balkendiagramm Schädigung je Klasse
        fig_miner = px.bar(
            df_miner, x="Klasse", y="D_i = n_i/N_f,i",
            color="D_i = n_i/N_f,i",
            color_continuous_scale=["green", "yellow", "red"],
            labels={"D_i = n_i/N_f,i": "Schädigung D_i"},
            title="Schadensanteil je Rainflow-Klasse",
        )
        fig_miner.add_hline(y=1.0, line_dash="dash", line_color="black",
                             annotation_text="D = 1 (Versagen)")
        fig_miner.update_layout(template="plotly_white")
        st.plotly_chart(fig_miner, use_container_width=True)


# ===========================================================================
# TAB 3 – PUNKTEBEWERTUNG
# ===========================================================================
with tab_score:
    st.header("Punktebewertung – qualitative Parameter")
    st.markdown(
        "Bewertung nicht-quantifizierbarer Merkmale mit **0–5 Punkten** je Parameter. "
        "Binäre Kriterien: **0 = fehlt / 1 = vorhanden**."
    )

    # Container für alle Scores (wird auch in Tab 4 verwendet)
    scores: dict[str, ComponentScore] = {}

    # -----------------------------------------------------------------------
    # Transistor-Score
    # -----------------------------------------------------------------------
    with st.expander("💡 Transistor (max. 55 Punkte)", expanded=True):
        sc_t = ComponentScore("Transistor", MAX_SCORES["Transistor"])

        c1, c2 = st.columns(2)
        with c1:
            mat = st.selectbox("Halbleitermaterial", ["SiC", "GaN", "Si (modern)", "Si (klassisch)"])
            sc_t.add("Halbleitermaterial", score_halbleitermaterial(mat))

            attach = st.selectbox("Chip-Attach-Technologie", ["Sintern", "Press-fit", "Weichlot (SnAg)"])
            sc_t.add("Chip-Attach", score_chip_attach(attach))

            bond = st.selectbox("Bond-Technologie", ["Ribbon / Cu (bondlos)", "Cu-Draht", "Al-Draht"])
            sc_t.add("Bond-Technologie", score_bond_tech(bond))

            sub = st.selectbox("Substratmaterial", ["Si₃N₄", "AlN", "Al₂O₃", "andere"])
            sc_t.add("Substrat", score_substrat(sub))

        with c2:
            bp = st.selectbox("Baseplatematerial", ["AlSiC", "Cu", "ohne Baseplate"])
            sc_t.add("Baseplate", score_baseplate(bp))

            aec = st.selectbox("AEC-Qualifikation", ["AEC-Q101 / Q102", "Industrie-Qualifikation", "keine"])
            sc_t.add("AEC-Qual.", score_aec(aec))

            msl = st.selectbox("MSL-Level", [1, 2, 3, 4], index=1)
            sc_t.add("MSL", score_msl(int(msl)))

            qg = st.number_input("Gate-Ladung Q_G [nC]", value=50.0, min_value=0.0)
            sc_t.add("Q_G", score_gate_charge(qg))

        # Binär
        st.markdown("**Binär (0/1)**")
        b1, b2, b3, b4 = st.columns(4)
        sc_t.add("Datenblatt",      float(b1.checkbox("Datenblatt vorhanden", value=True)))
        sc_t.add("LESIT/CIPS",      float(b2.checkbox("LESIT/CIPS-Daten",    value=False)))
        sc_t.add("CM-Parameter",    float(b3.checkbox("CM-Parameter",         value=False)))
        sc_t.add("Therm. RC-Kette", float(b4.checkbox("Therm. RC-Kette",      value=False)))

        scores["Transistor"] = sc_t
        color_t, status_t = sc_t.status
        st.markdown(
            f"**Transistor-Score: {sc_t.total:.0f} / {sc_t.max_points} Punkte "
            f"({sc_t.normalized*100:.0f} %) — "
            f"<span style='color:{color_t}'>{status_t}</span>**",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Kondensator-Score
    # -----------------------------------------------------------------------
    with st.expander("🔋 Kondensator (max. 35 Punkte)"):
        sc_c = ComponentScore("Kondensator", MAX_SCORES["Kondensator"])

        c1, c2 = st.columns(2)
        with c1:
            tol = st.number_input("Kapazitätstoleranz [%]", value=20.0, min_value=0.0)
            sc_c.add("Toleranz", score_kapazitaetstoleranz(tol))

            esr_score = st.slider("ESR-Bewertung (0=sehr hoch / 5=sehr niedrig)", 0, 5, 3)
            sc_c.add("ESR", float(esr_score))

            thb_h = st.number_input("THB-Testdauer [h]", value=1000, min_value=0)
            thb_score = 5 if thb_h >= 2000 else (4 if thb_h >= 1000 else (2 if thb_h >= 500 else 0))
            sc_c.add("THB-Dauer", float(thb_score))

        with c2:
            kk_c = st.selectbox("Klimaklasse", ["55/125/21", "40/85/21", "25/85/21", "unbekannt"])
            kk_s = 5 if kk_c == "55/125/21" else (4 if "85" in kk_c else 2)
            sc_c.add("Klimaklasse", float(kk_s))

            n_parallel = st.number_input("Anzahl Parallel-C", value=1, min_value=1)
            sc_c.add("Redundanz", min(5.0, float(n_parallel)))

        st.markdown("**Binär (0/1)**")
        b1c, b2c, b3c = st.columns(3)
        sc_c.add("Datenblatt-C", float(b1c.checkbox("Datenblatt C", value=True, key="ds_c")))
        sc_c.add("Typbez.-C",    float(b2c.checkbox("Typbezeichnung C", value=True, key="id_c")))
        sc_c.add("Hersteller-C", float(b3c.checkbox("Hersteller bekannt", value=True, key="mfr_c")))

        scores["Kondensator"] = sc_c
        color_c, status_c = sc_c.status
        st.markdown(
            f"**Kondensator-Score: {sc_c.total:.0f} / {sc_c.max_points} "
            f"({sc_c.normalized*100:.0f} %) — "
            f"<span style='color:{color_c}'>{status_c}</span>**",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Leiterplatten-Score
    # -----------------------------------------------------------------------
    with st.expander("🖥️ Leiterplatte PCB (max. 55 Punkte)"):
        sc_p = ComponentScore("Leiterplatte", MAX_SCORES["Leiterplatte"])

        c1, c2 = st.columns(2)
        with c1:
            pcb_mat = st.selectbox("PCB-Material",
                                    ["FR4-Standard", "FR4-High-Tg (≥170°C)", "Polyimid (Rogers)", "CEM"])
            sc_p.add("Material", score_pcb_material(pcb_mat))

            coating = st.selectbox("Coating-Typ", ["kein", "Acryl", "Urethan", "Silikon", "Parylene"])
            sc_p.add("Coating", score_coating_typ(coating))

            coat_proc = st.selectbox("Coating-Verfahren", ["Spray", "Dip", "Selektiv"])
            sc_p.add("Coat.-Verfahren", {"Spray": 2, "Dip": 3, "Selektiv": 5}.get(coat_proc, 2))

            coat_thick = st.number_input("Coating-Dicke [µm]", value=50, min_value=0)
            sc_p.add("Coat.-Dicke",
                      5 if coat_thick >= 100 else (4 if coat_thick >= 50 else (2 if coat_thick >= 25 else 1)))

        with c2:
            n_layer = st.number_input("Lagenzahl", value=4, min_value=1)
            sc_p.add("Lagen", 5 if n_layer >= 6 else (3 if n_layer >= 4 else 1))

            cu_thick = st.selectbox("Kupferdicke [µm]", [35, 70, 105], index=1)
            sc_p.add("Cu-Dicke", {35: 2, 70: 4, 105: 5}.get(cu_thick, 2))

            caf = st.selectbox("CAF-Risiko",
                                ["unbekannt", "hoch", "gering", "kein Risiko (getestet)"])
            sc_p.add("CAF", {"unbekannt": 0, "hoch": 1, "gering": 3, "kein Risiko (getestet)": 5}.get(caf, 0))

            sm_type = st.selectbox("Soldermask-Typ", ["keine", "Epoxid", "LPI (flüssig fotosensitiv)"])
            sc_p.add("Soldermask", {"keine": 0, "Epoxid": 3, "LPI (flüssig fotosensitiv)": 5}.get(sm_type, 0))

        st.markdown("**Binär (0/1)**")
        b1p, b2p = st.columns(2)
        sc_p.add("Datenblatt-PCB", float(b1p.checkbox("Fertigungsspezifikation", value=True, key="ds_p")))
        sc_p.add("IPC-Klasse",     float(b2p.checkbox("IPC-A-610 Nachweis",      value=False, key="ipc_p")))

        scores["Leiterplatte"] = sc_p
        color_p, status_p = sc_p.status
        st.markdown(
            f"**PCB-Score: {sc_p.total:.0f} / {sc_p.max_points} "
            f"({sc_p.normalized*100:.0f} %) — "
            f"<span style='color:{color_p}'>{status_p}</span>**",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # System-Score
    # -----------------------------------------------------------------------
    with st.expander("⚙️ System (max. 30 Punkte)"):
        sc_s = ComponentScore("System", MAX_SCORES["System"])

        c1, c2 = st.columns(2)
        with c1:
            eta_25 = st.number_input("Wirkungsgrad bei 25 °C / Nennlast [%]", value=97.5,
                                      min_value=80.0, max_value=100.0, step=0.1)
            sc_s.add("η (25°C)", score_wirkungsgrad_25(eta_25))

            eta_tmax = st.number_input("Wirkungsgrad bei T_max [%]", value=96.5,
                                        min_value=80.0, max_value=100.0, step=0.1)
            sc_s.add("η (T_max)", 5 if eta_tmax >= 97 else (4 if eta_tmax >= 95 else (2 if eta_tmax >= 93 else 0)))

            kuehl = st.selectbox("Kühlung", ["Naturkonvektion", "Forcierte Luft (Lüfter)", "Flüssigkühlung"])
            sc_s.add("Kühlung", score_kühlung(kuehl))

        with c2:
            i_leak = st.number_input("Leakage-Strom gegen PE [mA]", value=3.0, min_value=0.0)
            sc_s.add("Leakage",
                      5 if i_leak < 1 else (3 if i_leak <= 5 else (1 if i_leak <= 30 else 0)))

            rh_case = st.number_input("Rel. Feuchte im Gehäuse [%]", value=45.0, min_value=0.0, max_value=100.0)
            sc_s.add("RH Gehäuse",
                      5 if rh_case < 40 else (3 if rh_case <= 60 else (1 if rh_case <= 80 else 0)))

            l_loop = st.number_input("Schleifeninduktivität L_loop [nH]", value=30.0, min_value=0.0)
            sc_s.add("L_loop",
                      5 if l_loop < 10 else (3 if l_loop <= 50 else (1 if l_loop <= 200 else 0)))

        # Derating-Anzeige (informativ)
        st.subheader("Derating-Faktoren (informativ)")
        d1, d2, d3 = st.columns(3)
        V_derate  = d1.number_input("k_V = V_op/V_max [-]",    value=0.75, min_value=0.0, max_value=1.5, step=0.01)
        T_derate  = d2.number_input("k_T = T_op/T_max [-]",    value=0.80, min_value=0.0, max_value=1.5, step=0.01)
        I_derate  = d3.number_input("k_I = I_op/I_cont [-]",   value=0.70, min_value=0.0, max_value=1.5, step=0.01)

        for label, ratio, target in [("Spannungs-Derating", V_derate, 0.80),
                                       ("Temperatur-Derating", T_derate, 0.90),
                                       ("Strom-Derating", I_derate, 0.75)]:
            col_d, _, _ = derating_status(ratio, target)
            _, txt = derating_status(ratio, target)
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

    # Platzhalter für Spule und Relais (Basiswerte)
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

    # Sicherstellen dass scores befüllt sind (falls Tab 3 nicht besucht)
    for comp in ["Transistor", "Kondensator", "Leiterplatte", "System", "Spule", "Relais"]:
        if comp not in scores:
            sc_dummy = ComponentScore(comp, MAX_SCORES[comp])
            sc_dummy.add("Platzhalter", 0)
            scores[comp] = sc_dummy

    # Gesamtpunkte
    total_punkte = sum(sc.total for sc in scores.values())
    total_norm   = total_punkte / TOTAL_MAX

    # Ampelfarbe gesamt
    total_color = ampel_farbe(total_norm)
    total_label = ("GUT – Lebensdauer OK" if total_norm >= 0.80
                   else "MITTEL – Maßnahmen prüfen" if total_norm >= 0.50
                   else "KRITISCH – Sofort handeln")

    # Anzeige Gesamt-KPI
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Gesamtpunkte", f"{total_punkte:.0f} / {TOTAL_MAX}")
    kpi_col2.metric("Normierter Score", f"{total_norm*100:.1f} %")
    kpi_col3.markdown(
        f"<div style='background:{total_color};padding:16px;border-radius:8px;"
        f"color:white;font-weight:bold;font-size:1.1em;text-align:center;'>"
        f"{total_label}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Detailtabelle
    rows_g = []
    for comp, sc in scores.items():
        c_color, c_status = sc.status
        rows_g.append({
            "Komponente":      comp,
            "Punkte":          f"{sc.total:.0f} / {sc.max_points}",
            "Score [%]":       f"{sc.normalized*100:.0f} %",
            "Status":          c_status,
        })
    df_g = pd.DataFrame(rows_g)
    st.dataframe(df_g, use_container_width=True, hide_index=True)

    # Radar-Chart
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
    # Zielwert-Ring bei 80 %
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
        title="Komponenten-Score Übersicht (Radar)",
        template="plotly_white",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Balkendiagramm Score je Komponente
    colors = [ampel_farbe(sc.normalized) for sc in scores.values()]
    fig_bar = go.Figure(go.Bar(
        x=komponnenten,
        y=norm_vals,
        marker_color=colors,
        text=[f"{v:.0f} %" for v in norm_vals],
        textposition="outside",
    ))
    fig_bar.add_hline(y=80, line_dash="dash", line_color=FARBE_GUT,    annotation_text="GUT (80%)")
    fig_bar.add_hline(y=50, line_dash="dot",  line_color=FARBE_MITTEL, annotation_text="MITTEL (50%)")
    fig_bar.update_layout(
        title="Score je Komponente",
        yaxis=dict(range=[0, 110], title="Normierter Score [%]"),
        xaxis_title="Komponente",
        template="plotly_white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Handlungsempfehlungen
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

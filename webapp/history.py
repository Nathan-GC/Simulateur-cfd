"""Historique des calculs, comparaison et études paramétriques (balayages, convergence en maillage)."""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``uuid4`` : identifiant d'une étude ; ``SimpleNamespace`` : solveur minimal pour relire un état.
import uuid
from types import SimpleNamespace

# Types des fonctions passées en paramètre.
from collections.abc import Callable

# ``dataclass`` : description d'une étude paramétrique.
from dataclasses import dataclass

# ``Any`` : valeurs de types variés.
from typing import Any

import numpy as np

# Plotly : graphiques de comparaison et d'études.
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit : tableaux, widgets.
import streamlit as st

# Paquet : grille, état, visualisation, références.
from cfd2d import FlowState, StaggeredGrid
from cfd2d import visualization as viz
from cfd2d.validation import CYLINDER_REFERENCES, cavity_centerlines, williamson_strouhal

from . import store
from .analysis import thin_airfoil_cl
from .common import ADVECTION_SCHEMES, CASES, PLOTLY_CONFIG, format_duration, style_plotly, theme

# Paramètres d'étude proposés par cas : nom -> (libellé, valeurs par défaut, type).
STUDY_PARAMS: dict[str, dict[str, tuple[str, str, type]]] = {
    "obstacle": {
        "Re": ("Nombre de Reynolds", "40, 60, 100, 150", float),
        "cells": ("Résolution (cellules par D) : convergence en maillage", "12, 18, 27", int),
        "height": ("Hauteur du domaine (blocage)", "4, 8, 16", float),
        "incidence": ("Incidence α (°) — plaque ou profil", "0, 4, 8, 12", float),
        "advection": ("Schéma d'advection", "upwind, central, quick, tvd", str),
    },
    "cavity": {
        "Re": ("Nombre de Reynolds", "100, 400, 1000", float),
        "N": ("Cellules par côté : convergence en maillage", "32, 48, 72", int),
    },
    "channel": {
        "Re": ("Nombre de Reynolds", "10, 30, 60, 100", float),
        "Ny": ("Cellules sur la hauteur : convergence en maillage", "8, 16, 32", int),
    },
}


@dataclass
class StudySpec:
    """Étude paramétrique : paramètre balayé et valeurs."""

    param: str
    values: list[Any]


def parse_values(text: str, kind: type) -> list[Any]:
    """Valeurs séparées par des virgules (ou points-virgules) ; doublons retirés, ordre conservé."""
    values: list[Any] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        # Nombres : virgule décimale française acceptée (« 0,5 » s'écrit alors « 0.5 »).
        value = item if kind is str else kind(float(item))
        if value not in values:
            values.append(value)
    return values


def study_controls(case: str, params: dict[str, Any]) -> tuple[StudySpec | None, list[str]]:
    """Réglages d'une étude paramétrique (panneau de gauche) : (étude ou None, erreurs)."""
    s = st.session_state
    with st.expander("Étude paramétrique (série de calculs)", expanded=bool(s.get("study_on")), icon=":material/stacked_line_chart:"):
        st.toggle("Lancer une série de calculs au lieu d'un calcul unique", key="study_on")
        if not s.study_on:
            return None, []
        options = dict(STUDY_PARAMS[case])
        # L'incidence n'a de sens que pour une plaque ou un profil.
        if case == "obstacle" and params.get("shape") not in ("rectangle", "naca"):
            options.pop("incidence")
        param = st.selectbox("Paramètre étudié", list(options), format_func=lambda k: options[k][0],
                             key=f"study_param_{case}")
        label, default, kind = options[param]
        key = f"study_values_{case}_{param}"
        s.setdefault(key, default)
        st.text_input("Valeurs (séparées par des virgules)", key=key,
                      help="Les autres paramètres sont ceux du panneau ci-dessous. Une étude en résolution "
                      "doit compter au moins 3 maillages (rapport constant conseillé, ex. ×1,5).")
        errors: list[str] = []
        try:
            values = parse_values(s[key], kind)
        except ValueError:
            return None, ["Valeurs invalides : nombres séparés par des virgules, ex. 40, 60, 100."]
        if param == "advection":
            bad = [v for v in values if v not in ADVECTION_SCHEMES]
            if bad:
                errors.append(f"Schémas inconnus : {', '.join(bad)} (choix : {', '.join(ADVECTION_SCHEMES)}).")
        if len(values) < 2:
            errors.append("Une étude demande au moins deux valeurs.")
        st.caption(f"{len(values)} calculs, enchaînés en arrière-plan ; animation désactivée pour gagner du temps.")
        return (StudySpec(param, values) if not errors else None), errors


def study_params(case: str, params: dict[str, Any], spec: StudySpec) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Paramètres de chaque calcul d'une étude : liste de (paramètres, description de l'étude)."""
    study_id = uuid.uuid4().hex[:8]
    runs = []
    for value in spec.values:
        p = dict(params)
        if spec.param == "Re":
            # Balayage en Re : la viscosité est déduite de Re.
            p.update(visc_mode="re", Re=float(value), nu=None)
        elif spec.param == "N":
            # Cavité : N pair (faces de la grille sur les axes médians).
            value = int(2 * round(value / 2))
            p["N"] = value
        else:
            p[spec.param] = value
        if case == "obstacle":
            p["animation"] = False
        shown = f"{value:g}" if isinstance(value, (int, float)) else value
        runs.append((p, {"id": study_id, "param": spec.param, "value": value, "n": len(spec.values),
                         "label": f"{SHORT_NAMES[spec.param]} = {shown}"}))
    return runs


#: Noms courts des paramètres d'étude (libellés des calculs).
SHORT_NAMES = {"Re": "Re", "cells": "mailles/D", "height": "H", "incidence": "α", "advection": "schéma", "N": "N",
               "Ny": "mailles/H"}


# ================================================================================ historique
def _history_rows(runs: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Colonnes du tableau de l'historique."""
    def metric(meta: dict, name: str) -> float | None:
        value = (meta.get("metrics") or {}).get(name)
        return None if value is None else float(value)

    return {
        "Date": [m.get("created", "") for m in runs],
        "Calcul": [m.get("label", "") for m in runs],
        "Cas": [CASES.get(m.get("case"), m.get("case")) for m in runs],
        "Re": [metric(m, "Re") for m in runs],
        "Cellules": [m.get("cells") for m in runs],
        "St": [metric(m, "St") for m in runs],
        "Cd": [metric(m, "Cd") for m in runs],
        "Cl": [metric(m, "Cl") for m in runs],
        "Ghia u": [metric(m, "ghia_u") for m in runs],
        "Durée": [format_duration(m.get("run_time", float("nan"))) for m in runs],
        "Étude": [(m.get("study") or {}).get("label", "") for m in runs],
    }


def history_tab(on_show: Callable[[str], None], on_reload: Callable[[str], None]) -> None:
    """Historique des calculs enregistrés : sélection, affichage, rechargement, comparaison, suppression."""
    runs = store.list_runs()
    if not runs:
        st.info("Aucun calcul enregistré pour l'instant : chaque calcul terminé est ajouté ici (dossier "
                "`outputs/app_runs/`).", icon=":material/history:")
        return
    plural = "s" if len(runs) > 1 else ""
    st.caption(f"{len(runs)} calcul{plural} enregistré{plural} dans `outputs/{store.RUNS_DIR.name}`. "
               "Sélectionnez des lignes (case à gauche) pour afficher, recharger ou comparer.")
    event = st.dataframe(_history_rows(runs), hide_index=True, on_select="rerun", selection_mode="multi-row",
                         key="history_table", column_config={
                             "Re": st.column_config.NumberColumn(format="%.4g"),
                             "St": st.column_config.NumberColumn(format="%.4f"),
                             "Cd": st.column_config.NumberColumn(format="%.4f"),
                             "Cl": st.column_config.NumberColumn(format="%.4f"),
                             "Ghia u": st.column_config.NumberColumn(format="%.4f"),
                         })
    selected = [runs[i] for i in event.selection.rows] if event is not None else []
    with st.container(horizontal=True):
        single = len(selected) == 1
        st.button("Afficher les résultats", icon=":material/visibility:", disabled=not single,
                  on_click=on_show, args=(selected[0]["run_id"],) if single else None)
        st.button("Recharger ses paramètres", icon=":material/settings_backup_restore:", disabled=not single,
                  on_click=on_reload, args=(selected[0]["run_id"],) if single else None,
                  help="Remplace les réglages du panneau de gauche par ceux de ce calcul.")
        with st.popover("Supprimer", icon=":material/delete:", disabled=not selected):
            st.markdown(f"Supprimer définitivement {len(selected)} calcul(s) du dossier `outputs/app_runs/` ?")
            if st.button("Confirmer la suppression", type="primary", key="confirm_delete"):
                for meta in selected:
                    store.delete_run(meta["run_id"])
                st.rerun()
    if len(selected) >= 2:
        comparison(selected)


def _cavity_profiles(run_id: str):
    """Profils médians d'une cavité enregistrée, relus sans reconstruire tout le solveur."""
    with np.load(store.RUNS_DIR / run_id / "state.npz") as data:
        state = FlowState(data["u"], data["v"], data["p"], t=float(data["t"]))
        grid = StaggeredGrid(float(data["Lx"]), float(data["Ly"]), int(data["Nx"]), int(data["Ny"]))
    # cavity_centerlines n'utilise que la grille et l'état : un objet minimal suffit.
    return cavity_centerlines(SimpleNamespace(grid=grid, state=state))


def comparison(selected: list[dict[str, Any]]) -> None:
    """Comparaison de calculs : courbes superposées (même cas) et tableau des indicateurs."""
    cases = {m["case"] for m in selected}
    st.markdown(f"**Comparaison de {len(selected)} calculs**")
    if len(cases) > 1:
        st.info("Sélectionnez des calculs du même cas d'étude pour superposer leurs courbes.")
        return
    case = cases.pop()
    if case == "obstacle":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Traînée Cd(t)", "Portance Cl(t)"))
        for k, meta in enumerate(selected):
            history = store.load_history(meta["run_id"])
            if history is None:
                continue
            tau = viz.convective_time(history)
            step = max(1, tau.size // 3000)
            color = viz.SERIES[k % len(viz.SERIES)]
            for row, values in ((1, history.cd), (2, history.cl)):
                fig.add_trace(go.Scatter(x=tau[::step], y=values[::step], mode="lines", name=meta["label"],
                                         legendgroup=meta["run_id"], showlegend=row == 1,
                                         line=dict(color=color, width=1.6)), row=row, col=1)
        fig.update_xaxes(title_text="t U / L", row=2, col=1)
        fig.update_yaxes(title_text="Cd", row=1, col=1)
        fig.update_yaxes(title_text="Cl", row=2, col=1)
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(style_plotly(fig, 560, legend=True), theme=None, config=PLOTLY_CONFIG)
    elif case == "cavity":
        fig = make_subplots(rows=1, cols=2, subplot_titles=("u sur l'axe vertical", "v sur l'axe horizontal"))
        for k, meta in enumerate(selected):
            y, u, x, v = _cavity_profiles(meta["run_id"])
            color = viz.SERIES[k % len(viz.SERIES)]
            fig.add_trace(go.Scatter(x=u, y=y, mode="lines", name=meta["label"], line=dict(color=color, width=1.8),
                                     legendgroup=meta["run_id"]), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=v, mode="lines", name=meta["label"], line=dict(color=color, width=1.8),
                                     legendgroup=meta["run_id"], showlegend=False), row=1, col=2)
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(style_plotly(fig, 460, legend=True), theme=None, config=PLOTLY_CONFIG)
    rows = _history_rows(selected)
    st.dataframe({k: rows[k] for k in ("Calcul", "Re", "Cellules", "St", "Cd", "Cl", "Ghia u", "Durée")},
                 hide_index=True)


# ============================================================================ études paramétriques
def richardson(h: np.ndarray, phi: np.ndarray) -> dict[str, float] | None:
    """Extrapolation de Richardson sur les trois maillages les plus fins (Celik et al., 2008).

    ``h`` : tailles de maille ; ``phi`` : grandeur mesurée. Renvoie l'ordre observé ``p``, la
    valeur extrapolée, l'indice de convergence GCI du maillage fin (incertitude relative) et
    ``oscillatory`` (convergence oscillante). None si les écarts sont nuls.
    """
    order = np.argsort(h)[:3]
    h1, h2, h3 = h[order]
    f1, f2, f3 = phi[order]
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0 or r21 <= 1 or r32 <= 1:
        return None
    s = float(np.sign(e32 / e21))
    # Résolution par point fixe de p = |ln|e32/e21| + q(p)| / ln r21, q(p) = ln((r21^p - s)/(r32^p - s)).
    p = abs(np.log(abs(e32 / e21))) / np.log(r21)
    for _ in range(100):
        q = np.log((r21**p - s) / (r32**p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    extrapolated = (r21**p * f1 - f2) / (r21**p - 1.0)
    relative_error = abs((f1 - f2) / f1) if f1 != 0 else float("nan")
    return {"p": float(p), "extrapolated": float(extrapolated), "gci": float(1.25 * relative_error / (r21**p - 1.0)),
            "oscillatory": s < 0}


def _study_metric_options(case: str) -> dict[str, str]:
    """Grandeurs tracées dans une étude : nom de l'indicateur -> libellé."""
    if case == "obstacle":
        return {"St": "Strouhal St", "Cd": "Cd moyen", "Cl": "Cl moyen", "Cl_rms": "Cl rms", "Lr": "Recirculation Lr/L",
                "L_D": "Finesse L/D"}
    if case == "cavity":
        return {"ghia_u": "Écart à Ghia (u)", "ghia_v": "Écart à Ghia (v)", "psi_min": "ψ minimal", "vortex_x": "x du tourbillon",
                "vortex_y": "y du tourbillon"}
    return {"entry_length": "Longueur d'établissement / H", "dpdx_error": "Erreur sur dp/dx", "profile_error": "Écart à Poiseuille"}


def studies_tab() -> None:
    """Études paramétriques enregistrées : tableau, courbes, références et convergence en maillage."""
    runs = [m for m in store.list_runs() if m.get("study")]
    if not runs:
        st.info("Aucune étude enregistrée. Dans le panneau de gauche, ouvrez « Étude paramétrique », choisissez un "
                "paramètre et ses valeurs, puis lancez : les calculs s'enchaînent et leurs résultats s'affichent ici.",
                icon=":material/stacked_line_chart:")
        return
    # Regroupement par identifiant d'étude (le plus récent d'abord).
    studies: dict[str, list[dict]] = {}
    for meta in runs:
        studies.setdefault(meta["study"]["id"], []).append(meta)
    ids = list(studies)
    study_id = st.selectbox("Étude", ids, key="study_choice", format_func=lambda i: (
        f"{studies[i][0]['created'][:16]} · {CASES.get(studies[i][0]['case'])} · {studies[i][0]['study']['param']} "
        f"({len(studies[i])} calcul{'s' if len(studies[i]) > 1 else ''})"))
    members = sorted(studies[study_id], key=lambda m: _sort_key(m["study"]["value"]))
    case, param = members[0]["case"], members[0]["study"]["param"]
    expected = members[0]["study"].get("n", len(members))
    if len(members) < expected:
        plural = "s" if len(members) > 1 else ""
        st.caption(f"{len(members)} calcul{plural} sur {expected} terminé{plural}.")
    options = _study_metric_options(case)
    available = {k: v for k, v in options.items() if any((m.get("metrics") or {}).get(k) is not None for m in members)}
    if not available:
        st.info("Pas encore d'indicateur exploitable pour cette étude.")
        return
    metric = st.selectbox("Grandeur", list(available), format_func=available.get, key="study_metric")
    values = [m["study"]["value"] for m in members]
    y = np.array([np.nan if (m.get("metrics") or {}).get(metric) is None else float(m["metrics"][metric]) for m in members])
    if param in ("cells", "N", "Ny"):
        convergence_view(members, values, y, available[metric])
    else:
        parameter_view(members, case, param, values, y, metric, available[metric])
    rows = _history_rows(members)
    st.dataframe({"Valeur": values, "Calcul": rows["Calcul"], "St": rows["St"], "Cd": rows["Cd"], "Cl": rows["Cl"],
                  "Ghia u": rows["Ghia u"], "Durée": rows["Durée"]}, hide_index=True)


def _sort_key(value: Any) -> tuple:
    """Tri des valeurs d'étude : nombres puis textes."""
    return (0, float(value), "") if isinstance(value, (int, float)) else (1, 0.0, str(value))


def parameter_view(members: list[dict], case: str, param: str, values: list[Any], y: np.ndarray, metric: str,
                   label: str) -> None:
    """Grandeur en fonction du paramètre étudié, avec les références disponibles."""
    c = theme()
    fig = go.Figure()
    if param == "advection":
        fig.add_trace(go.Bar(x=[ADVECTION_SCHEMES.get(v, v).split(" (")[0] for v in values], y=y,
                             marker_color=viz.SERIES[0], hovertemplate="%{x}<br>%{y:.4f}<extra></extra>"))
    else:
        x = np.array(values, dtype=float)
        if param == "height":
            # Hauteur -> blocage (hauteur frontale / H), plus parlant.
            x = np.array([float((m.get("metrics") or {}).get("blockage") or np.nan) * 100 for m in members])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="cfd2d", line=dict(color=viz.SERIES[1], width=2),
                                 marker=dict(size=9, line=dict(color=c["surface"], width=2))))
        if param == "Re" and case == "obstacle" and metric == "St":
            re = np.linspace(max(47.0, x.min()), min(190.0, max(x.max(), 48.0)), 100)
            fig.add_trace(go.Scatter(x=re, y=[williamson_strouhal(r) for r in re], mode="lines", name="Williamson (1989)",
                                     line=dict(color=viz.SERIES[0], width=2, dash="dash")))
            # Strouhal corrigé du blocage par conservation du débit.
            corrected = [(m.get("metrics") or {}).get("St_blockage") for m in members]
            if any(v is not None for v in corrected):
                fig.add_trace(go.Scatter(x=x, y=[np.nan if v is None else v for v in corrected], mode="markers",
                                         name="cfd2d corrigé du blocage, St (1 − β)",
                                         marker=dict(color=viz.SERIES[2], size=8, symbol="diamond")))
        if param == "Re" and case == "obstacle" and metric in ("Cd", "Lr"):
            name = "Cd" if metric == "Cd" else "Lr"
            pts = [(r, v[name]) for r, v in CYLINDER_REFERENCES.items() if name in v]
            for k, (r, (lo, hi)) in enumerate(pts):
                fig.add_trace(go.Scatter(x=[r, r], y=[lo, hi], mode="lines", line=dict(color=viz.SERIES[2], width=5),
                                         name="littérature (milieu infini)", showlegend=k == 0))
        if param == "incidence" and metric == "Cl" and members[0].get("shape") == "naca":
            code = store.load_params(members[0]["run_id"]).get("naca_code", "0012")
            alpha = np.linspace(min(x.min(), 0.0), max(x.max(), 1.0), 50)
            fig.add_trace(go.Scatter(x=alpha, y=[thin_airfoil_cl(code, a) for a in alpha], mode="lines",
                                     name="profils minces (fluide parfait)", line=dict(color=viz.SERIES[0], dash="dash")))
        fig.update_xaxes(title_text={"Re": "Re", "height": "Blocage β (%)", "incidence": "Incidence α (°)"}.get(param, param))
    fig.update_yaxes(title_text=label)
    fig.update_layout(title=dict(text=f"{label} en fonction du paramètre"), legend=dict(x=0.01, y=0.99))
    st.plotly_chart(style_plotly(fig, 420, legend=param != "advection"), theme=None, config=PLOTLY_CONFIG)


def convergence_view(members: list[dict], values: list[Any], y: np.ndarray, label: str) -> None:
    """Convergence en maillage : grandeur en fonction de la taille de maille, extrapolation de Richardson."""
    c = theme()
    # Taille de maille relative : h = 1 / (nombre de cellules par longueur).
    h = 1.0 / np.array(values, dtype=float)
    ok = np.isfinite(y)
    fig = go.Figure(go.Scatter(x=h[ok], y=y[ok], mode="lines+markers", name="cfd2d", line=dict(color=viz.SERIES[1], width=2),
                               marker=dict(size=9, line=dict(color=c["surface"], width=2))))
    result = richardson(h[ok], y[ok]) if ok.sum() >= 3 else None
    if result is not None:
        fig.add_trace(go.Scatter(x=[0.0], y=[result["extrapolated"]], mode="markers", name="extrapolation (h → 0)",
                                 marker=dict(color=viz.SERIES[2], size=12, symbol="star")))
    fig.update_xaxes(title_text="taille de maille h = 1 / (cellules par longueur)", rangemode="tozero")
    fig.update_yaxes(title_text=label)
    fig.update_layout(title=dict(text=f"Convergence en maillage : {label}"), legend=dict(x=0.01, y=0.99))
    st.plotly_chart(style_plotly(fig, 420, legend=True), theme=None, config=PLOTLY_CONFIG)
    if result is None:
        st.info("Il faut au moins trois maillages (et des écarts non nuls) pour estimer l'ordre de convergence.")
        return
    cols = st.columns(3)
    cols[0].metric("Ordre observé p", f"{result['p']:.2f}", border=True,
                   help="Ordre de convergence mesuré sur les trois maillages les plus fins (1 : erreur ∝ h ; 2 : ∝ h²).")
    cols[1].metric("Valeur extrapolée (h → 0)", f"{result['extrapolated']:.4g}", border=True,
                   help="Extrapolation de Richardson : estimation de la valeur à maillage infiniment fin.")
    cols[2].metric("Incertitude GCI (maillage fin)", f"{100 * result['gci']:.2f} %", border=True,
                   help="Grid Convergence Index (Roache) : incertitude relative due au maillage, facteur de sécurité 1,25.")
    if result["oscillatory"]:
        st.caption("Convergence oscillante (les écarts changent de signe) : l'extrapolation est à prendre avec prudence.")
    st.caption("Méthode : Celik et al. (2008), « Procedure for estimation and reporting of uncertainty due to "
               "discretization in CFD applications », J. Fluids Eng. 130. Une frontière en escalier limite souvent l'ordre à 1.")

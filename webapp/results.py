"""Affichage des résultats d'un calcul : indicateurs, lecture commentée et sous-onglets."""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# Dossiers temporaires : les fonctions d'export du paquet écrivent dans des fichiers.
import tempfile

# Types des fonctions passées en paramètre.
from collections.abc import Callable

# Chemins de fichiers portables.
from pathlib import Path

# ``Any`` : valeurs de types variés.
from typing import Any

import numpy as np

# Streamlit : mise en page et widgets.
import streamlit as st

# Paquet : analyse, entrées/sorties, visualisation.
from cfd2d import analytics as an
from cfd2d import io as cio
from cfd2d import visualization as viz

from . import figures
from .analysis import SimulationResult, memo
from .common import FIELD_HELP, FIELDS, PLOTLY_CONFIG, format_duration, human_size, keep, thousands


def _fr(x: float) -> str:
    """Nombre à 3 chiffres significatifs avec virgule décimale (phrases en français)."""
    return f"{x:.3g}".replace(".", ",")


# ================================================================================ en-tête
def show_header(result: SimulationResult, params: dict[str, Any] | None, on_prolong: Callable[[float], None] | None) -> None:
    """Bandeau de fin de calcul, paramètres modifiés depuis, prolongation, messages du solveur."""
    s = result.solver
    origin = " (relu depuis l'historique)" if result.from_disk else ""
    text = (f"**{result.label}**{origin} : {thousands(s.state.step)} pas jusqu'à t = {s.time:.2f} en "
            f"{format_duration(result.run_time)} (grille {s.grid.Nx} × {s.grid.Ny}).")
    if result.stopped:
        st.warning(text + " Calcul arrêté avant la fin.", icon=":material/stop_circle:")
    else:
        st.success(text, icon=":material/check_circle:")
    if params is not None and _comparable(result.params) != _comparable(params):
        st.info("Les réglages actuels diffèrent de ceux de ce calcul : relancez pour mettre les résultats "
                "à jour.", icon=":material/sync_problem:")
    if on_prolong is not None:
        # Prolongation du calcul (moniteurs conservés) : utile quand le régime établi est court.
        with st.container(horizontal=True, vertical_alignment="bottom"):
            extra = st.number_input("Prolonger de (unités de temps)", min_value=1.0, max_value=500.0, value=40.0,
                                    step=10.0, format="%g", key="prolong_time", width=220)
            st.button("Prolonger le calcul", icon=":material/fast_forward:", on_click=on_prolong, args=(extra,),
                      help="Reprend le calcul à partir de son état final : efforts et moyennes continuent de s'accumuler.")
    if result.warnings:
        with st.expander(f"Messages du solveur ({len(result.warnings)})", icon=":material/warning:"):
            for message in result.warnings:
                st.markdown(f"- {message}")


def _comparable(p: dict[str, Any]) -> dict[str, Any]:
    """Paramètres comparés pour signaler un calcul périmé (sans l'aperçu, sans l'image, t_end arrondi)."""
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in p.items() if k not in ("live_preview", "image")}


def show_comments(result: SimulationResult) -> None:
    """Lecture commentée des résultats (régime, références, fiabilité, conseils)."""
    with st.container(border=True):
        st.markdown("**Lecture des résultats**")
        st.markdown("\n".join(f"- {note}" for note in result.comments))


# ============================================================================== indicateurs
def _window_series(history: an.ForceHistory, summary: an.ForceSummary | None, values: np.ndarray) -> np.ndarray:
    """Série restreinte à la fenêtre analysée (sans le pic du démarrage), sous-échantillonnée (~300 points)."""
    sel = history.time >= (summary.t_start if summary is not None else history.time[0])
    series = values[sel]
    return series[:: max(1, series.size // 300)]


#: Largeur des cartes d'indicateurs (pixels) : elles passent à la ligne quand la place manque ; deux
#: cartes et leur espacement (8 px) tiennent sur la largeur utile d'un téléphone (328 px à 360 px).
METRIC_WIDTH = 160


def metric_row():
    """Conteneur horizontal des cartes d'indicateurs (retour à la ligne automatique)."""
    return st.container(horizontal=True, wrap=True, gap="xsmall")


def show_obstacle_metrics(result: SimulationResult) -> None:
    """Indicateurs adaptés à la forme : corps non profilé (Cd, Cl rms, St, Lr) ou élancé (Cl, Cd, L/D, Cm)."""
    summary, history, ex = result.summary, result.history, result.extra
    w = METRIC_WIDTH
    with metric_row():
        if summary is None:
            st.metric("Cd moyen", "—", help="Historique trop court pour une synthèse.", width=w)
            return
        cd = _window_series(history, summary, history.cd)
        cl = _window_series(history, summary, history.cl)
        if result.slender:
            # Corps portant : portance moyenne, traînée, finesse, moment de tangage.
            st.metric("Cl moyen", f"{summary.cl_mean:.3f}", border=True, chart_data=cl, width=w,
                      help="Coefficient de portance moyen sur le régime analysé.")
            st.metric("Cd moyen", f"{summary.cd_mean:.3f}", border=True, chart_data=cd, width=w,
                      help="Traînée moyenne : pression + frottement.")
            st.metric("Finesse L/D", f"{ex.get('lift_to_drag', float('nan')):.2f}", border=True, width=w,
                      help="Rapport portance / traînée : efficacité aérodynamique du profil.")
            st.metric("Cm", f"{summary.cm_mean:.3f}", border=True, width=w,
                      help="Moment de tangage moyen autour du quart de corde, positif à cabrer.")
        else:
            st.metric("Cd moyen", f"{summary.cd_mean:.3f}", border=True, chart_data=cd, width=w,
                      help="Coefficient de traînée moyen sur le régime établi : pression + frottement.")
            st.metric("Cl rms", f"{summary.cl_rms:.3f}", border=True, chart_data=cl, width=w,
                      help="Écart-type de la portance (oscillations du lâcher tourbillonnaire).")
            if summary.unsteady:
                n = summary.n_periods
                # Fiabilité du Strouhal : nombre de périodes analysées (code couleur).
                color = "green" if n >= 10 else "orange" if n >= 5 else "red"
                st.metric("Strouhal", f"{summary.strouhal:.3f}", border=True, delta=f"{n:.1f} périodes".replace(".", ","),
                          delta_color=color, delta_arrow="off", width=w,
                          help="St = f L / U∞ (pic du spectre de Cl). Vert : au moins 10 périodes analysées ; orange : "
                          "5 à 10 ; rouge : moins de 5 (prolonger le calcul).")
            else:
                st.metric("Strouhal", "—", border=True, width=w, help="Écoulement stationnaire : pas d'oscillation.")
            lr = result.recirculation
            st.metric("Recirculation", f"{lr:.2f} L" if np.isfinite(lr) else "—", border=True, width=w,
                      help="Longueur Lr de la bulle de retour derrière l'obstacle (champ moyen), en longueurs de référence.")
        st.metric("max |∇·u|", f"{an.divergence_report(result.solver).history_max:.1e}", border=True, width=w,
                  help="Divergence discrète maximale sur tout le calcul : incompressibilité à la précision machine.")


def show_cavity_metrics(result: SimulationResult) -> None:
    """Indicateurs de la cavité : écarts à Ghia, tourbillon principal, stationnarité."""
    ex = result.extra
    ghia = "ghia_errors" in ex
    w = METRIC_WIDTH
    with metric_row():
        err_u, err_v = ex["ghia_errors"] if ghia else (float("nan"), float("nan"))
        for name, err in (("u", err_u), ("v", err_v)):
            st.metric(f"Écart à Ghia ({name})", f"{err:.4f}" if ghia else "—", border=True, width=w,
                      help=f"Écart absolu maximal de {name}/U sur l'axe médian (tables à Re = 100, 400 et 1000).")
        x, y, psi = ex["vortex"]
        reference = ex.get("ghia_vortex", (None, None, None))
        for label, value, ref in (("Tourbillon : x", f"{x:.3f}", reference[0]), ("Tourbillon : y", f"{y:.3f}", reference[1]),
                                  ("ψ min / (U L)", f"{psi:.4f}", reference[2])):
            st.metric(label, value, delta=f"Ghia {ref:g}" if ref is not None else None, delta_color="off",
                      delta_arrow="off", border=True, width=w,
                      help="Centre du tourbillon principal : minimum de la fonction de courant ψ (aux coins des cellules, "
                      "précision d'une maille).")
        st.metric("Variation de Ec", f"{100 * ex['steadiness']:.2g} %", border=True, width=w,
                  help="Variation relative de l'énergie cinétique sur les 10 % finaux : proche de 0 en régime stationnaire.")


def show_channel_metrics(result: SimulationResult) -> None:
    """Indicateurs du canal : vitesse maximale, gradient de pression, longueur d'établissement."""
    ex, g = result.extra, result.solver.grid
    w = METRIC_WIDTH + 30
    with metric_row():
        u_end = float(np.interp(0.9 * g.Lx, g.x_c, ex["u_axis"]))
        st.metric("u axe / U en sortie", f"{u_end:.4f}", delta=f"{u_end - 1.5:+.4f} vs 1,5", delta_color="off",
                  delta_arrow="off", border=True, width=w, help="Vitesse sur l'axe à x = 0,9 L ; Poiseuille : 1,5.")
        error = (ex["dpdx"] - ex["dpdx_analytic"]) / abs(ex["dpdx_analytic"])
        st.metric("dp/dx (zone établie)", f"{ex['dpdx']:.4g}", delta=f"{100 * error:+.2f} % vs Poiseuille",
                  delta_color="off", delta_arrow="off", border=True, width=w,
                  help=f"Analytique : −12 μ U / H² = {ex['dpdx_analytic']:.4g}.")
        entry = ex["entry_length"]
        st.metric("Établissement", f"{entry:.2f} H" if np.isfinite(entry) else "> L", border=True, width=w,
                  help="Abscisse où la vitesse sur l'axe atteint 99 % de 1,5 U.")
        st.metric("Écart à Poiseuille", f"{ex['profile_error']:.4f}", border=True, width=w,
                  help="max |u/U − 6 η (1 − η)| à la dernière station.")


# ============================================================================ sous-onglets
def lazy_tabs(views: dict[str, Callable[[SimulationResult], None]], result: SimulationResult) -> None:
    """Sous-onglets : seul l'onglet ouvert est calculé (on_change="rerun")."""
    tabs = st.tabs(list(views), key=f"result_tab_{result.case}", on_change="rerun")
    for tab, render in zip(tabs, views.values()):
        if tab.open:
            with tab:
                render(result)


def show_fields(result: SimulationResult) -> None:
    """Explorateur de champs : grandeur, champ instantané ou moyen, cadrage, surcouches, échelle."""
    case = result.case
    c1, c2, c3 = st.columns([1.2, 1.4, 1.4])
    quantity = c1.selectbox("Grandeur", list(FIELDS), format_func=FIELDS.get, **keep(f"field_{case}"))
    sources = {"final": f"Instantané (t = {result.fields.t:.2f})"}
    if result.mean is not None:
        sources["mean"] = "Moyenne temporelle"
        sources["both"] = "Les deux côte à côte"
    if st.session_state.get(f"source_{case}") not in sources:
        st.session_state[f"source_{case}"] = "final"
    source = c2.radio("Champ", list(sources), format_func=sources.get, horizontal=True, **keep(f"source_{case}"))
    frames = figures.frame_options(result)
    if st.session_state.get(f"frame_{case}") not in frames:
        st.session_state[f"frame_{case}"] = next(iter(frames))
    frame_name = c3.radio("Cadrage", list(frames), horizontal=True, **keep(f"frame_{case}"))
    frame = frames[frame_name]
    c1, c2 = st.columns([2, 1])
    overlays = c1.pills("Surcouches", ["streamlines", "vectors", "isolines"], selection_mode="multi",
                        format_func={"streamlines": "lignes de courant (ψ)", "vectors": "vecteurs vitesse",
                                     "isolines": "isovaleurs"}.get, **keep(f"overlays_{case}"))
    manual = c2.toggle("Échelle de couleurs imposée", **keep(f"manual_scale_{case}"))
    s = result.solver
    data, label, kind = figures.field_values(s, result.fields, quantity)
    zrange = None
    if manual:
        lo, hi = figures.auto_range(data, result.fields.solid, kind)
        m1, m2 = st.columns(2)
        zmin = m1.number_input(f"{label} minimal", value=float(lo), format="%.4g", key=f"zmin_{case}_{quantity}")
        zmax = m2.number_input(f"{label} maximal", value=float(hi), format="%.4g", key=f"zmax_{case}_{quantity}")
        zrange = (zmin, zmax) if zmax > zmin else None
    overlays = tuple(overlays or ())
    if source == "both":
        # Même échelle pour les deux cartes (comparaison directe) : celle du champ instantané.
        zrange = zrange or figures.auto_range(data, result.fields.solid, kind)
        left, right = st.columns(2)
        for column, fields, state, name in ((left, result.fields, s.state, "instantané"),
                                            (right, result.mean, result.mean_state, "moyen")):
            with column:
                fig = figures.field_figure(s, fields, quantity, frame, state=state, overlays=overlays, zrange=zrange,
                                           title=f"{FIELDS[quantity]} ({name})")
                st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=f"field_{name}")
    else:
        fields = result.mean if source == "mean" else result.fields
        state = result.mean_state if source == "mean" else s.state
        fig = figures.field_figure(s, fields, quantity, frame, state=state, overlays=overlays, zrange=zrange)
        st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG)
    st.caption(FIELD_HELP[quantity] + " Les lignes de courant sont les isovaleurs de la fonction de courant ψ "
               "(u = ∂ψ/∂y, v = −∂ψ/∂x) : le débit entre deux lignes voisines est constant.")


def show_forces(result: SimulationResult) -> None:
    """Efforts : Cd(t), Cl(t), spectre de Cl et synthèse chiffrée."""
    summary = result.summary
    if summary is None:
        st.info("Historique trop court pour analyser les efforts (allonger la durée simulée).")
        return
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(figures.forces_figure(result.history, summary), theme=None, config=PLOTLY_CONFIG)
    if result.spectrum is not None:
        c2.plotly_chart(figures.spectrum_figure(result), theme=None, config=PLOTLY_CONFIG)
    else:
        c2.info("Écoulement stationnaire : pas de spectre (portance constante).")
    ex = result.extra
    rows = [
        ("Fenêtre d'analyse", f"t ∈ [{summary.t_start:.2f}, {summary.t_end:.2f}]"),
        ("Cd moyen", f"{summary.cd_mean:.4f}"),
        ("↳ dont pression / frottement / convection", f"{summary.cd_pressure_mean:.4f} / {summary.cd_viscous_mean:.4f} / "
                                                     f"{summary.cd_convective_mean:.4f}"),
        ("Cd' rms", f"{summary.cd_rms:.4f}"),
        ("Cl moyen / rms / amplitude", f"{summary.cl_mean:.4f} / {summary.cl_rms:.4f} / {summary.cl_amplitude:.4f}"),
        ("Cm moyen (tangage)", f"{summary.cm_mean:.4f}"),
        ("Blocage β (hauteur frontale / H)", f"{100 * ex.get('blockage', 0.0):.1f} %"),
    ]
    if summary.unsteady:
        rows += [("Strouhal (pic FFT)", f"{summary.strouhal:.4f} ({summary.n_periods:.1f} périodes analysées)"),
                 ("Strouhal (passages par zéro)", f"{ex.get('st_crossing', float('nan')):.4f}"),
                 ("Strouhal corrigé du blocage, St (1 − β)", f"{ex.get('st_blockage', float('nan')):.4f}")]
    if "st_williamson" in ex:
        rows.append(("Strouhal, loi de Williamson (milieu infini)", f"{ex['st_williamson']:.4f}"))
    if "lift_to_drag" in ex:
        rows.append(("Finesse L/D", f"{ex['lift_to_drag']:.3f}"))
    if "cl_thin_airfoil" in ex:
        rows.append(("Cl, théorie des profils minces (fluide parfait)", f"{ex['cl_thin_airfoil']:.3f}"))
    if np.isfinite(summary.cd_cv_mean):
        rows.append(("Cd moyen par bilan de quantité de mouvement", f"{summary.cd_cv_mean:.4f}"))
    st.table({"Grandeur": [r[0] for r in rows], "Valeur": [r[1] for r in rows]}, hide_index=True)
    st.caption("Coefficients rapportés à ½ ρ U∞² L, temps en unités convectives t U∞/L. La zone teintée est le régime "
               "établi détecté automatiquement (extrema de Cl stables à 5 % près) ; le terme « convection » est le flux "
               "de quantité de mouvement absorbé par les faces de l'escalier (artefact discret, proche de 0).")


def show_wall_and_wake(result: SimulationResult) -> None:
    """Pression pariétale (θ ou x/c), profils de sillage et leurs indicateurs."""
    if result.mean_window is not None:
        t0, t1 = result.mean_window
        st.caption(f"Calculés sur le champ moyen, t ∈ [{t0:.1f} ; {t1:.1f}].")
    else:
        st.caption("Calculés sur le champ instantané final.")
    c1, c2 = st.columns(2)
    with c1:
        if result.cp is None:
            st.info("Contour de l'obstacle indisponible : pas de Cp pariétal.")
        else:
            st.plotly_chart(figures.cp_figure(result), theme=None, config=PLOTLY_CONFIG)
            cp = result.cp
            m1, m2, m3 = st.columns(3)
            # Point d'arrêt : pression maximale (et non le point θ = 0, faux pour un corps incliné).
            m1.metric("Cp au point d'arrêt (max)", f"{cp.cp.max():.3f}", help="Théorie (fluide parfait) : 1.")
            m2.metric("Cp minimal", f"{cp.cp.min():.3f}", help="Aspiration maximale.")
            if result.cp_chord is not None:
                k = int(np.argmin(result.cp_chord.cp_upper))
                m3.metric("Pic d'aspiration (extrados)", f"x/c = {result.cp_chord.x_upper[k]:.2f}")
            else:
                # np.interp : Cp au culot (θ = 180°), significatif pour un corps non profilé.
                m3.metric("Cp au culot (θ = 180°)", f"{np.interp(180.0, cp.theta, cp.cp):.3f}")
    with c2:
        if not result.profiles:
            st.info("Aucune station de sillage dans le domaine.")
        else:
            st.plotly_chart(figures.wake_figure(result), theme=None, config=PLOTLY_CONFIG)
            profiles = result.profiles
            st.table({
                "x / L": [f"{p.station:g}" for p in profiles],
                "u axe / U": [f"{p.centerline_velocity:.3f}" for p in profiles],
                "déficit": [f"{p.deficit:.3f}" for p in profiles],
                "demi-largeur / L": [f"{p.half_width:.3f}" for p in profiles],
            }, hide_index=True)


def show_animation(result: SimulationResult) -> None:
    """Animations à la demande (vorticité, vitesse, pression) et curseur temporel sur les instantanés."""
    states = result.snapshots
    fps = result.params.get("fps", 20)
    if not states and not result.animations:
        st.info("Pas d'instantanés : option « Enregistrer la fin du calcul pour l'animer » désactivée.")
        return
    labels = viz.ANIMATED_QUANTITIES
    c1, c2 = st.columns([1, 2])
    quantity = c1.radio("Grandeur animée", list(labels), format_func=labels.get, **keep("anim_quantity"))
    key = (quantity, fps)
    if key not in result.animations and states:
        # Rendu à la demande (environ 0,2 s par image), mémorisé et enregistré avec le calcul.
        with st.spinner(f"Rendu de l'animation ({len(states)} images)…", show_time=True):
            from .runner import render_animation
            from .store import save_run

            animation = render_animation(result, quantity, fps)
            if animation is not None:
                result.animations[key] = animation
                save_run(result)
    if key in result.animations:
        data, mime = result.animations[key]
        if mime == "video/mp4":
            st.video(data, format=mime, loop=True, autoplay=True, muted=True)
        else:
            st.image(data, width="stretch", alt=f"Animation : {labels[quantity].lower()} autour de l'obstacle")
        c2.download_button("Télécharger l'animation", data=data, file_name=f"animation_{quantity}.{mime.split('/')[1]}",
                           mime=mime, on_click="ignore", icon=":material/movie:", key=f"download_anim_{quantity}")
    elif not states:
        st.info("Instantanés non conservés pour ce calcul relu : seule l'animation de la vorticité est disponible.")
    if states:
        # Curseur temporel : carte interactive de n'importe quel instantané.
        st.markdown("**Parcourir la fin du calcul**")
        index = st.slider("Instantané", 0, len(states) - 1, len(states) - 1, key="snapshot_index",
                          format=f"%d / {len(states) - 1}")
        state = states[index]
        s = result.solver
        fields = an.fields_from_arrays(s.grid, s.solid, state.u.astype(float), state.v.astype(float),
                                       state.p.astype(float), state.t)
        field_name = {"vorticity": "vorticity", "speed": "speed", "pressure": "cp"}[quantity]
        fig = figures.field_figure(s, fields, field_name, figures.frame_options(result).get("Sillage (±3 L)"),
                                   title=f"{FIELDS[field_name]} à t = {state.t:.2f}")
        st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG)


def show_method(result: SimulationResult) -> None:
    """La méthode à l'œuvre : un pas de projection décomposé et la contrainte qui fixe Δt."""
    s = result.solver
    parts = memo(result, "anatomy", s.projection_anatomy)
    div_star, div_after = float(np.abs(parts["div_star"][s.fluid]).max()), float(np.abs(parts["div_after"][s.fluid]).max())
    st.markdown(
        "Un pas de temps de la méthode de **projection de Chorin**, rejoué à partir de l'état final "
        f"(prédiction d'Euler, Δt = {_fr(parts['dt'])}) : 1. la vitesse prédite u* ne respecte pas l'incompressibilité ; "
        "2. l'équation de Poisson donne la pression qui la corrige ; 3. après correction, la divergence tombe à "
        "l'erreur d'arrondi."
    )
    cols = st.columns(3)
    cols[0].metric("max |∇·u*| (prédiction)", f"{div_star:.2e}", border=True)
    cols[1].metric("max |∇·uⁿ⁺¹| (après projection)", f"{div_after:.2e}", border=True)
    cols[2].metric("Réduction", f"10^{np.log10(max(div_star, 1e-300) / max(div_after, 1e-300)):.0f}", border=True,
                   help="Rapport des divergences avant et après la projection.")
    frame = figures.frame_options(result)
    st.plotly_chart(figures.anatomy_figure(s, parts, next(iter(frame.values()))), theme=None, config=PLOTLY_CONFIG)
    # Contrainte de stabilité : part de l'advection et de la diffusion dans 1/Δt.
    advective, diffusive = s.stability_rates()
    share = advective / (advective + diffusive)
    st.markdown(
        f"**Pas de temps** : 1/Δt = advection + diffusion = {_fr(advective)} + {_fr(diffusive)} s⁻¹, soit un Δt stable "
        f"de {_fr(s.stable_dt())}. L'advection représente **{100 * share:.0f} %** de la contrainte : le pas est limité "
        f"par le **{'nombre de Courant' if share > 0.5 else 'nombre de Fourier'}**."
        + (" Raffiner la maille divise Δt par 2." if share > 0.5 else
           " Raffiner la maille divise Δt par 4 : la diffusion explicite coûte cher aux faibles Reynolds.")
    )


def show_dashboard(result: SimulationResult) -> None:
    """Tableau de bord du paquet : image (matplotlib) ou rapport interactif (Plotly)."""
    if result.summary is None:
        st.info("Historique trop court pour le tableau de bord.")
        return
    c1, c2 = st.columns(2)
    version = c1.radio("Version", ["png", "plotly"], horizontal=True, key="dashboard_version",
                       format_func={"png": "Image (Matplotlib)", "plotly": "Interactif (Plotly)"}.get)
    if version == "png":
        main = c2.radio("Panneau principal", ["vorticity", "streamlines"], horizontal=True, key="dashboard_main",
                        format_func={"vorticity": "vorticité", "streamlines": "lignes de courant"}.get)
        with st.spinner("Rendu du tableau de bord…"):
            png = memo(result, ("dashboard", main), lambda: figures.dashboard_png(result, main))
        st.image(png, width="stretch", alt="Tableau de bord : champ principal, coefficients d'effort, Cp et profils "
                 "de sillage")
    else:
        stations = tuple(p.station for p in result.profiles) or (1.0,)
        figure = memo(result, "report", lambda: viz.report_figure(result.history, solver=result.solver,
                                                                   mean_fields=result.mean, stations=stations))
        st.plotly_chart(figure, theme=None, config=PLOTLY_CONFIG)


def show_diagnostics(result: SimulationResult) -> None:
    """Santé numérique du calcul : pas de temps, CFL, divergence, énergie, solveur de pression."""
    s = result.solver
    d = s.diagnostics.as_arrays()
    cols = st.columns(4)
    cols[0].metric("Pas de temps", thousands(s.state.step), help="Nombre d'itérations en temps.")
    cols[1].metric("Δt moyen", f"{s.time / max(s.state.step, 1):.2e}")
    cols[2].metric("CFL maximal", f"{d['cfl'].max():.3f}")
    direct = s.config.numerics.pressure_solver == "direct"
    cols[3].metric("Itérations de pression / pas", "direct" if direct else f"{d['pressure_iterations'].mean():.1f}",
                   help="Solveur direct : une descente-remontée par pas (matrice factorisée une seule fois) ; "
                   "solveurs itératifs : nombre moyen d'itérations par résolution.")
    st.plotly_chart(figures.diagnostics_figure(s), theme=None, config=PLOTLY_CONFIG)
    st.caption("Δt suit le critère de stabilité (CFL et Fourier) ; la divergence reste à la précision machine grâce à la "
               "projection ; l'énergie cinétique se stabilise (régime stationnaire) ou oscille (régime périodique).")
    st.code(s.summary(), language=None, wrap_lines=True)


# ================================================================================== exports
def file_bytes(write: Callable[[Path], object], name: str) -> Callable[[], bytes]:
    """Fonction sans argument (pour ``st.download_button``) : écrit le fichier puis renvoie ses octets.

    Le fichier n'est produit qu'au clic (génération différée) dans un dossier temporaire.
    """
    def make() -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            write(path)
            return path.read_bytes()

    return make


def save_diagnostics(solver, path: Path) -> None:
    """Diagnostics de chaque pas en CSV (une colonne par indicateur)."""
    d = solver.diagnostics.as_arrays()
    np.savetxt(path, np.column_stack(list(d.values())), delimiter=",", header=",".join(d), comments="", fmt="%.10g")


def show_exports(result: SimulationResult) -> None:
    """Téléchargements regroupés (données, champs, figures, reprise), avec leur taille approximative."""
    s, r = result.solver, result
    cells, steps = s.grid.Nx * s.grid.Ny, s.state.step
    # Tailles approximatives : VTK binaire ~ 72 octets par cellule ; npz compressé ~ 40 ; CSV ~ 12 par valeur.
    groups: dict[str, list[tuple[str, str, Callable[[Path], object], float]]] = {
        "Données tabulées (CSV)": [("Diagnostics de chaque pas", "diagnostics.csv", lambda path: save_diagnostics(s, path), 84 * steps)],
        "Champs 2D": [
            ("Champs finaux (.npz)", "champs_final.npz", lambda path: cio.save_fields_npz(path, r.fields, Re=s.reynolds), 40 * cells),
            ("Champs finaux (.vtk, ParaView)", "champs_final.vtk", lambda path: cio.save_vtk(path, r.fields, s.grid), 72 * cells),
        ],
        "Reprise du calcul": [("Point de reprise (.npz)", "reprise.npz", lambda path: cio.save_checkpoint(path, s), 24 * cells)],
        "Figures et rapport": [],
    }
    if r.mean is not None:
        groups["Champs 2D"] += [
            ("Champs moyens (.npz)", "champs_moyens.npz", lambda path: cio.save_fields_npz(path, r.mean, Re=s.reynolds), 40 * cells),
            ("Champs moyens (.vtk)", "champs_moyens.vtk", lambda path: cio.save_vtk(path, r.mean, s.grid, title="cfd2d champ moyen"), 72 * cells),
        ]
    if r.history is not None:
        groups["Données tabulées (CSV)"].append(("Efforts Cd(t), Cl(t)", "efforts.csv", lambda path: r.history.save(path),
                                                 120 * r.history.time.size))
    if r.spectrum is not None:
        groups["Données tabulées (CSV)"].append(("Spectre de Cl", "spectre_cl.csv", lambda path: r.spectrum.save(path),
                                                 24 * r.spectrum.frequency.size))
    if r.cp_chord is not None:
        groups["Données tabulées (CSV)"].append(("Cp le long de la corde", "cp_corde.csv", lambda path: r.cp_chord.save(path), 2e4))
    elif r.cp is not None:
        groups["Données tabulées (CSV)"].append(("Cp pariétal", "cp.csv", lambda path: r.cp.save(path), 2e4))
    if r.profiles:
        groups["Données tabulées (CSV)"].append(("Profils de sillage", "sillage.csv",
                                                 lambda path: an.save_wake_profiles(path, r.profiles), 4e4 * len(r.profiles)))
    if r.summary is not None:
        stations = tuple(p.station for p in r.profiles) or (1.0,)
        groups["Figures et rapport"] += [
            ("Tableau de bord (.png)", "tableau_de_bord.png",
             lambda path: path.write_bytes(memo(r, ("dashboard", "vorticity"), lambda: figures.dashboard_png(r, "vorticity"))), 1.2e6),
            ("Rapport interactif (.html)", "rapport.html", lambda path: viz.interactive_report(
                r.history, path, solver=s, mean_fields=r.mean, stations=stations, include_plotlyjs=True), 5e6),
        ]
    st.caption("Fichiers produits par les fonctions d'export du paquet au moment du clic (tailles approximatives).")
    for title, items in groups.items():
        if not items and not (title == "Figures et rapport" and r.animations):
            continue
        st.markdown(f"**{title}**")
        with st.container(horizontal=True, wrap=True):
            for label, name, write, size in items:
                # on_click="ignore" : le téléchargement ne relance pas la page.
                st.download_button(f"{label} · ≈ {human_size(size)}", data=file_bytes(write, name), file_name=name,
                                   on_click="ignore", icon=":material/download:", key=f"download_{name}")
            if title == "Figures et rapport":
                for (quantity, fps), (data, mime) in r.animations.items():
                    st.download_button(f"Animation {viz.ANIMATED_QUANTITIES[quantity].lower()} · {human_size(len(data))}",
                                       data=data, file_name=f"animation_{quantity}.{mime.split('/')[1]}", mime=mime,
                                       on_click="ignore", icon=":material/movie:", key=f"download_anim_{quantity}_{fps}")
    st.caption("Les fichiers .vtk s'ouvrent dans ParaView (Open puis Apply) ; le point de reprise se recharge avec "
               "`cfd2d.io.load_checkpoint` pour prolonger le calcul hors de l'application. Tous les calculs sont aussi "
               "enregistrés dans `outputs/app_runs/`.")


# =============================================================================== validations
def show_cavity_validation(result: SimulationResult) -> None:
    """Cavité : profils médians comparés à Ghia et al."""
    st.plotly_chart(figures.cavity_figure(result), theme=None, config=PLOTLY_CONFIG)
    if "ghia_errors" not in result.extra:
        st.info("Tables de Ghia et al. (1982) disponibles à Re = 100, 400 et 1000.")
    st.caption("Profils de u sur l'axe vertical médian et de v sur l'axe horizontal médian (faces de la grille MAC "
               "exactement sur les axes, sans interpolation). Pour Re = 400, la valeur publiée de v en x = 0,9063 est une "
               "coquille connue : elle est écartée.")


def show_channel_profiles(result: SimulationResult) -> None:
    """Canal : développement du profil de Poiseuille."""
    st.plotly_chart(figures.channel_figure(result), theme=None, config=PLOTLY_CONFIG)
    st.caption("Le profil d'entrée se déforme sous l'effet du frottement pariétal jusqu'à la parabole de Poiseuille "
               "u = 6 U η (1 − η) ; la pression décroît alors linéairement avec la pente −12 μ U / H².")


# ================================================================================ assemblage
def show_results(result: SimulationResult, params: dict[str, Any] | None,
                 on_prolong: Callable[[float], None] | None = None) -> None:
    """Résultats complets d'un calcul : en-tête, indicateurs, lecture commentée, sous-onglets."""
    show_header(result, params, on_prolong)
    if result.case == "obstacle":
        show_obstacle_metrics(result)
        views = {"🌀 Champs": show_fields, "📈 Efforts": show_forces, "🧱 Paroi & sillage": show_wall_and_wake,
                 "🎞️ Animation": show_animation, "🔬 Méthode": show_method, "🖼️ Tableau de bord": show_dashboard,
                 "🩺 Diagnostics": show_diagnostics, "💾 Exports": show_exports}
    elif result.case == "cavity":
        show_cavity_metrics(result)
        views = {"✅ Validation (Ghia)": show_cavity_validation, "🌀 Champs": show_fields, "🔬 Méthode": show_method,
                 "🩺 Diagnostics": show_diagnostics, "💾 Exports": show_exports}
    else:
        show_channel_metrics(result)
        views = {"📐 Profils de Poiseuille": show_channel_profiles, "🌀 Champs": show_fields, "🔬 Méthode": show_method,
                 "🩺 Diagnostics": show_diagnostics, "💾 Exports": show_exports}
    show_comments(result)
    lazy_tabs(views, result)

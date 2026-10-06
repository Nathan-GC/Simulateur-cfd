"""Graphiques de l'application : cartes de champs, efforts, spectre, paroi, sillage, validations.

Les graphiques sont interactifs (Plotly : zoom, survol des valeurs) et suivent la charte du
paquet en thème clair ou sombre ; seul le tableau de bord complet reste une image matplotlib
(fonction ``plot_dashboard`` du paquet).
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``math`` : sous-échantillonnage des grandes grilles.
import math

# ``Any`` : valeurs de types variés.
from typing import Any

import numpy as np

# Plotly : graphiques interactifs ; ``figure_factory.create_quiver`` : champ de vecteurs.
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paquet : état, analyse, visualisation, données de référence.
from cfd2d import FlowState, NavierStokesSolver
from cfd2d import analytics as an
from cfd2d import visualization as viz
from cfd2d.validation import GHIA

from .analysis import SimulationResult
from .common import FIELDS, SHAPES, figure_png, plotly_colorscale, style_plotly, theme
from .runner import MATPLOTLIB_LOCK

# Nombre maximal de valeurs d'une carte envoyées au navigateur (sous-échantillonnage au-delà).
MAX_MAP_VALUES = 150_000


# ===================================================================== cartes de champs
def frame_options(result: SimulationResult) -> dict[str, tuple[float, float, float, float] | None]:
    """Cadrages proposés : libellé -> (xmin, xmax, ymin, ymax) ou None (domaine entier)."""
    solver = result.solver
    if not solver.obstacles:
        return {"Domaine entier": None}
    g, L = solver.grid, solver.L_ref
    xc, yc = solver.obstacles[0].center
    # Proche de l'obstacle : 1,5 L en amont, 5 L en aval, ±1,5 L (borné au domaine).
    near = (max(xc - 1.5 * L, 0.0), min(xc + 5.0 * L, g.Lx), max(yc - 1.5 * L, 0.0), min(yc + 1.5 * L, g.Ly))
    return {"Sillage (±3 L)": viz.default_view(solver), "Proche de l'obstacle": near, "Domaine entier": None}


def field_values(solver: NavierStokesSolver, fields: an.FlowFields, quantity: str) -> tuple[np.ndarray, str, str]:
    """Champ adimensionné à afficher : (valeurs (Nx, Ny), libellé, type d'échelle de couleurs)."""
    U, L = solver.U_ref, solver.L_ref
    if quantity == "vorticity":
        return fields.vorticity / (U / L), "ω L/U", "diverging"
    if quantity == "speed":
        return fields.speed / U, "‖u‖/U", "sequential"
    if quantity == "u":
        return fields.u / U, "u/U", "diverging"
    if quantity == "v":
        return fields.v / U, "v/U", "diverging"
    if quantity == "cp":
        # Pression de référence : moyenne à l'entrée (ou 0 sans entrée, cavité).
        p_ref = an.reference_pressure(solver, pressure=fields.p)
        return (fields.p - p_ref) / (0.5 * solver.rho * U**2), "Cp", "diverging"
    return fields.q_criterion / (U / L) ** 2, "Q (L/U)²", "diverging"


def auto_range(data: np.ndarray, solid: np.ndarray, kind: str) -> tuple[float, float]:
    """Bornes automatiques de l'échelle de couleurs (symétriques pour une grandeur signée)."""
    if kind == "diverging":
        # Percentile 99,5 de |valeur| dans le fluide : robuste aux pics pariétaux.
        vmax = viz.symmetric_limit(np.where(solid, np.nan, data))
        return -vmax, vmax
    return 0.0, float(np.nanmax(np.where(solid, np.nan, data)))


def field_figure(
    solver: NavierStokesSolver,
    fields: an.FlowFields,
    quantity: str,
    frame: tuple[float, float, float, float] | None,
    *,
    state: FlowState | None = None,
    overlays: tuple[str, ...] = (),
    zrange: tuple[float, float] | None = None,
    title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Carte interactive d'un champ avec le masque de l'obstacle, son contour exact et des surcouches.

    ``overlays`` : ``'streamlines'`` (isovaleurs de la fonction de courant ψ, calculée à partir
    des vitesses aux faces de ``state``), ``'vectors'`` (vecteurs vitesse), ``'isolines'``
    (isovaleurs du champ affiché). ``zrange`` impose les bornes de l'échelle de couleurs.
    """
    c = theme()
    g = solver.grid
    data, label, kind = field_values(solver, fields, quantity)
    zmin, zmax = zrange if zrange is not None else auto_range(data, fields.solid, kind)
    # Valeur neutre dans le solide (le masque gris la recouvre).
    data = np.where(fields.solid, 0.0, data)
    # Sous-échantillonnage des très grandes grilles (taille des données envoyées au navigateur).
    stride = max(1, math.ceil(math.sqrt(g.Nx * g.Ny / MAX_MAP_VALUES)))
    x, y = g.x_c[::stride], g.y_c[::stride]
    fig = go.Figure(go.Heatmap(
        x=x, y=y, z=data[::stride, ::stride].T, colorscale=plotly_colorscale(kind), zsmooth="best",
        zmin=zmin, zmax=zmax, zmid=0.0 if kind == "diverging" else None,
        colorbar=dict(title=dict(text=label), thickness=12),
        hovertemplate="x = %{x:.3f}<br>y = %{y:.3f}<br>" + label + " = %{z:.4f}<extra></extra>",
    ))
    if "isolines" in overlays:
        # Isovaleurs du champ affiché (15 niveaux entre les bornes de l'échelle).
        fig.add_trace(go.Contour(
            x=x, y=y, z=data[::stride, ::stride].T, contours=dict(coloring="none", start=zmin, end=zmax,
                                                                   size=(zmax - zmin) / 15 or 1.0),
            line=dict(color=c["ink2"], width=0.6), showscale=False, hoverinfo="skip",
        ))
    if "streamlines" in overlays and state is not None:
        # Lignes de courant = isovaleurs de ψ (u = ∂ψ/∂y, v = -∂ψ/∂x), aux coins des cellules.
        psi = an.stream_function(np.asarray(state.u, dtype=float), np.asarray(state.v, dtype=float), g)
        lo, hi = np.percentile(psi, [1, 99])
        fig.add_trace(go.Contour(
            x=g.x_f[::stride], y=g.y_f[::stride], z=psi[::stride, ::stride].T,
            contours=dict(coloring="none", start=lo, end=hi, size=(hi - lo) / 28 or 1.0),
            line=dict(color=c["ink"], width=0.8), showscale=False, hoverinfo="skip",
        ))
    if "vectors" in overlays:
        # Vecteurs vitesse sur une grille d'environ 28 flèches dans la largeur du cadrage.
        xmin, xmax, ymin, ymax = frame or (0.0, g.Lx, 0.0, g.Ly)
        step = max(1, int((xmax - xmin) / g.dx / 28))
        ii = np.arange(step // 2, g.Nx, step)
        jj = np.arange(step // 2, g.Ny, step)
        X, Y = np.meshgrid(g.x_c[ii], g.y_c[jj], indexing="ij")
        U = np.where(fields.solid[np.ix_(ii, jj)], 0.0, fields.u[np.ix_(ii, jj)]) / solver.U_ref
        V = np.where(fields.solid[np.ix_(ii, jj)], 0.0, fields.v[np.ix_(ii, jj)]) / solver.U_ref
        # Longueur d'une flèche de vitesse U : 90 % de l'espacement des flèches.
        quiver = ff.create_quiver(X.ravel(), Y.ravel(), U.ravel(), V.ravel(), scale=0.9 * step * g.dx,
                                  arrow_scale=0.3, line=dict(color=c["ink2"], width=1.0), hoverinfo="skip")
        fig.add_traces(quiver.data)
    if fields.solid.any():
        # Masque en marches d'escalier, tel que le voit le solveur (NaN = transparent).
        mask = np.where(fields.solid, 1.0, np.nan)[::stride, ::stride]
        fig.add_trace(go.Heatmap(x=x, y=y, z=mask.T, colorscale=[[0, c["solid"]], [1, c["solid"]]], showscale=False,
                                 hoverinfo="skip"))
    for obstacle in solver.obstacles:
        outline = obstacle.outline()
        if outline is not None:
            # Contour exact refermé (premier point répété).
            fig.add_trace(go.Scatter(x=np.append(outline[:, 0], outline[0, 0]), y=np.append(outline[:, 1], outline[0, 1]),
                                     mode="lines", line=dict(color=c["ink"], width=1.2), hoverinfo="skip"))
    xmin, xmax, ymin, ymax = frame or (0.0, g.Lx, 0.0, g.Ly)
    # Repère orthonormé : l'échelle y est liée à l'échelle x (scaleanchor), cadre réduit au besoin.
    fig.update_xaxes(range=[xmin, xmax], constrain="domain", title_text="x")
    fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1, constrain="domain", title_text="y")
    # Hauteur estimée pour une largeur utile d'environ 900 pixels.
    height = height or int(np.clip(900 * (ymax - ymin) / (xmax - xmin), 260, 640)) + 90
    fig.update_layout(title=dict(text=title or f"{FIELDS[quantity]} à t = {fields.t:.2f}"))
    return style_plotly(fig, height)


# ===================================================================== efforts et spectre
def forces_figure(history: an.ForceHistory, summary: an.ForceSummary | None) -> go.Figure:
    """Cd et Cl en fonction du temps convectif t U/L (deux graphiques superposés, axe x partagé)."""
    tau = viz.convective_time(history)
    scale = history.reference_velocity / history.reference_length
    # Au plus ~5000 points par courbe (affichage fluide).
    k = max(1, tau.size // 5000)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Traînée Cd(t)", "Portance Cl(t)"))
    for row, (values, name, color, symmetric) in enumerate(
        ((history.cd, "Cd", viz.SERIES[0], False), (history.cl, "Cl", viz.SERIES[1], True)), 1
    ):
        fig.add_trace(go.Scatter(x=tau[::k], y=values[::k], mode="lines", line=dict(color=color, width=1.8), name=name,
                                 hovertemplate="t U/L = %{x:.2f}<br>" + name + " = %{y:.4f}<extra></extra>"),
                      row=row, col=1)
        if summary is not None:
            # Fenêtre d'analyse (régime établi) : voile très léger de la couleur de la série.
            fig.add_vrect(x0=summary.t_start * scale, x1=summary.t_end * scale, fillcolor=color, opacity=0.08,
                          line_width=0, row=row, col=1)
        # Cadrage vertical sans le pic du démarrage (symétrique pour une portance oscillante).
        fig.update_yaxes(title_text=name, range=list(viz.focus_limits(history.time, values, symmetric)), row=row, col=1)
    fig.update_xaxes(title_text="t U / L", row=2, col=1)
    return style_plotly(fig, 470)


def spectrum_figure(result: SimulationResult) -> go.Figure:
    """Spectre d'amplitude de Cl en fonction du Strouhal, pic annoté, loi de Williamson."""
    c = theme()
    spectrum, history = result.spectrum, result.history
    L, U = history.reference_length, history.reference_velocity
    st_axis = spectrum.frequency * L / U
    peak = spectrum.peak_frequency * L / U
    shown = st_axis <= max(3.0 * peak, 0.5)
    fig = go.Figure(go.Scatter(x=st_axis[shown], y=spectrum.amplitude[shown], mode="lines",
                               line=dict(color=viz.SERIES[1], width=2),
                               hovertemplate="St = %{x:.4f}<br>amplitude = %{y:.4f}<extra></extra>"))
    # Pic annoté (anneau de la couleur du fond).
    fig.add_trace(go.Scatter(x=[peak], y=[spectrum.peak_amplitude], mode="markers+text", text=[f"St = {peak:.3f}"],
                             textposition="middle right", textfont=dict(color=c["ink"]), hoverinfo="skip",
                             marker=dict(color=viz.SERIES[1], size=9, line=dict(color=c["surface"], width=2))))
    if "st_williamson" in result.extra:
        # Référence en milieu infini (trait vertical pointillé).
        fig.add_vline(x=result.extra["st_williamson"], line=dict(color=c["muted"], dash="dot", width=1.2),
                      annotation_text="Williamson", annotation_position="top left",
                      annotation_font=dict(color=c["ink2"], size=11))
    fig.update_xaxes(title_text="St = f L / U")
    fig.update_yaxes(title_text="Amplitude de Cl", rangemode="tozero")
    fig.update_layout(title=dict(text="Spectre de Cl"))
    return style_plotly(fig, 470)


# ========================================================================= paroi et sillage
def cp_figure(result: SimulationResult) -> go.Figure:
    """Cp pariétal : en fonction de θ (corps non profilé) ou de x/c, extrados et intrados (corps élancé)."""
    c = theme()
    fig = go.Figure()
    if result.cp_chord is not None:
        cpc = result.cp_chord
        for x, values, side, color in ((cpc.x_upper, cpc.cp_upper, "extrados", viz.SERIES[0]),
                                       (cpc.x_lower, cpc.cp_lower, "intrados", viz.SERIES[1])):
            fig.add_trace(go.Scatter(x=x, y=values, mode="lines", line=dict(color=color, width=2), name=side,
                                     hovertemplate="x/c = %{x:.3f}<br>Cp = %{y:.3f}<extra>" + side + "</extra>"))
        # Convention aérodynamique : axe inversé (aspiration vers le haut).
        fig.update_yaxes(title_text="Cp (axe inversé)", autorange="reversed")
        fig.update_xaxes(title_text="x / c (depuis le bord d'attaque)", range=[0.0, 1.0])
        fig.update_layout(title=dict(text="Pression pariétale Cp(x/c)"))
        return style_plotly(fig, 420, legend=True)
    cp = result.cp
    fig.add_trace(go.Scatter(x=cp.theta, y=cp.cp, mode="lines", line=dict(color=viz.SERIES[0], width=2),
                             hovertemplate="θ = %{x:.0f}°<br>Cp = %{y:.3f}<extra></extra>"))
    # Points remarquables : arrêt (Cp maximal), aspiration maximale (Cp minimal).
    for k, text, position in ((int(np.argmax(cp.cp)), "arrêt", "top right"), (int(np.argmin(cp.cp)), "Cp min", "bottom right")):
        fig.add_trace(go.Scatter(x=[cp.theta[k]], y=[cp.cp[k]], mode="markers+text", text=[f"{text} {cp.cp[k]:.2f}"],
                                 textposition=position, textfont=dict(color=c["ink"]), hoverinfo="skip",
                                 marker=dict(color=viz.SERIES[0], size=8, line=dict(color=c["surface"], width=2))))
    fig.add_hline(y=0.0, line=dict(color=c["axis"], width=1))
    fig.update_xaxes(title_text="θ (° depuis l'amont)", tickvals=[0, 90, 180, 270, 360], range=[cp.theta[0], cp.theta[0] + 360])
    fig.update_yaxes(title_text="Cp")
    fig.update_layout(title=dict(text="Pression pariétale Cp(θ)"))
    return style_plotly(fig, 420)


def wake_figure(result: SimulationResult) -> go.Figure:
    """Profils u(y)/U aux stations du sillage (une couleur par station, légende)."""
    c = theme()
    fig = go.Figure()
    origin = "depuis l'arrière" if result.extra.get("wake_origin") == "rear" else "depuis le centre"
    for k, profile in enumerate(result.profiles):
        fig.add_trace(go.Scatter(x=profile.u, y=profile.eta, mode="lines", name=f"x/L = {profile.station:g}",
                                 line=dict(color=viz.SERIES[k % len(viz.SERIES)], width=2),
                                 hovertemplate="u/U = %{x:.3f}<br>(y-y_c)/L = %{y:.2f}<extra>" + f"x/L = {profile.station:g}</extra>"))
    # u = 0 : frontière de l'écoulement de retour.
    fig.add_vline(x=0.0, line=dict(color=c["axis"], width=1))
    fig.update_xaxes(title_text="u / U")
    fig.update_yaxes(title_text="(y − y_c) / L", range=[-3, 3])
    fig.update_layout(title=dict(text=f"Profils de sillage (stations {origin})"))
    return style_plotly(fig, 420, legend=len(result.profiles) > 1)


# ============================================================================ diagnostics
def diagnostics_figure(solver: NavierStokesSolver) -> go.Figure:
    """Indicateurs numériques enregistrés à chaque pas : Δt, CFL, max|∇·u|, énergie cinétique."""
    d = solver.diagnostics.as_arrays()
    # Au plus ~4000 points par courbe ; temps convectif t U / L.
    k = max(1, d["time"].size // 4000)
    t = d["time"][::k] * solver.U_ref / solver.L_ref
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.18, horizontal_spacing=0.09, subplot_titles=(
        "Pas de temps Δt", "Nombre de Courant effectif", "Divergence maximale |∇·u|", "Énergie cinétique Ec"))
    series = (
        (d["dt"], 1, 1, viz.SERIES[0]), (d["cfl"], 1, 2, viz.SERIES[1]),
        # np.maximum(..., 1e-20) : pas de zéro sur l'échelle logarithmique.
        (np.maximum(d["divergence_max"], 1e-20), 2, 1, viz.SERIES[2]), (d["kinetic_energy"], 2, 2, viz.SERIES[3]),
    )
    for values, row, col, color in series:
        fig.add_trace(go.Scatter(x=t, y=values[::k], mode="lines", line=dict(color=color, width=1.8),
                                 hovertemplate="t U/L = %{x:.3f}<br>%{y:.4g}<extra></extra>"), row=row, col=col)
        fig.update_xaxes(title_text="t U / L", row=row, col=col)
    fig.update_yaxes(type="log", exponentformat="e", row=2, col=1)
    return style_plotly(fig, 560)


def anatomy_figure(solver: NavierStokesSolver, parts: dict[str, Any], frame: tuple[float, float, float, float] | None) -> go.Figure:
    """Anatomie d'un pas de projection : log10|∇·u*|, pression de Poisson, log10|∇·u^(n+1)|."""
    c = theme()
    g = solver.grid
    stride = max(1, math.ceil(math.sqrt(g.Nx * g.Ny / (MAX_MAP_VALUES / 3))))
    x, y = g.x_c[::stride], g.y_c[::stride]
    fluid = solver.fluid
    # Trois cartes l'une sous l'autre (lisibles dans une colonne étroite).
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.09, subplot_titles=(
        "1. Prédiction : log₁₀ |∇·u*|", "2. Poisson : pression p (écart à la moyenne)",
        "3. Après correction : log₁₀ |∇·uⁿ⁺¹|"))
    # Divergences en échelle logarithmique commune (de 10⁻¹⁷ à 1) ; zéro exact remplacé par 10⁻¹⁷.
    for row, key in ((1, "div_star"), (3, "div_after")):
        log_div = np.log10(np.maximum(np.abs(np.where(fluid, parts[key], 0.0)), 1e-17))
        fig.add_trace(go.Heatmap(x=x, y=y, z=log_div[::stride, ::stride].T, zmin=-17, zmax=0,
                                 colorscale=plotly_colorscale("sequential"), zsmooth="best", showscale=row == 1,
                                 colorbar=dict(title=dict(text="log₁₀|∇·u|"), thickness=12, len=0.3, y=0.86),
                                 hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<br>log₁₀|∇·u| = %{z:.1f}<extra></extra>"),
                      row=row, col=1)
    # Pression : échelle divergente symétrique (le gris neutre est la moyenne).
    p = np.where(fluid, parts["pressure"] - parts["pressure"][fluid].mean(), 0.0)
    vmax = viz.symmetric_limit(np.where(fluid, p, np.nan))
    fig.add_trace(go.Heatmap(x=x, y=y, z=p[::stride, ::stride].T, zmin=-vmax, zmax=vmax, zmid=0.0,
                             colorscale=plotly_colorscale("diverging"), zsmooth="best",
                             colorbar=dict(title=dict(text="p"), thickness=12, len=0.3, y=0.5),
                             hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<br>p = %{z:.4f}<extra></extra>"),
                  row=2, col=1)
    xmin, xmax, ymin, ymax = frame or (0.0, g.Lx, 0.0, g.Ly)
    for row in (1, 2, 3):
        # Obstacles en gris (contour exact) sur chaque carte ; repère orthonormé.
        for obstacle in solver.obstacles:
            outline = obstacle.outline()
            if outline is not None:
                fig.add_trace(go.Scatter(x=np.append(outline[:, 0], outline[0, 0]), y=np.append(outline[:, 1], outline[0, 1]),
                                         mode="lines", fill="toself", fillcolor=c["solid"], line=dict(width=0),
                                         hoverinfo="skip"), row=row, col=1)
        fig.update_xaxes(range=[xmin, xmax], constrain="domain", row=row, col=1)
        fig.update_yaxes(range=[ymin, ymax], scaleanchor=f"x{row if row > 1 else ''}", scaleratio=1, constrain="domain",
                         row=row, col=1)
    panel = int(np.clip(760 * (ymax - ymin) / (xmax - xmin), 150, 320))
    return style_plotly(fig, 3 * panel + 150)


# ===================================================================== cavité et canal
def cavity_figure(result: SimulationResult) -> go.Figure:
    """Profils médians de la cavité, comparés aux tables de Ghia et al. (1982) si disponibles."""
    c = theme()
    y, u, x, v = result.extra["centerlines"]
    N = result.solver.grid.Nx
    ref = GHIA.get(result.extra.get("ghia_Re"))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("u sur l'axe vertical x = L/2", "v sur l'axe horizontal y = L/2"))
    fig.add_trace(go.Scatter(x=u, y=y, mode="lines", name=f"cfd2d ({N}×{N})", line=dict(color=viz.SERIES[0], width=2),
                             legendgroup="sim", hovertemplate="u/U = %{x:.4f}<br>y/L = %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=v, mode="lines", name=f"cfd2d ({N}×{N})", line=dict(color=viz.SERIES[0], width=2),
                             legendgroup="sim", showlegend=False,
                             hovertemplate="x/L = %{x:.3f}<br>v/U = %{y:.4f}<extra></extra>"), row=1, col=2)
    if ref is not None:
        # Référence : cercles creux de la deuxième couleur de série.
        marker = dict(color="rgba(0,0,0,0)", size=8, line=dict(color=viz.SERIES[1], width=2))
        label = f"Ghia et al. (1982), Re = {result.extra['ghia_Re']}"
        fig.add_trace(go.Scatter(x=ref["u"], y=ref["y"], mode="markers", name=label, marker=marker, legendgroup="ref",
                                 hovertemplate="u/U = %{x:.5f}<br>y/L = %{y:.4f}<extra>Ghia</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ref["x"], y=ref["v"], mode="markers", name=label, marker=marker, legendgroup="ref",
                                 showlegend=False, hovertemplate="x/L = %{x:.4f}<br>v/U = %{y:.5f}<extra>Ghia</extra>"),
                      row=1, col=2)
    fig.update_xaxes(title_text="u / U", row=1, col=1)
    fig.update_yaxes(title_text="y / L", row=1, col=1)
    fig.update_xaxes(title_text="x / L", row=1, col=2)
    fig.update_yaxes(title_text="v / U", row=1, col=2)
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0.0, font=dict(color=c["ink2"])))
    return style_plotly(fig, 460, legend=ref is not None)


def channel_figure(result: SimulationResult) -> go.Figure:
    """Canal : profils u(y) aux stations, vitesse sur l'axe et pression le long du canal."""
    c = theme()
    ex, g = result.extra, result.solver.grid
    eta = ex["eta"]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07,
                        subplot_titles=("Profils de vitesse", "Vitesse sur l'axe", "Pression sur l'axe"))
    # Profils aux stations (une couleur par station) et parabole de Poiseuille (tirets).
    for k, (station, u) in enumerate(ex["profiles"]):
        fig.add_trace(go.Scatter(x=u, y=eta, mode="lines", name=f"x/H = {station:g}",
                                 line=dict(color=viz.SERIES[k % len(viz.SERIES)], width=2),
                                 hovertemplate="u/U = %{x:.3f}<br>y/H = %{y:.3f}<extra>" + f"x/H = {station:g}</extra>"),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=6.0 * eta * (1.0 - eta), y=eta, mode="lines", name="Poiseuille",
                             line=dict(color=c["ink"], width=1.4, dash="dash")), row=1, col=1)
    x = g.x_c / g.Ly
    # Vitesse sur l'axe : tend vers 1,5 U (maximum de la parabole).
    fig.add_trace(go.Scatter(x=x, y=ex["u_axis"], mode="lines", name="u axe", showlegend=False,
                             line=dict(color=viz.SERIES[0], width=2),
                             hovertemplate="x/H = %{x:.2f}<br>u/U = %{y:.4f}<extra></extra>"), row=1, col=2)
    fig.add_hline(y=1.5, line=dict(color=c["muted"], dash="dot"), row=1, col=2)
    if np.isfinite(ex["entry_length"]):
        fig.add_vline(x=ex["entry_length"], line=dict(color=viz.SERIES[1], width=1.2), row=1, col=2,
                      annotation_text=f"99 % : {ex['entry_length']:.2f} H", annotation_font=dict(color=c["ink"]))
    # Pression sur l'axe et pente de Poiseuille passant par le dernier point.
    fig.add_trace(go.Scatter(x=x, y=ex["p_axis"], mode="lines", name="p (cfd2d)", showlegend=False,
                             line=dict(color=viz.SERIES[0], width=2),
                             hovertemplate="x/H = %{x:.2f}<br>p = %{y:.4f}<extra></extra>"), row=1, col=3)
    fig.add_trace(go.Scatter(x=x, y=ex["p_axis"][-1] + ex["dpdx_analytic"] * (g.x_c - g.x_c[-1]), mode="lines",
                             name="pente de Poiseuille", line=dict(color=viz.SERIES[1], width=1.4, dash="dash")),
                  row=1, col=3)
    for col, (xt, yt) in enumerate((("u / U", "y / H"), ("x / H", "u axe / U"), ("x / H", "p")), 1):
        fig.update_xaxes(title_text=xt, row=1, col=col)
        fig.update_yaxes(title_text=yt, row=1, col=col)
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0.0))
    return style_plotly(fig, 440, legend=True)


# =========================================================================== tableau de bord
def dashboard_png(result: SimulationResult, main: str) -> bytes:
    """Tableau de bord complet du paquet (15 x 14 pouces), en PNG."""
    stations = tuple(p.station for p in result.profiles) or (1.0,)
    with MATPLOTLIB_LOCK:
        fig = viz.plot_dashboard(result.solver, result.history, mean_fields=result.mean, stations=stations, main=main,
                                 title=f"{SHAPES[result.params['shape']]} : Re = {result.solver.reynolds:.4g}")
        return figure_png(fig, dpi=100)

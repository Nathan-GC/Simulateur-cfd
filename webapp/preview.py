"""Schéma du domaine (avant le calcul) et aperçu en direct (pendant le calcul)."""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``Any`` : valeurs de types variés.
from typing import Any

import numpy as np

# Streamlit : mise en cache des schémas.
import streamlit as st

# Contexte de style matplotlib, palette à deux couleurs, couleurs RGBA, cadre du domaine.
from matplotlib import rc_context
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as FramePatch

# Pillow : agrandissement de l'aperçu et dessin du contour exact de l'obstacle.
from PIL import Image, ImageDraw

# Paquet : conditions aux limites, grille, masque, analyse et charte graphique.
from cfd2d import Inlet, NavierStokesSolver, NoSlipWall, Outlet, SimulationConfig, StaggeredGrid, build_mask
from cfd2d import analytics as an
from cfd2d import visualization as viz

from .common import PROFILES, figure_png, thousands
from .params import build_config

# Bornes fixes des échelles de couleurs de l'aperçu en direct (pas de « clignotement ») :
# vorticité ω L/U dans [-3, 3], norme de la vitesse ‖u‖/U dans [0, 1.6].
LIVE_SCALE = {"vorticity": 3.0, "speed": 1.6}


# ======================================================================== schéma du domaine
def boundary_style(condition: Any) -> tuple[str, dict[str, Any]]:
    """Libellé de légende et style de trait d'une condition aux limites."""
    if isinstance(condition, Inlet):
        return f"Entrée {PROFILES[condition.profile].lower()}", dict(color=viz.SERIES[0], lw=3.0)
    if isinstance(condition, Outlet):
        return f"Sortie {'convective' if condition.kind == 'convective' else 'Neumann'}", dict(
            color=viz.SERIES[2], lw=3.0, ls="--")
    if isinstance(condition, NoSlipWall):
        if condition.velocity != 0:
            return "Paroi mobile", dict(color=viz.SERIES[1], lw=3.5)
        return "Paroi adhérente", dict(color=viz.INK_SECONDARY, lw=3.5)
    return "Paroi glissante", dict(color=viz.MUTED, lw=2.0, ls=":")


def draw_boundaries(ax, config: SimulationConfig) -> None:
    """Trace les quatre côtés du domaine selon leur condition aux limites (légende au-dessus)."""
    d, bc = config.domain, config.boundaries
    Lx, Ly = d.Lx, d.Ly
    # Segment (x, y) de chaque côté.
    segments = {"west": ([0, 0], [0, Ly]), "east": ([Lx, Lx], [0, Ly]), "south": ([0, Lx], [0, 0]),
                "north": ([0, Lx], [Ly, Ly])}
    labels: list[str] = []
    for side, (xs, ys) in segments.items():
        label, style = boundary_style(getattr(bc, side))
        # Une seule entrée de légende par type de condition (label=None : pas de doublon).
        ax.plot(xs, ys, solid_capstyle="butt", zorder=4, label=None if label in labels else label, **style)
        if label not in labels:
            labels.append(label)
    # Légende en une ligne au-dessus du schéma (bbox_to_anchor en coordonnées d'axes).
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=len(labels), handlelength=2.2, fontsize=8.5,
              borderaxespad=0.3)
    # Flèches du profil d'entrée (quiver : champ de vecteurs ; longueur proportionnelle au profil).
    if isinstance(bc.west, Inlet):
        y = np.linspace(0.06, 0.94, 9) * Ly
        speed = bc.west.profile_values(y, Ly, 1.0, 1e9) / bc.west.peak_factor()
        ax.quiver(np.zeros_like(y), y, speed * 0.07 * Lx, 0 * y, angles="xy", scale_units="xy", scale=1,
                  color=viz.SERIES[0], width=0.004, zorder=5)
    # Flèches du couvercle mobile.
    if isinstance(bc.north, NoSlipWall) and bc.north.velocity != 0:
        x = np.linspace(0.15, 0.75, 4) * Lx
        ax.quiver(x, np.full_like(x, 0.97 * Ly), np.full_like(x, 0.1 * Lx), 0 * x, angles="xy", scale_units="xy",
                  scale=1, color=viz.SERIES[1], width=0.006, zorder=5)


@st.cache_data(show_spinner=False, max_entries=48)
def domain_preview(key: dict[str, Any]) -> tuple[bytes, bytes | None]:
    """Schéma du domaine et zoom sur le maillage autour de l'obstacle, en PNG (deux images).

    ``key`` : paramètres géométriques (:func:`webapp.params.geometry_key`) ; deux images séparées
    se placent côte à côte sur un grand écran et l'une sous l'autre sur un téléphone.
    """
    config = build_config(key)
    d = config.domain
    grid = StaggeredGrid(d.Lx, d.Ly, d.Nx, d.Ny)
    # Masque binaire tel que le solveur le construira (même sur-échantillonnage).
    solid = build_mask(grid, config.obstacles, config.numerics.mask_supersampling)
    aspect = d.Ly / d.Lx
    with rc_context(viz.STYLE):
        # Taille en pouces proche de la taille affichée : textes lisibles ; largeur réduite pour un
        # domaine haut (cavité).
        panel_h = float(np.clip(6.0 * aspect, 0.9, 4.4))
        panel_w = min(6.0, panel_h / aspect)
        fig = Figure(figsize=(panel_w + 0.7, panel_h + 1.3), layout="constrained")
        ax = fig.subplots()
        # Fond du domaine, obstacles (contour exact rempli) et conditions aux limites.
        ax.add_patch(FramePatch((0, 0), d.Lx, d.Ly, facecolor="#f2f6fc", edgecolor="none", zorder=0))
        viz.draw_obstacles(ax, config.obstacles, grid, solid)
        draw_boundaries(ax, config)
        ax.set_aspect("equal")
        ax.grid(False)
        # Petite marge autour du domaine (traits des parois entièrement visibles).
        ax.set_xlim(-0.01 * d.Lx, 1.01 * d.Lx)
        ax.set_ylim(-0.02 * d.Ly, 1.02 * d.Ly)
        # Titre ; séparateur de milliers : espace simple (présente dans toutes les polices).
        fig.suptitle(f"Domaine {d.Lx:g} × {d.Ly:g} · grille {d.Nx} × {d.Ny} = {thousands(d.Nx * d.Ny, ' ')} cellules",
                     fontsize=10, color=viz.INK)
        domain_png = figure_png(fig, dpi=130)
        if not config.obstacles:
            return domain_png, None
        # Zoom : obstacle plus une marge (au moins 3 mailles).
        ob = config.obstacles[0]
        xmin, xmax, ymin, ymax = ob.bounds
        margin = 0.2 * max(xmax - xmin, ymax - ymin) + 3 * max(grid.dx, grid.dy)
        i0, i1 = max(int((xmin - margin) / grid.dx), 0), min(int(np.ceil((xmax + margin) / grid.dx)), grid.Nx)
        j0, j1 = max(int((ymin - margin) / grid.dy), 0), min(int(np.ceil((ymax + margin) / grid.dy)), grid.Ny)
        zoom_aspect = (j1 - j0) * grid.dy / ((i1 - i0) * grid.dx)
        fig = Figure(figsize=(3.4, float(np.clip(3.4 * zoom_aspect, 1.6, 4.4)) + 0.6), layout="constrained")
        ax = fig.subplots()
        # pcolormesh : une case par cellule (bords tracés : le maillage est visible s'il n'est pas trop fin).
        ax.pcolormesh(
            grid.x_f[i0 : i1 + 1], grid.y_f[j0 : j1 + 1], solid[i0:i1, j0:j1].T.astype(float),
            cmap=ListedColormap(["#f2f6fc", "#b9b8b0"]), vmin=0, vmax=1,
            edgecolors=viz.GRID_COLOR if (i1 - i0) < 90 else "none", linewidth=0.4,
        )
        # Contour exact de la géométrie (le solveur voit l'escalier gris).
        outline = ob.outline()
        if outline is not None:
            closed = np.vstack([outline, outline[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=viz.INK, lw=1.2)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlim(grid.x_f[i0], grid.x_f[i1])
        ax.set_ylim(grid.y_f[j0], grid.y_f[j1])
        ax.set_title(f"Maillage : {thousands(int(solid.sum()), ' ')} cellules solides", fontsize=9.5)
        return domain_png, figure_png(fig, dpi=130)


# =========================================================================== aperçu en direct
def live_frame(solver: NavierStokesSolver, quantity: str, max_width: int = 1000, max_height: int = 420) -> np.ndarray:
    """Image RGB rapide (sans matplotlib) du champ courant, à échelle de couleurs fixe."""
    state = solver.state
    if quantity == "vorticity":
        # Vorticité aux coins, adimensionnée par U/L ; échelle fixe [-3, 3].
        data = an.vorticity_nodes(state.u, state.v, solver.grid, solver.solid) / (solver.U_ref / solver.L_ref)
        rgba = viz.DIVERGING(np.clip(0.5 + 0.5 * data / LIVE_SCALE["vorticity"], 0.0, 1.0))
    else:
        # Norme de la vitesse aux centres, rapportée à la vitesse de référence ; échelle fixe [0, 1.6].
        uc, vc = solver.cell_velocity()
        rgba = viz.SEQUENTIAL(np.clip(np.hypot(uc, vc) / solver.U_ref / LIVE_SCALE["speed"], 0.0, 1.0))
    # Octets RGB ; transposition (lignes = y) et retournement vertical (y vers le haut).
    pixels = np.ascontiguousarray((255 * rgba[..., :3]).astype(np.uint8).transpose(1, 0, 2)[::-1])
    # Agrandissement bilinéaire à la taille d'affichage (proportions du domaine conservées).
    g = solver.grid
    scale = min(max_width / g.Lx, max_height / g.Ly)
    size = (max(1, round(g.Lx * scale)), max(1, round(g.Ly * scale)))
    image = Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR)
    # Obstacles dessinés par leur contour exact (couvre les coins de l'escalier du masque).
    draw = ImageDraw.Draw(image)
    solid = tuple(int(255 * c) for c in to_rgba(viz.SOLID_COLOR)[:3])
    for obstacle in solver.obstacles:
        outline = obstacle.outline()
        if outline is not None:
            # Coordonnées physiques -> pixels (axe vertical inversé : ligne 0 en haut de l'image).
            points = [(float(x) / g.Lx * size[0], (1.0 - float(y) / g.Ly) * size[1]) for x, y in outline]
            draw.polygon(points, fill=solid)
    return np.asarray(image)


def live_caption(quantity: str) -> str:
    """Légende de l'échelle de couleurs de l'aperçu en direct."""
    if quantity == "vorticity":
        return "vorticité ω L/U, échelle fixe [−3, 3] (bleu : rotation horaire, rouge : trigonométrique)"
    return "vitesse ‖u‖/U, échelle fixe [0 ; 1,6] (clair : lent, foncé : rapide)"

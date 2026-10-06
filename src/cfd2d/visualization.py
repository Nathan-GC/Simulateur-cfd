"""Visualisation : tableau de bord (PNG), animations (MP4/GIF), rapport interactif (HTML).

* :func:`plot_dashboard` : carte de vorticité (ou lignes de courant) avec l'obstacle,
  ``Cd(t)`` et ``Cl(t)``, spectre de ``Cl`` et nombre de Strouhal, ``Cp(θ)``, profils de
  sillage et indicateurs clés ;
* :class:`FrameRecorder` + :func:`animate` : animation de la vorticité (ou de la vitesse)
  en MP4 (ffmpeg, via ffmpeg-python) ou en GIF (Pillow) ;
* :func:`interactive_report` : les mêmes graphiques en HTML interactif (Plotly).

Charte graphique : fond clair, encre sombre, grille discrète en traits pleins et fins ;
séries catégorielles dans un ordre fixe ; échelle divergente bleu/rouge à milieu gris neutre
pour les grandeurs signées (vorticité) et séquentielle bleue pour les normes (vitesse).
Les figures sont créées sans pyplot (``matplotlib.figure.Figure``) : le module fonctionne
quel que soit le backend graphique et ne laisse pas de fenêtres ouvertes.
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# Journalisation.
import logging

# ``shutil.which`` : recherche d'un exécutable (ffmpeg) dans le PATH.
import shutil

# Types génériques (séquence de stations, itérateur d'images).
from collections.abc import Iterator, Sequence

# Chemins de fichiers portables.
from pathlib import Path

import numpy as np

# ``rc_context`` : applique temporairement un jeu de paramètres de style matplotlib.
from matplotlib import rc_context

# Moteur de rendu « Agg » (images matricielles) : produit les pixels des animations.
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ``LinearSegmentedColormap`` : palette de couleurs continue définie par des couleurs clés.
from matplotlib.colors import LinearSegmentedColormap

# ``Figure`` : figure matplotlib indépendante de pyplot.
from matplotlib.figure import Figure

# Outils d'analyse (champs dérivés, efforts, profils...).
from .analytics import (
    ChordwiseDistribution,
    FlowFields,
    ForceHistory,
    ForceSummary,
    Spectrum,
    SurfaceDistribution,
    WakeProfile,
    chord_line,
    chordwise_pressure,
    compute_fields,
    divergence_report,
    recirculation_length,
    reference_pressure,
    surface_pressure,
    vorticity_nodes,
    wake_profiles,
)

# Obstacles (contours à dessiner).
from .geometry import Obstacle

# Grille décalée.
from .grid import StaggeredGrid

# Vitesses au centre des cellules (animation de la norme de la vitesse).
from .operators import cell_centered_velocity

# Solveur et état de l'écoulement (instantanés animés après coup).
from .solver import FlowState, NavierStokesSolver

# Journal du module.
logger = logging.getLogger(__name__)

# ======================================================================= charte graphique
# Surface des graphiques (fond), encres du texte (principale, secondaire, atténuée),
# grille et axes (un cran seulement plus foncés que le fond : discrets).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
# Couleurs catégorielles, à utiliser dans cet ordre fixe (bleu, orange, turquoise, jaune...).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
# Obstacles : gris neutre (pas une couleur de série).
SOLID_COLOR = "#52514e"
# Couleurs clés de l'échelle divergente : bleu foncé -> gris neutre -> rouge foncé.
DIVERGING_COLORS = ("#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a39c", "#e34948", "#8a1f1e")
# Échelle divergente (vorticité : sens de rotation opposés, zéro en gris).
DIVERGING = LinearSegmentedColormap.from_list("cfd2d_diverging", DIVERGING_COLORS)
# Échelle séquentielle à une seule teinte (bleu clair -> bleu foncé) pour les normes affichées
# en aplats (le bleu très clair des faibles valeurs se fond volontairement dans le fond).
SEQUENTIAL = LinearSegmentedColormap.from_list("cfd2d_sequential", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
# Même teinte pour des traits fins (lignes de courant) : départ plus foncé pour rester visible.
SEQUENTIAL_LINES = LinearSegmentedColormap.from_list("cfd2d_sequential_lines", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# Paramètres matplotlib appliqués pendant la création des figures (rc_context).
STYLE = {
    # Fonds.
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    # Axes : bords discrets, pas de cadre haut/droite.
    "axes.edgecolor": AXIS_COLOR,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Textes : titres à l'encre principale, étiquettes à l'encre secondaire.
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.labelcolor": INK_SECONDARY,
    "axes.labelsize": 9.5,
    "text.color": INK,
    # Grille : traits pleins et fins, sous les données.
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    # Graduations.
    "xtick.color": AXIS_COLOR,
    "ytick.color": AXIS_COLOR,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    # Courbes : traits fins aux extrémités arrondies.
    "lines.linewidth": 1.4,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    # Légende sans cadre.
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK_SECONDARY,
    # Police sans empattement du système.
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
}


# ======================================================================= utilitaires
def corner_solid(solid: np.ndarray) -> np.ndarray:
    """Coins de cellules entièrement entourés de solide, forme ``(Nx+1, Ny+1)``."""
    # np.pad(..., mode="edge") recopie les bords : chaque coin a ainsi quatre cellules voisines.
    padded = np.pad(solid, 1, mode="edge")
    # Coin solide si ses quatre cellules voisines le sont (& = ET logique).
    return padded[:-1, :-1] & padded[1:, :-1] & padded[:-1, 1:] & padded[1:, 1:]


def symmetric_limit(values: np.ndarray, percentile: float = 99.5) -> float:
    """Borne symétrique de l'échelle de couleurs (percentile de |valeurs|, robuste aux pics)."""
    # Valeurs finies seulement (les NaN marquent le solide).
    finite = np.abs(values[np.isfinite(values)])
    # Aucune valeur : borne arbitraire.
    if finite.size == 0:
        return 1.0
    # np.percentile : valeur sous laquelle se trouvent 99.5 % des données.
    return float(max(np.percentile(finite, percentile), 1e-12))


def draw_obstacles(ax, obstacles: Sequence[Obstacle], grid: StaggeredGrid, solid: np.ndarray) -> None:
    """Dessine les obstacles : contour exact si connu, sinon contour du masque binaire."""
    # Faut-il tracer le masque (obstacle sans contour connu) ?
    use_mask = not obstacles and bool(solid.any())
    for obstacle in obstacles:
        outline = obstacle.outline()
        if outline is None:
            use_mask = True
            continue
        # ax.fill : polygone plein ; zorder=3 le place au-dessus de la carte de couleurs.
        ax.fill(outline[:, 0], outline[:, 1], color=SOLID_COLOR, linewidth=0, zorder=3)
    if use_mask:
        # contourf : remplissage de la zone où le masque vaut 1 (niveaux entre 0.5 et 1.5).
        ax.contourf(grid.x_c, grid.y_c, solid.T.astype(float), levels=[0.5, 1.5], colors=[SOLID_COLOR], zorder=3)


def _colorbar(ax, mappable, label: str) -> None:
    """Barre de couleurs discrète attachée à ``ax``."""
    # fraction/pad : largeur et écart de la barre ; shrink : hauteur relative.
    bar = ax.figure.colorbar(mappable, ax=ax, fraction=0.025, pad=0.01, shrink=0.9)
    # Titre de la barre, cadre masqué, graduations aux couleurs de la charte.
    bar.set_label(label, color=INK_SECONDARY)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=AXIS_COLOR, labelcolor=INK_SECONDARY, labelsize=8)


def _frame_axes(ax, grid: StaggeredGrid, view: tuple[float, float, float, float] | None) -> None:
    """Axes d'une carte 2D : repère orthonormé, pas de grille, cadrage."""
    # Même échelle en x et en y (les formes ne sont pas déformées).
    ax.set_aspect("equal")
    # Pas de grille sur une carte de couleurs.
    ax.grid(False)
    # Cadrage : vue demandée ou domaine entier.
    xmin, xmax, ymin, ymax = view or (0.0, grid.Lx, 0.0, grid.Ly)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def field_image(
    ax,
    data: np.ndarray,
    grid: StaggeredGrid,
    solid: np.ndarray,
    obstacles: Sequence[Obstacle],
    *,
    kind: str = "vorticity",
    vmax: float | None = None,
    view: tuple[float, float, float, float] | None = None,
    colorbar: bool = True,
):
    """Affiche un champ 2D avec les obstacles.

    ``kind`` : ``'vorticity'`` (aux coins, échelle divergente), ``'speed'`` (aux centres,
    échelle séquentielle) ou ``'pressure'`` (coefficient de pression aux centres, divergente).
    """
    if kind == "pressure":
        # Coefficient de pression aux centres ; valeur neutre (0) dans le solide.
        data = np.where(solid, 0.0, data)
        extent = (0.0, grid.Lx, 0.0, grid.Ly)
        # Échelle divergente symétrique : surpression en rouge, dépression en bleu.
        vmax = vmax or symmetric_limit(data)
        norm = {"cmap": DIVERGING, "vmin": -vmax, "vmax": vmax}
        label = "Cp"
    elif kind == "vorticity":
        # Vorticité aux coins ; valeur neutre (0, gris) dans le solide : les coins de l'escalier
        # non recouverts par le contour exact de l'obstacle restent ainsi discrets.
        data = np.where(corner_solid(solid), 0.0, data)
        # Les pixels de l'image sont centrés sur les coins : étendue décalée d'une demi-maille.
        extent = (-0.5 * grid.dx, grid.Lx + 0.5 * grid.dx, -0.5 * grid.dy, grid.Ly + 0.5 * grid.dy)
        # Échelle divergente symétrique autour de 0.
        vmax = vmax or symmetric_limit(data)
        norm = {"cmap": DIVERGING, "vmin": -vmax, "vmax": vmax}
        label = "ω L / U"
    else:
        # Norme de la vitesse aux centres des cellules (nulle dans le solide).
        data = np.where(solid, 0.0, data)
        extent = (0.0, grid.Lx, 0.0, grid.Ly)
        # Échelle séquentielle de 0 au maximum.
        vmax = vmax or float(np.nanmax(data))
        norm = {"cmap": SEQUENTIAL, "vmin": 0.0, "vmax": vmax}
        label = "‖u‖ / U"
    # imshow attend (lignes = y, colonnes = x) : transposition ; origin="lower" place y = 0 en bas ;
    # interpolation bilinéaire pour un rendu lissé.
    image = ax.imshow(data.T, origin="lower", extent=extent, interpolation="bilinear", **norm)
    # Obstacles par-dessus.
    draw_obstacles(ax, obstacles, grid, solid)
    _frame_axes(ax, grid, view)
    if colorbar:
        _colorbar(ax, image, label)
    return image


# ============================================================ graphiques élémentaires
def plot_vorticity(
    ax,
    solver: NavierStokesSolver,
    *,
    vmax: float | None = None,
    view: tuple[float, float, float, float] | None = None,
    colorbar: bool = True,
):
    """Carte de vorticité adimensionnée ``ω L / U`` de l'état courant."""
    # Vorticité aux coins (condition miroir aux parois des obstacles).
    omega = vorticity_nodes(solver.state.u, solver.state.v, solver.grid, solver.solid)
    # Adimensionnement par U/L.
    scale = solver.U_ref / solver.L_ref
    return field_image(
        ax, omega / scale, solver.grid, solver.solid, solver.obstacles, vmax=vmax, view=view, colorbar=colorbar
    )


def plot_streamlines(
    ax,
    solver: NavierStokesSolver,
    *,
    fields: FlowFields | None = None,
    density: float = 1.6,
    view: tuple[float, float, float, float] | None = None,
    colorbar: bool = True,
):
    """Lignes de courant colorées par la norme de la vitesse (champ courant ou ``fields``)."""
    f = compute_fields(solver) if fields is None else fields
    g = solver.grid
    # Vitesses masquées dans le solide : les lignes de courant s'y arrêtent. streamplot attend des
    # tableaux (y, x) : transposition.
    u = np.ma.masked_array(f.u, mask=f.solid).T
    v = np.ma.masked_array(f.v, mask=f.solid).T
    # Couleur des lignes : norme de la vitesse adimensionnée.
    speed = (np.hypot(f.u, f.v) / solver.U_ref).T
    # streamplot : intègre et trace les lignes de courant ; density règle leur espacement.
    lines = ax.streamplot(
        g.x_c, g.y_c, u, v, color=speed, cmap=SEQUENTIAL_LINES, density=density, linewidth=0.8, arrowsize=0.7
    )
    draw_obstacles(ax, solver.obstacles, g, solver.solid)
    _frame_axes(ax, g, view)
    if colorbar:
        # lines.lines : collection de segments portant l'échelle de couleurs.
        _colorbar(ax, lines.lines, "‖u‖ / U")
    return lines


def focus_limits(t: np.ndarray, values: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    """Bornes verticales ignorant le pic du démarrage impulsif (5 % initiaux du temps)."""
    # Échantillons après les premiers 5 % de la durée.
    keep = t >= t[0] + 0.05 * (t[-1] - t[0])
    selected = values[keep] if keep.sum() > 1 else values
    lo, hi = float(selected.min()), float(selected.max())
    if symmetric and lo < 0.0 < hi:
        # Axe symétrique autour de 0 quand le signal change de signe (portance oscillante d'un corps
        # symétrique) ; une portance de signe constant (profil) garde un cadrage serré.
        bound = max(abs(lo), abs(hi), 1e-6)
        lo, hi = -bound, bound
    # Marge de 10 % (au moins 1e-6 pour un signal constant).
    margin = max(0.1 * (hi - lo), 1e-6)
    return lo - margin, hi + margin


# Ancien nom (privé), conservé pour compatibilité.
_focus_limits = focus_limits


def convective_time(history: ForceHistory) -> np.ndarray:
    """Temps adimensionné ``t U / L`` des séries d'efforts (égal au temps physique si U = L = 1)."""
    return history.time * history.reference_velocity / history.reference_length


def plot_force_history(ax_cd, ax_cl, history: ForceHistory, summary: ForceSummary | None = None) -> None:
    """``Cd(t)`` et ``Cl(t)`` en petits multiples (deux axes superposés, une série chacun)."""
    # Abscisse : temps convectif t U / L ; facteur de conversion des instants de la fenêtre d'analyse.
    tau = convective_time(history)
    scale = history.reference_velocity / history.reference_length
    for ax, values, name, color, symmetric in (
        (ax_cd, history.cd, "Cd", SERIES[0], False),
        (ax_cl, history.cl, "Cl", SERIES[1], True),
    ):
        # Courbe du coefficient.
        ax.plot(tau, values, color=color)
        # Fenêtre d'analyse (régime établi) : voile très léger de la couleur de la série.
        if summary is not None:
            ax.axvspan(summary.t_start * scale, summary.t_end * scale, color=color, alpha=0.08, linewidth=0)
        # Nom du coefficient en étiquette verticale (une seule série : pas de légende).
        ax.set_ylabel(name)
        # Cadrage vertical sans le pic de démarrage.
        ax.set_ylim(*_focus_limits(history.time, values, symmetric))
    # Titre sur l'axe du haut, étiquette de temps sur l'axe du bas.
    ax_cd.set_title("Coefficients aérodynamiques (zone teintée : régime analysé)")
    # Les graduations x de l'axe du haut sont masquées (axe partagé).
    ax_cd.tick_params(labelbottom=False)
    ax_cl.set_xlabel("t U / L")


def plot_spectrum(ax, spectrum: Spectrum | None, length: float, velocity: float) -> None:
    """Spectre d'amplitude de ``Cl`` en fonction du nombre de Strouhal, pic annoté."""
    ax.set_title("Spectre de Cl")
    if spectrum is None:
        # Écoulement stationnaire : pas de spectre exploitable.
        ax.text(0.5, 0.5, "Écoulement stationnaire", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    # Fréquences converties en nombres de Strouhal St = f L / U.
    strouhal = spectrum.frequency * length / velocity
    peak = spectrum.peak_frequency * length / velocity
    # Fenêtre d'affichage : jusqu'à 3 fois le pic (harmoniques 2 et 3 visibles).
    shown = strouhal <= max(3.0 * peak, 0.5)
    ax.plot(strouhal[shown], spectrum.amplitude[shown], color=SERIES[1])
    # Point du pic (anneau de la couleur du fond pour rester lisible sur la courbe).
    ax.plot([peak], [spectrum.peak_amplitude], "o", color=SERIES[1], markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    # Étiquette du pic (texte à l'encre, décalé de 8 points à droite).
    ax.annotate(f"St = {peak:.3f}", xy=(peak, spectrum.peak_amplitude), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9.5)
    ax.set_xlabel("St = f L / U")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(bottom=0.0)


def plot_cp(ax, cp: SurfaceDistribution | None) -> None:
    """Coefficient de pression le long de la paroi, en fonction de l'angle depuis l'amont."""
    ax.set_title("Pression pariétale Cp(θ)")
    if cp is None:
        ax.text(0.5, 0.5, "Contour indisponible", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    # Courbe Cp(θ).
    ax.plot(cp.theta, cp.cp, color=SERIES[0])
    # Référence Cp = 0 (pression de l'écoulement amont).
    ax.axhline(0.0, color=AXIS_COLOR, linewidth=0.8)
    # Minimum de pression (point le plus aspiré) : marqué et annoté.
    k = int(np.argmin(cp.cp))
    ax.plot([cp.theta[k]], [cp.cp[k]], "o", color=SERIES[0], markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    # Étiquette du côté intérieur du graphique : à droite du point dans la première moitié,
    # à gauche dans la seconde (le texte ne sort jamais du cadre).
    first_half = cp.theta[k] < cp.theta[0] + 180.0
    ax.annotate(f"Cp min = {cp.cp[k]:.2f} ({cp.theta[k]:.0f}°)", xy=(cp.theta[k], cp.cp[k]),
                xytext=(10, 0) if first_half else (-10, 0), textcoords="offset points",
                ha="left" if first_half else "right", va="center", color=INK, fontsize=9)
    # Graduations tous les 90° : avant, dessus, culot, dessous.
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlim(cp.theta[0], cp.theta[0] + 360.0)
    ax.set_xlabel("θ (° depuis l'amont)")
    ax.set_ylabel("Cp")


def plot_cp_chordwise(ax, cp: ChordwiseDistribution | None) -> None:
    """Cp le long de la corde d'un corps élancé, extrados et intrados (axe vertical inversé).

    Convention aérodynamique : les Cp négatifs (aspiration) sont vers le haut ; l'aire entre les
    deux courbes est proportionnelle à la portance due à la pression.
    """
    ax.set_title("Pression pariétale Cp(x/c)")
    if cp is None:
        ax.text(0.5, 0.5, "Contour indisponible", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    # Deux séries (couleurs fixes de la charte) et une légende.
    ax.plot(cp.x_upper, cp.cp_upper, color=SERIES[0], label="extrados")
    ax.plot(cp.x_lower, cp.cp_lower, color=SERIES[1], label="intrados")
    # Référence Cp = 0 (pression de l'écoulement amont).
    ax.axhline(0.0, color=AXIS_COLOR, linewidth=0.8)
    # Axe inversé : aspiration vers le haut.
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("x / c (depuis le bord d'attaque)")
    ax.set_ylabel("Cp (axe inversé)")
    ax.legend(loc="best")


def wake_origin(solver: NavierStokesSolver) -> str:
    """Origine des stations de sillage : arrière d'un corps élancé, centre d'un corps non profilé."""
    return "rear" if solver.obstacles and chord_line(solver.obstacles[0]) is not None else "center"


def plot_wake_profiles(ax, profiles: Sequence[WakeProfile], eta_max: float = 3.0, origin: str = "center") -> None:
    """Profils ``u(y)/U∞`` aux stations du sillage (une couleur fixe par station, légende).

    ``origin`` rappelle dans le titre d'où sont mesurées les stations (``'center'`` ou ``'rear'``).
    """
    ax.set_title("Profils de sillage" + (" (x depuis l'arrière)" if origin == "rear" else ""))
    for k, profile in enumerate(profiles):
        # Couleur catégorielle attribuée dans l'ordre fixe (station k -> couleur k).
        ax.plot(profile.u, profile.eta, color=SERIES[k % len(SERIES)], label=f"x/L = {profile.station:g}")
    # u = 0 : frontière de l'écoulement de retour.
    ax.axvline(0.0, color=AXIS_COLOR, linewidth=0.8)
    # Région du sillage.
    ax.set_ylim(-eta_max, eta_max)
    ax.set_xlabel("u / U")
    ax.set_ylabel("(y - y_c) / L")
    # Légende (toujours présente dès deux séries).
    if len(profiles) > 1:
        ax.legend(loc="lower left")


def plot_indicators(ax, rows: Sequence[tuple[str, str]]) -> None:
    """Tableau d'indicateurs (libellé à gauche, valeur à droite, filets de séparation)."""
    ax.set_title("Indicateurs")
    # Pas d'axes : zone de texte.
    ax.axis("off")
    n = len(rows)
    for k, (label, value) in enumerate(rows):
        # Ordonnée du centre de la ligne k (du haut vers le bas), en coordonnées d'axes (0..1).
        y = 1.0 - (k + 0.5) / n
        ax.text(0.0, y, label, color=INK_SECONDARY, fontsize=10, va="center", transform=ax.transAxes)
        ax.text(1.0, y, value, color=INK, fontsize=11, fontweight="semibold", ha="right", va="center",
                transform=ax.transAxes)
        # Filet fin sous chaque ligne sauf la dernière.
        if k < n - 1:
            ax.plot([0.0, 1.0], [y - 0.5 / n] * 2, color=GRID_COLOR, linewidth=0.6, transform=ax.transAxes)


# ================================================================== tableau de bord
def default_view(solver: NavierStokesSolver, half_height: float = 3.0) -> tuple[float, float, float, float]:
    """Cadrage de la carte : domaine entier en x, bande de ±``half_height`` L autour de l'obstacle."""
    g = solver.grid
    if not solver.obstacles:
        return (0.0, g.Lx, 0.0, g.Ly)
    # Ordonnée du centre du premier obstacle.
    yc = solver.obstacles[0].center[1]
    # Demi-hauteur de la bande, limitée au domaine.
    h = min(half_height * solver.L_ref, 0.5 * g.Ly)
    return (0.0, g.Lx, max(yc - h, 0.0), min(yc + h, g.Ly))


def dashboard_indicators(
    solver: NavierStokesSolver, summary: ForceSummary, recirculation: float | None = None
) -> list[tuple[str, str]]:
    """Lignes du tableau d'indicateurs."""
    rows = [("Reynolds", f"{solver.reynolds:.4g}")]
    # Strouhal seulement pour un écoulement instationnaire.
    if summary.unsteady:
        rows.append(("Nombre de Strouhal", f"{summary.strouhal:.3f}"))
    rows += [
        ("Cd moyen", f"{summary.cd_mean:.3f}"),
        ("dont pression / frottement", f"{summary.cd_pressure_mean:.3f} / {summary.cd_viscous_mean:.3f}"),
        ("Cl rms", f"{summary.cl_rms:.3f}"),
    ]
    # Longueur de recirculation si elle est définie.
    if recirculation is not None and np.isfinite(recirculation):
        rows.append(("Recirculation Lr / L", f"{recirculation:.2f}"))
    # Incompressibilité.
    rows.append(("max |div u|", f"{divergence_report(solver).history_max:.1e}"))
    return rows


def plot_dashboard(
    solver: NavierStokesSolver,
    history: ForceHistory,
    *,
    mean_fields: FlowFields | None = None,
    stations: Sequence[float] = (1.0, 2.0, 5.0, 10.0),
    main: str = "vorticity",
    view: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    path: str | Path | None = None,
    dpi: int = 120,
) -> Figure:
    """Tableau de bord complet ; enregistré en PNG si ``path`` est fourni.

    ``mean_fields`` (champs moyens de :class:`~cfd2d.analytics.FieldAverager`) sert au ``Cp``
    et aux profils de sillage ; à défaut, le champ instantané est utilisé. ``main`` :
    ``'vorticity'`` ou ``'streamlines'`` pour le panneau principal.
    """
    # Synthèse des efforts et spectre (si instationnaire).
    summary = history.summary()
    spectrum = history.spectrum() if summary.unsteady else None
    # Cp pariétal (seulement si l'obstacle a un contour connu) ; le long de la corde pour un corps élancé.
    obstacle_has_outline = bool(solver.obstacles) and solver.obstacles[0].outline() is not None
    cp = surface_pressure(solver, fields=mean_fields) if obstacle_has_outline else None
    chord = chord_line(solver.obstacles[0]) if solver.obstacles else None
    cp_chord = chordwise_pressure(cp, *chord) if cp is not None and chord is not None else None
    # Profils de sillage (depuis l'arrière d'un corps élancé) et longueur de recirculation (corps non
    # profilé seulement : sur un corps incliné, l'axe horizontal ne suit pas le sillage).
    profiles = wake_profiles(solver, tuple(stations), fields=mean_fields, origin=wake_origin(solver)) if solver.obstacles else []
    recirculation = recirculation_length(solver, fields=mean_fields) if solver.obstacles and chord is None else None
    # Cadrage de la carte.
    view = view or default_view(solver)
    # Toute la figure est construite avec la charte graphique.
    with rc_context(STYLE):
        # Figure indépendante de pyplot (15 x 14 pouces).
        fig = Figure(figsize=(15.0, 14.0))
        # Grille de mise en page 3 x 3 ; la ligne 0, plus haute, laisse la carte (repère
        # orthonormé) occuper toute la largeur.
        layout = fig.add_gridspec(
            3, 3, height_ratios=(1.8, 1.0, 1.0), hspace=0.4, wspace=0.28, left=0.06, right=0.97, top=0.93, bottom=0.05
        )
        # Panneau principal sur toute la largeur.
        ax_main = fig.add_subplot(layout[0, :])
        # Sous-grille 2 x 1 pour Cd(t) et Cl(t) (axes x partagés).
        forces_layout = layout[1, :2].subgridspec(2, 1, hspace=0.12)
        ax_cd = fig.add_subplot(forces_layout[0])
        ax_cl = fig.add_subplot(forces_layout[1], sharex=ax_cd)
        # Spectre, Cp, profils, indicateurs.
        ax_spectrum = fig.add_subplot(layout[1, 2])
        ax_cp = fig.add_subplot(layout[2, 0])
        ax_wake = fig.add_subplot(layout[2, 1])
        ax_info = fig.add_subplot(layout[2, 2])

        # Panneau principal.
        if main == "streamlines":
            plot_streamlines(ax_main, solver, view=view)
            ax_main.set_title(f"Lignes de courant à t = {solver.time:.1f}")
        else:
            plot_vorticity(ax_main, solver, view=view)
            ax_main.set_title(f"Vorticité à t = {solver.time:.1f}")
        # Panneaux secondaires.
        plot_force_history(ax_cd, ax_cl, history, summary)
        plot_spectrum(ax_spectrum, spectrum, history.reference_length, history.reference_velocity)
        if cp_chord is not None:
            plot_cp_chordwise(ax_cp, cp_chord)
        else:
            plot_cp(ax_cp, cp)
        plot_wake_profiles(ax_wake, profiles, origin=wake_origin(solver))
        plot_indicators(ax_info, dashboard_indicators(solver, summary, recirculation))
        # Titre général aligné à gauche.
        fig.suptitle(
            title or f"{solver.config.name} : Re = {solver.reynolds:g}",
            x=0.06, ha="left", fontsize=15, fontweight="semibold", color=INK,
        )
        # Enregistrement éventuel.
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    return fig


# ========================================================================= animation
#: Grandeurs animables et titres des animations correspondantes.
ANIMATED_QUANTITIES = {"vorticity": "Vorticité", "speed": "Vitesse", "pressure": "Pression (Cp)"}


class FrameRecorder:
    """Moniteur : mémorise la vorticité, la norme de la vitesse ou Cp tous les ``every`` pas.

    Les images sont stockées en simple précision (float32) ; ~0.3 Mo par image sur une
    grille 400 x 200. ``t_start`` évite d'enregistrer le régime transitoire. :func:`animate`
    les assemble ensuite en vidéo. :meth:`record` ajoute l'image d'un état quelconque (par
    exemple un instantané conservé pendant le calcul), ce qui permet d'animer après coup.
    """

    def __init__(
        self,
        solver: NavierStokesSolver,
        *,
        every: int = 25,
        quantity: str = "vorticity",
        t_start: float = 0.0,
        attach: bool = True,
    ) -> None:
        # Trois grandeurs disponibles.
        if quantity not in ANIMATED_QUANTITIES:
            raise ValueError("quantity doit valoir 'vorticity', 'speed' ou 'pressure'.")
        self.quantity = quantity
        # Instant à partir duquel les images sont enregistrées.
        self.t_start = float(t_start)
        # Éléments nécessaires au rendu sans le solveur.
        self.grid, self.solid, self.obstacles = solver.grid, solver.solid, list(solver.obstacles)
        # Échelle d'adimensionnement : U/L (vorticité), U (vitesse), ½ ρ U² (pression).
        self.scale = {
            "vorticity": solver.U_ref / solver.L_ref,
            "speed": solver.U_ref,
            "pressure": 0.5 * solver.rho * solver.U_ref**2,
        }[quantity]
        # Instants et images enregistrés.
        self.times: list[float] = []
        self.frames: list[np.ndarray] = []
        # Inscription auprès du solveur.
        if attach:
            solver.add_callback(self, every)

    def __len__(self) -> int:
        # Nombre d'images enregistrées.
        return len(self.frames)

    def __call__(self, solver: NavierStokesSolver) -> None:
        # Image de l'état courant du solveur.
        self.record(solver.state, solver)

    def record(self, state: FlowState, solver: NavierStokesSolver) -> None:
        """Ajoute l'image de ``state`` (état courant ou instantané enregistré, même grille)."""
        # Régime transitoire : rien à enregistrer.
        if state.t < self.t_start:
            return
        if self.quantity == "vorticity":
            # Vorticité aux coins, adimensionnée.
            data = vorticity_nodes(state.u, state.v, self.grid, self.solid) / self.scale
        elif self.quantity == "speed":
            # Norme de la vitesse aux centres, adimensionnée.
            uc, vc = cell_centered_velocity(state.u, state.v)
            data = np.hypot(uc, vc) / self.scale
        else:
            # Coefficient de pression, référence choisie comme pour Cp pariétal (entrée ou sortie).
            data = (state.p - reference_pressure(solver, pressure=state.p)) / self.scale
        # Stockage en float32 (moitié de la mémoire d'un float64).
        self.frames.append(np.asarray(data, dtype=np.float32))
        self.times.append(float(state.t))


def _render_frames(
    recorder: FrameRecorder,
    *,
    vmax: float | None,
    view: tuple[float, float, float, float] | None,
    dpi: int,
    title: str,
) -> Iterator[np.ndarray]:
    """Génère les images RGB (tableaux (hauteur, largeur, 3) d'octets) de l'animation."""
    g = recorder.grid
    xmin, xmax, ymin, ymax = view or (0.0, g.Lx, 0.0, g.Ly)
    # Largeur 10 pouces, hauteur selon le rapport d'aspect du cadrage (+ place pour le titre).
    width = 10.0
    height = max(2.5, width * (ymax - ymin) / (xmax - xmin) + 0.9)
    # Échelle de couleurs fixe pour toute l'animation (évite le « clignotement »), calculée
    # sur les dernières images (régime établi).
    if vmax is None:
        tail = np.stack(recorder.frames[-min(5, len(recorder)):])
        # Grandeurs signées (vorticité, Cp) : borne symétrique ; norme de la vitesse : maximum.
        vmax = float(np.nanmax(tail)) if recorder.quantity == "speed" else symmetric_limit(tail)
    with rc_context(STYLE):
        # layout="constrained" : marges ajustées automatiquement (titre, barre de couleurs).
        fig = Figure(figsize=(width, height), dpi=dpi, layout="constrained")
        # Canevas Agg : rendu en mémoire, accès direct aux pixels.
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        # Première image : crée l'objet image, mis à jour ensuite par set_data (rapide).
        image = field_image(
            ax, recorder.frames[0].astype(float), g, recorder.solid, recorder.obstacles,
            kind=recorder.quantity, vmax=vmax, view=view,
        )
        heading = ax.set_title("")
        # Masque des nœuds solides (coins pour la vorticité, cellules pour la vitesse).
        hidden = corner_solid(recorder.solid) if recorder.quantity == "vorticity" else recorder.solid
        for k, (data, t) in enumerate(zip(recorder.frames, recorder.times)):
            # Nouvelle image (valeur neutre dans le solide) ; .T : convention (y, x) d'imshow.
            image.set_data(np.where(hidden, 0.0, data).T)
            heading.set_text(f"{title} : t = {t:.2f}")
            # Rendu, puis lecture des pixels RGBA ; [..., :3] ne garde que R, G, B.
            canvas.draw()
            # Mise en page calculée à la première image puis figée (moteur « none ») : sinon le
            # moteur « constrained » recalculerait toutes les marges à chaque image (~10 fois plus lent).
            if k == 0:
                fig.set_layout_engine("none")
            rgb = np.asarray(canvas.buffer_rgba())[..., :3]
            # H.264 exige des dimensions paires : on complète d'une ligne/colonne si besoin.
            yield np.pad(rgb, ((0, rgb.shape[0] % 2), (0, rgb.shape[1] % 2), (0, 0)), mode="edge")


def _write_mp4(frames: Iterator[np.ndarray], path: Path, fps: int) -> None:
    """Encode les images en MP4 (H.264) en les envoyant à ffmpeg par un tube (ffmpeg-python)."""
    # ffmpeg-python : construit et lance la commande ffmpeg (import local : dépendance optionnelle).
    import ffmpeg

    # L'exécutable ffmpeg doit être accessible.
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Executable ffmpeg introuvable : l'installer ou choisir un fichier .gif.")
    # Première image : dimensions de la vidéo.
    first = next(frames)
    height, width = first.shape[:2]
    # Entrée : images brutes RGB 24 bits lues sur l'entrée standard ("pipe:") ; sortie : H.264,
    # format yuv420p (compatible avec tous les lecteurs), qualité crf=20, lecture progressive ;
    # -loglevel error limite les messages (évite de remplir le tube d'erreurs).
    process = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}", framerate=fps)
        .output(str(path), pix_fmt="yuv420p", vcodec="libx264", crf=20, movflags="+faststart")
        .global_args("-loglevel", "error")
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )
    # Envoi des octets de chaque image (première comprise).
    process.stdin.write(first.tobytes())
    for rgb in frames:
        process.stdin.write(rgb.tobytes())
    # Fin du flux, puis attente de la fin de l'encodage ; communicate() lit les messages d'erreur.
    process.stdin.close()
    _, errors = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Echec de ffmpeg : {errors.decode(errors='replace')}")


def _write_gif(frames: Iterator[np.ndarray], path: Path, fps: int) -> None:
    """Assemble les images en GIF animé (Pillow)."""
    # Pillow (dépendance de matplotlib) : manipulation d'images.
    from PIL import Image

    # Conversion en images à palette de 128 couleurs (format GIF), palette adaptée à chaque image.
    images = [Image.fromarray(rgb).convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for rgb in frames]
    # save_all + append_images : GIF multi-images ; duration en millisecondes ; loop=0 : boucle infinie.
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(round(1000 / fps)), loop=0)


def animate(
    recorder: FrameRecorder,
    path: str | Path,
    *,
    fps: int = 20,
    vmax: float | None = None,
    view: tuple[float, float, float, float] | None = None,
    dpi: int = 100,
    title: str | None = None,
) -> Path:
    """Crée une animation ``.mp4`` (ffmpeg) ou ``.gif`` (Pillow) à partir d'un :class:`FrameRecorder`."""
    path = Path(path)
    # Au moins une image enregistrée.
    if not recorder.frames:
        raise ValueError("Aucune image enregistree : attacher le FrameRecorder avant solver.run().")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Générateur d'images (produites une à une : mémoire maîtrisée).
    heading = title or ANIMATED_QUANTITIES[recorder.quantity]
    frames = _render_frames(recorder, vmax=vmax, view=view, dpi=dpi, title=heading)
    # Choix du format selon l'extension (.lower() : insensible à la casse).
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        _write_mp4(frames, path, fps)
    elif suffix == ".gif":
        _write_gif(frames, path, fps)
    else:
        raise ValueError("Format d'animation non supporte (attendu : .mp4 ou .gif).")
    logger.info("Animation de %d images ecrite dans %s.", len(recorder), path)
    return path


# ==================================================================== rapport Plotly
def _plotly_colorscale() -> list[list[float | str]]:
    """Échelle divergente au format Plotly : [[position, couleur], ...]."""
    n = len(DIVERGING_COLORS)
    # Positions régulièrement réparties de 0 à 1.
    return [[k / (n - 1), color] for k, color in enumerate(DIVERGING_COLORS)]


def interactive_report(
    history: ForceHistory,
    path: str | Path,
    *,
    solver: NavierStokesSolver | None = None,
    mean_fields: FlowFields | None = None,
    stations: Sequence[float] = (1.0, 2.0, 5.0, 10.0),
    stride: int = 2,
    include_plotlyjs: bool | str = True,
) -> Path:
    """Rapport HTML interactif (Plotly) : vorticité, Cd(t), Cl(t), spectre, Cp(θ), sillage, ∇·u.

    ``stride`` sous-échantillonne la carte (taille du fichier) ; ``include_plotlyjs=True``
    produit un fichier autonome (≈ 4 Mo, lisible hors ligne), ``'cdn'`` un fichier léger
    qui charge la bibliothèque depuis internet.
    """
    # Construction de la figure Plotly (voir report_figure).
    fig = report_figure(history, solver=solver, mean_fields=mean_fields, stations=stations, stride=stride)
    # Écriture du fichier HTML.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=include_plotlyjs)
    return path


def report_figure(
    history: ForceHistory,
    *,
    solver: NavierStokesSolver | None = None,
    mean_fields: FlowFields | None = None,
    stations: Sequence[float] = (1.0, 2.0, 5.0, 10.0),
    stride: int = 2,
):
    """Figure Plotly du rapport interactif (affichable directement dans un notebook Jupyter)."""
    # Plotly : graphiques interactifs (import local, dépendance optionnelle).
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Synthèse, spectre, grandeurs de référence.
    summary = history.summary()
    spectrum = history.spectrum() if summary.unsteady else None
    L, U = history.reference_length, history.reference_velocity
    # Corps élancé (profil, plaque) : Cp le long de la corde, sillage mesuré depuis l'arrière.
    chord = chord_line(solver.obstacles[0]) if solver is not None and solver.obstacles else None
    # Grille de sous-graphiques : carte sur toute la largeur, puis 3 lignes de 2 graphiques.
    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}], [{}, {}]],
        row_heights=[0.34, 0.22, 0.22, 0.22],
        vertical_spacing=0.07,
        horizontal_spacing=0.09,
        subplot_titles=(
            "Vorticité ω L / U",
            "Traînée Cd(t)",
            "Portance Cl(t)",
            "Spectre de Cl",
            "Pression pariétale Cp(x/c)" if chord is not None else "Pression pariétale Cp(θ)",
            "Profils de sillage u / U",
            "Divergence max |∇·u|",
        ),
    )
    # --- carte de vorticité
    if solver is not None:
        g = solver.grid
        fields = compute_fields(solver)
        # Vorticité adimensionnée (valeur neutre 0 dans le solide), sous-échantillonnée ([::stride]).
        omega = np.where(fields.solid, 0.0, fields.vorticity) / (U / L)
        vmax = symmetric_limit(omega)
        fig.add_trace(
            go.Heatmap(
                x=g.x_c[::stride],
                y=g.y_c[::stride],
                z=omega[::stride, ::stride].T,
                colorscale=_plotly_colorscale(),
                zmid=0.0,
                zmin=-vmax,
                zmax=vmax,
                # zsmooth="best" : lissage de l'affichage (pas de pixels visibles).
                zsmooth="best",
                colorbar=dict(title=dict(text="ω L/U"), len=0.32, y=0.84, thickness=12),
                hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<br>ω L/U = %{z:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        # Obstacles : polygones pleins (fill="toself" referme le contour).
        for obstacle in solver.obstacles:
            outline = obstacle.outline()
            if outline is not None:
                fig.add_trace(
                    go.Scatter(
                        x=np.append(outline[:, 0], outline[0, 0]),
                        y=np.append(outline[:, 1], outline[0, 1]),
                        fill="toself",
                        fillcolor=SOLID_COLOR,
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
        # Cadrage identique au tableau de bord (bande autour de l'obstacle) ; repère orthonormé
        # (scaleanchor lie l'échelle y à l'échelle x) obtenu en réduisant le cadre du graphique
        # (constrain="domain") plutôt qu'en élargissant les plages affichées.
        xmin, xmax, ymin, ymax = default_view(solver)
        fig.update_xaxes(range=[xmin, xmax], constrain="domain", row=1, col=1)
        fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1, constrain="domain", row=1, col=1)
    # --- Cd(t) et Cl(t) : une série par graphique (petits multiples), en temps convectif t U / L
    tau = convective_time(history)
    for col, (values, name, color) in enumerate(((history.cd, "Cd", SERIES[0]), (history.cl, "Cl", SERIES[1])), 1):
        fig.add_trace(
            go.Scatter(
                x=tau,
                y=values,
                mode="lines",
                line=dict(color=color, width=2),
                name=name,
                showlegend=False,
                hovertemplate=f"t U/L = %{{x:.2f}}<br>{name} = %{{y:.4f}}<extra></extra>",
            ),
            row=2,
            col=col,
        )
        # Zone du régime analysé (rectangle très léger).
        fig.add_vrect(x0=summary.t_start * U / L, x1=summary.t_end * U / L, fillcolor=color, opacity=0.08, line_width=0,
                      row=2, col=col)
        fig.update_xaxes(title_text="t U / L", row=2, col=col)
        # Cadrage vertical sans le pic du démarrage impulsif (portance : axe symétrique).
        fig.update_yaxes(range=list(_focus_limits(history.time, values, symmetric=name == "Cl")), row=2, col=col)
    # --- spectre
    if spectrum is not None:
        st = spectrum.frequency * L / U
        peak = spectrum.peak_frequency * L / U
        shown = st <= max(3.0 * peak, 0.5)
        fig.add_trace(
            go.Scatter(
                x=st[shown], y=spectrum.amplitude[shown], mode="lines", line=dict(color=SERIES[1], width=2),
                showlegend=False, hovertemplate="St = %{x:.3f}<br>amplitude = %{y:.4f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        # Pic annoté.
        fig.add_trace(
            go.Scatter(
                x=[peak], y=[spectrum.peak_amplitude], mode="markers+text", text=[f"St = {peak:.3f}"],
                textposition="middle right", marker=dict(color=SERIES[1], size=9, line=dict(color=SURFACE, width=2)),
                textfont=dict(color=INK), showlegend=False, hoverinfo="skip",
            ),
            row=3,
            col=1,
        )
        fig.update_xaxes(title_text="St = f L / U", row=3, col=1)
    # --- Cp et sillage (nécessitent le solveur)
    if solver is not None and solver.obstacles:
        if solver.obstacles[0].outline() is not None:
            cp = surface_pressure(solver, fields=mean_fields)
            if chord is not None:
                # Corps élancé : extrados et intrados en fonction de x/c, axe vertical inversé.
                cpc = chordwise_pressure(cp, *chord)
                for x, values, side, color in ((cpc.x_upper, cpc.cp_upper, "extrados", SERIES[0]),
                                               (cpc.x_lower, cpc.cp_lower, "intrados", SERIES[1])):
                    # legend="legend2" : seconde légende, placée dans le graphique de Cp (voir plus bas).
                    fig.add_trace(
                        go.Scatter(
                            x=x, y=values, mode="lines", line=dict(color=color, width=2), name=side, legend="legend2",
                            hovertemplate="x/c = %{x:.3f}<br>Cp = %{y:.3f}<extra>" + side + "</extra>",
                        ),
                        row=3,
                        col=2,
                    )
                fig.update_xaxes(title_text="x / c", range=[0.0, 1.0], row=3, col=2)
                fig.update_yaxes(title_text="Cp (axe inversé)", autorange="reversed", row=3, col=2)
            else:
                fig.add_trace(
                    go.Scatter(
                        x=cp.theta, y=cp.cp, mode="lines", line=dict(color=SERIES[0], width=2), showlegend=False,
                        hovertemplate="θ = %{x:.0f}°<br>Cp = %{y:.3f}<extra></extra>",
                    ),
                    row=3,
                    col=2,
                )
                fig.update_xaxes(title_text="θ (° depuis l'amont)", tickvals=[0, 90, 180, 270, 360], row=3, col=2)
        for k, profile in enumerate(wake_profiles(solver, tuple(stations), fields=mean_fields, origin=wake_origin(solver))):
            fig.add_trace(
                go.Scatter(
                    x=profile.u, y=profile.eta, mode="lines", line=dict(color=SERIES[k % len(SERIES)], width=2),
                    name=f"x/L = {profile.station:g}", legendgroup="wake",
                    hovertemplate="u/U = %{x:.3f}<br>(y-y_c)/L = %{y:.2f}<extra>" + f"x/L = {profile.station:g}</extra>",
                ),
                row=4,
                col=1,
            )
        fig.update_xaxes(title_text="u / U", row=4, col=1)
        fig.update_yaxes(title_text="(y - y_c) / L", range=[-3, 3], row=4, col=1)
        # --- suivi de la divergence (échelle logarithmique)
        diag = solver.diagnostics.as_arrays()
        fig.add_trace(
            go.Scatter(
                x=diag["time"] * U / L, y=np.maximum(diag["divergence_max"], 1e-20), mode="lines",
                line=dict(color=SERIES[0], width=2), showlegend=False,
                hovertemplate="t U/L = %{x:.2f}<br>max|div u| = %{y:.2e}<extra></extra>",
            ),
            row=4,
            col=2,
        )
        fig.update_yaxes(type="log", exponentformat="e", row=4, col=2)
        fig.update_xaxes(title_text="t U / L", row=4, col=2)
    # --- mise en forme générale (charte graphique)
    # La légende (profils de sillage) est placée dans le coin haut-gauche de leur graphique :
    # get_subplot(4, 1) donne ses axes, dont le « domain » est la position dans la page (0..1).
    wake_axes = fig.get_subplot(4, 1)
    legend_x, legend_y = wake_axes.xaxis.domain[0] + 0.01, wake_axes.yaxis.domain[1] - 0.005
    # Seconde légende (extrados / intrados) dans le coin haut-droit du graphique de Cp.
    cp_axes = fig.get_subplot(3, 2)
    fig.update_layout(
        legend2=dict(
            x=cp_axes.xaxis.domain[1] - 0.01, y=cp_axes.yaxis.domain[1] - 0.005, xanchor="right", yanchor="top",
            bgcolor="rgba(252,252,251,0.85)", font=dict(color=INK_SECONDARY),
        )
    )
    fig.update_layout(
        height=1500,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_SECONDARY, size=12),
        title=dict(text="cfd2d : rapport d'analyse", font=dict(color=INK, size=20), x=0.02),
        legend=dict(
            x=legend_x, y=legend_y, xanchor="left", yanchor="top",
            bgcolor="rgba(252,252,251,0.85)", font=dict(color=INK_SECONDARY),
        ),
        margin=dict(l=60, r=40, t=80, b=50),
        hovermode="closest",
    )
    # Axes : grille fine et discrète.
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=AXIS_COLOR, linecolor=AXIS_COLOR, showline=True)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=AXIS_COLOR, linecolor=AXIS_COLOR, showline=True)
    # Titres des sous-graphiques à l'encre principale.
    fig.update_annotations(font=dict(color=INK, size=14))
    return fig

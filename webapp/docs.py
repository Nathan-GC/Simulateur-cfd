"""Onglet « Théorie & Documentation » : théorie illustrée, README, tutoriel, exercices, code source."""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``ast`` : lecture du docstring d'un module sans l'exécuter ; ``json`` : mesures des régimes.
import ast
import json

# Expressions régulières : liens et titres des documents Markdown.
import re

# Chemins de fichiers portables.
from pathlib import Path

import numpy as np

# Plotly : courbe St(Re).
import plotly.graph_objects as go

# Streamlit : mise en page, widgets, rendu Markdown et LaTeX.
import streamlit as st

# Style matplotlib et figure indépendante de pyplot (schémas).
from matplotlib import rc_context
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

# Paquet : grille, cylindre, masque, références du cylindre et charte graphique.
from cfd2d import Cylinder, StaggeredGrid
from cfd2d import visualization as viz
from cfd2d.validation import CYLINDER_REFERENCES, williamson_strouhal

from .codegen import code_popover
from .common import ASSETS, PLOTLY_CONFIG, ROOT, SRC, figure_png, style_plotly, theme

# Lien Markdown [texte](cible), sauf images ![...](...) : (?<!!) refuse un « ! » juste avant.
LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
# Début ou fin de bloc de code (``` ou ~~~), éventuellement indenté.
FENCE = re.compile(r"^\s*(```|~~~)")
# Titre Markdown : 1 à 6 « # », puis le texte (les « # » de fermeture éventuels sont ignorés).
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# Ordre de lecture conseillé du code (tutoriel, partie C) : fichier -> rôle.
READING_ORDER = (
    ("src/cfd2d/grid.py", "grille MAC : p aux centres, u et v aux faces"),
    ("src/cfd2d/geometry.py", "obstacles et masque binaire"),
    ("src/cfd2d/boundary.py", "conditions aux limites"),
    ("src/cfd2d/operators.py", "advection, laplacien, divergence"),
    ("src/cfd2d/pressure.py", "équation de Poisson et ses solveurs"),
    ("src/cfd2d/config.py", "paramètres d'une simulation"),
    ("src/cfd2d/solver.py", "projection de Chorin, boucle en temps"),
    ("src/cfd2d/presets.py", "cas types"),
    ("src/cfd2d/validation.py", "données de référence (Ghia, cylindre)"),
    ("src/cfd2d/analytics.py", "efforts, Strouhal, Cp, sillage"),
    ("src/cfd2d/visualization.py", "tableau de bord, animations, rapport"),
    ("src/cfd2d/io.py", "exports .npz / .vtk, reprise"),
)


# ========================================================================= documents Markdown
@st.cache_data(show_spinner=False)
def read_text(path: str, mtime: float) -> str:
    """Contenu d'un fichier texte ; ``mtime`` (date de modification) invalide le cache après une édition."""
    return Path(path).read_text(encoding="utf-8")


def load_text(path: Path) -> str:
    """Lit un fichier du projet en profitant du cache (relu seulement s'il a changé)."""
    return read_text(str(path), path.stat().st_mtime)


def documents() -> dict[str, Path]:
    """Documents Markdown à la racine du projet (README en premier) : libellé -> chemin."""
    found = sorted(ROOT.glob("*.md"), key=lambda p: (p.name.lower() != "readme.md", p.name.lower()))
    # Libellé : « README » ou nom du fichier en minuscules avec majuscule initiale (« Tutoriel »).
    return {("README" if p.stem.upper() == "README" else p.stem.capitalize()): p for p in found}


def localize_links(markdown: str, docs: dict[str, Path]) -> str:
    """Adapte les liens relatifs : autre document -> indication du sélecteur ; fichier -> chemin en code."""
    labels = {path.name.lower(): label for label, path in docs.items()}

    def replace(match: re.Match[str]) -> str:
        text, target = match.group(1), match.group(2)
        # Lien web (http:, https:, mailto:...) ou ancre interne (#...) : inchangé.
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE) or target.startswith("#"):
            return match.group(0)
        name = target.split("#")[0]
        label = labels.get(Path(name).name.lower())
        if label is not None:
            return f"**{text}** (document « {label} » ci-dessus)"
        # Fichier du projet : chemin en police de code (consultable dans « Code source »).
        return f"`{name}`" if text.strip("`") == name else f"{text} (`{name}`)"

    lines, in_code = [], False
    for line in markdown.splitlines():
        # Les délimiteurs de blocs de code basculent l'état « dans un bloc de code ».
        if FENCE.match(line):
            in_code = not in_code
        elif not in_code:
            line = LINK.sub(replace, line)
        lines.append(line)
    return "\n".join(lines)


def headings(markdown: str) -> list[tuple[int, str, int]]:
    """Titres de niveaux 2 et 3 hors blocs de code : (niveau, texte, numéro de ligne)."""
    found, in_code = [], False
    for number, line in enumerate(markdown.splitlines()):
        if FENCE.match(line):
            in_code = not in_code
            continue
        match = None if in_code else HEADING.match(line)
        if match and len(match.group(1)) in (2, 3):
            found.append((len(match.group(1)), match.group(2), number))
    return found


def section_text(markdown: str, heads: list[tuple[int, str, int]], index: int) -> str:
    """Texte de la section ``index`` : de son titre au titre suivant de niveau inférieur ou égal."""
    lines = markdown.splitlines()
    level, _, start = heads[index]
    end = next((line for lvl, _, line in heads[index + 1 :] if lvl <= level), len(lines))
    return "\n".join(lines[start:end])


def render_markdown_document(label: str, path: Path, docs: dict[str, Path]) -> None:
    """Affiche un document Markdown (entier ou une section), formules LaTeX comprises."""
    text = load_text(path)
    heads = headings(text)
    choice = st.selectbox(
        "Aller à la section", [-1, *range(len(heads))],
        # Les titres de niveau 3 sont décalés (espaces cadratin) sous leur titre de niveau 2.
        format_func=lambda k: "Document complet" if k < 0 else (" " * (heads[k][0] - 2)) + heads[k][1],
        key=f"doc_section_{label}",
    )
    body = text if choice < 0 else section_text(text, heads, choice)
    with st.container(border=True):
        st.markdown(localize_links(body, docs))


# ========================================================================= théorie illustrée
#: Résumés théoriques des sections de paramètres (fenêtres « Théorie »).
THEORY_SNIPPETS = {
    "flow": (
        "**Nombre de Reynolds** : rapport inertie / viscosité, $Re = U_\\infty L / \\nu$. Deux écoulements de "
        "même géométrie et de même Re sont semblables (similitude) : à Re fixé, changer U ne change que "
        "l'échelle de temps.\n\n"
        "**Cylindre** : Re < 5 sans décollement ; 5 < Re < 47 deux tourbillons attachés ; 47 < Re < 190 allée "
        "de Von Kármán ; au-delà, le sillage devient tridimensionnel.\n\n"
        "**Masse volumique** : en incompressible, ρ ne fixe que l'échelle de la pression ; les coefficients "
        "$C_d = F_x / (\\tfrac12 \\rho U^2 L)$, $C_p$... n'en dépendent pas."
    ),
    "geometry": (
        "**Frontière immergée** : la grille cartésienne ne suit pas la paroi ; une cellule est solide si au "
        "moins la moitié de sa surface est dans l'obstacle (masque en *marches d'escalier*). Les faces touchant "
        "le solide ont une vitesse nulle (adhérence), la pression y vérifie $\\partial p/\\partial n = 0$.\n\n"
        "**Résolution** : il faut des mailles assez fines pour les couches limites, d'épaisseur "
        "$\\delta \\sim L/\\sqrt{Re}$ ; repère : Reynolds de maille $Re_\\Delta = U\\Delta x/\\nu \\lesssim 10$, "
        "et au moins 8 mailles dans l'épaisseur d'un corps mince.\n\n"
        "**Blocage** $\\beta = h/H$ (hauteur frontale / hauteur du domaine) : les parois accélèrent "
        "l'écoulement autour de l'obstacle ; au-delà de quelques %, Cd et St augmentent."
    ),
    "boundary": (
        "**Entrée** : vitesse imposée (uniforme ou parabolique). **Sortie** : pression nulle et vitesse "
        "extrapolée, $\\partial u/\\partial x = 0$ (Neumann) ou transportée, $\\partial u/\\partial t + U_c\\, "
        "\\partial u/\\partial x = 0$ (convective : les tourbillons sortent sans se réfléchir).\n\n"
        "**Paroi adhérente** : $u = v = 0$ ; **glissante** : vitesse normale nulle, cisaillement nul "
        "(soufflerie idéale). Sur la grille MAC, la vitesse tangentielle est imposée par des *cellules "
        "fantômes* : valeur miroir $u_g = 2u_{paroi} - u_{int}$.\n\n"
        "**Condition initiale** : la perturbation brise la symétrie haut/bas et déclenche plus tôt le lâcher "
        "tourbillonnaire ; un démarrage progressif évite le choc d'un démarrage impulsif."
    ),
    "time": (
        "**Stabilité d'un schéma explicite** : le pas de temps est limité par l'advection (nombre de Courant) "
        "et par la diffusion (nombre de Fourier) :\n\n"
        "$$\\frac{1}{\\Delta t} = \\frac{1}{\\mathrm{CFL}}\\left(\\frac{|u|}{\\Delta x} + \\frac{|v|}{\\Delta y}\\right)"
        " + \\frac{\\nu}{\\mathrm{Fo}}\\left(\\frac{1}{\\Delta x^2} + \\frac{1}{\\Delta y^2}\\right)$$\n\n"
        "**Schémas** : Euler (ordre 1, pédagogique, CFL ≈ 0,2), Adams–Bashforth 2 (ordre 2, une évaluation par "
        "pas, CFL ≈ 0,4), Runge–Kutta SSP 3 (ordre 3, trois évaluations par pas, CFL ≈ 0,8). Au-delà de la "
        "limite, les erreurs s'amplifient à chaque pas et le calcul diverge."
    ),
    "numerics": (
        "**Advection** $\\nabla\\cdot(\\mathbf{u}\\otimes\\mathbf{u})$ : la valeur transportée sur une face est "
        "reconstruite à partir des cellules voisines. *Décentré amont* : ordre 1, très diffusif (viscosité "
        "numérique $\\sim U\\Delta x/2$) ; *centré* : ordre 2, oscille si $Re_\\Delta > 2$ ; *QUICK* : "
        "parabole décentrée, ordre 3, bon compromis ; *TVD* : limiteur de pente, sans oscillation.\n\n"
        "**Poisson** $\\nabla^2 p = (\\rho/\\Delta t)\\,\\nabla\\cdot\\mathbf{u}^*$ : système linéaire creux, "
        "symétrique défini positif. Le solveur direct le factorise une fois (matrice constante) ; les "
        "solveurs itératifs (CG, SOR, Jacobi) repartent à chaque pas."
    ),
}


def theory_popover(topic: str) -> None:
    """Fenêtre surgissante « Théorie » d'une section de paramètres."""
    text = THEORY_SNIPPETS.get(topic)
    if text:
        with st.popover("Théorie", icon=":material/school:", help="Les notions derrière ces réglages."):
            st.markdown(text)


def topic_extras(topic: str) -> None:
    """En tête de section : fenêtres « Théorie » et « Code » côte à côte."""
    with st.container(horizontal=True):
        theory_popover(topic)
        code_popover(topic)


@st.cache_data(show_spinner=False)
def mac_grid_figure() -> bytes:
    """Schéma de la grille MAC : p aux centres, u aux faces verticales, v aux faces horizontales."""
    with rc_context(viz.STYLE):
        fig = Figure(figsize=(6.4, 3.6), layout="constrained")
        ax = fig.subplots()
        nx, ny = 3, 2
        # Cellules fantômes (sous la paroi du bas et à gauche de la paroi gauche) : bande grisée.
        ax.fill_between([-1, nx], -1, 0, color="#ecebe6", zorder=0)
        ax.fill_between([-1, 0], -1, ny, color="#ecebe6", zorder=0)
        # Lignes de la grille (faces).
        for i in range(-1, nx + 1):
            ax.plot([i, i], [-1, ny], color=viz.GRID_COLOR, lw=1.0, zorder=1)
        for j in range(-1, ny + 1):
            ax.plot([-1, nx], [j, j], color=viz.GRID_COLOR, lw=1.0, zorder=1)
        # Parois du domaine (x = 0 et y = 0).
        ax.plot([0, 0], [0, ny], color=viz.INK_SECONDARY, lw=3)
        ax.plot([0, nx], [0, 0], color=viz.INK_SECONDARY, lw=3)
        for i in range(nx):
            for j in range(ny):
                # Pression au centre de la cellule (i, j).
                ax.plot(i + 0.5, j + 0.5, "o", color=viz.SERIES[2], ms=7, zorder=3)
                ax.annotate(f"p[{i},{j}]", (i + 0.5, j + 0.5), xytext=(0, -11), textcoords="offset points",
                            ha="center", va="top", fontsize=7.5, color=viz.INK_SECONDARY)
        for i in range(nx + 1):
            for j in range(ny):
                # u sur les faces verticales (flèches horizontales).
                ax.annotate("", (i + 0.18, j + 0.5), (i - 0.18, j + 0.5),
                            arrowprops=dict(arrowstyle="->", color=viz.SERIES[0], lw=1.6))
        for i in range(nx):
            for j in range(ny + 1):
                # v sur les faces horizontales (flèches verticales).
                ax.annotate("", (i + 0.5, j + 0.18), (i + 0.5, j - 0.18),
                            arrowprops=dict(arrowstyle="->", color=viz.SERIES[1], lw=1.6))
        # Valeurs fantômes de u sous la paroi (miroir de la première ligne intérieure).
        for i in range(nx + 1):
            ax.annotate("", (i + 0.18, -0.5), (i - 0.18, -0.5),
                        arrowprops=dict(arrowstyle="->", color=viz.SERIES[0], lw=1.2, ls="--"))
        ax.text(nx - 0.05, -0.62, "u fantômes : u_g = −u_int (paroi fixe)", ha="right", fontsize=8, color=viz.INK_SECONDARY)
        ax.text(0.04, ny + 0.08, "p (centres)", color=viz.SERIES[2], fontsize=9, va="bottom")
        ax.text(1.2, ny + 0.08, "u (faces verticales)", color=viz.SERIES[0], fontsize=9, va="bottom")
        ax.text(2.45, ny + 0.08, "v (faces horizontales)", color=viz.SERIES[1], fontsize=9, va="bottom")
        ax.set_xlim(-1.05, nx + 0.6)
        ax.set_ylim(-1.05, ny + 0.45)
        ax.set_aspect("equal")
        ax.axis("off")
        return figure_png(fig, dpi=130)


@st.cache_data(show_spinner=False)
def staircase_figure() -> bytes:
    """Masque en marches d'escalier d'un cylindre à deux résolutions (6 et 16 mailles par diamètre)."""
    with rc_context(viz.STYLE):
        fig = Figure(figsize=(6.4, 3.2), layout="constrained")
        axes = fig.subplots(1, 2)
        circle = np.linspace(0.0, 2.0 * np.pi, 200)
        for ax, n in zip(axes, (6, 16)):
            # Domaine 2 x 2 centré sur un cylindre de diamètre 1, n mailles par diamètre.
            grid = StaggeredGrid(2.0, 2.0, 2 * n, 2 * n)
            mask = Cylinder(1.0, 1.0, 1.0).mask(grid, 4)
            ax.pcolormesh(grid.x_f, grid.y_f, mask.T.astype(float), cmap=ListedColormap(["#f2f6fc", "#b9b8b0"]),
                          vmin=0, vmax=1, edgecolors=viz.GRID_COLOR, linewidth=0.4)
            ax.plot(1.0 + 0.5 * np.cos(circle), 1.0 + 0.5 * np.sin(circle), color=viz.INK, lw=1.3)
            ax.set_title(f"{n} mailles par diamètre", fontsize=10)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
        return figure_png(fig, dpi=130)


#: Organigramme d'un pas de temps (langage DOT, rendu par st.graphviz_chart dans le navigateur).
TIME_STEP_GRAPH = """
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#f2f6fc", color="#2a78d6", fontname="Segoe UI", fontsize=11];
  edge [color="#52514e", fontname="Segoe UI", fontsize=10];
  un [label="uⁿ\\n(divergence nulle)"];
  pred [label="1. Prédiction\\nu* = uⁿ + Δt H(uⁿ)\\nadvection + diffusion"];
  bc [label="Conditions aux limites\\nfaces des obstacles = 0"];
  poisson [label="2. Poisson\\n∇²p = (ρ/Δt) ∇·u*"];
  corr [label="3. Correction\\nuⁿ⁺¹ = u* − (Δt/ρ) ∇p"];
  un -> pred -> bc -> poisson -> corr;
  corr -> un [label="pas suivant (Δt recalculé : CFL, Fourier)", style=dashed];
}
"""


def stability_calculator() -> None:
    """Calculateur interactif : nombres de Courant, de Fourier et de Reynolds de maille."""
    c1, c2, c3, c4 = st.columns(4)
    U = c1.slider("Vitesse U", 0.1, 3.0, 1.0, 0.1, key="calc_U")
    dx = c2.select_slider("Maille Δx", [0.01, 0.02, 0.04, 0.0625, 0.0833, 0.125, 0.2], 0.0833,
                          format_func=lambda v: f"{v:g}", key="calc_dx")
    nu = c3.select_slider("Viscosité ν", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1], 1e-2,
                          format_func=lambda v: f"{v:g}", key="calc_nu")
    dt = c4.select_slider("Pas de temps Δt", [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.02, 0.05, 0.1], 0.01,
                          format_func=lambda v: f"{v:g}", key="calc_dt")
    # Grandeurs adimensionnelles (mailles carrées, écoulement selon x).
    cfl, fourier, cell_re = U * dt / dx, nu * dt * 2.0 / dx**2, U * dx / nu
    # Pas maximal d'AB2 selon le critère combiné du solveur (CFL 0,4 ; Fourier 0,9 x 0,25).
    dt_max = 1.0 / (U / dx / 0.4 + nu * 2.0 / dx**2 / (0.9 * 0.25))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Courant U Δt / Δx", f"{cfl:.3g}", border=True)
    m2.metric("Fourier ν Δt (1/Δx² + 1/Δy²)", f"{fourier:.3g}", border=True)
    m3.metric("Reynolds de maille U Δx / ν", f"{cell_re:.3g}", border=True)
    ratio = dt / dt_max
    m4.metric("Δt / Δt max (AB2)", f"{ratio:.2f}", border=True,
              delta="stable" if ratio <= 1 else "instable", delta_color="green" if ratio <= 1 else "red",
              delta_arrow="off")
    share = (U / dx / 0.4) / (U / dx / 0.4 + nu * 2.0 / dx**2 / 0.225)
    st.caption(f"Contrainte limitante : advection {100 * share:.0f} %, diffusion {100 * (1 - share):.0f} %. "
               "Raffiner la maille divise Δt max par 2 (advection) ou par 4 (diffusion) : le coût d'un calcul croît "
               "comme la résolution au cube.")


@st.cache_data(show_spinner=False)
def regimes_data() -> dict[str, dict]:
    """Mesures des calculs de régimes pré-calculés (``webapp/assets/regimes.json``)."""
    path = ASSETS / "regimes.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def strouhal_figure() -> go.Figure:
    """Loi St(Re) de Williamson, fourchettes de la littérature et calculs de l'application."""
    c = theme()
    Re = np.linspace(47.0, 190.0, 200)
    fig = go.Figure(go.Scatter(x=Re, y=[williamson_strouhal(r) for r in Re], mode="lines", name="Williamson (1989), milieu infini",
                               line=dict(color=viz.SERIES[0], width=2)))
    # Fourchettes de la littérature (segments verticaux).
    for r, values in CYLINDER_REFERENCES.items():
        if "St" in values:
            lo, hi = values["St"]
            fig.add_trace(go.Scatter(x=[r, r], y=[lo, hi], mode="lines+markers", line=dict(color=viz.SERIES[2], width=4),
                                     marker=dict(size=5), name="simulations 2D publiées", showlegend=r == 100))
    # Calculs de l'application (domaine 16 D x 8 D : blocage 12,5 %).
    data = [v for v in regimes_data().values() if v.get("St")]
    if data:
        fig.add_trace(go.Scatter(x=[v["Re"] for v in data], y=[v["St"] for v in data], mode="markers",
                                 name="cfd2d (blocage 12,5 %)",
                                 marker=dict(color=viz.SERIES[1], size=11, line=dict(color=c["surface"], width=2))))
    fig.update_xaxes(title_text="Re")
    fig.update_yaxes(title_text="St = f D / U∞")
    # Légende en bas à droite (la courbe monte de bas à gauche vers le haut à droite).
    fig.update_layout(title=dict(text="Fréquence du lâcher tourbillonnaire"),
                      legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"))
    return style_plotly(fig, 380, legend=True)


def theory_page() -> None:
    """Théorie illustrée : équations, grille MAC, pas de temps, masque, stabilité, régimes, grandeurs."""
    st.markdown("#### 1. Les équations")
    st.markdown("Écoulement **incompressible**, masse volumique ρ et viscosité cinématique ν constantes :")
    st.latex(
        r"\frac{\partial \mathbf{u}}{\partial t} + \nabla\cdot(\mathbf{u}\otimes\mathbf{u})"
        r" = -\frac{1}{\rho}\nabla p + \nu\,\nabla^2\mathbf{u}, \qquad \nabla\cdot\mathbf{u} = 0"
    )
    st.markdown(
        "La pression n'a pas d'équation d'évolution : c'est le **multiplicateur** qui maintient la divergence "
        "nulle à chaque instant. Le **nombre de Reynolds** $Re = U_\\infty L/\\nu$ compare l'inertie à la viscosité."
    )
    st.markdown("#### 2. La grille décalée (MAC)")
    c1, c2 = st.columns([3, 2])
    c1.image(mac_grid_figure(), width="stretch", alt="Grille décalée MAC : pression au centre des cellules, u sur les "
             "faces verticales, v sur les faces horizontales, cellules fantômes en gris")
    c2.markdown(
        "La pression est stockée au **centre** des cellules, u sur les **faces verticales** et v sur les faces "
        "**horizontales** (Harlow et Welch, 1965).\n\n"
        "- Le bilan de masse d'une cellule utilise exactement ses quatre faces : "
        "$(u_e - u_w)/\\Delta x + (v_n - v_s)/\\Delta y$.\n"
        "- Un champ de pression en damier a un gradient discret non nul : il est vu (et corrigé) par le solveur.\n"
        "- Les **cellules fantômes** (en gris) portent les conditions aux limites."
    )
    st.markdown("#### 3. Un pas de temps : la projection de Chorin")
    st.graphviz_chart(TIME_STEP_GRAPH, width="stretch")
    st.latex(r"\nabla\cdot\mathbf{u}^{n+1} = 0 \;\Longrightarrow\; \nabla^2 p = \frac{\rho}{\Delta t}\,\nabla\cdot\mathbf{u}^*")
    st.markdown("Dans l'application, l'onglet **Méthode** des résultats montre ces trois étapes sur votre calcul : "
                "la divergence de u* (de l'ordre de 10⁻²) tombe à 10⁻¹⁵ après la correction.")
    st.markdown("#### 4. Les obstacles : un masque en marches d'escalier")
    c1, c2 = st.columns([3, 2])
    c1.image(staircase_figure(), width="stretch", alt="Cylindre représenté par des cellules solides en marches "
             "d'escalier, avec son contour exact superposé")
    c2.markdown(
        "La grille reste cartésienne : une cellule est **solide** si au moins la moitié de sa surface est dans "
        "l'obstacle. Le contour vu par le solveur (gris) approche le contour exact (trait noir) d'autant mieux "
        "que la maille est fine.\n\n"
        "Les efforts sont intégrés sur ce contour en escalier, de façon cohérente avec le solveur (vérifiée par "
        "un bilan de quantité de mouvement)."
    )
    st.markdown("#### 5. Stabilité : nombres de Courant et de Fourier")
    st.markdown("Le schéma est **explicite** : le pas de temps est borné. Faites varier les grandeurs :")
    stability_calculator()
    st.markdown("#### 6. Les régimes de sillage du cylindre")
    data = regimes_data()
    images = [(Re, ASSETS / f"regime_Re{Re}.png") for Re in (20, 40, 100, 200)]
    cols = st.columns(2)
    for k, (Re, path) in enumerate(images):
        if path.is_file():
            info = data.get(str(Re), {})
            if info.get("unsteady"):
                caption = f"Re = {Re} : périodique, St = {info['St']:.3f}, Cd = {info['Cd']:.2f}"
            else:
                caption = f"Re = {Re} : stationnaire, Lr/D = {info.get('Lr', float('nan')):.2f}, Cd = {info.get('Cd', float('nan')):.2f}"
            cols[k % 2].image(path.read_bytes(), caption=caption, width="stretch",
                              alt=f"Champ de vorticité derrière un cylindre à Re = {Re}")
    st.caption("Calculs de l'application (16 mailles par D, domaine 16 D × 8 D : blocage 12,5 %, d'où Cd et St un peu "
               "supérieurs aux valeurs en milieu infini). Le lâcher apparaît vers Re ≈ 47 (bifurcation de Hopf).")
    st.plotly_chart(strouhal_figure(), theme=None, config=PLOTLY_CONFIG)
    st.markdown("#### 7. Les grandeurs d'analyse")
    st.latex(
        r"C_d = \frac{F_x}{\tfrac12 \rho U_\infty^2 L}, \qquad C_l = \frac{F_y}{\tfrac12 \rho U_\infty^2 L},"
        r" \qquad C_p = \frac{p - p_\infty}{\tfrac12 \rho U_\infty^2}, \qquad St = \frac{f\,L}{U_\infty}"
    )
    st.latex(
        r"\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}, \qquad"
        r" Q = \tfrac12\left(\lVert\boldsymbol{\Omega}\rVert^2 - \lVert\mathbf{S}\rVert^2\right)"
    )
    st.markdown("Canal plan : profil de **Poiseuille** $u(y) = 6U\\eta(1-\\eta)$, $\\eta = y/H$, et "
                "$dp/dx = -12\\mu U/H^2$. Cavité : référence de **Ghia et al. (1982)** à Re = 100, 400 et 1000.")


# ================================================================================ exercices
#: Exercices du tutoriel (partie C.12) : énoncé, réalisation dans l'application, éléments de correction.
EXERCISES = (
    ("Diffusion numérique",
     "Cavité N = 32 avec le schéma décentré amont puis QUICK, en comparant les écarts à Ghia. Refaire le cylindre "
     "à Re = 100 en décentré amont : que devient le lâcher tourbillonnaire ?",
     "Scénario guidé « Diffusion numérique », ou cas Cavité puis Obstacle avec le schéma d'advection choisi dans "
     "« Méthodes numériques ».",
     "Le schéma décentré ajoute une viscosité numérique d'ordre U Δx / 2 : tout se passe comme si le Reynolds "
     "était plus faible. Les écarts à Ghia sont nettement plus grands qu'avec QUICK ; sur le cylindre, "
     "l'amplitude de Cl diminue et, sur une maille grossière, le lâcher peut même disparaître."),
    ("Transition",
     "Cylindre à Re = 30, 40, 50, 60 avec la perturbation par défaut et une durée longue (150). Pour quels Re "
     "l'oscillation de la portance s'amortit-elle ? Mesurer la recirculation à Re = 40.",
     "Étude paramétrique en Re (valeurs 30, 40, 50, 60), puis l'onglet Études.",
     "Sous Re ≈ 47 la perturbation s'amortit (écoulement stationnaire, deux tourbillons attachés) ; au-dessus, "
     "elle croît jusqu'à un cycle limite : c'est une bifurcation de Hopf. À Re = 40, Lr/D ≈ 2,2 à 2,35 en "
     "milieu infini (un peu moins avec le blocage)."),
    ("Loi St(Re)",
     "Tracer St pour Re = 60, 80, 100, 150 et comparer à la loi de Williamson.",
     "Étude paramétrique en Re : l'onglet Études superpose automatiquement la loi de Williamson.",
     "St croît avec Re en suivant la loi de Williamson ; le confinement (blocage) décale les points vers le haut "
     "de quelques % : la correction par conservation du débit, St (1 − β), réduit cet écart."),
    ("Convergence en maillage",
     "Cd et St pour 12, 18 et 27 cellules par D.",
     "Étude paramétrique en résolution : l'onglet Études calcule l'ordre observé et la valeur extrapolée "
     "(Richardson).",
     "Les écarts entre résolutions successives diminuent : la solution converge. L'ordre observé est en général "
     "compris entre 1 et 2 : la frontière en escalier limite la précision près de la paroi."),
    ("Solveurs de pression",
     "Chronométrer une cavité N = 48 avec les solveurs direct, CG et SOR ; comparer les itérations.",
     "Cas Cavité, « Méthodes numériques » : solveur de Poisson ; durées dans l'onglet Historique, itérations "
     "dans l'onglet Diagnostics.",
     "Le solveur direct est le plus rapide (matrice factorisée une seule fois, une descente-remontée par pas) ; "
     "CG préconditionné converge en quelques itérations, SOR en plusieurs dizaines ou centaines."),
    ("Profil d'aile",
     "NACA 0012 à Re = 1000 pour α = 0, 4 et 8° : Cl augmente-t-il linéairement avec α ?",
     "Étude paramétrique en incidence (forme : profil NACA, au moins 48 mailles par corde).",
     "Cl croît à peu près linéairement aux petites incidences, avec une pente nettement inférieure à la valeur "
     "de la théorie des profils minces (2π par radian, environ 0,11 par degré) : à Re = 1000, les couches "
     "limites épaisses réduisent la circulation."),
    ("Nouvel obstacle",
     "Écrire une classe Ellipse(Obstacle) sur le modèle de Cylinder (méthode contains et propriétés "
     "reference_length, center, bounds), puis un test.",
     "Dans le code : src/cfd2d/geometry.py (onglet Code source, fichier 2).",
     "contains(x, y) renvoie ((x − xc)/a)² + ((y − yc)/b)² ≤ 1 ; reference_length = 2b (hauteur frontale) ; "
     "bounds = (xc − a, xc + a, yc − b, yc + b). Test : l'aire du masque tend vers π a b."),
)


def exercises_page() -> None:
    """Exercices du tutoriel, avec la façon de les faire dans l'application et des éléments de correction."""
    st.markdown("Exercices de la partie C.12 du tutoriel. Chacun se fait dans l'application ; la correction ne "
                "donne que les tendances attendues : à vous de mesurer les valeurs.")
    for k, (title, statement, how, answer) in enumerate(EXERCISES, 1):
        with st.container(border=True):
            st.markdown(f"**{k}. {title}.** {statement}")
            st.caption(f"Dans l'application : {how}")
            with st.expander("Éléments de correction"):
                st.markdown(answer)


# ============================================================================== code source
def source_files() -> list[tuple[Path, str]]:
    """Fichiers Python du projet, dans l'ordre de lecture conseillé : (chemin, rôle)."""
    ordered = [(ROOT / rel, role) for rel, role in READING_ORDER if (ROOT / rel).exists()]
    known = {path for path, _ in ordered}
    others = [(p, "module du paquet") for p in sorted((SRC / "cfd2d").glob("*.py")) if p not in known]
    examples = [(p, "exemple") for p in sorted((ROOT / "examples").glob("*.py"))]
    tests = [(p, "test") for p in sorted((ROOT / "tests").glob("*.py"))]
    app = [(ROOT / "app.py", "application web")] + [(p, "application web") for p in sorted((ROOT / "webapp").glob("*.py"))]
    return ordered + others + examples + tests + app


def render_source_viewer() -> None:
    """Lecteur de code : docstring du module (le « pourquoi »), source numérotée, ouverture dans VS Code."""
    files = source_files()
    index = st.selectbox(
        "Fichier (ordre de lecture conseillé par le tutoriel)", range(len(files)),
        format_func=lambda k: f"{k + 1:2d}. {files[k][0].relative_to(ROOT).as_posix()} : {files[k][1]}",
        key="source_file",
    )
    path = files[index][0]
    source = load_text(path)
    try:
        docstring = ast.get_docstring(ast.parse(source))
    except SyntaxError:
        docstring = None
    if docstring:
        with st.container(border=True):
            st.markdown(docstring)
    # Lien vscode://file/... : ouvre le fichier dans VS Code (si installé) sur cet ordinateur.
    st.caption(f"{len(source.splitlines())} lignes, commentées ligne à ligne · "
               f"[ouvrir dans VS Code](vscode://file/{path.as_posix()})")
    st.code(source, language="python", line_numbers=True, height=650)


# ================================================================================ onglet
def documentation_tab() -> None:
    """Onglet 1 : théorie illustrée (par défaut), documents du projet, exercices, code source."""
    docs = documents()
    st.markdown(
        "Ordre conseillé : la **théorie illustrée** (équations, grille, méthode), le **README** (vue d'ensemble), le "
        "**Tutoriel** (installation, paramètres, lecture guidée du code), les **exercices**, puis le **code source**."
    )
    options = ["Théorie illustrée", *docs, "Exercices", "Code source"]
    choice = st.segmented_control("Document", options, default=options[0], required=True, key="doc_choice",
                                  label_visibility="collapsed")
    if choice == "Théorie illustrée":
        theory_page()
    elif choice in docs:
        render_markdown_document(choice, docs[choice], docs)
    elif choice == "Exercices":
        exercises_page()
    else:
        render_source_viewer()

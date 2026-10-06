"""Éléments communs de l'application : chemins, libellés, mise en forme, charte des figures."""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# Expressions régulières : traduction des messages du solveur.
import re

# Tampon mémoire : images PNG produites sans passer par le disque.
from io import BytesIO

# Chemins de fichiers portables.
from pathlib import Path

import numpy as np

# Plotly : graphiques interactifs.
import plotly.graph_objects as go

# Streamlit : thème de la page (clair ou sombre).
import streamlit as st

# Contexte de style matplotlib et conversion de couleurs.
from matplotlib import rc_context
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

# Charte graphique du paquet (couleurs, palettes, style matplotlib).
from cfd2d import visualization as viz

# ============================================================================ chemins
# Dossier du projet (parent du dossier webapp/), sources du paquet, images de l'application.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ASSETS = Path(__file__).resolve().parent / "assets"
# Calculs enregistrés par l'application (dossier ignoré par git, comme tout outputs/).
RUNS_DIR = ROOT / "outputs" / "app_runs"

# =========================================================================== libellés
# Cas d'étude : identifiant -> libellé, et description courte.
CASES = {"obstacle": "Obstacle en soufflerie", "cavity": "Cavité entraînée", "channel": "Canal plan"}
CASE_HELP = {
    "obstacle": "Cylindre, carré, plaque, profil NACA ou image : sillage, Cd, Cl, Strouhal.",
    "cavity": "Cas test de référence, comparé à Ghia, Ghia & Shin (1982) à Re = 100, 400 et 1000.",
    "channel": "Établissement du profil de Poiseuille entre deux parois adhérentes.",
}
# Formes d'obstacles (classes du module geometry).
SHAPES = {
    "cylinder": "Cylindre",
    "square": "Carré",
    "rectangle": "Rectangle / plaque",
    "naca": "Profil NACA 4 chiffres",
    "image": "Image noir et blanc",
}
# Schémas temporels (solver.py), d'advection (operators.py), solveurs de pression (pressure.py).
TIME_SCHEMES = {
    "ab2": "Adams–Bashforth 2 (défaut, économique)",
    "rk3": "Runge–Kutta SSP 3 (robuste)",
    "euler": "Euler explicite (pédagogique)",
}
ADVECTION_SCHEMES = {
    "quick": "QUICK (précis, défaut)",
    "tvd": "TVD van Leer (sans oscillations)",
    "upwind": "Décentré amont (robuste, diffusif)",
    "central": "Centré (oscille si la maille est grossière)",
}
PRESSURE_SOLVERS = {
    "direct": "LU direct (le plus rapide)",
    "cg": "Gradient conjugué (CG)",
    "bicgstab": "BiCGSTAB",
    "sor": "SOR rouge-noir",
    "jacobi": "Jacobi (très lent, pédagogique)",
}
# Préconditionneurs des méthodes de Krylov ; la multigrille n'est proposée que si pyamg est installé.
PRECONDITIONERS = {"ilu": "ILU (factorisation incomplète)", "jacobi": "Jacobi (diagonale)", "none": "Aucun"}
try:
    # Simple test de présence (le paquet n'est importé que s'il existe).
    import importlib.util

    if importlib.util.find_spec("pyamg") is not None:
        PRECONDITIONERS["amg"] = "Multigrille algébrique (pyamg)"
except ImportError:  # pragma: no cover - importlib fait partie de la bibliothèque standard
    pass
# Conditions aux limites (boundary.py) : libellés courts, détails dans les bulles d'aide.
WALLS = {"slip": "Glissante", "noslip": "Adhérente"}
WALLS_HELP = "Glissante : soufflerie idéale, sans frottement (∂u/∂n = 0). Adhérente : paroi fixe, u = v = 0."
OUTLETS = {"convective": "Convective", "neumann": "Neumann"}
OUTLETS_HELP = ("Pression nulle en sortie. Convective : ∂u/∂t + U ∂u/∂x = 0, laisse sortir les tourbillons ; "
                "Neumann : ∂u/∂x = 0.")
PROFILES = {"uniform": "Uniforme", "parabolic": "Parabolique"}
PROFILES_HELP = "Uniforme : U partout. Parabolique : profil de Poiseuille de même débit (maximum 1,5 U)."
# Stations du sillage proposées (x/L) et stations des profils du canal (x/H).
WAKE_STATIONS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0)
CHANNEL_STATIONS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
# Grandeurs de l'explorateur de champs.
FIELDS = {
    "vorticity": "Vorticité ω",
    "speed": "Norme de la vitesse ‖u‖",
    "u": "Vitesse longitudinale u",
    "v": "Vitesse transverse v",
    "cp": "Pression (Cp)",
    "q": "Critère Q",
}
# Explication affichée sous chaque carte.
FIELD_HELP = {
    "vorticity": "ω = ∂v/∂x − ∂u/∂y mesure la rotation locale du fluide : rouge, rotation dans le sens "
    "trigonométrique ; bleu, sens horaire. Les couches cisaillées issues des parois s'enroulent en "
    "tourbillons (allée de Von Kármán derrière un obstacle).",
    "speed": "Norme de la vitesse rapportée à la vitesse de référence : accélération sur les flancs d'un "
    "obstacle, fluide lent dans son sillage et près des parois adhérentes.",
    "u": "Composante longitudinale u/U : les zones bleues (u < 0) sont des écoulements de retour "
    "(bulle de recirculation).",
    "v": "Composante transverse v/U : son alternance haut/bas signale le lâcher tourbillonnaire.",
    "cp": "Coefficient de pression Cp = (p − p∞)/(½ ρ U²) : Cp ≈ 1 au point d'arrêt, dépression (Cp < 0) "
    "sur les flancs et au culot d'un obstacle.",
    "q": "Critère Q = ½(‖Ω‖² − ‖S‖²) : positif (rouge) là où la rotation l'emporte sur la déformation, "
    "c'est-à-dire dans les cœurs des tourbillons.",
}
# Options communes des graphiques Plotly (pas de logo, export PNG en double résolution).
PLOTLY_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


def keep(key: str) -> dict[str, str]:
    """Arguments communs des widgets : clé stable, valeur conservée quand le widget est masqué."""
    return {"key": key, "persist_state": "page"}


# ========================================================================= mise en forme
def thousands(n: int, sep: str = " ") -> str:
    """Entier avec séparateur de milliers à la française (espace fine insécable par défaut)."""
    # {n:,} sépare les milliers par des virgules, remplacées ensuite par l'espace demandée.
    return f"{n:,}".replace(",", sep)


def format_duration(seconds: float) -> str:
    """Durée lisible : « 45 s », « 2 min 05 s », « 1 h 03 min »."""
    # Durée inconnue (NaN) ou infinie.
    if not np.isfinite(seconds):
        return "?"
    # divmod(a, b) renvoie (quotient, reste).
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


def human_size(n_bytes: float) -> str:
    """Taille de fichier lisible (ko, Mo, Go)."""
    for unit, factor in (("Go", 1e9), ("Mo", 1e6), ("ko", 1e3)):
        if n_bytes >= factor:
            return f"{n_bytes / factor:.1f} {unit}".replace(".", ",")
    return f"{int(n_bytes)} o"


def figure_png(fig: Figure, dpi: int = 110) -> bytes:
    """Rendu d'une figure matplotlib en octets PNG (fond de la charte graphique)."""
    buffer = BytesIO()
    # savefig dans le contexte de style : mêmes paramètres qu'à la création de la figure.
    with rc_context(viz.STYLE):
        fig.savefig(buffer, format="png", dpi=dpi, facecolor=viz.SURFACE)
    return buffer.getvalue()


# ================================================================= charte des figures Plotly
# Couleurs de fond, d'encre et de grille pour chaque thème (les couleurs de séries et les
# palettes de la charte du paquet conviennent aux deux).
_LIGHT = {"surface": viz.SURFACE, "ink": viz.INK, "ink2": viz.INK_SECONDARY, "muted": viz.MUTED,
          "grid": viz.GRID_COLOR, "axis": viz.AXIS_COLOR, "solid": viz.SOLID_COLOR, "legend": "rgba(252,252,251,0.85)"}
_DARK = {"surface": "#16171a", "ink": "#ecebe7", "ink2": "#bdbcb5", "muted": "#8f8e88",
         "grid": "#2e2f33", "axis": "#4a4b50", "solid": "#8f8e88", "legend": "rgba(22,23,26,0.85)"}


def is_dark() -> bool:
    """Vrai si la page est affichée en thème sombre (réglage du navigateur ou du menu Streamlit)."""
    try:
        # st.context.theme.type vaut "light" ou "dark" (None hors d'une session Streamlit).
        return getattr(st.context.theme, "type", None) == "dark"
    except (AttributeError, RuntimeError):
        return False


def theme() -> dict[str, str]:
    """Couleurs de fond, d'encre et de grille du thème courant."""
    return _DARK if is_dark() else _LIGHT


def plotly_colorscale(kind: str) -> list[list[float | str]]:
    """Échelle de couleurs Plotly de la charte : ``'diverging'`` (signée) ou ``'sequential'``."""
    if kind == "diverging":
        colors = list(viz.DIVERGING_COLORS)
    else:
        # Échelle séquentielle échantillonnée en 5 couleurs (to_hex : couleur -> "#rrggbb").
        colors = [to_hex(viz.SEQUENTIAL(k / 4)) for k in range(5)]
    # Format Plotly : [[position entre 0 et 1, couleur], ...].
    return [[k / (len(colors) - 1), color] for k, color in enumerate(colors)]


def style_plotly(fig: go.Figure, height: int, *, legend: bool = False) -> go.Figure:
    """Applique la charte graphique (thème clair ou sombre) à une figure Plotly."""
    c = theme()
    fig.update_layout(
        height=height,
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        font=dict(family="Segoe UI, system-ui, sans-serif", color=c["ink2"], size=12),
        margin=dict(l=60, r=30, t=50, b=50),
        hovermode="closest",
        showlegend=legend,
        legend=dict(bgcolor=c["legend"], font=dict(color=c["ink2"])),
        title_font=dict(color=c["ink"], size=15),
    )
    # Grille fine et discrète, axes visibles.
    fig.update_xaxes(gridcolor=c["grid"], zerolinecolor=c["axis"], linecolor=c["axis"], showline=True)
    fig.update_yaxes(gridcolor=c["grid"], zerolinecolor=c["axis"], linecolor=c["axis"], showline=True)
    # Titres des sous-graphiques (annotations) à l'encre principale.
    fig.update_annotations(font=dict(color=c["ink"], size=13))
    return fig


# ============================================================ messages du solveur
# Messages du paquet (sans accents : consoles Windows) -> texte français et conseil.
_MESSAGES = (
    (re.compile(r"Seulement ([\d.]+) periodes analysees"),
     lambda m: f"Seulement {m[1]} périodes analysées : Strouhal peu précis. Allongez la durée simulée (10 périodes "
               "au moins) ou prolongez le calcul."),
    (re.compile(r"Regime etabli non detecte"),
     lambda m: "Régime établi non détecté : l'analyse porte sur la seconde moitié du signal. Allongez la durée "
               "simulée ou prolongez le calcul."),
    (re.compile(r"dt impose \(([^)]+)\) > dt stable \(([^)]+)\) a t = ([^ ]+) : dt reduit"),
     lambda m: f"Pas de temps imposé ({m[1]}) supérieur au pas stable ({m[2]}) à t = {m[3]} : il a été réduit "
               "automatiquement."),
    (re.compile(r"Solveur de pression non converge au pas (\d+) \(residu ([^)]+)\)"),
     lambda m: f"Solveur de pression non convergé au pas {m[1]} (résidu {m[2]}) : augmentez les itérations "
               "maximales ou choisissez le solveur direct."),
    (re.compile(r"Station x/L = ([^ ]+) hors du domaine"),
     lambda m: f"Station de sillage x/L = {m[1]} hors du domaine : ignorée."),
    (re.compile(r"(\d+) cellule\(s\) fluide\(s\) isolee\(s\)"),
     lambda m: f"{m[1]} cellule(s) fluide(s) enfermée(s) dans l'obstacle comblée(s) (sinon l'équation de Poisson "
               "serait singulière)."),
    (re.compile(r"Obstacle .* non resolu par la grille"),
     lambda m: "Obstacle plus petit qu'une maille : il n'apparaît pas sur la grille. Raffinez le maillage."),
    (re.compile(r"Obstacle .* peu resolu : ([\d.]+) cellules"),
     lambda m: f"Obstacle peu résolu ({m[1]} mailles par longueur de référence) : résultats peu fiables."),
)


def friendly_message(message: str) -> str:
    """Traduit un message du paquet en français accentué, avec un conseil (sinon le renvoie tel quel)."""
    for pattern, translate in _MESSAGES:
        match = pattern.search(message)
        if match:
            return translate(match)
    return message

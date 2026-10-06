"""Données de référence et comparaisons pour la validation du solveur."""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``TYPE_CHECKING`` vaut False à l'exécution et True pour les vérificateurs de types :
# l'import ci-dessous ne sert qu'aux annotations (évite un import circulaire).
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .solver import NavierStokesSolver

#: Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, cavité entraînée à Re = 100 :
#: profil u(y) sur l'axe vertical x = 0.5 et profil v(x) sur l'axe horizontal y = 0.5.
#: (Voir aussi :data:`GHIA`, qui regroupe Re = 100, 400 et 1000.)
GHIA_RE100 = {
    # Ordonnées des points de mesure de u (axe vertical médian).
    "y": np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000,
                   0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]),
    # Vitesse horizontale u/U correspondante.
    "u": np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
                   -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000]),
    # Abscisses des points de mesure de v (axe horizontal médian).
    "x": np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5000,
                   0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000]),
    # Vitesse verticale v/U correspondante.
    "v": np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454,
                   -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000]),
}

#: Ghia, Ghia & Shin (1982), tables I et II : profils médians à Re = 100, 400 et 1000 (mêmes
#: positions de mesure que :data:`GHIA_RE100`). La valeur de v publiée pour Re = 400 en
#: x = 0.9063 (-0.23827) est une coquille connue (elle rompt la régularité du profil) : elle est
#: remplacée par NaN et ignorée dans les comparaisons.
GHIA = {
    100: GHIA_RE100,
    400: {
        "y": GHIA_RE100["y"],
        "u": np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477,
                       0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.00000]),
        "x": GHIA_RE100["x"],
        "v": np.array([0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203, 0.30174, 0.05186,
                       -0.38598, -0.44993, np.nan, -0.22847, -0.19254, -0.15663, -0.12146, 0.00000]),
    },
    1000: {
        "y": GHIA_RE100["y"],
        "u": np.array([0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648, -0.06080,
                       0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1.00000]),
        "x": GHIA_RE100["x"],
        "v": np.array([0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235, 0.02526,
                       -0.31966, -0.42665, -0.51500, -0.39188, -0.33714, -0.27669, -0.21388, 0.00000]),
    },
}

#: Tourbillon principal de la cavité (Ghia et al. 1982, table III) : Re -> (x, y, ψ minimal),
#: ψ adimensionnée par U L (u = ∂ψ/∂y, ψ = 0 sur les parois).
GHIA_PRIMARY_VORTEX = {
    100: (0.6172, 0.7344, -0.103423),
    400: (0.5547, 0.6055, -0.113909),
    1000: (0.5313, 0.5625, -0.117929),
}

#: Valeurs de référence pour le cylindre en milieu infini (fourchettes de la littérature) :
#: Re -> {grandeur: (min, max)}. Re = 20 et 40 (stationnaires) : Dennis & Chang (1970) et
#: Fornberg (1980) ; Re = 100 et 200 : simulations 2D publiées (Cd moyen, amplitude de Cl, St).
#: ``Lr`` est la longueur de recirculation mesurée depuis l'arrière du cylindre, en diamètres.
CYLINDER_REFERENCES: dict[int, dict[str, tuple[float, float]]] = {
    20: {"Cd": (2.00, 2.05), "Lr": (0.91, 0.94)},
    40: {"Cd": (1.50, 1.54), "Lr": (2.24, 2.35)},
    100: {"Cd": (1.33, 1.35), "Cl_amplitude": (0.32, 0.34), "St": (0.164, 0.166)},
    200: {"Cd": (1.31, 1.34), "Cl_amplitude": (0.65, 0.70), "St": (0.19, 0.20)},
}


def williamson_strouhal(Re: float) -> float:
    """Loi de Williamson (1989) du nombre de Strouhal d'un cylindre en milieu infini.

    ``St = -3.3265/Re + 0.1816 + 1.6e-4 Re``, établie pour le lâcher parallèle laminaire
    (49 < Re < 178) ; renvoie NaN hors de 47 <= Re <= 190.
    """
    if not 47.0 <= Re <= 190.0:
        return float("nan")
    return -3.3265 / Re + 0.1816 + 1.6e-4 * Re


def cylinder_reference(Re: float, tol: float = 0.02) -> dict[str, tuple[float, float]] | None:
    """Fourchettes de référence du cylindre si ``Re`` est l'un des nombres tabulés (à ``tol`` près)."""
    for tabulated, values in CYLINDER_REFERENCES.items():
        # Comparaison relative (Re = 100.0 saisi au clavier : égalité exacte).
        if abs(Re - tabulated) <= tol * tabulated:
            return values
    return None


def cavity_centerlines(solver: NavierStokesSolver) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Profils ``(y, u(x=L/2, y), x, v(x, y=L/2))`` d'une cavité (Nx et Ny pairs).

    Les faces de la grille MAC tombent exactement sur les axes : aucune interpolation.
    """
    g, st = solver.grid, solver.state
    # Avec Nx pair, la face u d'indice Nx/2 est exactement en x = L/2 (idem pour v et Ny).
    if g.Nx % 2 or g.Ny % 2:
        raise ValueError("Nx et Ny doivent être pairs pour extraire les profils médians.")
    # Ordonnées : paroi du bas (0), centres des cellules, paroi du haut (Ly).
    y = np.concatenate([[0.0], g.y_c, [g.Ly]])
    # Colonne de u sur la face médiane, lignes fantômes comprises.
    u = st.u[g.Nx // 2]
    # Valeurs aux parois = moyenne fantôme/intérieur (0 en bas, vitesse du couvercle en haut),
    # encadrant les valeurs physiques u[1:-1].
    u = np.concatenate([[0.5 * (u[0] + u[1])], u[1:-1], [0.5 * (u[-2] + u[-1])]])  # valeurs aux parois
    # Abscisses : paroi gauche, centres des cellules, paroi droite.
    x = np.concatenate([[0.0], g.x_c, [g.Lx]])
    # Ligne de v sur la face médiane horizontale, colonnes fantômes comprises.
    v = st.v[:, g.Ny // 2]
    # Même traitement des valeurs aux parois (nulles).
    v = np.concatenate([[0.5 * (v[0] + v[1])], v[1:-1], [0.5 * (v[-2] + v[-1])]])
    return y, u, x, v


def ghia_errors(solver: NavierStokesSolver, Re: int = 100) -> tuple[float, float]:
    """Écarts maximaux ``(max|u - u_Ghia|, max|v - v_Ghia|)`` sur les axes médians.

    ``Re`` choisit la table de référence (100, 400 ou 1000) ; les vitesses sont rapportées à
    la vitesse de référence du solveur (vitesse du couvercle).
    """
    # Profils simulés, adimensionnés par la vitesse du couvercle.
    y, u, x, v = cavity_centerlines(solver)
    ref = GHIA[Re]
    U = solver.U_ref
    # np.interp(points, abscisses, valeurs) : interpolation linéaire des profils simulés aux
    # positions de la référence ; np.nanmax ignore les points de référence écartés (NaN).
    return (
        float(np.nanmax(np.abs(np.interp(ref["y"], y, u / U) - ref["u"]))),
        float(np.nanmax(np.abs(np.interp(ref["x"], x, v / U) - ref["v"]))),
    )

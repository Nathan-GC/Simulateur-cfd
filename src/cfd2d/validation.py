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


def ghia_errors(solver: NavierStokesSolver) -> tuple[float, float]:
    """Écarts maximaux ``(max|u - u_Ghia|, max|v - v_Ghia|)`` sur les axes médians (Re = 100)."""
    # Profils simulés.
    y, u, x, v = cavity_centerlines(solver)
    ref = GHIA_RE100
    # np.interp(points, abscisses, valeurs) : interpolation linéaire des profils simulés aux
    # positions de la référence ; écart absolu maximal.
    return (
        float(np.abs(np.interp(ref["y"], y, u) - ref["u"]).max()),
        float(np.abs(np.interp(ref["x"], x, v) - ref["v"]).max()),
    )

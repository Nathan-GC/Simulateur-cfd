"""Tests de la grille décalée (MAC) : pas, formes des tableaux, coordonnées."""

import numpy as np

# pytest : cadre de test ; ``pytest.approx`` compare des flottants avec une tolérance,
# ``pytest.raises`` vérifie qu'une exception est levée.
import pytest

from cfd2d import StaggeredGrid


def test_spacing_and_shapes():
    # Domaine 2 x 1 découpé en 40 x 10 cellules.
    g = StaggeredGrid(Lx=2.0, Ly=1.0, Nx=40, Ny=10)
    # Pas d'espace : 2/40 et 1/10.
    assert g.dx == pytest.approx(0.05)
    assert g.dy == pytest.approx(0.1)
    # Pression aux centres : (Nx, Ny).
    assert g.shape_p == (40, 10)
    # u : Nx+1 faces verticales, Ny lignes + 2 fantômes.
    assert g.shape_u == (41, 12)
    # v : Nx colonnes + 2 fantômes, Ny+1 faces horizontales.
    assert g.shape_v == (42, 11)


def test_coordinates_are_staggered():
    g = StaggeredGrid(Lx=1.0, Ly=1.0, Nx=8, Ny=4)
    # Centres des cellules en (i + 0.5) dx ; assert_allclose compare des tableaux.
    np.testing.assert_allclose(g.x_c, (np.arange(8) + 0.5) / 8)
    # Faces horizontales en j dy.
    np.testing.assert_allclose(g.y_f, np.arange(5) / 4)
    # Maillages 2D des nœuds de u et de v.
    xu, yu = g.u_nodes()
    xv, yv = g.v_nodes()
    # Formes attendues : (Nx+1, Ny) pour u, (Nx, Ny+1) pour v.
    assert xu.shape == (9, 4) and xv.shape == (8, 5)
    # Premier nœud de u : sur la face x = 0, à mi-hauteur de la première cellule.
    assert xu[0, 0] == 0.0 and yu[0, 0] == pytest.approx(0.125)
    # Premier nœud de v : à mi-largeur de la première cellule, sur la face y = 0.
    assert xv[0, 0] == pytest.approx(0.0625) and yv[0, 0] == 0.0


def test_coordinates_are_read_only():
    g = StaggeredGrid(1.0, 1.0, 8, 8)
    # Toute écriture dans un tableau de coordonnées doit échouer (protection _readonly).
    with pytest.raises(ValueError):
        g.x_c[0] = 1.0


def test_locate_clips_to_domain():
    g = StaggeredGrid(1.0, 1.0, 10, 10)
    # Le point (0.55, 0.05) est dans la cellule (5, 0).
    assert g.locate(0.55, 0.05) == (5, 0)
    # Un point hors du domaine est ramené sur la cellule de bord la plus proche.
    assert g.locate(-1.0, 2.0) == (0, 9)


# parametrize : le test est exécuté une fois pour chaque jeu d'arguments invalides
# (longueur nulle, puis nombre de cellules trop faible).
@pytest.mark.parametrize("args", [(0.0, 1.0, 8, 8), (1.0, 1.0, 2, 8)])
def test_invalid_grid(args):
    # *args dépaquette le tuple en arguments positionnels.
    with pytest.raises(ValueError):
        StaggeredGrid(*args)

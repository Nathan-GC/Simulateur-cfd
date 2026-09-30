"""Tests des opérateurs discrets : reconstruction aux faces, advection, laplacien, divergence."""

import numpy as np
import pytest

from cfd2d import StaggeredGrid
from cfd2d.operators import advection, divergence, laplacian_u, reconstruct

# 11 nœuds régulièrement espacés sur [0, 1].
X = np.linspace(0.0, 1.0, 11)  # nœuds
# Positions des 10 faces, au milieu de deux nœuds consécutifs.
XF = 0.5 * (X[:-1] + X[1:])  # faces


# Test répété pour une vitesse advectante positive puis négative.
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_quick_is_exact_for_quadratics(sign):
    # np.full(forme, valeur) : vitesse advectante constante de signe « sign ».
    face = reconstruct(3 * X**2 - X + 2, np.full(XF.shape, sign), 0, "quick")
    # QUICK interpole une parabole exactement sur les faces intérieures (stencil complet).
    np.testing.assert_allclose(face[1:-1], (3 * XF**2 - XF + 2)[1:-1], atol=1e-12)


@pytest.mark.parametrize("scheme", ["central", "tvd"])
def test_second_order_schemes_are_exact_for_linear_data(scheme):
    # Donnée linéaire : les schémas d'ordre 2 la reproduisent exactement.
    face = reconstruct(2 * X + 1, np.ones(XF.shape), 0, scheme)
    np.testing.assert_allclose(face[1:-1], (2 * XF + 1)[1:-1], atol=1e-12)


def test_upwind_takes_upstream_node():
    # Nœuds 0, 1, 2, 3, 4.
    q = np.arange(5.0)
    # Vitesses alternées : +, -, +, -.
    a = np.array([1.0, -1.0, 1.0, -1.0])
    # Face k prend le nœud k si a > 0, le nœud k+1 sinon.
    np.testing.assert_array_equal(reconstruct(q, a, 0, "upwind"), [0.0, 2.0, 2.0, 4.0])


def test_tvd_does_not_overshoot_a_step():
    # Marche (discontinuité) entre 0 et 1.
    q = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    for sign in (1.0, -1.0):
        face = reconstruct(q, np.full(5, sign), 0, "tvd")
        # Pas de dépassement : les valeurs restent dans [0, 1] (propriété TVD).
        assert face.min() >= 0.0 and face.max() <= 1.0


def test_reconstruct_along_axis_1_matches_axis_0():
    # Générateur aléatoire à graine fixe (résultats reproductibles).
    rng = np.random.default_rng(1)
    # Données (7 nœuds x 5) et vitesses (6 faces x 5) aléatoires.
    q, a = rng.standard_normal((7, 5)), rng.standard_normal((6, 5))
    for scheme in ("upwind", "central", "quick", "tvd"):
        # Reconstruire selon l'axe 1 des transposées = transposée de la reconstruction selon l'axe 0.
        np.testing.assert_allclose(reconstruct(q.T, a.T, 1, scheme), reconstruct(q, a, 0, scheme).T)


def test_invalid_scheme():
    # Un nom de schéma inconnu lève ValueError.
    with pytest.raises(ValueError):
        reconstruct(X, np.ones(XF.shape), 0, "weno")


def test_uniform_field_has_no_advection_or_diffusion():
    g = StaggeredGrid(2.0, 1.0, 16, 8)
    # Écoulement uniforme u = 1.5, v = 0 (fantômes compris).
    u, v = np.full(g.shape_u, 1.5), np.zeros(g.shape_v)
    for scheme in ("upwind", "central", "quick", "tvd"):
        # Un champ uniforme ne transporte aucune quantité de mouvement nette.
        adv_u, adv_v = advection(u, v, g, scheme)
        assert np.abs(adv_u).max() < 1e-12 and np.abs(adv_v).max() < 1e-12
    # Laplacien d'une constante : nul.
    assert np.abs(laplacian_u(u, g)).max() < 1e-10
    # Divergence d'un champ uniforme : nulle.
    assert np.abs(divergence(u, v, g)).max() < 1e-12


def test_laplacian_is_exact_for_quadratics():
    g = StaggeredGrid(1.0, 1.0, 8, 8)
    # Abscisses des faces u (colonne) et ordonnées des lignes u, fantômes compris (ligne).
    x = g.x_f[:, None]
    y = (np.arange(g.Ny + 2) - 0.5)[None, :] * g.dy  # lignes fantômes incluses
    # u = x² + 3 y² : laplacien exact = 2 + 6 = 8 ; le schéma à 5 points est exact pour
    # les polynômes de degré 2.
    u = x**2 + 3 * y**2
    np.testing.assert_allclose(laplacian_u(u, g), 8.0, atol=1e-9)

"""Tests de l'équation de Poisson et de la projection (5 solveurs de pression)."""

import numpy as np
import pytest

from cfd2d import NavierStokesSolver, NumericsConfig, StaggeredGrid, presets
from cfd2d.operators import divergence
from cfd2d.pressure import PoissonOperator, make_pressure_solver


def cylinder_solver(cells_per_diameter: int = 8, **numerics) -> NavierStokesSolver:
    """Petit cas « cylindre » (domaine 8 x 4 D) avec des options numériques au choix."""
    # Configuration type réduite.
    cfg = presets.cylinder_flow(cells_per_diameter=cells_per_diameter, length=8.0, height=4.0, x_center=2.0)
    # **numerics transmet les arguments nommés restants (solveur de pression, tolérance...).
    cfg.numerics = NumericsConfig(**numerics)
    return NavierStokesSolver(cfg)


def random_predicted_field(solver: NavierStokesSolver, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Champ aléatoire respectant les conditions aux limites (non solénoïdal)."""
    # Générateur aléatoire à graine fixe.
    rng = np.random.default_rng(seed)
    # Vitesses aléatoires (loi normale) aux formes des tableaux du solveur.
    u = rng.standard_normal(solver.state.u.shape)
    v = rng.standard_normal(solver.state.v.shape)
    # Mêmes contraintes qu'un champ prédit : CL normales, sorties extrapolées, faces solides nulles.
    solver.boundary.apply_normal(u, v, 0.0)
    solver.boundary.extrapolate_outlets(u, v)
    solver._zero_solid_faces(u, v)
    return u, v


def test_poisson_matrix_is_symmetric_positive_definite():
    # Petite grille : la matrice peut être convertie en matrice pleine (toarray).
    s = cylinder_solver(cells_per_diameter=4)
    A = s.poisson.matrix.toarray()
    # Symétrie A = Aᵀ.
    np.testing.assert_allclose(A, A.T)
    # eigvalsh : valeurs propres d'une matrice symétrique ; toutes > 0 = définie positive.
    assert np.linalg.eigvalsh(A).min() > 0.0


# Test répété pour chaque couple (méthode, préconditionneur).
@pytest.mark.parametrize(
    ("method", "preconditioner"),
    [("direct", "none"), ("cg", "jacobi"), ("cg", "ilu"), ("bicgstab", "ilu"), ("sor", "none"), ("jacobi", "none")],
)
def test_projection_removes_divergence(method, preconditioner):
    # Solveur avec la méthode testée, tolérance serrée.
    s = cylinder_solver(
        cells_per_diameter=4,
        pressure_solver=method,
        preconditioner=preconditioner,
        pressure_tol=1e-10,
        pressure_maxiter=200_000,
    )
    # Champ non solénoïdal et sa divergence initiale.
    u, v = random_predicted_field(s)
    div0 = np.abs(divergence(u, v, s.grid)).max()
    # Projection (u et v modifiés en place).
    p, info = s._project(u, v, dt_eff=0.1)
    assert info.converged
    # La divergence est réduite d'au moins 8 ordres de grandeur.
    assert np.abs(divergence(u, v, s.grid)).max() < 1e-8 * div0
    # les faces figées (solide, parois, entrée) ne sont pas modifiées par la projection
    assert np.all(u[:, 1:-1][s._u_fixed] == 0.0)
    assert np.all(v[1:-1, :][s._v_fixed] == 0.0)
    # Référence : même champ projeté avec le solveur direct.
    reference = cylinder_solver(cells_per_diameter=4)
    u_ref, v_ref = random_predicted_field(reference)
    p_ref, _ = reference._project(u_ref, v_ref, dt_eff=0.1)
    # Toutes les méthodes donnent la même pression.
    np.testing.assert_allclose(p, p_ref, atol=1e-6 * np.abs(p_ref).max())


def test_projection_conserves_mass_through_outlet():
    s = cylinder_solver()
    u, v = random_predicted_field(s)
    s._project(u, v, dt_eff=0.1)
    # Débits entrant (face x = 0) et sortant (face x = Lx) après projection.
    inflow = u[0, 1:-1].sum() * s.grid.dy
    outflow = u[-1, 1:-1].sum() * s.grid.dy
    # Conservation globale de la masse.
    assert outflow == pytest.approx(inflow, rel=1e-10)


def test_projection_in_closed_cavity_is_well_posed():
    # Cavité fermée : problème de Neumann pur (singulier) régularisé.
    s = NavierStokesSolver(presets.lid_driven_cavity(N=24))
    assert s.poisson.singular
    u, v = random_predicted_field(s)
    p, _ = s._project(u, v, dt_eff=0.1)
    # Champ projeté à divergence nulle.
    assert np.abs(divergence(u, v, s.grid)).max() < 1e-10
    # Pression de moyenne nulle (choix de la constante).
    assert abs(p.mean()) < 1e-12


def test_projection_is_idempotent():
    s = cylinder_solver()
    u, v = random_predicted_field(s)
    # Première projection.
    s._project(u, v, dt_eff=0.1)
    # Copie du champ projeté.
    u1, v1 = u.copy(), v.copy()
    # Seconde projection : elle ne doit rien changer (pression nulle).
    p, _ = s._project(u, v, dt_eff=0.1)
    np.testing.assert_allclose(u, u1, atol=1e-12)
    np.testing.assert_allclose(v, v1, atol=1e-12)
    assert np.abs(p).max() < 1e-10


def _poisson_error(N: int, closed: bool) -> float:
    """Erreur max de la solution discrète de -∇²p = f pour une solution analytique."""
    g = StaggeredGrid(1.0, 1.0, N, N)
    # Aucun solide.
    solid = np.zeros(g.shape_p, dtype=bool)
    # Toutes les faces intérieures sont actives.
    u_act = np.zeros((N + 1, N), dtype=bool)
    u_act[1:-1] = True
    v_act = np.zeros((N, N + 1), dtype=bool)
    v_act[:, 1:-1] = True
    # Faces de Dirichlet (sortie), aucune par défaut.
    u_dir = np.zeros_like(u_act)
    v_dir = np.zeros_like(v_act)
    X, Y = g.cell_centers()
    if closed:  # Neumann partout : p = cos(πx) cos(πy)
        exact = np.cos(np.pi * X) * np.cos(np.pi * Y)
        # -∇²p = 2π² p.
        lam = 2.0 * np.pi**2
    else:  # p = 0 à l'est (sortie), Neumann ailleurs : p = cos(πx/2) cos(πy)
        # La face est devient une face de sortie (Dirichlet p = 0), active.
        u_dir[-1] = u_act[-1] = True
        exact = np.cos(0.5 * np.pi * X) * np.cos(np.pi * Y)
        # -∇²p = (π²/4 + π²) p.
        lam = 1.25 * np.pi**2
    # Opérateur discret.
    op = PoissonOperator(g, solid, u_act, v_act, u_dir, v_dir)
    # Second membre f = λ p_exact.
    b = lam * exact.ravel()
    # Cas singulier : second membre compatible et cellule de référence.
    if op.singular:
        b -= b.mean()
        b[op.pin] = 0.0
    # Résolution directe.
    p, _ = make_pressure_solver("direct", op).solve(b)
    # Cas singulier : solutions comparées à constante près (moyennes retirées).
    if closed:
        p, exact = p - p.mean(), exact.ravel() - exact.mean()
    # Erreur maximale.
    return float(np.abs(p - exact.ravel()).max())


@pytest.mark.parametrize("closed", [False, True])
def test_poisson_operator_is_second_order(closed):
    # Erreurs sur deux grilles (16² puis 32²).
    e1, e2 = _poisson_error(16, closed), _poisson_error(32, closed)
    # Ordre de convergence log2(e1/e2) ≈ 2 : l'erreur est divisée par 4 quand h est divisé par 2.
    assert np.log2(e1 / e2) == pytest.approx(2.0, abs=0.15)

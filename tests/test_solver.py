"""Tests du solveur : écoulements analytiques, validation (Ghia), schémas, pas de temps."""

import logging

import numpy as np
import pytest

from cfd2d import (
    BoundaryConfig,
    DomainConfig,
    FlowConfig,
    Inlet,
    NavierStokesSolver,
    NoSlipWall,
    NumericsConfig,
    SimulationConfig,
    TimeConfig,
    presets,
)
from cfd2d.validation import ghia_errors


def test_uniform_flow_is_preserved_exactly():
    # Canal à parois glissantes sans obstacle : l'écoulement uniforme est solution exacte.
    cfg = SimulationConfig(
        domain=DomainConfig(Lx=4.0, Ly=1.0, Nx=64, Ny=16),
        flow=FlowConfig(U_inf=2.0, Re=100.0),
        boundaries=BoundaryConfig.channel(walls="slip"),
        time=TimeConfig(t_end=1.0),
    )
    s = NavierStokesSolver(cfg)
    # 50 pas de temps.
    s.run(max_steps=50)
    # u reste égal à 2 partout, v à 0, la pression à 0 (erreurs d'arrondi seulement).
    np.testing.assert_allclose(s.state.u[:, 1:-1], 2.0, atol=1e-12)
    np.testing.assert_allclose(s.state.v[1:-1, :], 0.0, atol=1e-12)
    assert np.abs(s.state.p).max() < 1e-10


def test_poiseuille_flow_develops_in_channel():
    # Canal à parois adhérentes, entrée uniforme, Re = 10.
    s = NavierStokesSolver(presets.channel_flow(Re=10.0, Nx=64, Ny=16, t_end=15.0))
    s.run()
    g = s.grid
    # Colonne de faces u aux 3/4 du canal.
    i = int(0.75 * g.Nx)  # x = 3H : écoulement établi
    # Ordonnée réduite des lignes physiques.
    eta = g.y_c / g.Ly
    # Profil établi : parabole de Poiseuille u = 6 U η(1-η) (débit unitaire).
    np.testing.assert_allclose(s.state.u[i, 1:-1], 6.0 * eta * (1.0 - eta), atol=0.01)
    # Gradient de pression moyen entre deux colonnes de cellules.
    dpdx = np.mean((s.state.p[i] - s.state.p[i - 1]) / g.dx)
    # Valeur analytique dp/dx = -12 μ U / H².
    assert dpdx == pytest.approx(-12.0 * s.nu * s.rho * 1.0 / g.Ly**2, rel=0.02)
    # Incompressibilité respectée à chaque pas.
    assert max(s.diagnostics.divergence_max) < 1e-10


def test_lid_driven_cavity_matches_ghia():
    # Cavité entraînée à Re = 100, grille 48 x 48, jusqu'à l'état stationnaire.
    s = NavierStokesSolver(presets.lid_driven_cavity(Re=100.0, N=48, t_end=25.0))
    s.run()
    # Écarts maximaux aux profils de référence de Ghia et al. (1982).
    err_u, err_v = ghia_errors(s)
    assert err_u < 0.015 and err_v < 0.015


# Deux paramétrisations empilées : 3 schémas temporels x 4 schémas d'advection = 12 cas.
@pytest.mark.parametrize("scheme", ["euler", "ab2", "rk3"])
@pytest.mark.parametrize("advection", ["upwind", "central", "quick", "tvd"])
def test_schemes_run_divergence_free_around_cylinder(scheme, advection):
    # Petit cylindre (8 cellules par diamètre).
    cfg = presets.cylinder_flow(cells_per_diameter=8, length=10.0, height=5.0, x_center=3.0)
    cfg.time = TimeConfig(t_end=2.0, scheme=scheme)
    cfg.numerics = NumericsConfig(advection=advection)
    s = NavierStokesSolver(cfg)
    s.run()
    # Le calcul atteint exactement l'instant final.
    assert s.time == pytest.approx(2.0)
    # Divergence à la précision machine à chaque pas.
    assert max(s.diagnostics.divergence_max) < 1e-10
    # Vitesse nulle dans le solide.
    uc, vc = s.cell_velocity()
    assert np.all(uc[s.solid] == 0.0) and np.all(vc[s.solid] == 0.0)
    assert np.abs(uc).max() < 3.0  # pas d'explosion numérique


def test_convective_outlet_keeps_global_mass_balance():
    s = NavierStokesSolver(presets.cylinder_flow(cells_per_diameter=8, length=10.0, height=5.0, x_center=3.0))
    s.run(t_end=3.0)
    dy = s.grid.dy
    # Débit sortant = débit entrant (sortie convective + projection).
    assert s.state.u[-1, 1:-1].sum() * dy == pytest.approx(s.state.u[0, 1:-1].sum() * dy, rel=1e-10)


def test_imposed_dt_is_reduced_when_unstable(caplog):
    # dt imposé beaucoup trop grand pour la stabilité.
    cfg = presets.lid_driven_cavity(N=16, t_end=0.5)
    cfg.time = TimeConfig(t_end=0.5, dt=0.5)
    s = NavierStokesSolver(cfg)
    with caplog.at_level(logging.WARNING):
        s.run()
    # Avertissement émis et pas effectivement réduit.
    assert "dt reduit" in caplog.text
    assert max(s.diagnostics.dt) < 0.5
    # Le nombre de Courant reste sous la valeur par défaut d'AB2 (0.4).
    assert max(s.diagnostics.cfl) <= 0.4 + 1e-12


def test_stable_imposed_dt_is_kept():
    # dt imposé stable : utilisé tel quel.
    cfg = presets.lid_driven_cavity(N=16, t_end=0.1)
    cfg.time = TimeConfig(t_end=0.1, dt=0.005)
    s = NavierStokesSolver(cfg)
    s.run()
    np.testing.assert_allclose(s.diagnostics.dt, 0.005)
    # 0.1 / 0.005 = 20 pas.
    assert len(s.diagnostics) == 20


def test_unconverged_pressure_solver_is_reported(caplog):
    # SOR limité à 2 itérations : il ne peut pas converger.
    cfg = presets.lid_driven_cavity(N=16)
    cfg.numerics = NumericsConfig(pressure_solver="sor", pressure_maxiter=2)
    s = NavierStokesSolver(cfg)
    with caplog.at_level(logging.WARNING):
        s.run(max_steps=3)
    assert "non converge" in caplog.text


def test_callbacks_are_called_at_requested_interval():
    s = NavierStokesSolver(presets.lid_driven_cavity(N=16))
    # Liste des numéros de pas vus par le rappel.
    calls = []
    # lambda : petite fonction anonyme ajoutant le numéro de pas à la liste, tous les 5 pas.
    s.add_callback(lambda solver: calls.append(solver.state.step), every=5)
    s.run(max_steps=23)
    # Appels aux pas 5, 10, 15 et 20.
    assert calls == [5, 10, 15, 20]


def test_ramped_inlet_starts_from_rest():
    # Entrée avec montée en vitesse sur une unité de temps.
    cfg = presets.channel_flow(t_end=1.0)
    cfg.boundaries = BoundaryConfig.channel(walls="no-slip", ramp_time=1.0)
    s = NavierStokesSolver(cfg)
    # À t = 0 le fluide est au repos.
    assert np.abs(s.state.u).max() == 0.0
    s.run(t_end=0.5)
    # À mi-rampe : U ½ (1 - cos(π/2)) = 0.5.
    assert s.state.u[0, 1:-1] == pytest.approx(0.5)  # 0.5 (1 - cos(π/2))


def test_configuration_errors():
    # nu et Re donnés ensemble : ambigu.
    with pytest.raises(ValueError):
        FlowConfig(nu=0.01, Re=100.0)
    # Ni nu ni Re : viscosité inconnue.
    with pytest.raises(ValueError):
        FlowConfig()
    # Entrée sans sortie : incompatible avec l'incompressibilité.
    with pytest.raises(ValueError):
        BoundaryConfig(west=Inlet(), east=NoSlipWall()).validate()
    # Schémas inconnus.
    with pytest.raises(ValueError):
        NumericsConfig(advection="weno")
    with pytest.raises(ValueError):
        TimeConfig(scheme="rk4")

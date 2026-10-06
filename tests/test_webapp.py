"""Tests des fonctions de l'application web sans interface : paramètres, configuration, diagnostic,
script généré, extrapolation de Richardson, enregistrement et relecture d'un calcul."""

import numpy as np
import pytest

# Application web : ignorée si Streamlit n'est pas installé.
pytest.importorskip("streamlit")

from cfd2d import NACA4, Cylinder, Rectangle  # noqa: E402
from webapp import analysis, history, runner, store  # noqa: E402
from webapp.codegen import python_script  # noqa: E402
from webapp.common import friendly_message  # noqa: E402
from webapp.params import DEFAULTS, build_config, check_params, estimate_cost, state_values  # noqa: E402


def obstacle_params(**changes):
    """Paramètres du cas « obstacle » par défaut (ceux de l'interface), modifiés par ``changes``."""
    p = {
        "case": "obstacle", "visc_mode": "re", "Re": 100.0, "nu": None, "U": 1.0, "rho": 1.0, "t_end": 80.0,
        "scheme": "ab2", "dt": None, "cfl": None, "fourier": None, "advection": "quick", "pressure_solver": "direct",
        "pressure_tol": 1e-8, "pressure_maxiter": 10_000, "preconditioner": "ilu", "sor_omega": None,
        "live_preview": False, "shape": "cylinder", "rotation": 0.0, "incidence": 0.0, "rect_width": 2.0,
        "rect_height": 0.4, "naca_code": "2412", "image": None, "image_width": 2.0, "threshold": 0.5, "invert": False,
        "upstream": 4.0, "downstream": 12.0, "height": 8.0, "y_offset": 0.0, "cells": 12, "supersampling": 4,
        "perturbation": 0.05, "ramp_time": 0.0, "inlet_profile": "uniform", "outlet": "convective", "north": "slip",
        "south": "slip", "avg_start": None, "stations": (1.0, 2.0), "control_volume": False, "animation": False,
        "anim_duration": 20.0, "fps": 20,
    }
    return p | changes


def test_build_config_follows_the_parameters():
    config = build_config(obstacle_params())
    # Domaine 16 D x 8 D à 12 mailles par D, cylindre de diamètre 1 en (4, 4).
    assert (config.domain.Nx, config.domain.Ny) == (192, 96)
    assert isinstance(config.obstacles[0], Cylinder) and config.obstacles[0].center == (4.0, 4.0)
    assert config.reynolds == pytest.approx(100.0)
    # Plaque à incidence positive : rotation de sens horaire ; profil : quart de corde au centre.
    plate = build_config(obstacle_params(shape="rectangle", incidence=20.0)).obstacles[0]
    assert isinstance(plate, Rectangle) and plate.angle_deg == -20.0
    naca = build_config(obstacle_params(shape="naca", incidence=8.0)).obstacles[0]
    assert isinstance(naca, NACA4) and naca.pivot_point == pytest.approx((4.0, 4.0))


def test_blockage_uses_the_frontal_height():
    # Profil NACA 2412 à 8° dans un domaine haut de 4 cordes : blocage réel ≈ 4,6 % (et non 25 %).
    config, errors, advice = check_params(obstacle_params(shape="naca", incidence=8.0, height=4.0, cells=24))
    assert not errors
    assert not any("Blocage élevé" in a for a in advice)
    # 24 mailles par corde : l'épaisseur (12 %) n'est résolue que par ~3 mailles, ce qui est signalé.
    assert any("Épaisseur résolue" in a for a in advice)


def test_invalid_parameters_are_reported():
    _, errors, _ = check_params(obstacle_params(shape="naca", naca_code="12"))
    assert errors and "NACA" in errors[0]
    _, errors, _ = check_params(obstacle_params(upstream=1.5, height=2.0, y_offset=0.8))
    assert errors and "bord du domaine" in errors[0]


def test_state_values_invert_read_params():
    # Les paramètres d'une étude ou d'un scénario se traduisent en valeurs de widgets.
    values = state_values("obstacle", {"Re": 40.0, "cfl": 0.8, "ramp_time": 2.0, "incidence": 5.0, "avg_start": None})
    assert values["obs_visc_mode"] == "re" and values["obs_re"] == 40.0
    assert values["obs_cfl_auto"] is False and values["obs_cfl"] == 0.8
    assert values["obs_start"] == "ramp" and values["obs_ramp"] == 2.0
    assert values["obs_rect_alpha"] == values["obs_naca_alpha"] == 5.0
    assert values["obs_avg_mode"] == "auto"
    # Toutes les clés produites existent dans les valeurs initiales des widgets.
    assert set(values) - {"case"} <= set(DEFAULTS)


def test_cost_estimate_scales_with_resolution():
    coarse = estimate_cost(build_config(obstacle_params(cells=12)), 1.6)
    fine = estimate_cost(build_config(obstacle_params(cells=24)), 1.6)
    # Deux fois plus de mailles par direction : 2 à 4 fois plus de pas (advection ∝ 1/Δx, diffusion ∝ 1/Δx²)
    # et quatre fois plus de cellules.
    assert 2.0 <= fine["steps"] / coarse["steps"] <= 4.0
    assert fine["base_seconds"] > 5 * coarse["base_seconds"]
    assert 0.0 < coarse["advective_share"] < 1.0


@pytest.mark.parametrize("params", [
    obstacle_params(), obstacle_params(shape="naca", incidence=4.0), obstacle_params(shape="rectangle"),
    {"case": "cavity", "visc_mode": "re", "Re": 100.0, "nu": None, "U": 1.0, "rho": 1.0, "t_end": 25.0, "scheme": "ab2",
     "dt": None, "cfl": None, "fourier": None, "advection": "quick", "pressure_solver": "direct", "pressure_tol": 1e-8,
     "pressure_maxiter": 10_000, "preconditioner": "ilu", "sor_omega": None, "live_preview": False, "N": 32},
])
def test_generated_script_rebuilds_the_same_configuration(params):
    script = python_script(params, "test")
    # Le script est du Python valide ; la configuration qu'il définit est celle de l'application.
    namespace: dict = {}
    head = script.split("# Messages de progression")[0]
    exec(compile(head, "mon_calcul.py", "exec"), namespace)
    assert namespace["config"] == build_config(params)


def test_richardson_recovers_a_second_order_series():
    # φ(h) = 1 + 0.5 h² : ordre 2, limite 1 (rapport de raffinement 1,5).
    h = np.array([1 / 8, 1 / 12, 1 / 18])
    result = history.richardson(h, 1.0 + 0.5 * h**2)
    assert result["p"] == pytest.approx(2.0, rel=1e-6)
    assert result["extrapolated"] == pytest.approx(1.0, rel=1e-9)


def test_thin_airfoil_lift():
    # Profil symétrique : Cl = 2π α ; NACA 2412 : incidence de portance nulle ≈ -2,1°.
    assert analysis.thin_airfoil_cl("0012", 5.0) == pytest.approx(2 * np.pi * np.radians(5.0))
    assert analysis.thin_airfoil_cl("2412", -2.08) == pytest.approx(0.0, abs=0.01)


def test_solver_messages_are_translated():
    text = friendly_message("Seulement 1.2 periodes analysees : Strouhal peu precis.")
    assert "périodes analysées" in text and "Prolong" in text.title()


def test_run_is_saved_reloaded_and_prolonged(tmp_path, monkeypatch):
    # Dossier d'enregistrement temporaire.
    monkeypatch.setattr(store, "RUNS_DIR", tmp_path)
    p = obstacle_params(cells=8, upstream=2.0, downstream=6.0, height=4.0, t_end=3.0, animation=True,
                        anim_duration=1.0, stations=(1.0,))
    config, errors, _ = check_params(p)
    assert not errors
    job = runner.Job([runner.RunRequest(params=p, config=config, label="test")])
    runner.run_job(job)
    assert job.status == "done" and not job.errors
    result = job.results[0]
    # Moyenne choisie automatiquement, instantanés et animation par défaut disponibles.
    assert result.mean_state is not None and result.snapshots and ("vorticity", 20) in result.animations
    # Relecture depuis le disque : mêmes efforts, même état final.
    loaded = store.load_run(result.run_id)
    np.testing.assert_allclose(loaded.history.cd, result.history.cd)
    assert loaded.solver.time == pytest.approx(result.solver.time)
    # Prolongation du calcul relu (nouveaux moniteurs, historiques concaténés).
    job = runner.Job([runner.RunRequest(params=loaded.params, config=None, label=loaded.label, base=loaded,
                                        extra_time=1.0)])
    runner.run_job(job)
    prolonged = job.results[0]
    assert prolonged.solver.time == pytest.approx(4.0)
    assert prolonged.history.time[-1] == pytest.approx(4.0) and prolonged.history.time.size > result.history.time.size

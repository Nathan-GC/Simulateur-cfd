"""Tests de l'interface (app.py) sans navigateur, avec ``streamlit.testing.v1.AppTest``.

Les calculs y sont exécutés dans le fil de la page (variable d'environnement CFD2D_SYNC_RUNS)
et enregistrés dans un dossier temporaire.
"""

from pathlib import Path

import pytest

# AppTest : exécute le script Streamlit sans navigateur ; tests ignorés si Streamlit est absent.
testing = pytest.importorskip("streamlit.testing.v1")

from webapp import store  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"

# Petits réglages rapides de chaque cas (clés de widgets -> valeurs).
SMALL = {
    "obstacle": {"obs_cells": 8, "obs_upstream": 2.0, "obs_downstream": 5.0, "obs_height": 4.0, "obs_t_end": 6.0,
                 "obs_anim_duration": 3.0, "obs_stations": [1.0, 2.0], "obs_live": False},
    "cavity": {"cav_N": 16, "cav_t_end": 2.0, "cav_live": False},
    "channel": {"chn_Ny": 8, "chn_length": 3.0, "chn_t_end": 2.0, "chn_live": False},
}
# Sous-onglets de résultats de chaque cas.
RESULT_TABS = {
    "obstacle": ["🌀 Champs", "📈 Efforts", "🧱 Paroi & sillage", "🎞️ Animation", "🔬 Méthode", "🖼️ Tableau de bord",
                 "🩺 Diagnostics", "💾 Exports"],
    "cavity": ["✅ Validation (Ghia)", "🌀 Champs", "🔬 Méthode", "🩺 Diagnostics", "💾 Exports"],
    "channel": ["📐 Profils de Poiseuille", "🌀 Champs", "🔬 Méthode", "🩺 Diagnostics", "💾 Exports"],
}


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Application prête à être exécutée (calculs synchrones, enregistrements temporaires)."""
    monkeypatch.setenv("CFD2D_SYNC_RUNS", "1")
    monkeypatch.setattr(store, "RUNS_DIR", tmp_path)
    return testing.AppTest.from_file(str(APP), default_timeout=600)


def check(at):
    """Aucune exception affichée par la page."""
    assert not at.exception, [e.value for e in at.exception]


def test_documentation_pages(app):
    app.run()
    check(app)
    assert [t.label for t in app.tabs[:2]] == ["📚 Théorie & Documentation", "🚀 Simulation Interactive"]
    for choice in ("README", "Tutoriel", "Exercices", "Code source", "Théorie illustrée"):
        app.segmented_control(key="doc_choice").set_value(choice).run()
        check(app)


@pytest.mark.parametrize("case", ["obstacle", "cavity", "channel"])
def test_quick_simulation_and_result_tabs(app, case):
    for key, value in SMALL[case].items():
        app.session_state[key] = value
    app.session_state["case"] = case
    app.run()
    check(app)
    app.button(key="run").click().run()
    check(app)
    result = app.session_state["result"]
    assert result.case == case and result.solver.state.step > 10
    # Chaque sous-onglet de résultats s'affiche sans erreur.
    for label in RESULT_TABS[case]:
        app.session_state[f"result_tab_{case}"] = label
        app.run()
        check(app)
    # Historique : le calcul y figure.
    app.session_state["right_tabs"] = "🗂️ Historique"
    app.run()
    check(app)
    assert len(store.list_runs()) == 1


def test_airfoil_shows_lift_metrics_and_chordwise_pressure(app):
    # Profil NACA 2412 à 6° (corps élancé) : indicateurs de portance et Cp le long de la corde.
    settings = SMALL["obstacle"] | {"obs_shape": "naca", "obs_naca_alpha": 6.0, "obs_cells": 16, "obs_t_end": 4.0,
                                    "obs_anim": False}
    for key, value in settings.items():
        app.session_state[key] = value
    app.run()
    app.button(key="run").click().run()
    check(app)
    result = app.session_state["result"]
    assert result.slender and result.cp_chord is not None and "lift_to_drag" in result.extra
    # Cartes adaptées au corps portant (la page Théorie affiche aussi des cartes : on cherche celles-ci).
    labels = [m.label for m in app.metric]
    assert {"Cl moyen", "Finesse L/D", "Cm"} <= set(labels) and "Strouhal" not in labels
    app.session_state["result_tab_obstacle"] = "🧱 Paroi & sillage"
    app.run()
    check(app)


def test_scenario_loads_its_settings(app):
    app.session_state["scenario"] = "regimes"
    app.run()
    check(app)
    # Étape 1 du scénario « régimes » : Re = 20, sans perturbation.
    app.button(key="scenario_regimes_1").click().run()
    check(app)
    assert app.session_state["obs_re"] == 20.0 and app.session_state["obs_perturbation"] == 0.0


def test_parametric_study(app):
    settings = SMALL["cavity"] | {"case": "cavity", "study_on": True, "study_param_cavity": "Re",
                                  "study_values_cavity_Re": "50, 100"}
    for key, value in settings.items():
        app.session_state[key] = value
    app.run()
    check(app)
    app.button(key="run").click().run()
    check(app)
    # Deux calculs enregistrés, appartenant à la même étude ; l'onglet Études s'affiche.
    runs = store.list_runs()
    assert len(runs) == 2 and len({r["study"]["id"] for r in runs}) == 1
    app.session_state["right_tabs"] = "📈 Études"
    app.run()
    check(app)

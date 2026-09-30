"""Tests de la visualisation : tableau de bord, animations GIF/MP4, rapport HTML."""

# ``shutil.which`` : présence de l'exécutable ffmpeg (test MP4).
import shutil

import numpy as np
import pytest

# Pillow : relecture des images produites.
from PIL import Image

from cfd2d import FieldAverager, ForceMonitor, NavierStokesSolver, presets
from cfd2d.visualization import (
    FrameRecorder,
    animate,
    corner_solid,
    interactive_report,
    plot_dashboard,
    report_figure,
)


@pytest.fixture(scope="module")
def run():
    """Petit calcul de cylindre avec moniteurs (partagé par les tests du module)."""
    # scope="module" : exécuté une seule fois pour tous les tests de ce fichier.
    s = NavierStokesSolver(presets.cylinder_flow(cells_per_diameter=8, length=10.0, height=5.0, x_center=3.0))
    forces = ForceMonitor(s)
    averager = FieldAverager(s, t_start=1.0)
    recorder = FrameRecorder(s, every=10)
    s.run(t_end=3.0)
    return s, forces.history(), averager.fields(), recorder


def test_corner_solid_marks_only_fully_surrounded_corners():
    # Bloc solide de 2 x 2 cellules au milieu d'une grille 4 x 4.
    solid = np.zeros((4, 4), dtype=bool)
    solid[1:3, 1:3] = True
    corners = corner_solid(solid)
    # Seul le coin central (2, 2) est entouré de 4 cellules solides.
    assert corners.shape == (5, 5)
    assert corners.sum() == 1 and corners[2, 2]


@pytest.mark.parametrize("main", ["vorticity", "streamlines"])
def test_dashboard_is_rendered(tmp_path, run, main):
    s, history, mean, _ = run
    path = tmp_path / f"dashboard_{main}.png"
    fig = plot_dashboard(s, history, mean_fields=mean, main=main, path=path)
    # 7 axes de données + barre(s) de couleurs.
    assert len(fig.axes) >= 7
    # Image PNG valide et de taille attendue (15 x 14 pouces à 120 ppp).
    with Image.open(path) as image:
        assert image.size == (1800, 1680)


def test_gif_animation(tmp_path, run):
    s, _, _, recorder = run
    path = animate(recorder, tmp_path / "vorticity.gif", fps=10)
    # Le GIF contient une image par instant enregistré (n_frames : nombre d'images).
    with Image.open(path) as gif:
        assert gif.n_frames == len(recorder)


# skipif : test ignoré si ffmpeg n'est pas installé sur la machine.
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent")
def test_mp4_animation(tmp_path, run):
    s, _, _, recorder = run
    path = animate(recorder, tmp_path / "vorticity.mp4", fps=10)
    # Fichier MP4 non vide ; les octets 4 à 8 contiennent la signature « ftyp » du conteneur.
    data = path.read_bytes()
    assert len(data) > 1000 and data[4:8] == b"ftyp"


def test_unknown_animation_format(tmp_path, run):
    _, _, _, recorder = run
    with pytest.raises(ValueError):
        animate(recorder, tmp_path / "vorticity.avi")


def test_report_figure_layout(run):
    s, history, mean, _ = run
    layout = report_figure(history, solver=s, mean_fields=mean).layout
    # Carte cadrée sur le domaine en x, en repère orthonormé (échelle y liée à x).
    assert tuple(layout.xaxis.range) == (0.0, s.grid.Lx) and layout.yaxis.scaleanchor == "x"
    # La légende (profils de sillage) est dans le cadre du graphique des profils (axes n° 6).
    assert layout.xaxis6.domain[0] <= layout.legend.x <= layout.xaxis6.domain[1]
    assert layout.yaxis6.domain[0] <= layout.legend.y <= layout.yaxis6.domain[1]


def test_interactive_report(tmp_path, run):
    s, history, mean, _ = run
    # include_plotlyjs="cdn" : fichier léger (bibliothèque chargée depuis internet).
    path = interactive_report(history, tmp_path / "report.html", solver=s, mean_fields=mean, include_plotlyjs="cdn")
    html = path.read_text(encoding="utf-8")
    # Le fichier contient les données Plotly et le titre du rapport.
    assert "plotly" in html.lower() and "rapport d'analyse" in html

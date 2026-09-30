"""Tests des entrées/sorties : archives npz, fichiers VTK, séries temporelles, points de reprise."""

# ``json`` : lecture de l'index de série temporelle.
import json

import numpy as np
import pytest

from cfd2d import NavierStokesSolver, compute_fields, presets
from cfd2d.io import (
    VTK_SCALARS,
    VTKSeriesWriter,
    load_checkpoint,
    load_fields_npz,
    save_checkpoint,
    save_fields_npz,
    save_vtk,
)


@pytest.fixture
def solver() -> NavierStokesSolver:
    """Petit cylindre ayant avancé de quelques pas (champs non triviaux)."""
    # Une « fixture » pytest : chaque test qui la demande en argument reçoit un solveur neuf.
    s = NavierStokesSolver(presets.cylinder_flow(cells_per_diameter=8, length=10.0, height=5.0, x_center=3.0))
    s.run(max_steps=10)
    return s


def read_vtk_block(path, keyword: bytes, count: int, dtype: str) -> np.ndarray:
    """Lit le bloc binaire qui suit la ligne d'en-tête ``keyword`` dans un fichier VTK legacy."""
    # Contenu complet du fichier (octets).
    content = path.read_bytes()
    # Position du mot-clé, puis début des données juste après.
    start = content.index(keyword) + len(keyword)
    # np.frombuffer interprète ``count`` valeurs du type donné (">f8" : flottant 64 bits gros-boutiste).
    return np.frombuffer(content, dtype=dtype, count=count, offset=start)


def test_fields_npz_roundtrip(tmp_path, solver):
    fields = compute_fields(solver)
    # Écriture avec une grandeur supplémentaire, puis relecture.
    path = save_fields_npz(tmp_path / "fields.npz", fields, Re=100.0)
    loaded = load_fields_npz(path)
    # Mêmes champs et même instant.
    np.testing.assert_array_equal(loaded.vorticity, fields.vorticity)
    np.testing.assert_array_equal(loaded.solid, fields.solid)
    assert loaded.t == pytest.approx(fields.t)
    # La grandeur supplémentaire est dans l'archive.
    assert float(np.load(path)["Re"]) == 100.0


def test_binary_vtk_contains_the_fields(tmp_path, solver):
    fields = compute_fields(solver)
    g = solver.grid
    path = save_vtk(tmp_path / "flow.vtk", fields, g)
    n = g.Nx * g.Ny
    # En-tête : dimensions en points (Nx+1, Ny+1, 1) et nombre de cellules.
    header = path.read_bytes()[:300].decode("ascii", errors="replace")
    assert f"DIMENSIONS {g.Nx + 1} {g.Ny + 1} 1" in header and f"CELL_DATA {n}" in header
    # Bloc de pression : ordre VTK (x le plus rapide) -> reshape (Ny, Nx) puis transposition.
    pressure = read_vtk_block(path, b"SCALARS pressure double 1\nLOOKUP_TABLE default\n", n, ">f8")
    np.testing.assert_array_equal(pressure.reshape(g.Ny, g.Nx).T, fields.p)
    # Bloc vitesse : triplets (u, v, 0).
    velocity = read_vtk_block(path, b"VECTORS velocity double\n", 3 * n, ">f8").reshape(n, 3)
    np.testing.assert_array_equal(velocity[:, 0].reshape(g.Ny, g.Nx).T, fields.u)
    assert np.all(velocity[:, 2] == 0.0)
    # Masque solide en entiers 32 bits.
    solid = read_vtk_block(path, b"SCALARS solid int 1\nLOOKUP_TABLE default\n", n, ">i4")
    assert solid.sum() == fields.solid.sum()


def test_ascii_vtk_lists_every_value(tmp_path, solver):
    fields = compute_fields(solver)
    g = solver.grid
    path = save_vtk(tmp_path / "flow_ascii.vtk", fields, g, binary=False)
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[2] == "ASCII"
    # Lignes d'en-tête (8) + pour chaque scalaire : 2 lignes de mots-clés + Nx*Ny valeurs,
    # puis le masque (même structure), puis le vecteur : 1 ligne + Nx*Ny triplets.
    n = g.Nx * g.Ny
    expected = 8 + (len(VTK_SCALARS) + 1) * (2 + n) + 1 + n
    assert len(lines) == expected


def test_vtk_series_writer_indexes_the_snapshots(tmp_path):
    s = NavierStokesSolver(presets.lid_driven_cavity(N=16))
    # Un fichier VTK tous les 3 pas.
    writer = VTKSeriesWriter(s, tmp_path / "vtk", every=3, prefix="cavity")
    s.run(max_steps=10)
    # Pas 3, 6 et 9 : trois fichiers.
    assert len(writer.files) == 3
    assert sorted(p.name for p in (tmp_path / "vtk").glob("cavity_*.vtk")) == [
        "cavity_00000.vtk",
        "cavity_00001.vtk",
        "cavity_00002.vtk",
    ]
    # Index JSON : noms et instants croissants.
    series = json.loads(writer.series_path.read_text(encoding="utf-8"))
    times = [entry["time"] for entry in series["files"]]
    assert series["file-series-version"] == "1.0" and times == sorted(times)


def test_checkpoint_restores_the_state(tmp_path, solver):
    # Sauvegarde de l'état après 10 pas.
    path = save_checkpoint(tmp_path / "state.npz", solver)
    # Nouveau solveur de même configuration, puis reprise.
    other = NavierStokesSolver(solver.config)
    load_checkpoint(path, other)
    # État identique (vitesses, pression, temps, pas).
    np.testing.assert_array_equal(other.state.u, solver.state.u)
    np.testing.assert_array_equal(other.state.p, solver.state.p)
    assert other.state.t == solver.state.t and other.state.step == solver.state.step
    # Le calcul peut continuer après la reprise.
    other.run(max_steps=5)
    assert other.state.step == solver.state.step + 5


def test_checkpoint_rejects_another_grid(tmp_path, solver):
    path = save_checkpoint(tmp_path / "state.npz", solver)
    # Grille différente (cavité 16 x 16) : la reprise est refusée.
    cavity = NavierStokesSolver(presets.lid_driven_cavity(N=16))
    with pytest.raises(ValueError):
        load_checkpoint(path, cavity)

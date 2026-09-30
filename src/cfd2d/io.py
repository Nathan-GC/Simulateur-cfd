"""Entrées/sorties : champs (.npz, .vtk pour ParaView), séries temporelles, points de reprise.

* :func:`save_fields_npz` / :func:`load_fields_npz` : champs 2D dans une archive NumPy ;
* :func:`save_vtk` : champs au format VTK « legacy » (``STRUCTURED_POINTS``), que ParaView,
  VisIt ou PyVista ouvrent directement ;
* :class:`VTKSeriesWriter` : moniteur écrivant un fichier VTK tous les N pas et un index
  ``.vtk.series`` (JSON) que ParaView lit comme une animation temporelle ;
* :func:`save_checkpoint` / :func:`load_checkpoint` : état complet pour reprendre un calcul.
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``json`` : écriture de l'index de série temporelle lu par ParaView.
import json

# Journalisation.
import logging

# Chemins de fichiers portables.
from pathlib import Path

import numpy as np

# Champs dérivés (vorticité, critère Q...) calculés à partir de l'état du solveur.
from .analytics import FlowFields, compute_fields

# Grille décalée (dimensions et pas pour l'en-tête VTK).
from .grid import StaggeredGrid

# État de l'écoulement et solveur (points de reprise).
from .solver import FlowState, NavierStokesSolver

# Journal du module.
logger = logging.getLogger(__name__)

# Champs scalaires exportés vers VTK : nom affiché dans ParaView -> attribut de FlowFields.
VTK_SCALARS = {
    "pressure": "p",
    "speed": "speed",
    "vorticity": "vorticity",
    "q_criterion": "q_criterion",
    "divergence": "divergence",
}


def _prepare(path: str | Path) -> Path:
    """Convertit en Path et crée le dossier parent si besoin."""
    path = Path(path)
    # parents=True : crée aussi les dossiers intermédiaires ; exist_ok : pas d'erreur s'il existe.
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================= NPZ
def save_fields_npz(path: str | Path, fields: FlowFields, **extra: float | np.ndarray) -> Path:
    """Enregistre des champs 2D (et leurs coordonnées) dans une archive ``.npz`` compressée.

    ``extra`` : grandeurs supplémentaires à stocker (ex. ``Re=100.0``).
    """
    path = _prepare(path)
    # fields.as_dict() : {"t", "x", "y", "u", "v", "p", "vorticity", ..., "speed"} ;
    # np.savez_compressed écrit chaque tableau sous son nom, compressé (zip).
    np.savez_compressed(path, **fields.as_dict(), **{key: np.asarray(value) for key, value in extra.items()})
    return path


def load_fields_npz(path: str | Path) -> FlowFields:
    """Relit une archive écrite par :func:`save_fields_npz`."""
    # np.load ouvre l'archive ; « with » garantit sa fermeture.
    with np.load(path) as data:
        # Reconstruction de l'objet FlowFields ; float(...) convertit le tableau 0-D du temps.
        return FlowFields(
            t=float(data["t"]),
            x=data["x"],
            y=data["y"],
            u=data["u"],
            v=data["v"],
            p=data["p"],
            vorticity=data["vorticity"],
            q_criterion=data["q_criterion"],
            divergence=data["divergence"],
            # astype(bool) : le masque est relu en booléens.
            solid=data["solid"].astype(bool),
        )


# ============================================================================= VTK
def _vtk_order(field: np.ndarray) -> np.ndarray:
    """Réordonne un champ (Nx, Ny) en vecteur « x le plus rapide » attendu par VTK."""
    # field.T a la forme (Ny, Nx) ; ravel() parcourt alors i (x) le plus vite, puis j (y).
    return np.asarray(field).T.ravel()


def _write_values(handle, values: np.ndarray, binary: bool) -> None:
    """Écrit un bloc de données VTK (binaire gros-boutiste ou texte)."""
    if binary:
        # Le format VTK legacy binaire impose l'ordre des octets « big-endian » (">") ;
        # ">i4" = entier 32 bits, ">f8" = flottant 64 bits ; tobytes() donne les octets bruts,
        # ligne par ligne (un tableau (n, 3) donne bien les triplets consécutifs).
        dtype = ">i4" if np.issubdtype(values.dtype, np.integer) else ">f8"
        handle.write(values.astype(dtype).tobytes())
        # Saut de ligne séparant le bloc binaire du mot-clé suivant.
        handle.write(b"\n")
    else:
        # Texte : une valeur (ou un triplet pour les vecteurs) par ligne ; fmt adapté au type.
        fmt = "%d" if np.issubdtype(values.dtype, np.integer) else "%.10g"
        # reshape(len, -1) : 1 colonne pour un scalaire, 3 pour un vecteur ; -1 = déduit.
        np.savetxt(handle, values.reshape(len(values), -1), fmt=fmt)


def save_vtk(
    path: str | Path, fields: FlowFields, grid: StaggeredGrid, *, binary: bool = True, title: str | None = None
) -> Path:
    """Champs aux centres des cellules au format VTK « legacy » (``STRUCTURED_POINTS``).

    Le fichier contient les scalaires de :data:`VTK_SCALARS`, le masque ``solid`` (0/1) et
    le vecteur ``velocity`` = (u, v, 0), en données de cellules (``CELL_DATA``). Dans
    ParaView : *Open* puis *Apply* ; le filtre *Threshold* sur ``solid`` masque l'obstacle.
    """
    path = _prepare(path)
    # Nombre de cellules.
    nx, ny = grid.Nx, grid.Ny
    n_cells = nx * ny
    # En-tête : version, titre (255 caractères max), encodage, type de maillage, dimensions en
    # POINTS (Nx+1 x Ny+1 x 1 : coins des cellules), origine et pas, nombre de cellules.
    header = [
        "# vtk DataFile Version 3.0",
        (title or f"cfd2d t={fields.t:.6g}")[:255],
        "BINARY" if binary else "ASCII",
        "DATASET STRUCTURED_POINTS",
        f"DIMENSIONS {nx + 1} {ny + 1} 1",
        "ORIGIN 0 0 0",
        f"SPACING {grid.dx!r} {grid.dy!r} 1",
        f"CELL_DATA {n_cells}",
    ]
    # Ouverture en mode binaire ("wb") : l'en-tête texte est encodé en ASCII.
    with open(path, "wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        # Chaque champ scalaire : mot-clé SCALARS nom type nb_composantes, table de couleurs, données.
        for vtk_name, attribute in VTK_SCALARS.items():
            handle.write(f"SCALARS {vtk_name} double 1\nLOOKUP_TABLE default\n".encode("ascii"))
            # getattr(fields, "p") lit le champ ; conversion en flottants et ordre VTK.
            _write_values(handle, _vtk_order(getattr(fields, attribute)).astype(float), binary)
        # Masque solide en entiers 0/1 (filtrable dans ParaView).
        handle.write(b"SCALARS solid int 1\nLOOKUP_TABLE default\n")
        _write_values(handle, _vtk_order(fields.solid).astype(np.int32), binary)
        # Vecteur vitesse 3D (composante z nulle) : tableau (n_cells, 3) écrit ligne par ligne
        # (u0 v0 0 u1 v1 0 ... en binaire, un triplet par ligne en texte).
        handle.write(b"VECTORS velocity double\n")
        velocity = np.column_stack([_vtk_order(fields.u), _vtk_order(fields.v), np.zeros(n_cells)])
        _write_values(handle, velocity, binary)
    return path


class VTKSeriesWriter:
    """Moniteur : fichier VTK tous les ``every`` pas + index ``<prefix>.vtk.series``.

    Ouvrir le fichier ``.vtk.series`` dans ParaView donne une animation dont les instants
    sont les temps physiques de la simulation. ``t_start`` évite d'écrire le transitoire
    (chaque fichier binaire pèse ~0.7 Mo pour 10 000 cellules).
    """

    def __init__(
        self,
        solver: NavierStokesSolver,
        directory: str | Path,
        *,
        every: int = 100,
        prefix: str = "flow",
        binary: bool = True,
        t_start: float = 0.0,
        attach: bool = True,
    ) -> None:
        # Dossier de sortie (créé si besoin).
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # Préfixe des noms de fichiers, encodage et instant de début d'écriture.
        self.prefix, self.binary, self.t_start = prefix, binary, float(t_start)
        # Grille (en-tête VTK).
        self.grid = solver.grid
        # Liste des fichiers écrits : [{"name": ..., "time": ...}, ...].
        self.files: list[dict[str, float | str]] = []
        # Inscription auprès du solveur (appel tous les ``every`` pas).
        if attach:
            solver.add_callback(self, every)

    def __call__(self, solver: NavierStokesSolver) -> None:
        # Régime transitoire : rien à écrire.
        if solver.state.t < self.t_start:
            return
        # Champs dérivés de l'état courant.
        fields = compute_fields(solver)
        # Nom numéroté sur 5 chiffres (:05d), ex. flow_00012.vtk.
        name = f"{self.prefix}_{len(self.files):05d}.vtk"
        # Écriture du fichier VTK.
        save_vtk(self.directory / name, fields, self.grid, binary=self.binary)
        # Ajout à l'index et réécriture de celui-ci (toujours à jour, même si le calcul s'arrête).
        self.files.append({"name": name, "time": float(fields.t)})
        self._write_series()

    @property
    def series_path(self) -> Path:
        # Chemin de l'index de série temporelle.
        return self.directory / f"{self.prefix}.vtk.series"

    def _write_series(self) -> None:
        # Format JSON « file series » de ParaView.
        series = {"file-series-version": "1.0", "files": self.files}
        # json.dumps(..., indent=2) : texte JSON indenté ; write_text l'écrit en UTF-8.
        self.series_path.write_text(json.dumps(series, indent=2), encoding="utf-8")


# ============================================================= points de reprise
def save_checkpoint(path: str | Path, solver: NavierStokesSolver) -> Path:
    """Enregistre l'état complet (vitesses avec fantômes, pression, temps, pas, grille)."""
    path = _prepare(path)
    st, g = solver.state, solver.grid
    # Les dimensions de la grille sont stockées pour vérifier la compatibilité à la relecture.
    np.savez_compressed(
        path, u=st.u, v=st.v, p=st.p, t=st.t, step=st.step, Lx=g.Lx, Ly=g.Ly, Nx=g.Nx, Ny=g.Ny
    )
    return path


def load_checkpoint(path: str | Path, solver: NavierStokesSolver) -> None:
    """Recharge un état dans un solveur de même grille (AB2 redémarre par un pas d'Euler)."""
    g = solver.grid
    with np.load(path) as data:
        # Vérification des dimensions (nombre de cellules et taille du domaine).
        same_cells = (int(data["Nx"]), int(data["Ny"])) == (g.Nx, g.Ny)
        # np.allclose : égalité des longueurs aux arrondis près.
        same_size = np.allclose([float(data["Lx"]), float(data["Ly"])], [g.Lx, g.Ly])
        if not (same_cells and same_size):
            raise ValueError("Point de reprise incompatible avec la grille du solveur.")
        # Reconstruction de l'état et transmission au solveur.
        state = FlowState(data["u"], data["v"], data["p"], t=float(data["t"]), step=int(data["step"]))
    solver.restore_state(state)
    logger.info("Reprise a t = %.6g (pas %d) depuis %s.", state.t, state.step, path)

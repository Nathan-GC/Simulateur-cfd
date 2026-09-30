"""cfd2d — solveur Navier–Stokes 2D incompressible (grille MAC, frontières immergées)."""

# Ce fichier fait du dossier ``cfd2d`` un paquet Python et définit son API publique :
# ``from cfd2d import NavierStokesSolver`` fonctionne grâce aux imports ci-dessous.

# Module des configurations types (cfd2d.presets.cylinder_flow(...)).
from . import presets

# Conditions aux limites.
from .boundary import BoundaryConfig, Inlet, NoSlipWall, Outlet, Side, SlipWall

# Blocs de configuration.
from .config import DomainConfig, FlowConfig, NumericsConfig, SimulationConfig, TimeConfig

# Obstacles et construction du masque.
from .geometry import NACA4, Cylinder, ImageObstacle, Obstacle, PolygonObstacle, Rectangle, build_mask

# Grille décalée.
from .grid import StaggeredGrid

# Solveur, état, diagnostics, exception de divergence.
from .solver import Diagnostics, FlowState, NavierStokesSolver, SimulationDivergedError

# Module d'analyse, importé après le solveur dont il dépend (noqa : ordre d'import volontaire).
from . import analytics  # noqa: E402  (dépend du solveur)

# Outils d'analyse les plus utilisés, exposés au premier niveau.
from .analytics import FieldAverager, ForceMonitor, compute_fields  # noqa: E402

# Entrées/sorties (npz, VTK, points de reprise) et visualisation (tableau de bord, animations).
from . import io, visualization  # noqa: E402

# Version du paquet.
__version__ = "0.1.0"

# Noms exportés par ``from cfd2d import *``.
__all__ = [
    "BoundaryConfig",
    "Cylinder",
    "Diagnostics",
    "DomainConfig",
    "FieldAverager",
    "FlowConfig",
    "FlowState",
    "ForceMonitor",
    "ImageObstacle",
    "Inlet",
    "NACA4",
    "NavierStokesSolver",
    "NoSlipWall",
    "NumericsConfig",
    "Obstacle",
    "Outlet",
    "PolygonObstacle",
    "Rectangle",
    "Side",
    "SimulationConfig",
    "SimulationDivergedError",
    "SlipWall",
    "StaggeredGrid",
    "TimeConfig",
    "analytics",
    "build_mask",
    "compute_fields",
    "io",
    "presets",
    "visualization",
]

"""Grille cartésienne décalée (MAC) pour un domaine rectangulaire 2D."""

# Active l'évaluation différée des annotations de type (PEP 563) : les annotations comme
# ``tuple[int, int]`` ne sont pas évaluées à l'exécution, ce qui évite des erreurs sur les
# anciennes versions de Python et accélère légèrement l'import.
from __future__ import annotations

# ``dataclass`` génère automatiquement __init__, __repr__, __eq__ (et __hash__ si frozen)
# à partir des attributs annotés de la classe.
from dataclasses import dataclass

# ``cached_property`` transforme une méthode en attribut calculé une seule fois : le résultat
# est mémorisé dans l'instance au premier accès puis réutilisé.
from functools import cached_property

# NumPy : bibliothèque de calcul sur tableaux multidimensionnels (base de tout le solveur).
import numpy as np


def _readonly(a: np.ndarray) -> np.ndarray:
    """Rend un tableau NumPy non modifiable et le retourne (protection des coordonnées)."""
    # ``flags.writeable = False`` : toute tentative d'écriture (a[0] = ...) lèvera ValueError.
    a.flags.writeable = False
    # On renvoie le même tableau (aucune copie) pour pouvoir chaîner l'appel.
    return a


# ``frozen=True`` : les attributs ne peuvent plus être modifiés après la création (la grille
# est immuable), et la classe devient hachable (utilisable comme clé de dictionnaire).
@dataclass(frozen=True)
class StaggeredGrid:
    """Grille MAC uniforme sur ``[0, Lx] x [0, Ly]`` comportant ``Nx x Ny`` cellules.

    Indexation ``[i, j]`` : ``i`` suit l'axe x (axe 0 des tableaux), ``j`` l'axe y.

    ============  ======================  ==================  ================================
    Champ         Position                Forme               Cellules fantômes
    ============  ======================  ==================  ================================
    ``p[i, j]``   ``((i+½)Δx, (j+½)Δy)``  ``(Nx, Ny)``        aucune
    ``u[i, J]``   ``(iΔx, (J-½)Δy)``      ``(Nx+1, Ny+2)``    lignes ``J = 0`` et ``J = Ny+1``
    ``v[I, j]``   ``((I-½)Δx, jΔy)``      ``(Nx+2, Ny+1)``    colonnes ``I = 0`` et ``I = Nx+1``
    ============  ======================  ==================  ================================

    Les vitesses physiques sont donc ``u[:, 1:-1]`` (faces verticales, forme
    ``(Nx+1, Ny)``) et ``v[1:-1, :]`` (faces horizontales, forme ``(Nx, Ny+1)``) ;
    les rangées fantômes servent à imposer les conditions aux limites tangentielles.
    """

    # Longueur du domaine selon x (unité de longueur quelconque, cohérente avec U et nu).
    Lx: float
    # Hauteur du domaine selon y.
    Ly: float
    # Nombre de cellules selon x.
    Nx: int
    # Nombre de cellules selon y.
    Ny: int

    def __post_init__(self) -> None:
        """Validation des paramètres, appelée automatiquement après le __init__ généré."""
        # Les dimensions physiques doivent être strictement positives (sinon Δx ≤ 0).
        if not (self.Lx > 0 and self.Ly > 0):
            raise ValueError("Les dimensions Lx et Ly doivent être strictement positives.")
        # ``int(x) != x`` détecte un nombre non entier (par exemple 64.5) passé pour Nx ou Ny.
        if int(self.Nx) != self.Nx or int(self.Ny) != self.Ny:
            raise TypeError("Nx et Ny doivent être des entiers.")
        # Les stencils (QUICK, ghost cells) exigent un minimum de cellules par direction.
        if self.Nx < 4 or self.Ny < 4:
            raise ValueError("La grille doit comporter au moins 4 cellules dans chaque direction.")

    # ------------------------------------------------------------------ pas et formes
    # ``@property`` : méthode appelée comme un attribut (grid.dx et non grid.dx()).
    @property
    def dx(self) -> float:
        # Pas d'espace selon x : largeur d'une cellule.
        return self.Lx / self.Nx

    @property
    def dy(self) -> float:
        # Pas d'espace selon y : hauteur d'une cellule.
        return self.Ly / self.Ny

    @property
    def cell_area(self) -> float:
        # Aire d'une cellule (volume de contrôle 2D par unité d'envergure).
        return self.dx * self.dy

    @property
    def shape_p(self) -> tuple[int, int]:
        # Forme des tableaux définis au centre des cellules (pression, divergence...).
        return (self.Nx, self.Ny)

    @property
    def shape_u(self) -> tuple[int, int]:
        # Nx+1 faces verticales selon x ; Ny lignes physiques + 2 lignes fantômes selon y.
        return (self.Nx + 1, self.Ny + 2)

    @property
    def shape_v(self) -> tuple[int, int]:
        # Nx colonnes physiques + 2 colonnes fantômes selon x ; Ny+1 faces horizontales selon y.
        return (self.Nx + 2, self.Ny + 1)

    # ------------------------------------------------------------ coordonnées 1D
    @cached_property
    def x_c(self) -> np.ndarray:
        """Abscisses des centres de cellules, ``(Nx,)``."""
        # np.arange(Nx) = [0, 1, ..., Nx-1] ; +0.5 place le point au milieu de chaque cellule ;
        # le produit par dx donne la coordonnée physique ; _readonly empêche toute modification.
        return _readonly((np.arange(self.Nx) + 0.5) * self.dx)

    @cached_property
    def y_c(self) -> np.ndarray:
        """Ordonnées des centres de cellules, ``(Ny,)``."""
        # Même construction que x_c, selon y.
        return _readonly((np.arange(self.Ny) + 0.5) * self.dy)

    @cached_property
    def x_f(self) -> np.ndarray:
        """Abscisses des faces verticales (nœuds de u), ``(Nx+1,)``."""
        # Faces verticales en x = 0, dx, 2dx, ..., Lx : Nx+1 valeurs, bords du domaine inclus.
        return _readonly(np.arange(self.Nx + 1) * self.dx)

    @cached_property
    def y_f(self) -> np.ndarray:
        """Ordonnées des faces horizontales (nœuds de v), ``(Ny+1,)``."""
        # Faces horizontales en y = 0, dy, ..., Ly.
        return _readonly(np.arange(self.Ny + 1) * self.dy)

    # ------------------------------------------------------------ maillages 2D
    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordonnées ``(X, Y)`` des centres de cellules, formes ``(Nx, Ny)``."""
        # np.meshgrid combine deux vecteurs 1D en deux tableaux 2D de coordonnées ;
        # indexing="ij" garantit X[i, j] = x_c[i] et Y[i, j] = y_c[j] (convention [i, j] = [x, y]).
        return np.meshgrid(self.x_c, self.y_c, indexing="ij")

    def u_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordonnées des nœuds physiques de u, formes ``(Nx+1, Ny)``."""
        # u est porté par les faces verticales (x_f) à mi-hauteur des cellules (y_c).
        return np.meshgrid(self.x_f, self.y_c, indexing="ij")

    def v_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordonnées des nœuds physiques de v, formes ``(Nx, Ny+1)``."""
        # v est porté par les faces horizontales (y_f) à mi-largeur des cellules (x_c).
        return np.meshgrid(self.x_c, self.y_f, indexing="ij")

    def corner_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordonnées des coins de cellules (nœuds de vorticité), formes ``(Nx+1, Ny+1)``."""
        # Les coins sont à l'intersection des faces verticales et horizontales.
        return np.meshgrid(self.x_f, self.y_f, indexing="ij")

    def locate(self, x: float, y: float) -> tuple[int, int]:
        """Indices ``(i, j)`` de la cellule contenant le point ``(x, y)`` (bornés au domaine)."""
        # np.floor(x / dx) : numéro de la cellule contenant x (partie entière inférieure) ;
        # np.clip(..., 0, Nx-1) ramène un point hors domaine sur la cellule de bord ;
        # int(...) convertit le flottant NumPy en entier Python utilisable comme indice.
        i = int(np.clip(np.floor(x / self.dx), 0, self.Nx - 1))
        # Même calcul selon y.
        j = int(np.clip(np.floor(y / self.dy), 0, self.Ny - 1))
        # Retourne le couple d'indices (tuple).
        return i, j

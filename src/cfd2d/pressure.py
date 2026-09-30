"""Équation de Poisson pour la pression et solveurs linéaires associés.

L'opérateur est assemblé une seule fois (le masque et les conditions aux limites sont
fixes) ; le solveur direct le factorise donc une fois pour toutes, et chaque pas de
temps ne coûte qu'une descente-remontée triangulaire.
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``inspect`` : introspection (lecture de la signature d'une fonction SciPy).
import inspect

# Journalisation des avertissements.
import logging

# Classe de base abstraite commune aux solveurs de pression.
from abc import ABC, abstractmethod

# ``dataclass`` : petit enregistrement du bilan de résolution.
from dataclasses import dataclass

import numpy as np

# Matrices creuses (seuls les coefficients non nuls sont stockés : ~5 par ligne ici).
import scipy.sparse as sp

# Algèbre linéaire creuse : factorisation LU (splu), ILU (spilu), gradient conjugué (cg)...
import scipy.sparse.linalg as spla

# Grille décalée (dimensions, pas d'espace).
from .grid import StaggeredGrid

# Journal du module.
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolveInfo:
    """Bilan d'une résolution : itérations, résidu relatif ``‖b - Ax‖/‖b‖``, convergence."""

    # Nombre d'itérations effectuées (1 pour le solveur direct).
    iterations: int
    # Résidu relatif final (NaN s'il n'est pas calculé).
    residual: float
    # Vrai si la tolérance demandée est atteinte.
    converged: bool


class PoissonOperator:
    """Opérateur ``A = -∇·∇`` (symétrique défini positif) restreint aux cellules fluides.

    ``A = -D G`` où ``G`` est le gradient aux faces *actives* (faces corrigées par la
    projection) et ``D`` la divergence discrète, si bien que la vitesse projetée est
    exactement à divergence nulle (à la précision du solveur près) :

    * face active intérieure : couplage à 5 points classique ;
    * face de sortie (``p = 0`` sur le bord, à ``Δ/2`` du centre) : terme diagonal ``2/Δ²`` ;
    * face figée (paroi, entrée, face touchant un solide) : flux nul (Neumann homogène).

    Les cellules solides sont découplées (ligne identité, ``p = 0``). Sans sortie, le
    problème de Neumann pur est singulier : la pression d'une cellule est fixée à 0 et
    le second membre est rendu compatible (moyenne nulle sur le fluide).
    """

    def __init__(
        self,
        grid: StaggeredGrid,
        solid: np.ndarray,
        u_active: np.ndarray,
        v_active: np.ndarray,
        u_dirichlet: np.ndarray,
        v_dirichlet: np.ndarray,
    ) -> None:
        # Mémorisation de la grille et du masque solide (Nx, Ny).
        self.grid = grid
        self.solid = solid
        # Versions aplaties (vecteur de longueur Nx*Ny) : l'inconnue du système linéaire est
        # le vecteur des pressions, cellule k = i*Ny + j (ordre « C » de NumPy).
        self._solid_flat = solid.ravel()
        # ~ : négation booléenne élément par élément (cellules fluides).
        self._fluid_flat = ~self._solid_flat
        # Nombre total d'inconnues.
        n = grid.Nx * grid.Ny
        # Numéro de ligne matricielle de chaque cellule (i, j) : np.arange(n).reshape(Nx, Ny).
        idx = np.arange(n).reshape(grid.shape_p)
        # Coefficients du laplacien : 1/dx² et 1/dy².
        cx, cy = 1.0 / grid.dx**2, 1.0 / grid.dy**2

        # Diagonale de A, accumulée face par face.
        diag = np.zeros(n)
        # Listes des indices de ligne, de colonne et des valeurs des coefficients hors diagonale
        # (format « coordonnées » d'une matrice creuse).
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        def couple(a: np.ndarray, b: np.ndarray, c: float) -> None:
            """Ajoute le couplage de faces actives entre les cellules a[k] et b[k], de poids c."""
            # Contribution diagonale : chaque face active ajoute c aux deux cellules voisines.
            diag[a] += c
            diag[b] += c
            # Coefficients hors diagonale A[a, b] = A[b, a] = -c (matrice symétrique).
            rows.extend([a, b])
            cols.extend([b, a])
            # np.full(taille, valeur) : tableau constant de la bonne longueur.
            vals.extend([np.full(a.size, -c), np.full(a.size, -c)])

        # Faces u intérieures actives (entre la cellule (i-1, j) et la cellule (i, j)).
        act = u_active[1:-1, :]
        # idx[:-1, :][act] : cellules de gauche des faces actives ; idx[1:, :][act] : de droite.
        couple(idx[:-1, :][act], idx[1:, :][act], cx)
        # Faces v intérieures actives (entre (i, j-1) et (i, j)).
        act = v_active[:, 1:-1]
        couple(idx[:, :-1][act], idx[:, 1:][act], cy)
        # Faces de sortie (Dirichlet p = 0 sur le bord, à une demi-maille du centre) : le flux
        # (0 - p)/(Δ/2) divisé par Δ ajoute 2/Δ² sur la diagonale, sans voisin.
        diag[idx[0, :][u_dirichlet[0, :]]] += 2.0 * cx
        diag[idx[-1, :][u_dirichlet[-1, :]]] += 2.0 * cx
        diag[idx[:, 0][v_dirichlet[:, 0]]] += 2.0 * cy
        diag[idx[:, -1][v_dirichlet[:, -1]]] += 2.0 * cy

        # Cellules fluides sans aucune face active (cas pathologique) : ligne nulle.
        isolated = self._fluid_flat & (diag == 0.0)
        if isolated.any():
            logger.warning("%d cellule(s) fluide(s) sans face active : pression fixee a 0.", isolated.sum())
        # Cellules solides et isolées : ligne identité (p = 0), découplée du reste.
        diag[self._solid_flat | isolated] = 1.0

        # np.concatenate fusionne les listes de tableaux en vecteurs uniques (vides si aucune face).
        r = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        c = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
        w = np.concatenate(vals) if vals else np.zeros(0)

        #: nombre d'extrémités en Dirichlet selon x et selon y (0, 1 ou 2)
        # int(True) = 1 : on compte les côtés ouest/est puis sud/nord portant une sortie.
        self.dirichlet_ends = (
            int(u_dirichlet[0].any()) + int(u_dirichlet[-1].any()),
            int(v_dirichlet[:, 0].any()) + int(v_dirichlet[:, -1].any()),
        )
        # Aucune face de sortie : problème de Neumann pur, défini à une constante près.
        self.singular = not (u_dirichlet.any() or v_dirichlet.any())
        # Cellule dont la pression est fixée à 0 (seulement dans le cas singulier).
        self.pin: int | None = None
        if self.singular:
            # np.flatnonzero : indices des éléments vrais ; on prend la première cellule fluide.
            self.pin = int(np.flatnonzero(self._fluid_flat & ~isolated)[0])
            # On supprime la ligne ET la colonne de cette cellule (la matrice reste symétrique).
            keep = (r != self.pin) & (c != self.pin)
            r, c, w = r[keep], c[keep], w[keep]
            # Ligne identité : p[pin] = b[pin] = 0.
            diag[self.pin] = 1.0

        # sp.coo_matrix((valeurs, (lignes, colonnes))) construit la matrice creuse des termes
        # hors diagonale (les doublons éventuels sont additionnés).
        offdiag = sp.coo_matrix((w, (r, c)), shape=(n, n))
        # sp.diags(diag) : matrice diagonale ; .tocsr() convertit au format CSR (lignes
        # compressées), efficace pour les produits matrice-vecteur.
        self.matrix: sp.csr_matrix = (offdiag + sp.diags(diag)).tocsr()

    def rhs(self, div: np.ndarray, scale: float) -> np.ndarray:
        """Second membre ``b = -scale · ∇·u*`` (``scale = ρ/Δt``), nul dans le solide."""
        # A = -∇² donc A p = -(ρ/Δt) ∇·u* ; .ravel() aplatit la divergence (Nx, Ny) en vecteur.
        b = -scale * div.ravel()
        # Cellules solides : p = 0.
        b[self._solid_flat] = 0.0
        if self.singular:
            # Compatibilité du problème de Neumann : la somme du second membre doit être nulle
            # (sinon pas de solution) ; on retire la moyenne sur le fluide (erreurs d'arrondi).
            b[self._fluid_flat] -= b[self._fluid_flat].mean()
            # Cellule de référence : p = 0.
            b[self.pin] = 0.0
        return b

    def gauge(self, p: np.ndarray) -> np.ndarray:
        """Ramène une pression dans la jauge du système (``p[pin] = 0`` si singulier)."""
        # Vecteur aplati.
        flat = p.ravel()
        # Cas singulier : on soustrait la valeur à la cellule de référence (solution définie
        # à une constante près) ; sinon la pression est utilisée telle quelle.
        return flat - flat[self.pin] if self.singular else flat


# ============================================================================ solveurs
class PressureSolver(ABC):
    """Interface des solveurs linéaires ``A x = b`` pour la pression."""

    # Nom affiché dans les journaux (redéfini par les sous-classes).
    name: str = "abstract"

    def __init__(self, A: sp.csr_matrix) -> None:
        # Matrice du système.
        self.A = A

    @abstractmethod
    def solve(self, b: np.ndarray, x0: np.ndarray | None = None) -> tuple[np.ndarray, SolveInfo]:
        """Résout ``A x = b`` (``x0`` : estimation initiale, ignorée par le solveur direct)."""

    def relative_residual(self, x: np.ndarray, b: np.ndarray) -> float:
        """Résidu relatif ``‖b - A x‖ / ‖b‖`` (norme euclidienne)."""
        # np.linalg.norm : norme euclidienne du vecteur.
        bnorm = float(np.linalg.norm(b))
        # ``A @ x`` : produit matrice creuse - vecteur ; résidu nul par convention si b = 0.
        return float(np.linalg.norm(b - self.A @ x)) / bnorm if bnorm > 0 else 0.0


class DirectPressureSolver(PressureSolver):
    """Factorisation LU creuse (SuperLU) calculée une fois, réutilisée à chaque pas.

    La solution est exacte aux erreurs d'arrondi près : le résidu n'est pas recalculé
    (``SolveInfo.residual`` vaut NaN) pour économiser un produit matrice-vecteur par pas.
    """

    name = "direct"

    def __init__(self, A: sp.csr_matrix) -> None:
        # Appel du constructeur de la classe parente (mémorise A).
        super().__init__(A)
        # spla.splu : factorisation A = L U (bibliothèque SuperLU), au format CSC (colonnes
        # compressées) requis. Options :
        # - permc_spec="MMD_AT_PLUS_A" : renumérotation des inconnues (degré minimum sur A+Aᵀ)
        #   qui limite le « remplissage » des facteurs pour une matrice symétrique ;
        # - diag_pivot_thresh=0 : pas de pivotage (inutile, A est définie positive) ;
        # - SymmetricMode : SuperLU exploite la symétrie de structure.
        self._lu = spla.splu(
            A.tocsc(),
            permc_spec="MMD_AT_PLUS_A",  # ordonnancement symétrique : peu de remplissage
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )

    def solve(self, b: np.ndarray, x0: np.ndarray | None = None) -> tuple[np.ndarray, SolveInfo]:
        # _lu.solve : descente (L) puis remontée (U) triangulaires, coût ~ nombre de non-zéros.
        return self._lu.solve(b), SolveInfo(iterations=1, residual=float("nan"), converged=True)


def _tolerance_kwargs(fn: object, tol: float) -> dict[str, float]:
    """``rtol`` (SciPy >= 1.12) ou ``tol`` (versions antérieures)."""
    # inspect.signature(fn).parameters : noms des paramètres acceptés par la fonction SciPy.
    params = inspect.signature(fn).parameters  # type: ignore[arg-type]
    # Nom du paramètre de tolérance relative selon la version de SciPy ; atol = 0 : pas de
    # tolérance absolue (seule la tolérance relative compte).
    return {"rtol": tol, "atol": 0.0} if "rtol" in params else {"tol": tol, "atol": 0.0}


class KrylovPressureSolver(PressureSolver):
    """Gradient conjugué (``cg``) ou BiCGSTAB préconditionné, démarré depuis la pression précédente."""

    def __init__(
        self,
        A: sp.csr_matrix,
        method: str = "cg",
        preconditioner: str = "ilu",
        tol: float = 1e-8,
        maxiter: int = 10_000,
    ) -> None:
        super().__init__(A)
        # Correspondance nom -> fonction SciPy (cg : matrices symétriques définies positives ;
        # bicgstab : matrices quelconques).
        methods = {"cg": spla.cg, "bicgstab": spla.bicgstab}
        if method not in methods:
            raise ValueError(f"Méthode de Krylov inconnue : {method!r}.")
        # Nom affiché, ex. "cg+ilu".
        self.name = f"{method}+{preconditioner}"
        # Fonction de résolution choisie.
        self._method = methods[method]
        # Arguments de tolérance adaptés à la version de SciPy.
        self._kwargs = _tolerance_kwargs(self._method, tol)
        # Nombre maximal d'itérations.
        self.maxiter = maxiter
        # Préconditionneur M ≈ A⁻¹ (accélère la convergence), construit une seule fois.
        self._M = self._preconditioner(preconditioner)

    def _preconditioner(self, kind: str) -> spla.LinearOperator | None:
        """Construit le préconditionneur demandé."""
        A = self.A
        # Pas de préconditionnement.
        if kind == "none":
            return None
        # Jacobi : M = diag(A)⁻¹ (A.diagonal() extrait la diagonale), symétrique défini positif.
        if kind == "jacobi":
            return sp.diags(1.0 / A.diagonal()).tocsr()
        if kind == "ilu":
            # Ordonnancement symétrique sans pivotage : le préconditionneur reste (quasi)
            # symétrique, condition indispensable à la convergence du gradient conjugué.
            # spla.spilu : factorisation LU INCOMPLÈTE (les petits termes, < drop_tol, sont
            # abandonnés ; fill_factor borne le remplissage).
            ilu = spla.spilu(
                A.tocsc(),
                drop_tol=1e-4,
                fill_factor=10,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
            # LinearOperator : objet « matrice » défini par son action (x -> ilu.solve(x)).
            return spla.LinearOperator(A.shape, ilu.solve)
        if kind == "amg":
            # Multigrille algébrique (optionnel) : bibliothèque pyamg, importée à la demande.
            try:
                import pyamg
            except ImportError as exc:
                raise ImportError("Le préconditionneur 'amg' nécessite le paquet pyamg (pip install pyamg).") from exc
            # Solveur multigrille par agrégation lissée, utilisé comme préconditionneur (cycle en V).
            return pyamg.smoothed_aggregation_solver(A).aspreconditioner(cycle="V")
        raise ValueError(f"Préconditionneur inconnu : {kind!r}.")

    def solve(self, b: np.ndarray, x0: np.ndarray | None = None) -> tuple[np.ndarray, SolveInfo]:
        # Compteur d'itérations, incrémenté par le rappel (callback) de SciPy.
        count = 0

        def callback(_: np.ndarray) -> None:
            # ``nonlocal`` : modifie la variable count de la fonction englobante.
            nonlocal count
            count += 1

        # Appel de cg/bicgstab : x0 = point de départ (pression du pas précédent), M =
        # préconditionneur ; info == 0 signifie « convergé ».
        x, info = self._method(self.A, b, x0=x0, M=self._M, maxiter=self.maxiter, callback=callback, **self._kwargs)
        # Résidu recalculé explicitement (celui de SciPy peut dériver par arrondi).
        return x, SolveInfo(iterations=count, residual=self.relative_residual(x, b), converged=info == 0)


class _StationarySolver(PressureSolver):
    """Base des méthodes itératives stationnaires (Jacobi, SOR)."""

    def __init__(self, A: sp.csr_matrix, tol: float, maxiter: int, check_every: int) -> None:
        super().__init__(A)
        # Tolérance relative sur le résidu.
        self.tol = tol
        # Nombre maximal d'itérations.
        self.maxiter = maxiter
        # Fréquence du test de convergence (un résidu coûte un produit matrice-vecteur).
        self.check_every = check_every
        # Diagonale de A (utilisée par les deux méthodes).
        self._diag = A.diagonal()


class JacobiPressureSolver(_StationarySolver):
    """Méthode de Jacobi pondérée (pédagogique : convergence en O(N) itérations)."""

    name = "jacobi"

    def __init__(self, A: sp.csr_matrix, tol: float = 1e-8, maxiter: int = 10_000, omega: float = 1.0) -> None:
        # Test de convergence à chaque itération (le résidu est de toute façon calculé).
        super().__init__(A, tol, maxiter, check_every=1)
        # Facteur de relaxation (1 = Jacobi classique).
        self.omega = omega

    def solve(self, b: np.ndarray, x0: np.ndarray | None = None) -> tuple[np.ndarray, SolveInfo]:
        # Point de départ : zéro ou copie de x0 (astype(..., copy=True) évite de modifier x0).
        x = np.zeros_like(b) if x0 is None else x0.astype(float, copy=True)
        # Norme du second membre (``or 1.0`` évite une division par zéro si b = 0).
        bnorm = float(np.linalg.norm(b)) or 1.0
        # Résidu initial inconnu.
        res = np.inf
        for k in range(1, self.maxiter + 1):
            # Résidu r = b - A x.
            r = b - self.A @ x
            # Résidu relatif.
            res = float(np.linalg.norm(r)) / bnorm
            # Convergence atteinte : k-1 itérations ont été nécessaires.
            if res < self.tol:
                return x, SolveInfo(k - 1, res, True)
            # Mise à jour de Jacobi : x += ω D⁻¹ r (toutes les inconnues en même temps).
            x += self.omega * r / self._diag
        # Nombre maximal d'itérations atteint sans convergence.
        return x, SolveInfo(self.maxiter, res, False)


class SORPressureSolver(_StationarySolver):
    """Gauss-Seidel / SOR rouge-noir vectorisé.

    Avec le stencil à 5 points, les cellules « rouges » (``i+j`` pair) ne dépendent que
    des « noires » : chaque demi-balayage est une opération vectorielle. ``omega=1``
    donne Gauss-Seidel ; par défaut ``omega`` est l'optimum théorique de Young.
    """

    name = "sor"

    def __init__(
        self,
        A: sp.csr_matrix,
        grid: StaggeredGrid,
        tol: float = 1e-8,
        maxiter: int = 10_000,
        omega: float | None = None,
        check_every: int = 10,
        dirichlet_ends: tuple[int, int] = (2, 2),
    ) -> None:
        super().__init__(A, tol, maxiter, check_every)
        # np.indices : tableaux des indices i et j de chaque cellule.
        i, j = np.indices(grid.shape_p)
        # Coloriage en damier : rouge si i + j est pair ; aplati comme le vecteur inconnu.
        red = ((i + j) % 2 == 0).ravel()
        # Indices (dans le vecteur inconnu) des cellules rouges et noires.
        self._red, self._black = np.flatnonzero(red), np.flatnonzero(~red)
        # Format CSR pour l'extraction de sous-matrices.
        A = A.tocsr()
        # Couplages rouge-rouge hors diagonale : ils doivent être nuls pour un stencil à 5 points.
        same = A[self._red][:, self._red] - sp.diags(self._diag[self._red])
        # abs(same).sum() : somme des valeurs absolues des termes restants.
        if abs(same).sum() > 0:
            raise ValueError("La matrice n'admet pas de découpage rouge-noir (stencil non 5 points).")
        # Sous-matrices de couplage rouge <- noir et noir <- rouge.
        self._A_rb = A[self._red][:, self._black].tocsr()
        self._A_br = A[self._black][:, self._red].tocsr()
        # Paramètre de relaxation : imposé ou estimé.
        self.omega = float(omega) if omega is not None else self.optimal_omega(grid, dirichlet_ends)

    @staticmethod
    def optimal_omega(grid: StaggeredGrid, dirichlet_ends: tuple[int, int]) -> float:
        """Paramètre de Young ``2 / (1 + √(1 - ρ_J²))`` pour le laplacien sur rectangle.

        Le rayon spectral de Jacobi ``ρ_J = 1 - λ_min / λ_diag`` dépend des conditions aux
        limites : le mode le plus lent d'une direction a pour nombre d'onde ``π/N`` avec
        Dirichlet aux deux extrémités, ``π/(2N)`` avec une seule, 0 en Neumann pur.
        """
        # Coefficients du laplacien.
        cx, cy = 1.0 / grid.dx**2, 1.0 / grid.dy**2
        # Plus petites valeurs propres (mode le plus lent) de chaque direction.
        modes = []
        for c, n, ends in ((cx, grid.Nx, dirichlet_ends[0]), (cy, grid.Ny, dirichlet_ends[1])):
            # Nombre d'onde du mode le plus lent selon le nombre d'extrémités en Dirichlet.
            theta = {0: np.pi / n, 1: np.pi / (2 * n), 2: np.pi / n}[ends]
            # Valeur propre discrète du laplacien 1D : 4 c sin²(θ/2).
            modes.append((ends, 4.0 * c * np.sin(0.5 * theta) ** 2))
        if dirichlet_ends == (0, 0):  # Neumann pur : plus petit mode non constant
            lam_min = min(lam for _, lam in modes)
        else:  # les directions en Neumann pur contribuent par leur mode constant (λ = 0)
            lam_min = sum(lam for ends, lam in modes if ends > 0)
        # Rayon spectral de l'itération de Jacobi (diagonale = 2 cx + 2 cy).
        rho_jacobi = 1.0 - lam_min / (2.0 * cx + 2.0 * cy)
        # Formule de Young pour le ω optimal de SOR.
        return float(2.0 / (1.0 + np.sqrt(1.0 - rho_jacobi**2)))

    def solve(self, b: np.ndarray, x0: np.ndarray | None = None) -> tuple[np.ndarray, SolveInfo]:
        # Point de départ (copie).
        x = np.zeros_like(b) if x0 is None else x0.astype(float, copy=True)
        # Raccourcis : indices rouges/noirs et relaxation.
        r_idx, b_idx, w = self._red, self._black, self.omega
        # Inconnues et second membre séparés par couleur (copies par indexation avancée).
        xr, xb = x[r_idx], x[b_idx]
        br, bb = b[r_idx], b[b_idx]
        # Diagonales correspondantes.
        dr, db = self._diag[r_idx], self._diag[b_idx]
        # Point de départ déjà convergé (fréquent près d'un état stationnaire).
        res = self.relative_residual(x, b)
        if res < self.tol:
            return x, SolveInfo(0, res, True)
        for k in range(1, self.maxiter + 1):
            # Demi-balayage rouge : valeur de Gauss-Seidel (b - A_rb x_noir)/diag, puis
            # relaxation x += ω (valeur_GS - x), sur toutes les cellules rouges à la fois.
            xr += w * ((br - self._A_rb @ xb) / dr - xr)
            # Demi-balayage noir, avec les valeurs rouges tout juste mises à jour.
            xb += w * ((bb - self._A_br @ xr) / db - xb)
            # Test de convergence périodique (et à la dernière itération).
            if k % self.check_every == 0 or k == self.maxiter:
                # Réassemblage du vecteur complet.
                x[r_idx], x[b_idx] = xr, xb
                res = self.relative_residual(x, b)
                if res < self.tol:
                    return x, SolveInfo(k, res, True)
        # Non convergé : on renvoie la dernière estimation.
        x[r_idx], x[b_idx] = xr, xb
        return x, SolveInfo(self.maxiter, res, False)


def make_pressure_solver(
    method: str,
    operator: PoissonOperator,
    tol: float = 1e-8,
    maxiter: int = 10_000,
    preconditioner: str = "ilu",
    omega: float | None = None,
) -> PressureSolver:
    """Fabrique le solveur ``method`` ('direct', 'cg', 'bicgstab', 'sor', 'jacobi') pour ``operator``.

    Le solveur direct est de loin le plus rapide tant que la factorisation tient en
    mémoire (quelques 10⁵ cellules) ; les méthodes itératives démarrent de la pression
    du pas précédent. ``cg`` exige un préconditionneur symétrique ('jacobi', 'ilu', 'amg').
    """
    # Matrice du système.
    A = operator.matrix
    # Aiguillage selon la méthode demandée (fonction « fabrique »).
    if method == "direct":
        return DirectPressureSolver(A)
    if method in ("cg", "bicgstab"):
        return KrylovPressureSolver(A, method, preconditioner, tol, maxiter)
    if method == "sor":
        # Le SOR a besoin de la grille et des conditions aux limites pour estimer ω.
        return SORPressureSolver(A, operator.grid, tol, maxiter, omega, dirichlet_ends=operator.dirichlet_ends)
    if method == "jacobi":
        # omega sert ici de facteur de relaxation de Jacobi (1 par défaut).
        return JacobiPressureSolver(A, tol, maxiter, 1.0 if omega is None else omega)
    raise ValueError(f"Solveur de pression inconnu : {method!r}.")

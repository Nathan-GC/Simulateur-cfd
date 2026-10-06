"""Analyse des écoulements : champs dérivés, efforts, spectres, profils.

Organisation du module :

1. **Champs 2D** : vitesse, pression, vorticité ``ω = ∂v/∂x - ∂u/∂y``, critère Q,
   fonction de courant, divergence (:func:`compute_fields`) ;
2. **Efforts** : intégration de la pression et du frottement sur le contour en marches
   d'escalier de l'obstacle (:class:`BodySurface`), vérifiée par un bilan de quantité de
   mouvement sur un volume de contrôle (:class:`ControlVolume`) ;
3. **Analyse spectrale** : détection du régime établi, FFT, nombre de Strouhal ;
4. **Profils** : ``Cp`` le long de la paroi, profils de sillage ``u(y)/U∞``, longueur de
   recirculation ;
5. **Moniteurs** (callbacks du solveur) : :class:`ForceMonitor` (``Cd(t)``, ``Cl(t)``,
   ``Cm(t)``) et :class:`FieldAverager` (champs moyens, tensions de Reynolds) ;
6. **Incompressibilité** : suivi de ``max|∇·u|``.

Les résultats tabulaires s'exportent par ``save(path)`` au format ``.csv`` ou ``.npz``.
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# Journalisation des avertissements (régime non établi, station hors domaine...).
import logging

# ``dataclass`` : conteneurs de résultats ; ``fields`` (renommé pour éviter la confusion avec
# nos « champs » physiques) liste les attributs d'une dataclass.
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields

# Chemins de fichiers portables (exports).
from pathlib import Path

# ``Literal`` : paramètres texte restreints.
from typing import Literal

import numpy as np

# ``integrate.trapezoid`` : intégrale par la méthode des trapèzes ;
# ``ndimage`` : transformée de distance (cellule fluide la plus proche).
from scipy import integrate, ndimage

# ``scipy.signal`` : détection de pics, suppression de tendance, fenêtres d'apodisation.
from scipy import signal as sps

# Type d'entrée (choix de la pression de référence).
from .boundary import Inlet

# Classe de base des obstacles ; profil NACA et rectangle (corps élancés : Cp le long de la corde).
from .geometry import NACA4, Obstacle, Rectangle

# Grille décalée.
from .grid import StaggeredGrid

# Opérateurs discrets du solveur (réutilisés pour rester cohérent avec lui).
from .operators import advective_fluxes, cell_centered_velocity, divergence

# État de l'écoulement et solveur.
from .solver import FlowState, NavierStokesSolver

# Journal du module.
logger = logging.getLogger(__name__)


def _save_table(path: str | Path, columns: dict[str, np.ndarray], metadata: dict[str, float] | None = None) -> Path:
    """Écrit des colonnes de même longueur en ``.csv`` (une ligne d'en-tête) ou ``.npz``.

    Les métadonnées (grandeurs de référence) ne sont conservées que dans le format ``.npz``.
    """
    # Conversion en objet Path (accepte une chaîne).
    path = Path(path)
    # Création du dossier parent si nécessaire (parents=True : dossiers intermédiaires ;
    # exist_ok=True : pas d'erreur s'il existe déjà).
    path.parent.mkdir(parents=True, exist_ok=True)
    # Format binaire NumPy compressible.
    if path.suffix == ".npz":
        # {**a, **b} fusionne deux dictionnaires ; chaque valeur devient un tableau NumPy.
        arrays = {k: np.asarray(v) for k, v in {**(metadata or {}), **columns}.items()}
        # np.savez : un tableau par clé dans l'archive .npz.
        np.savez(path, **arrays)
    # Format texte lisible par un tableur.
    elif path.suffix == ".csv":
        # np.column_stack : colonnes côte à côte -> tableau (n_lignes, n_colonnes).
        data = np.column_stack([np.asarray(c, dtype=float) for c in columns.values()])
        # np.savetxt : écriture texte ; header = noms des colonnes séparés par des virgules ;
        # comments="" : pas de « # » devant l'en-tête ; fmt : 10 chiffres significatifs.
        np.savetxt(path, data, delimiter=",", header=",".join(columns), comments="", fmt="%.10g")
    else:
        raise ValueError(f"Format non supporté : {path.suffix!r} (attendu : .csv ou .npz).")
    # Chemin écrit (pratique pour l'afficher).
    return path


def _time_mean(t: np.ndarray, y: np.ndarray) -> float:
    """Moyenne temporelle (trapèzes), valable pour un échantillonnage non uniforme."""
    # Moins de deux instants (ou durée nulle) : simple moyenne arithmétique.
    if len(t) < 2 or t[-1] == t[0]:
        return float(np.mean(y))
    # Moyenne = intégrale de y(t) sur la durée / durée ; trapezoid tient compte des pas
    # de temps inégaux (pas adaptatif).
    return float(integrate.trapezoid(y, t) / (t[-1] - t[0]))


def _obstacle(solver: NavierStokesSolver, obstacle: int | Obstacle) -> Obstacle:
    """Obstacle désigné par un objet ou par son indice dans la liste du solveur."""
    # Objet fourni directement.
    if isinstance(obstacle, Obstacle):
        return obstacle
    # Indice demandé mais aucun obstacle défini.
    if not solver.obstacles:
        raise ValueError("La simulation ne comporte aucun obstacle.")
    # Accès par indice (0 = premier obstacle).
    return solver.obstacles[obstacle]


# ======================================================================= 1. champs 2D
def _buried_faces(solid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Faces u et v (tableaux avec fantômes) enfouies dans un obstacle : toutes leurs cellules
    voisines sont solides (une seule voisine pour les faces du bord du domaine)."""
    # Dimensions de la grille.
    nx, ny = solid.shape
    # Faces u, mêmes dimensions que le tableau u (lignes fantômes comprises, jamais enfouies).
    bu = np.zeros((nx + 1, ny + 2), dtype=bool)
    # Face intérieure enfouie : cellules gauche ET droite solides.
    bu[1:-1, 1:-1] = solid[:-1] & solid[1:]
    # Faces du bord du domaine : enfouies si leur unique cellule voisine est solide.
    bu[0, 1:-1], bu[-1, 1:-1] = solid[0], solid[-1]
    # Même construction pour les faces v (colonnes fantômes comprises).
    bv = np.zeros((nx + 2, ny + 1), dtype=bool)
    bv[1:-1, 1:-1] = solid[:, :-1] & solid[:, 1:]
    bv[1:-1, 0], bv[1:-1, -1] = solid[:, 0], solid[:, -1]
    return bu, bv


def _mirror_difference(lower: np.ndarray, upper: np.ndarray, lower_buried: np.ndarray, upper_buried: np.ndarray) -> np.ndarray:
    """``upper - lower``, une valeur enfouie étant remplacée par l'opposé de sa voisine fluide.

    La vitesse s'annule ainsi à mi-distance des deux nœuds, c'est-à-dire sur la paroi.
    """
    # Nœud inférieur enfoui (et supérieur fluide) : valeur miroir -upper.
    lo = np.where(lower_buried & ~upper_buried, -upper, lower)
    # Nœud supérieur enfoui (et inférieur fluide) : valeur miroir -lower.
    up = np.where(upper_buried & ~lower_buried, -lower, upper)
    # Différence (les deux enfouis : 0 - 0 = 0, intérieur du solide).
    return up - lo


def corner_derivatives(
    u: np.ndarray, v: np.ndarray, grid: StaggeredGrid, solid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Dérivées croisées ``(∂u/∂y, ∂v/∂x)`` aux coins des cellules, formes ``(Nx+1, Ny+1)``.

    Avec ``solid``, l'adhérence est imposée par valeurs miroirs sur les parois des
    obstacles, ce qui donne la vorticité pariétale correcte (et non sa moitié).
    """
    if solid is None or not solid.any():
        # Sans obstacle : différences entre lignes consécutives de u (np.diff axe 1) et entre
        # colonnes consécutives de v (axe 0) ; les fantômes portent déjà les CL du domaine.
        du, dv = np.diff(u, axis=1), np.diff(v, axis=0)
    else:
        # Faces enfouies dans les obstacles.
        bu, bv = _buried_faces(solid)
        # Différences avec substitution miroir près des parois des obstacles.
        du = _mirror_difference(u[:, :-1], u[:, 1:], bu[:, :-1], bu[:, 1:])
        dv = _mirror_difference(v[:-1, :], v[1:, :], bv[:-1, :], bv[1:, :])
    # Division par l'écart entre les nœuds : dérivées.
    return du / grid.dy, dv / grid.dx


def corners_to_centers(a: np.ndarray) -> np.ndarray:
    """Moyenne des quatre coins de chaque cellule : ``(Nx+1, Ny+1) -> (Nx, Ny)``."""
    # Coins sud-ouest, sud-est, nord-ouest et nord-est de chaque cellule.
    return 0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])


def vorticity_nodes(u: np.ndarray, v: np.ndarray, grid: StaggeredGrid, solid: np.ndarray | None = None) -> np.ndarray:
    """Vorticité ``ω = ∂v/∂x - ∂u/∂y`` aux coins des cellules (sa position naturelle sur grille MAC)."""
    # Les deux dérivées croisées sont naturellement centrées aux coins.
    du_dy, dv_dx = corner_derivatives(u, v, grid, solid)
    return dv_dx - du_dy


def velocity_gradients(
    u: np.ndarray, v: np.ndarray, grid: StaggeredGrid, solid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gradient de vitesse ``(∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y)`` au centre des cellules."""
    # Dérivées croisées aux coins.
    du_dy, dv_dx = corner_derivatives(u, v, grid, solid)
    # ∂u/∂x au centre : (u_est - u_ouest)/dx, exact sur grille décalée.
    du_dx = np.diff(u[:, 1:-1], axis=0) / grid.dx
    # ∂v/∂y au centre : (v_nord - v_sud)/dy.
    dv_dy = np.diff(v[1:-1, :], axis=1) / grid.dy
    # Dérivées croisées ramenées au centre par moyenne des 4 coins.
    return du_dx, corners_to_centers(du_dy), corners_to_centers(dv_dx), dv_dy


def q_criterion(u: np.ndarray, v: np.ndarray, grid: StaggeredGrid, solid: np.ndarray | None = None) -> np.ndarray:
    """Critère Q = ½(‖Ω‖² - ‖S‖²) au centre des cellules.

    En 2D, ``Q = -½(u_x² + v_y²) - u_y v_x`` (égal à ``det ∇u`` si ``∇·u = 0``) : Q > 0
    signale un cœur tourbillonnaire, où la rotation l'emporte sur la déformation.
    """
    # Les quatre composantes du gradient de vitesse.
    ux, uy, vx, vy = velocity_gradients(u, v, grid, solid)
    # Q = -½ tr(∇u · ∇u).
    return -0.5 * (ux**2 + vy**2) - uy * vx


def stream_function(u: np.ndarray, v: np.ndarray, grid: StaggeredGrid) -> np.ndarray:
    """Fonction de courant ``ψ`` aux coins (``u = ∂ψ/∂y``, ``v = -∂ψ/∂x``, ``ψ(0, 0) = 0``).

    Le champ discret étant à divergence nulle, l'intégration est indépendante du chemin :
    les obstacles et les parois sont exactement des lignes de courant.
    """
    # ψ aux coins des cellules.
    psi = np.zeros((grid.Nx + 1, grid.Ny + 1))
    # Le long du bord sud : ψ(x_{i+1}) = ψ(x_i) - v dx ; np.cumsum = somme cumulée.
    psi[1:, 0] = -np.cumsum(v[1:-1, 0]) * grid.dx
    # Vers le haut dans chaque colonne : ψ(y_{j+1}) = ψ(y_j) + u dy (faces u de la colonne) ;
    # psi[:, :1] (colonne 2D) est diffusé sur toutes les lignes.
    psi[:, 1:] = psi[:, :1] + np.cumsum(u[:, 1:-1], axis=1) * grid.dy
    return psi


@dataclass
class FlowFields:
    """Champs aux centres des cellules (instantanés ou moyens), formes ``(Nx, Ny)``.

    ``vorticity`` et ``q_criterion`` sont mis à zéro dans le solide ; :meth:`masked`
    masque les cellules solides pour l'affichage.
    """

    # Instant des champs.
    t: float
    # Coordonnées 1D des centres des cellules.
    x: np.ndarray
    y: np.ndarray
    # Vitesses au centre des cellules.
    u: np.ndarray
    v: np.ndarray
    # Pression.
    p: np.ndarray
    # Vorticité ω (moyenne des 4 coins).
    vorticity: np.ndarray
    # Critère Q.
    q_criterion: np.ndarray
    # Divergence discrète (contrôle de l'incompressibilité).
    divergence: np.ndarray
    # Masque solide.
    solid: np.ndarray

    @property
    def speed(self) -> np.ndarray:
        """Norme de la vitesse ``‖u‖``."""
        # np.hypot(a, b) = √(a² + b²), sans dépassement de capacité.
        return np.hypot(self.u, self.v)

    def masked(self, name: str) -> np.ma.MaskedArray:
        """Champ ``name`` (ou ``'speed'``) masqué dans le solide."""
        # np.ma.masked_array : tableau dont les éléments masqués (solide) sont ignorés par les
        # calculs et laissés vides par matplotlib.
        return np.ma.masked_array(getattr(self, name), mask=self.solid)

    def as_dict(self) -> dict[str, np.ndarray]:
        """Dictionnaire {nom: tableau} (export .npz)."""
        # dataclass_fields(self) : liste des attributs déclarés ; getattr lit chacun.
        out = {f.name: np.asarray(getattr(self, f.name)) for f in dataclass_fields(self)}
        # Ajout de la norme de la vitesse (propriété calculée).
        out["speed"] = self.speed
        return out


def fields_from_arrays(
    grid: StaggeredGrid, solid: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray, t: float = 0.0
) -> FlowFields:
    """Champs dérivés d'un état ``(u, v, p)`` sur grille MAC."""
    # Vitesses interpolées au centre des cellules.
    uc, vc = cell_centered_velocity(u, v)
    # Vorticité aux coins (avec miroir aux parois) ramenée aux centres.
    omega = corners_to_centers(vorticity_nodes(u, v, grid, solid))
    # Critère Q aux centres.
    q = q_criterion(u, v, grid, solid)
    # Valeurs sans signification dans le solide : mises à zéro.
    omega[solid] = 0.0
    q[solid] = 0.0
    # p.copy() : copie pour que le résultat ne change pas quand le solveur avance.
    return FlowFields(t, grid.x_c, grid.y_c, uc, vc, p.copy(), omega, q, divergence(u, v, grid), solid)


def compute_fields(solver: NavierStokesSolver, state: FlowState | None = None) -> FlowFields:
    """Champs 2D de l'état courant du solveur (ou de ``state``)."""
    # État courant par défaut.
    st = solver.state if state is None else state
    return fields_from_arrays(solver.grid, solver.solid, st.u, st.v, st.p, st.t)


def interpolate(
    field: np.ndarray, grid: StaggeredGrid, x: np.ndarray, y: np.ndarray, fluid: np.ndarray | None = None
) -> np.ndarray:
    """Interpolation bilinéaire d'un champ défini aux centres des cellules.

    Avec ``fluid`` (masque booléen), seules les cellules fluides contribuent (poids
    renormalisés) ; un point dont les quatre voisines sont solides prend la valeur de la
    cellule fluide la plus proche. Les points extérieurs au domaine sont ramenés au bord.
    """
    # Coordonnées des points (formes rendues compatibles).
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    # Position en « indices de cellule » (centre i à x = (i + 0.5) dx), bornée au domaine.
    fx = np.clip(x / grid.dx - 0.5, 0.0, grid.Nx - 1.0)
    fy = np.clip(y / grid.dy - 0.5, 0.0, grid.Ny - 1.0)
    # Indice de la cellule inférieure gauche du carré d'interpolation (au plus N-2 pour que
    # la cellule i0+1 existe) ; astype(int64) tronque vers le bas (valeurs positives).
    i0 = np.minimum(fx.astype(np.int64), grid.Nx - 2)
    j0 = np.minimum(fy.astype(np.int64), grid.Ny - 2)
    # Coordonnées locales dans le carré, entre 0 et 1.
    ax, ay = fx - i0, fy - j0
    # Numérateur (somme pondérée) et dénominateur (somme des poids).
    num = np.zeros(x.shape)
    den = np.zeros(x.shape)
    # Les 4 coins du carré et leurs poids bilinéaires.
    for di, dj, w in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)), (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
        # Les cellules solides reçoivent un poids nul.
        if fluid is not None:
            w = w * fluid[i0 + di, j0 + dj]
        # Accumulation (indexation avancée : une valeur par point).
        num += w * field[i0 + di, j0 + dj]
        den += w
    # Division là où au moins un poids est non nul ; NaN ailleurs (tous voisins solides).
    out = np.divide(num, den, out=np.full(x.shape, np.nan), where=den > 1e-12)
    # Points restés sans valeur.
    bad = np.isnan(out)
    if fluid is not None and bad.any():
        # distance_transform_edt(~fluid, return_indices=True) : pour chaque cellule, indices
        # (i, j) de la cellule fluide la plus proche (distance euclidienne).
        _, (near_i, near_j) = ndimage.distance_transform_edt(~fluid, return_indices=True)
        # Cellule contenant chaque point problématique.
        ic = np.clip((x[bad] / grid.dx).astype(np.int64), 0, grid.Nx - 1)
        jc = np.clip((y[bad] / grid.dy).astype(np.int64), 0, grid.Ny - 1)
        # Valeur de la cellule fluide la plus proche.
        out[bad] = field[near_i[ic, jc], near_j[ic, jc]]
    return out


# ========================================================================= 2. efforts
@dataclass(frozen=True)
class SurfaceForces:
    """Efforts du fluide sur un corps, par unité d'envergure.

    Décomposition : pression, frottement visqueux et flux convectif absorbé par la paroi
    (terme numérique du contour en marches d'escalier, d'ordre Δ). ``moment`` est pris
    autour du point de référence, positif dans le sens trigonométrique.
    """

    # Composantes x et y des trois contributions.
    fx_pressure: float
    fy_pressure: float
    fx_viscous: float
    fy_viscous: float
    fx_convective: float
    fy_convective: float
    # Moment autour du point de référence.
    moment: float

    @property
    def fx(self) -> float:
        # Force totale selon x (traînée pour un écoulement selon +x).
        return self.fx_pressure + self.fx_viscous + self.fx_convective

    @property
    def fy(self) -> float:
        # Force totale selon y (portance).
        return self.fy_pressure + self.fy_viscous + self.fy_convective


def _nonzero(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices (i, j) des éléments vrais d'un masque 2D, sous forme de tuple."""
    # np.nonzero renvoie un tuple de tableaux d'indices, un par dimension.
    return tuple(np.nonzero(mask))  # type: ignore[return-value]


class BodySurface:
    """Contour en marches d'escalier d'un corps et efforts que le fluide lui transmet.

    Les efforts sont la somme, sur le contour, des échanges que le solveur réalise entre
    les faces fluides et les faces figées du corps, si bien qu'ils vérifient exactement le
    bilan de quantité de mouvement discret (cf. :class:`ControlVolume`) :

    * **pression** : celle de la cellule fluide adjacente à chaque face du contour ;
    * **frottement** : flux visqueux ``-μ ∂u/∂n`` vers les faces figées, avec la condition
      miroir d'adhérence (``μ u / (Δ/2)`` le long d'une paroi) ;
    * **convection** : flux ``ρ u u`` vers les faces figées, terme propre au contour en
      marches d'escalier, qui tend vers 0 avec le raffinement.

    Une intégration « continue » du seul frottement ``μ ∂u_t/∂n`` sur les faces parallèles
    à l'écoulement ignore les échanges avec les contremarches et sous-estime la traînée
    (de l'ordre de 10 % à Re = 20), sans converger avec le maillage.

    Les calculs sont restreints à une fenêtre entourant le corps (``margin`` mailles, au
    moins la largeur du stencil convectif) : le coût par appel est indépendant de la taille
    du domaine.
    """

    def __init__(
        self,
        grid: StaggeredGrid,
        solid: np.ndarray,
        body: np.ndarray | None = None,
        scheme: str = "quick",
        margin: int = 4,
    ) -> None:
        # Corps étudié : tout le solide par défaut, sinon la partie du solide désignée.
        body = solid if body is None else (body & solid)
        if not body.any():
            raise ValueError("Le corps ne contient aucune cellule solide.")
        # Indices des cellules du corps.
        rows, cols = np.nonzero(body)
        # Fenêtre [i0, i1[ x [j0, j1[ : boîte englobante du corps élargie de ``margin`` mailles,
        # bornée au domaine (max/min).
        i0, i1 = max(int(rows.min()) - margin, 0), min(int(rows.max()) + margin + 1, grid.Nx)
        j0, j1 = max(int(cols.min()) - margin, 0), min(int(cols.max()) + margin + 1, grid.Ny)
        # Mémorisation de la grille, du schéma d'advection (le même que le solveur) et de la fenêtre.
        self.grid, self.scheme, self._window = grid, scheme, (i0, i1, j0, j1)
        # Masques restreints à la fenêtre (tous les indices ci-dessous sont locaux).
        solid, body = solid[i0:i1, j0:j1], body[i0:i1, j0:j1]
        fluid = ~solid
        nx, ny = solid.shape
        # Coordonnées locales : centres, faces.
        self._xc, self._yc = grid.x_c[i0:i1], grid.y_c[j0:j1]
        self._xf, self._yf = grid.x_f[i0 : i1 + 1], grid.y_f[j0 : j1 + 1]
        # Pression : faces du contour groupées par normale sortante du corps.
        self._east = _nonzero(body[:-1, :] & fluid[1:, :])  # corps (i, j) | fluide (i+1, j)
        self._west = _nonzero(fluid[:-1, :] & body[1:, :])  # fluide (i, j) | corps (i+1, j)
        self._north = _nonzero(body[:, :-1] & fluid[:, 1:])  # corps (i, j) sous fluide (i, j+1)
        self._south = _nonzero(fluid[:, :-1] & body[:, 1:])  # fluide (i, j) sous corps (i, j+1)
        # Aucune face de contact (corps entièrement noyé dans un autre solide) : erreur.
        if not sum(idx[0].size for idx in (self._east, self._west, self._north, self._south)):
            raise ValueError("Le corps n'a aucune face en contact avec le fluide.")

        # Faces u (nx+1, ny) : actives (fluide des deux côtés), figées par le corps, enfouies.
        u_act = np.zeros((nx + 1, ny), dtype=bool)
        # Face u active : ses deux cellules voisines sont fluides (comme dans le solveur).
        u_act[1:-1] = fluid[:-1] & fluid[1:]
        # Face u figée par le corps : au moins une cellule voisine appartient au corps.
        u_body = np.zeros_like(u_act)
        u_body[1:-1] = body[:-1] | body[1:]
        u_body[0], u_body[-1] = body[0], body[-1]
        # Face u enfouie : toutes ses cellules voisines sont solides (condition miroir).
        u_bur = np.zeros_like(u_act)
        u_bur[1:-1] = solid[:-1] & solid[1:]
        u_bur[0], u_bur[-1] = solid[0], solid[-1]
        # Faces v (nx, ny+1).
        v_act = np.zeros((nx, ny + 1), dtype=bool)
        v_act[:, 1:-1] = fluid[:, :-1] & fluid[:, 1:]
        v_body = np.zeros_like(v_act)
        v_body[:, 1:-1] = body[:, :-1] | body[:, 1:]
        v_body[:, 0], v_body[:, -1] = body[:, 0], body[:, -1]
        v_bur = np.zeros_like(v_act)
        v_bur[:, 1:-1] = solid[:, :-1] & solid[:, 1:]
        v_bur[:, 0], v_bur[:, -1] = solid[:, 0], solid[:, -1]

        def ghost_rows(a: np.ndarray) -> np.ndarray:  # (nx+1, ny) -> colonnes intérieures (nx-1, ny+2)
            # Ajoute des lignes fantômes (fausses) pour aligner le masque sur le tableau u
            # complet, puis ne garde que les faces u intérieures (comme le flux Fuy).
            out = np.zeros((nx + 1, ny + 2), dtype=bool)
            out[:, 1:-1] = a
            return out[1:-1]

        def ghost_cols(a: np.ndarray) -> np.ndarray:  # (nx, ny+1) -> lignes intérieures (nx+2, ny-1)
            # Même principe pour v : colonnes fantômes ajoutées, faces v intérieures gardées.
            out = np.zeros((nx + 2, ny + 1), dtype=bool)
            out[1:-1, :] = a
            return out[:, 1:-1]

        # Échanges : « pos » = face active du côté négatif de la face de volume de contrôle
        # et face du corps du côté positif, « neg » = l'inverse.
        # u, échanges selon x : faces de volume de contrôle au centre des cellules.
        self._ux = (_nonzero(u_act[:-1] & u_body[1:]), _nonzero(u_body[:-1] & u_act[1:]))  # centres (nx, ny)
        # v, échanges selon y : faces de volume de contrôle au centre des cellules.
        self._vy = (_nonzero(v_act[:, :-1] & v_body[:, 1:]), _nonzero(v_body[:, :-1] & v_act[:, 1:]))
        # u, échanges selon y : faces de volume de contrôle aux coins.
        ua, ub, uw = ghost_rows(u_act), ghost_rows(u_body), ghost_rows(u_bur)  # coins (nx-1, ny+1)
        self._uy = (_nonzero(ua[:, :-1] & ub[:, 1:]), _nonzero(ub[:, :-1] & ua[:, 1:]))
        # Le voisin figé est-il enfoui (miroir) ? Pour « pos » c'est la face du dessus, pour
        # « neg » celle du dessous ; l'indexation par le tuple d'indices extrait les valeurs.
        self._uy_mirror = (uw[:, 1:][self._uy[0]], uw[:, :-1][self._uy[1]])
        # v, échanges selon x : faces de volume de contrôle aux coins.
        va, vb, vw = ghost_cols(v_act), ghost_cols(v_body), ghost_cols(v_bur)  # coins (nx+1, ny-1)
        self._vx = (_nonzero(va[:-1] & vb[1:]), _nonzero(vb[:-1] & va[1:]))
        self._vx_mirror = (vw[1:][self._vx[0]], vw[:-1][self._vx[1]])

    def forces(
        self,
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        mu: float,
        rho: float = 1.0,
        center: tuple[float, float] = (0.0, 0.0),
    ) -> SurfaceForces:
        """Efforts de pression, de frottement et de convection, et moment autour de ``center``."""
        # Restriction des champs à la fenêtre (vues, sans copie) ; u et v gardent une rangée
        # fantôme locale de chaque côté.
        i0, i1, j0, j1 = self._window
        u, v, p = u[i0 : i1 + 1, j0 : j1 + 2], v[i0 : i1 + 2, j0 : j1 + 1], p[i0:i1, j0:j1]
        # Pas d'espace.
        dx, dy = self.grid.dx, self.grid.dy
        # Coordonnées locales (centres et faces).
        xc_, yc_, xf, yf = self._xc, self._yc, self._xf, self._yf
        # Point de référence du moment.
        xm, ym = center
        # Flux convectifs locaux, calculés avec le même schéma que le solveur.
        fux, fuy, fvx, fvy = advective_fluxes(u, v, self.scheme)
        # Accumulateurs par direction et par type d'effort (p = pression, v = visqueux,
        # c = convectif).
        fx = {"p": 0.0, "v": 0.0, "c": 0.0}
        fy = {"p": 0.0, "v": 0.0, "c": 0.0}
        moment = 0.0

        def add(axis: dict[str, float], kind: str, force: np.ndarray, x: np.ndarray, y: np.ndarray, along_x: bool) -> None:
            """Ajoute des forces élémentaires et leur moment (bras de levier = position (x, y))."""
            nonlocal moment
            # Somme des forces élémentaires.
            axis[kind] += float(force.sum())
            # Moment en z : -(y - ym) Fx pour une force selon x, (x - xm) Fy pour une force selon y.
            moment += float(np.sum(-(y - ym) * force if along_x else (x - xm) * force))

        # --- pression de la cellule fluide adjacente à chaque face du contour
        (ie, je), (iw, jw), (i_n, j_n), (i_s, j_s) = self._east, self._west, self._north, self._south
        # Normale +x (fluide à droite) : force -p dy ; position : face x_f[i+1].
        add(fx, "p", -p[ie + 1, je] * dy, xf[ie + 1], yc_[je], True)
        # Normale -x (fluide à gauche) : force +p dy.
        add(fx, "p", p[iw, jw] * dy, xf[iw + 1], yc_[jw], True)
        # Normale +y (fluide au-dessus) : force -p dx.
        add(fy, "p", -p[i_n, j_n + 1] * dx, xc_[i_n], yf[j_n + 1], False)
        # Normale -y (fluide au-dessous) : force +p dx.
        add(fy, "p", p[i_s, j_s] * dx, xc_[i_s], yf[j_s + 1], False)

        # --- quantité de mouvement x, faces de VC selon x (centres des cellules, sans miroir)
        # ∂u/∂x au centre des cellules (le solveur n'applique pas de miroir selon x pour u).
        du_dx = np.diff(u[:, 1:-1], axis=0) / dx
        # zip associe le signe (+1 pour « pos », -1 pour « neg ») à chaque liste d'indices.
        for sign, (i, j) in zip((1.0, -1.0), self._ux):
            # Flux convectif ρ u u sortant de la face active vers la face du corps.
            add(fx, "c", sign * rho * fux[i, j] * dy, xc_[i], yc_[j], True)
            # Flux visqueux -μ ∂u/∂x (même convention de signe).
            add(fx, "v", -sign * mu * du_dx[i, j] * dy, xc_[i], yc_[j], True)
        # --- quantité de mouvement x, faces de VC selon y (coins, miroir si voisin enfoui)
        # Faces u intérieures (lignes fantômes comprises), alignées avec fuy.
        uc = u[1:-1, :]
        # Indices des échanges « pos » (ip, jp) et « neg » (im, jm).
        (ip, jp), (im, jm) = self._uy
        below, above = uc[ip, jp], uc[ip, jp + 1]  # face active dessous, corps dessus
        # ∂u/∂y vu par la face active : voisin du dessus remplacé par -u si enfoui.
        du_dy = (np.where(self._uy_mirror[0], -below, above) - below) / dy
        # Flux sortant vers le haut (dans le corps) : convectif + visqueux.
        add(fx, "c", rho * fuy[ip, jp] * dx, xf[ip + 1], yf[jp], True)
        add(fx, "v", -mu * du_dy * dx, xf[ip + 1], yf[jp], True)
        below, above = uc[im, jm], uc[im, jm + 1]  # corps dessous, face active dessus
        # ∂u/∂y vu par la face active : voisin du dessous remplacé par -u si enfoui.
        du_dy = (above - np.where(self._uy_mirror[1], -above, below)) / dy
        # Flux sortant vers le bas : signe opposé.
        add(fx, "c", -rho * fuy[im, jm] * dx, xf[im + 1], yf[jm], True)
        add(fx, "v", mu * du_dy * dx, xf[im + 1], yf[jm], True)

        # --- quantité de mouvement y, faces de VC selon y (centres des cellules, sans miroir)
        # ∂v/∂y au centre des cellules.
        dv_dy = np.diff(v[1:-1, :], axis=1) / dy
        for sign, (i, j) in zip((1.0, -1.0), self._vy):
            add(fy, "c", sign * rho * fvy[i, j] * dx, xc_[i], yc_[j], False)
            add(fy, "v", -sign * mu * dv_dy[i, j] * dx, xc_[i], yc_[j], False)
        # --- quantité de mouvement y, faces de VC selon x (coins, miroir si voisin enfoui)
        # Faces v intérieures (colonnes fantômes comprises), alignées avec fvx.
        vc = v[:, 1:-1]
        (ip, jp), (im, jm) = self._vx
        left, right = vc[ip, jp], vc[ip + 1, jp]  # face active à gauche, corps à droite
        # ∂v/∂x vu par la face active (miroir si le voisin de droite est enfoui).
        dv_dx = (np.where(self._vx_mirror[0], -left, right) - left) / dx
        add(fy, "c", rho * fvx[ip, jp] * dy, xf[ip], yf[jp + 1], False)
        add(fy, "v", -mu * dv_dx * dy, xf[ip], yf[jp + 1], False)
        left, right = vc[im, jm], vc[im + 1, jm]  # corps à gauche, face active à droite
        # ∂v/∂x vu par la face active (miroir si le voisin de gauche est enfoui).
        dv_dx = (right - np.where(self._vx_mirror[1], -right, left)) / dx
        add(fy, "c", -rho * fvx[im, jm] * dy, xf[im], yf[jm + 1], False)
        add(fy, "v", mu * dv_dx * dy, xf[im], yf[jm + 1], False)

        # Assemblage du résultat.
        return SurfaceForces(fx["p"], fy["p"], fx["v"], fy["v"], fx["c"], fy["c"], moment)


class ControlVolume:
    """Bilan de quantité de mouvement sur un rectangle entourant le corps.

    Pour le fluide contenu dans le rectangle ::

        F = ∮ σ·n dS - ∮ ρ u (u·n) dS - d/dt ∫ ρ u dV,      σ = -p I + μ (∇u + ∇uᵀ)

    Cette méthode, peu sensible à la représentation en marches d'escalier, sert de
    vérification indépendante de :class:`BodySurface`. Le rectangle est calé sur les faces
    de la grille et ne doit ni toucher le bord du domaine ni intersecter d'obstacle.
    """

    def __init__(self, grid: StaggeredGrid, solid: np.ndarray, box: tuple[float, float, float, float]) -> None:
        # Rectangle demandé.
        xmin, xmax, ymin, ymax = box
        # Indices des faces de grille les plus proches (round = arrondi à l'entier le plus proche).
        ia, ib = round(xmin / grid.dx), round(xmax / grid.dx)
        ja, jb = round(ymin / grid.dy), round(ymax / grid.dy)
        # Il faut au moins une cellule de marge avec le bord (dérivées centrées sur le contour).
        if not (1 <= ia < ib <= grid.Nx - 1 and 1 <= ja < jb <= grid.Ny - 1):
            raise ValueError("Le volume de contrôle doit être strictement intérieur au domaine.")
        # Cadre de deux cellules d'épaisseur autour du contour : il doit être entièrement fluide.
        frame = np.zeros_like(solid)
        frame[ia - 1 : ib + 1, ja - 1 : jb + 1] = True
        frame[ia + 1 : ib - 1, ja + 1 : jb - 1] = False
        if (solid & frame).any():
            raise ValueError("Le contour du volume de contrôle ne doit pas intersecter d'obstacle.")
        # Mémorisation de la grille, du rectangle effectif et des indices.
        self.grid = grid
        self.bounds = (ia * grid.dx, ib * grid.dx, ja * grid.dy, jb * grid.dy)
        self._ia, self._ib, self._ja, self._jb = ia, ib, ja, jb

    def surface_forces(self, u: np.ndarray, v: np.ndarray, p: np.ndarray, rho: float, mu: float) -> tuple[float, float]:
        """Flux de quantité de mouvement et contraintes à travers le contour (sans terme instationnaire)."""
        dx, dy = self.grid.dx, self.grid.dy
        ia, ib, ja, jb = self._ia, self._ib, self._ja, self._jb
        # Composantes de la force accumulées sur les quatre côtés.
        fx = fy = 0.0
        rows = slice(ja + 1, jb + 1)  # lignes u (J = j+1) des cellules ja..jb-1
        for i, sign in ((ib, 1.0), (ia, -1.0)):  # faces est (n = +x) et ouest (n = -x)
            # Vitesse normale u sur le côté (portée exactement par les faces u d'indice i).
            un = u[i, rows]
            # v sur le côté : moyenne des 4 faces v voisines.
            vn = 0.25 * (v[i, ja:jb] + v[i + 1, ja:jb] + v[i, ja + 1 : jb + 1] + v[i + 1, ja + 1 : jb + 1])
            # Pression sur le côté : moyenne des deux cellules adjacentes.
            pn = 0.5 * (p[i - 1, ja:jb] + p[i, ja:jb])
            # Dérivées centrées nécessaires aux contraintes visqueuses.
            du_dx = (u[i + 1, rows] - u[i - 1, rows]) / (2.0 * dx)
            du_dy = (u[i, ja + 2 : jb + 2] - u[i, ja:jb]) / (2.0 * dy)
            dv_dx = 0.5 * (v[i + 1, ja:jb] - v[i, ja:jb] + v[i + 1, ja + 1 : jb + 1] - v[i, ja + 1 : jb + 1]) / dx
            # σ_xx - ρ u² (contrainte normale moins flux de quantité de mouvement), orienté par le signe.
            fx += sign * float(np.sum(-pn + 2.0 * mu * du_dx - rho * un * un)) * dy
            # σ_xy - ρ u v.
            fy += sign * float(np.sum(mu * (du_dy + dv_dx) - rho * un * vn)) * dy
        cols = slice(ia + 1, ib + 1)  # colonnes v (I = i+1) des cellules ia..ib-1
        for j, sign in ((jb, 1.0), (ja, -1.0)):  # faces nord (n = +y) et sud (n = -y)
            # Vitesse normale v sur le côté.
            vn = v[cols, j]
            # u sur le côté : moyenne des 4 faces u voisines.
            un = 0.25 * (u[ia:ib, j] + u[ia + 1 : ib + 1, j] + u[ia:ib, j + 1] + u[ia + 1 : ib + 1, j + 1])
            # Pression sur le côté.
            pn = 0.5 * (p[ia:ib, j - 1] + p[ia:ib, j])
            # Dérivées centrées.
            dv_dy = (v[cols, j + 1] - v[cols, j - 1]) / (2.0 * dy)
            dv_dx = (v[ia + 2 : ib + 2, j] - v[ia:ib, j]) / (2.0 * dx)
            du_dy = 0.5 * (u[ia:ib, j + 1] - u[ia:ib, j] + u[ia + 1 : ib + 1, j + 1] - u[ia + 1 : ib + 1, j]) / dy
            # σ_yx - ρ u v.
            fx += sign * float(np.sum(mu * (du_dy + dv_dx) - rho * un * vn)) * dx
            # σ_yy - ρ v².
            fy += sign * float(np.sum(-pn + 2.0 * mu * dv_dy - rho * vn * vn)) * dx
        return fx, fy

    def momentum(self, u: np.ndarray, v: np.ndarray, rho: float) -> tuple[float, float]:
        """Quantité de mouvement ``∫ ρ u dV`` contenue dans le rectangle."""
        ia, ib, ja, jb = self._ia, self._ib, self._ja, self._jb
        # Vitesses au centre des cellules du rectangle.
        uc = 0.5 * (u[ia:ib, ja + 1 : jb + 1] + u[ia + 1 : ib + 1, ja + 1 : jb + 1])
        vc = 0.5 * (v[ia + 1 : ib + 1, ja:jb] + v[ia + 1 : ib + 1, ja + 1 : jb + 1])
        # Aire d'une cellule.
        area = self.grid.cell_area
        # Somme ρ u dA (les cellules solides ont une vitesse nulle).
        return rho * float(uc.sum()) * area, rho * float(vc.sum()) * area


# ============================================================== 3. analyse spectrale
def stationary_start(t: np.ndarray, signal: np.ndarray, rel_tol: float = 0.05, n_last: int = 4) -> float:
    """Instant à partir duquel les oscillations de ``signal`` sont établies.

    Les extrema successifs sont comparés aux valeurs finales (médiane des ``n_last``
    derniers maxima et minima) : le régime est établi dès que tous les extrema suivants
    restent à ``rel_tol`` x amplitude (demi-crête-à-crête) près. À défaut d'oscillations
    exploitables, la seconde moitié de l'enregistrement est retenue.
    """
    # Conversion en tableaux de flottants.
    t, s = np.asarray(t, dtype=float), np.asarray(signal, dtype=float)
    # Valeur de repli : milieu de l'enregistrement.
    fallback = float(t[0] + 0.5 * (t[-1] - t[0]))
    # Étendue (max - min) du signal sur sa seconde moitié (échelle des oscillations finales).
    span = float(np.ptp(s[len(s) // 2 :])) if len(s) > 1 else 0.0
    # Signal constant : pas d'oscillation.
    if span == 0.0:
        return fallback
    # find_peaks : indices des maxima locaux ; prominence élimine les petites bosses de bruit.
    maxima, _ = sps.find_peaks(s, prominence=0.02 * span)
    # Minima = maxima du signal opposé.
    minima, _ = sps.find_peaks(-s, prominence=0.02 * span)
    # Trop peu d'extrema pour juger de la stabilisation.
    if min(len(maxima), len(minima)) < 2 * n_last:
        return fallback
    # Niveaux finaux : médiane (robuste) des n_last derniers maxima et minima.
    hi, lo = np.median(s[maxima[-n_last:]]), np.median(s[minima[-n_last:]])
    # Tolérance absolue : rel_tol x amplitude (demi-crête-à-crête).
    tol = rel_tol * 0.5 * (hi - lo)
    # Début de la stabilisation estimé séparément sur les maxima et sur les minima.
    starts = []
    for idx, final in ((maxima, hi), (minima, lo)):
        # Extrema encore trop éloignés du niveau final.
        bad = np.flatnonzero(np.abs(s[idx] - final) > tol)
        # Un des derniers extrema est hors tolérance : régime non établi.
        if bad.size and bad[-1] >= len(idx) - n_last:
            logger.warning("Regime etabli non detecte : seconde moitie du signal retenue.")
            return fallback
        # Premier extremum après le dernier extremum hors tolérance.
        starts.append(idx[bad[-1] + 1] if bad.size else idx[0])
    # Le plus tardif des deux critères.
    return float(t[max(starts)])


@dataclass
class Spectrum:
    """Spectre d'amplitude d'un signal rééchantillonné uniformément sur ``[t_start, t_end]``."""

    # Fréquences (axe du spectre).
    frequency: np.ndarray
    # Amplitude de chaque composante fréquentielle.
    amplitude: np.ndarray
    # Fréquence et amplitude du pic principal (interpolées).
    peak_frequency: float
    peak_amplitude: float
    # Fenêtre temporelle analysée.
    t_start: float
    t_end: float

    @property
    def resolution(self) -> float:
        """Résolution fréquentielle intrinsèque ``1/T`` (avant bourrage de zéros)."""
        return 1.0 / (self.t_end - self.t_start)

    @property
    def n_periods(self) -> float:
        """Nombre de périodes du pic contenues dans la fenêtre d'analyse."""
        return (self.t_end - self.t_start) * self.peak_frequency

    def save(self, path: str | Path) -> Path:
        # Export des colonnes (fréquence, amplitude) et des métadonnées (npz seulement).
        return _save_table(
            path,
            {"frequency": self.frequency, "amplitude": self.amplitude},
            {"peak_frequency": self.peak_frequency, "t_start": self.t_start, "t_end": self.t_end},
        )


def amplitude_spectrum(
    t: np.ndarray,
    signal: np.ndarray,
    t_start: float | None = None,
    t_end: float | None = None,
    window: str = "hann",
    zero_padding: int = 4,
) -> Spectrum:
    """Spectre d'amplitude par FFT sur ``[t_start, t_end]``.

    Le signal (échantillonné à pas variable) est interpolé sur une grille uniforme,
    privé de sa tendance linéaire, fenêtré puis complété de zéros ; le pic est localisé
    par interpolation parabolique du logarithme de l'amplitude.
    """
    t, s = np.asarray(t, dtype=float), np.asarray(signal, dtype=float)
    # Bornes de la fenêtre d'analyse (tout le signal par défaut).
    t0 = float(t[0] if t_start is None else t_start)
    t1 = float(t[-1] if t_end is None else t_end)
    # Échantillons de la fenêtre.
    sel = (t >= t0) & (t <= t1)
    if sel.sum() < 8:
        raise ValueError("Fenêtre d'analyse trop courte (moins de 8 échantillons).")
    # Nombre de points de rééchantillonnage : puissance de 2 (FFT efficace) au moins égale au
    # nombre d'échantillons, et au moins 1024 ; 1 << k = 2**k.
    n = max(1024, 1 << int(np.ceil(np.log2(sel.sum()))))
    # Instants régulièrement espacés.
    tu = np.linspace(t0, t1, n)
    # np.interp : interpolation linéaire sur la grille régulière ; detrend(type="linear")
    # retire la moyenne et la tendance linéaire (dérive lente).
    su = sps.detrend(np.interp(tu, t[sel], s[sel]), type="linear")
    # Fenêtre d'apodisation (Hann) : limite les fuites spectrales dues aux bords de la fenêtre.
    w = sps.get_window(window, n)
    # Longueur de FFT avec bourrage de zéros (spectre plus finement échantillonné).
    nfft = zero_padding * n
    # np.fft.rfft : FFT d'un signal réel (fréquences positives) ; normalisation 2/Σw pour
    # obtenir l'amplitude d'une sinusoïde.
    amplitude = 2.0 * np.abs(np.fft.rfft(su * w, nfft)) / w.sum()
    # Fréquences correspondantes (pas d'échantillonnage tu[1] - tu[0]).
    frequency = np.fft.rfftfreq(nfft, tu[1] - tu[0])
    # Indice du pic en excluant la fréquence nulle (amplitude[1:], d'où le +1).
    k = int(np.argmax(amplitude[1:]) + 1)
    # Estimation brute : fréquence et amplitude du point de maximum.
    peak_f, peak_a = frequency[k], amplitude[k]
    # Affinage par une parabole passant par les 3 points (log) autour du maximum.
    if k + 1 < amplitude.size and min(amplitude[k - 1 : k + 2]) > 0:
        a, b, c = np.log(amplitude[k - 1 : k + 2])
        # Décalage du sommet de la parabole, en fraction d'intervalle fréquentiel.
        delta = 0.5 * (a - c) / (a - 2.0 * b + c)
        # Fréquence et amplitude du sommet.
        peak_f = frequency[k] + delta * (frequency[1] - frequency[0])
        peak_a = float(np.exp(b - 0.25 * (a - c) * delta))
    return Spectrum(frequency, amplitude, float(peak_f), float(peak_a), t0, t1)


def crossing_frequency(t: np.ndarray, signal: np.ndarray, t_start: float | None = None) -> float:
    """Fréquence estimée par les passages ascendants par la valeur moyenne (vérification de la FFT)."""
    t, s = np.asarray(t, dtype=float), np.asarray(signal, dtype=float)
    # Échantillons postérieurs à t_start.
    sel = t >= (t[0] if t_start is None else t_start)
    # Signal centré sur sa moyenne temporelle.
    t, s = t[sel], s[sel] - _time_mean(t[sel], s[sel])
    # Indices k où le signal passe de négatif à positif entre k et k+1.
    k = np.flatnonzero((s[:-1] < 0.0) & (s[1:] >= 0.0))
    # Il faut au moins deux passages pour mesurer une période.
    if k.size < 2:
        return float("nan")
    # Instant précis de chaque passage par interpolation linéaire.
    crossings = t[k] - s[k] * (t[k + 1] - t[k]) / (s[k + 1] - s[k])
    # Fréquence = nombre de périodes complètes / durée correspondante.
    return float((crossings.size - 1) / (crossings[-1] - crossings[0]))


def strouhal_number(
    t: np.ndarray, signal: np.ndarray, length: float, velocity: float, t_start: float | None = None
) -> tuple[float, Spectrum]:
    """Nombre de Strouhal ``St = f L / U`` du pic spectral de ``signal`` (en général ``Cl``).

    ``t_start=None`` : début du régime établi détecté par :func:`stationary_start`.
    """
    # Détection automatique du régime établi.
    if t_start is None:
        t_start = stationary_start(t, signal)
    # Spectre sur le régime établi.
    spectrum = amplitude_spectrum(t, signal, t_start)
    # Moins de 5 périodes : résolution fréquentielle médiocre.
    if spectrum.n_periods < 5:
        logger.warning("Seulement %.1f periodes analysees : Strouhal peu precis.", spectrum.n_periods)
    # St = f L / U.
    return spectrum.peak_frequency * length / velocity, spectrum


# ======================================================================= 4. profils
def reference_pressure(
    solver: NavierStokesSolver,
    where: str | float | tuple[float, float] = "auto",
    fields: FlowFields | None = None,
    pressure: np.ndarray | None = None,
) -> float:
    """Pression de référence ``p∞`` pour les coefficients de pression.

    ``'inlet'`` : moyenne sur la première colonne de cellules (amont) ; ``'outlet'`` : 0
    (pression imposée en sortie) ; un nombre ; ou un point ``(x, y)``. ``'auto'`` choisit
    ``'inlet'`` si le côté ouest est une entrée, ``'outlet'`` sinon. Le champ utilisé est
    ``pressure`` s'il est fourni, sinon ``fields.p``, sinon la pression courante du solveur.
    """
    # Champ de pression : fourni, moyen, ou instantané.
    p = pressure if pressure is not None else (solver.state.p if fields is None else fields.p)
    # Choix automatique.
    if where == "auto":
        where = "inlet" if isinstance(solver.config.boundaries.west, Inlet) else "outlet"
    # Valeur numérique explicite.
    if isinstance(where, (int, float)):
        return float(where)
    # Pression imposée en sortie.
    if where == "outlet":
        return 0.0
    # Moyenne sur la colonne de cellules d'entrée (cellules fluides seulement).
    if where == "inlet":
        return float(p[0][solver.fluid[0]].mean())
    # Chaîne inconnue.
    if isinstance(where, str):
        raise ValueError(f"Référence de pression inconnue : {where!r}.")
    # Point (x, y) : interpolation restreinte au fluide.
    x, y = where
    return float(interpolate(p, solver.grid, np.array([x]), np.array([y]), solver.fluid)[0])


def _resample_closed(points: np.ndarray, n: int) -> np.ndarray:
    """Rééchantillonne un contour fermé en ``n`` points équirépartis en abscisse curviligne."""
    # Contour refermé (premier point répété à la fin).
    closed = np.vstack([points, points[:1]])
    # Abscisse curviligne cumulée : longueurs des segments (np.hypot des écarts dx, dy).
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    # n abscisses régulièrement espacées (sans répéter le point de départ).
    targets = np.linspace(0.0, s[-1], n, endpoint=False)
    # Interpolation linéaire de x(s) et y(s).
    return np.column_stack([np.interp(targets, s, closed[:, 0]), np.interp(targets, s, closed[:, 1])])


@dataclass
class SurfaceDistribution:
    """Coefficient de pression le long de la paroi d'un obstacle.

    Le parcours part du point le plus proche de l'arrêt amont et longe d'abord
    l'extrados (dessus). ``theta`` est l'angle, en degrés, mesuré depuis la direction
    amont autour du centre de l'obstacle et croissant le long du parcours (0 = avant,
    90 = dessus, 180 = culot, 270 = dessous) ; ``s`` l'abscisse curviligne.
    """

    # Points du contour.
    x: np.ndarray
    y: np.ndarray
    # Abscisse curviligne.
    s: np.ndarray
    # Angle depuis l'amont (degrés).
    theta: np.ndarray
    # Coefficient de pression.
    cp: np.ndarray

    def save(self, path: str | Path) -> Path:
        return _save_table(path, {"theta_deg": self.theta, "s": self.s, "x": self.x, "y": self.y, "cp": self.cp})


def surface_pressure(
    solver: NavierStokesSolver,
    obstacle: int | Obstacle = 0,
    *,
    n: int = 360,
    offset: float = 0.5,
    p_ref: str | float | tuple[float, float] = "auto",
    fields: FlowFields | None = None,
) -> SurfaceDistribution:
    """Distribution de ``Cp = (p - p∞) / (½ ρ U∞²)`` le long du contour exact de l'obstacle.

    La pression est échantillonnée à ``offset`` maille à l'extérieur de la paroi, par
    interpolation restreinte aux cellules fluides (``∂p/∂n ≈ 0`` à la paroi). ``fields``
    permet d'utiliser un champ moyen (:meth:`FieldAverager.fields`) pour obtenir ``<Cp>``.
    """
    # Obstacle et son contour exact (géométrie, pas l'escalier).
    ob = _obstacle(solver, obstacle)
    outline = ob.outline()
    if outline is None:
        raise ValueError(f"Contour inconnu pour l'obstacle {ob!r}.")
    g = solver.grid
    # n points équirépartis ; [::-1] inverse l'ordre : sens horaire (de l'amont vers le dessus).
    pts = _resample_closed(outline, n)[::-1]  # sens horaire : de l'amont vers l'extrados
    # Tangente par différence centrée (np.roll : voisins suivant et précédent, contour bouclé).
    tangent = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    # Normale sortante : tangente tournée de +90° pour un parcours horaire.
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])  # normale sortante (parcours horaire)
    # Normalisation à la longueur 1 (keepdims=True garde la forme (n, 1) pour la division).
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    # Centre de l'obstacle.
    xc, yc = ob.center
    # Angle mesuré depuis la direction amont (-x) : arctan2(dy, -dx).
    angle = np.arctan2(pts[:, 1] - yc, -(pts[:, 0] - xc))  # 0 à l'amont, dans ]-π, π]
    # Point le plus proche de l'arrêt amont.
    start = int(np.argmin(np.abs(angle)))
    # Rotation des tableaux pour commencer à ce point (expression génératrice dépaquetée).
    pts, normal, angle = (np.roll(a, -start, axis=0) for a in (pts, normal, angle))
    # np.unwrap supprime les sauts de 2π : angle continu et croissant ; conversion en degrés.
    theta = np.degrees(np.unwrap(angle))
    # Points de sonde décalés de offset mailles vers le fluide.
    probes = pts + offset * max(g.dx, g.dy) * normal
    # Champ de pression (instantané ou moyen).
    p = solver.state.p if fields is None else fields.p
    # Pression interpolée en n'utilisant que les cellules fluides.
    p_wall = interpolate(p, g, probes[:, 0], probes[:, 1], solver.fluid)
    # Pression dynamique de référence ½ ρ U².
    q = 0.5 * solver.rho * solver.U_ref**2
    # Coefficient de pression.
    cp = (p_wall - reference_pressure(solver, p_ref, fields)) / q
    # Abscisse curviligne depuis le point de départ.
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    return SurfaceDistribution(pts[:, 0], pts[:, 1], s, theta, cp)


def chord_line(obstacle: Obstacle) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Bord d'attaque et bord de fuite d'un corps élancé, ``None`` pour un corps non profilé.

    Corps élancés : profils NACA et rectangles d'allongement au moins 2 (plaques). Pour eux,
    la pression pariétale se lit le long de la corde (:func:`chordwise_pressure`) plutôt qu'en
    fonction de l'angle autour du centre, et la recirculation sur l'axe n'a pas de sens.
    """
    # Profil : corde définie par sa géométrie.
    if isinstance(obstacle, NACA4):
        return obstacle.leading_edge, obstacle.trailing_edge
    # Plaque : milieux des deux petits côtés (un carré ou un rectangle trapu n'est pas élancé).
    if isinstance(obstacle, Rectangle) and obstacle.aspect_ratio >= 2.0:
        return obstacle.leading_edge, obstacle.trailing_edge
    return None


@dataclass
class ChordwiseDistribution:
    """Coefficient de pression le long de la corde d'un corps élancé, extrados et intrados.

    ``x_upper`` et ``x_lower`` sont les abscisses réduites ``x/c`` le long de la corde
    (0 au bord d'attaque, 1 au bord de fuite), croissantes.
    """

    # Extrados (face supérieure) : abscisses réduites et Cp.
    x_upper: np.ndarray
    cp_upper: np.ndarray
    # Intrados (face inférieure).
    x_lower: np.ndarray
    cp_lower: np.ndarray

    def save(self, path: str | Path) -> Path:
        # Table longue : face (1 = extrados, 0 = intrados), x/c, Cp.
        return _save_table(
            path,
            {
                "upper": np.concatenate([np.ones(self.x_upper.size), np.zeros(self.x_lower.size)]),
                "x_c": np.concatenate([self.x_upper, self.x_lower]),
                "cp": np.concatenate([self.cp_upper, self.cp_lower]),
            },
        )


def chordwise_pressure(
    distribution: SurfaceDistribution, leading_edge: tuple[float, float], trailing_edge: tuple[float, float]
) -> ChordwiseDistribution:
    """Répartit une distribution pariétale entre extrados et intrados, en fonction de ``x/c``.

    Le contour est coupé en deux arcs au bord d'attaque (``x/c`` minimal) et au bord de
    fuite (``x/c`` maximal) ; l'extrados est l'arc situé en moyenne au-dessus de la corde.
    """
    le, te = np.asarray(leading_edge, dtype=float), np.asarray(trailing_edge, dtype=float)
    # Vecteur corde et carré de sa longueur.
    chord = te - le
    length2 = float(chord @ chord)
    # Points du contour relatifs au bord d'attaque.
    rel = np.column_stack([distribution.x, distribution.y]) - le
    # Abscisse réduite (projection sur la corde) et distance signée à la corde (produit vectoriel :
    # positive à gauche de la corde orientée de l'amont vers l'aval, c'est-à-dire au-dessus).
    xi = rel @ chord / length2
    eta = (chord[0] * rel[:, 1] - chord[1] * rel[:, 0]) / np.sqrt(length2)
    n = xi.size
    # Indices du bord d'attaque et du bord de fuite le long du contour (fermé, parcouru en boucle).
    i_le, i_te = int(np.argmin(xi)), int(np.argmax(xi))

    def arc(start: int, stop: int) -> np.ndarray:
        # Indices de start à stop inclus en tournant dans le sens du parcours (% n : retour au début).
        return np.arange(start, start + (stop - start) % n + 1) % n

    first, second = arc(i_le, i_te), arc(i_te, i_le)
    # L'arc situé en moyenne au-dessus de la corde est l'extrados.
    upper, lower = (first, second) if eta[first].mean() >= eta[second].mean() else (second, first)

    def sort_by_x(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Points de l'arc rangés par x/c croissant (np.argsort donne l'ordre de tri).
        order = np.argsort(xi[idx], kind="stable")
        return xi[idx][order], distribution.cp[idx][order]

    return ChordwiseDistribution(*sort_by_x(upper), *sort_by_x(lower))


@dataclass
class WakeProfile:
    """Profil transverse de vitesse à la station ``x = x_ref + station · L`` du sillage.

    ``eta = (y - y_c)/L`` ; ``u`` et ``v`` sont adimensionnés par ``U∞`` (NaN dans le solide).
    """

    # Station x/L.
    station: float
    # Abscisse physique de la station.
    x: float
    # Ordonnée réduite (y - y_c)/L.
    eta: np.ndarray
    # Vitesses adimensionnées.
    u: np.ndarray
    v: np.ndarray

    @property
    def centerline_velocity(self) -> float:
        """Vitesse ``u/U∞`` sur l'axe ``eta = 0``."""
        # Points valides (np.isfinite exclut les NaN du solide).
        ok = np.isfinite(self.u)
        # Interpolation linéaire en eta = 0.
        return float(np.interp(0.0, self.eta[ok], self.u[ok]))

    @property
    def deficit(self) -> float:
        """Déficit de vitesse : écoulement extérieur moins minimum du sillage."""
        # np.nanmax / np.nanmin ignorent les NaN.
        return float(np.nanmax(self.u) - np.nanmin(self.u))

    @property
    def half_width(self) -> float:
        """Demi-largeur du sillage : demi-distance entre les points à mi-déficit."""
        u = self.u
        # Indice du minimum de vitesse (centre du sillage).
        k = int(np.nanargmin(u))
        # Niveau à mi-déficit.
        half = u[k] + 0.5 * self.deficit
        # Points au-dessus du centre ayant remonté au niveau « half ».
        above = np.flatnonzero(u[k:] >= half)
        # Points au-dessous du centre (jusqu'au centre inclus) au niveau « half ».
        below = np.flatnonzero(u[: k + 1] >= half)
        # Pas de croisement de part et d'autre : largeur non définie.
        if not above.size or not below.size:
            return float("nan")
        # Premier point au-dessus et dernier point au-dessous franchissant le niveau.
        j = k + above[0]
        i = below[-1]
        # Interpolation linéaire des positions exactes de franchissement.
        eta_up = np.interp(half, [u[j - 1], u[j]], [self.eta[j - 1], self.eta[j]])
        eta_down = np.interp(half, [u[i + 1], u[i]], [self.eta[i + 1], self.eta[i]])
        # Demi-distance entre les deux.
        return float(0.5 * (eta_up - eta_down))


def wake_profiles(
    solver: NavierStokesSolver,
    stations: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0),
    *,
    fields: FlowFields | None = None,
    obstacle: int | Obstacle = 0,
    origin: Literal["center", "rear"] = "center",
) -> list[WakeProfile]:
    """Profils ``u(y)/U∞`` aux stations ``x/L`` (mesurées depuis le centre ou l'arrière de l'obstacle).

    Pour un sillage instationnaire, passer les champs moyens (:meth:`FieldAverager.fields`).
    """
    # Champs instantanés par défaut.
    f = compute_fields(solver) if fields is None else fields
    ob = _obstacle(solver, obstacle)
    # Grille, longueur et vitesse de référence.
    g, L, U = solver.grid, ob.reference_length, solver.U_ref
    xc, yc = ob.center
    # Origine des stations : centre de l'obstacle ou son arrière (bounds[1] = xmax).
    x0 = xc if origin == "center" else ob.bounds[1]
    # Vitesses mises à NaN dans le solide.
    u = np.where(f.solid, np.nan, f.u)
    v = np.where(f.solid, np.nan, f.v)
    profiles = []
    for station in stations:
        # Abscisse de la station.
        x = x0 + station * L
        # Station hors du domaine : ignorée.
        if not g.x_c[0] <= x <= g.x_c[-1]:
            logger.warning("Station x/L = %g hors du domaine : ignoree.", station)
            continue
        # Position en indices de cellule, colonne de gauche i0 et poids linéaire w.
        a = x / g.dx - 0.5
        i0 = min(int(a), g.Nx - 2)
        w = a - i0
        # Profil interpolé linéairement entre les colonnes i0 et i0+1, adimensionné.
        profiles.append(
            WakeProfile(
                station=float(station),
                x=float(x),
                eta=(g.y_c - yc) / L,
                u=((1.0 - w) * u[i0] + w * u[i0 + 1]) / U,
                v=((1.0 - w) * v[i0] + w * v[i0 + 1]) / U,
            )
        )
    return profiles


def save_wake_profiles(path: str | Path, profiles: list[WakeProfile]) -> Path:
    """Exporte des profils de sillage en table longue (station, eta, u, v)."""
    # np.full(taille, station) répète le numéro de station sur chaque ligne du profil ;
    # np.concatenate met les profils bout à bout.
    return _save_table(
        path,
        {
            "station": np.concatenate([np.full(p.eta.size, p.station) for p in profiles]),
            "eta": np.concatenate([p.eta for p in profiles]),
            "u": np.concatenate([p.u for p in profiles]),
            "v": np.concatenate([p.v for p in profiles]),
        },
    )


def recirculation_length(
    solver: NavierStokesSolver, *, fields: FlowFields | None = None, obstacle: int | Obstacle = 0
) -> float:
    """Longueur de la zone de recirculation derrière l'obstacle, en longueurs de référence.

    Mesurée sur l'axe ``y = y_c`` depuis l'arrière de l'obstacle jusqu'au point où la
    vitesse longitudinale redevient positive (utiliser un champ stationnaire ou moyen) ;
    vaut 0 en l'absence d'écoulement de retour.
    """
    f = compute_fields(solver) if fields is None else fields
    ob = _obstacle(solver, obstacle)
    g = solver.grid
    # Arrière de l'obstacle et ordonnée de son axe.
    x_rear, (_, yc) = ob.bounds[1], ob.center
    # Centres des cellules situées derrière l'obstacle.
    xs = g.x_c[g.x_c > x_rear]
    # Vitesse u le long de l'axe (interpolation restreinte au fluide).
    u_axis = interpolate(f.u, g, xs, np.full(xs.shape, yc), ~f.solid)
    # Écoulement de retour (u < 0) juste derrière l'obstacle ?
    negative = np.flatnonzero(u_axis[: max(3, xs.size // 4)] < 0.0)
    if not negative.size:
        return 0.0
    # Début de la zone de retour.
    k0 = negative[0]
    # Premier point, au-delà, où u redevient positive (fin de la bulle).
    positive = np.flatnonzero(u_axis[k0:] >= 0.0)
    if not positive.size:
        return float("nan")
    k = k0 + positive[0]
    # Position exacte du changement de signe par interpolation linéaire.
    x_end = xs[k - 1] - u_axis[k - 1] * (xs[k] - xs[k - 1]) / (u_axis[k] - u_axis[k - 1])
    # Longueur rapportée à la longueur de référence.
    return float((x_end - x_rear) / ob.reference_length)


# ===================================================================== 5. moniteurs
# Colonnes des séries temporelles d'efforts.
_FORCE_COLUMNS = (
    "time", "cd", "cl", "cd_pressure", "cd_viscous", "cd_convective",
    "cl_pressure", "cl_viscous", "cl_convective", "cm",
)


@dataclass(frozen=True)
class ForceSummary:
    """Synthèse des coefficients aérodynamiques sur la fenêtre ``[t_start, t_end]``."""

    # Fenêtre d'analyse.
    t_start: float
    t_end: float
    # Écoulement instationnaire (oscillations de Cl) ?
    unsteady: bool
    # Traînée : moyenne, fluctuation rms et décomposition moyenne.
    cd_mean: float
    cd_rms: float
    cd_pressure_mean: float
    cd_viscous_mean: float
    cd_convective_mean: float
    # Portance : moyenne, rms, amplitude (demi-crête-à-crête).
    cl_mean: float
    cl_rms: float
    cl_amplitude: float
    # Moment de tangage moyen.
    cm_mean: float
    # Nombre de Strouhal et nombre de périodes analysées.
    strouhal: float
    n_periods: float
    # Traînée moyenne par volume de contrôle (NaN si non calculée).
    cd_cv_mean: float = float("nan")

    def __str__(self) -> str:
        # Texte multiligne (sans accents : consoles Windows) utilisé par print(summary).
        lines = [
            f"Fenetre d'analyse : t = [{self.t_start:.2f}, {self.t_end:.2f}]"
            + (f" ({self.n_periods:.1f} periodes)" if self.unsteady else " (ecoulement stationnaire)"),
            f"Cd moyen = {self.cd_mean:.4f} (pression {self.cd_pressure_mean:.4f} + frottement "
            f"{self.cd_viscous_mean:.4f} + convection paroi {self.cd_convective_mean:.4f}), "
            f"Cd' rms = {self.cd_rms:.4f}",
            f"Cl moyen = {self.cl_mean:.4f}, Cl' rms = {self.cl_rms:.4f}, amplitude Cl = {self.cl_amplitude:.4f}",
            f"Cm moyen = {self.cm_mean:.4f}",
        ]
        if self.unsteady:
            lines.append(f"Nombre de Strouhal St = {self.strouhal:.4f}")
        if np.isfinite(self.cd_cv_mean):
            lines.append(f"Cd moyen (bilan de quantite de mouvement) = {self.cd_cv_mean:.4f}")
        # "\n".join : lignes séparées par des retours à la ligne.
        return "\n".join(lines)


@dataclass
class ForceHistory:
    """Séries temporelles des coefficients aérodynamiques."""

    # Instants d'enregistrement.
    time: np.ndarray
    # Coefficients de traînée et de portance totaux.
    cd: np.ndarray
    cl: np.ndarray
    # Décomposition de la traînée.
    cd_pressure: np.ndarray
    cd_viscous: np.ndarray
    cd_convective: np.ndarray
    # Décomposition de la portance.
    cl_pressure: np.ndarray
    cl_viscous: np.ndarray
    cl_convective: np.ndarray
    # Moment de tangage.
    cm: np.ndarray
    # Grandeurs de référence des coefficients.
    reference_length: float
    reference_velocity: float
    # Coefficients par volume de contrôle (optionnels).
    cd_cv: np.ndarray | None = None
    cl_cv: np.ndarray | None = None

    def is_unsteady(self, threshold: float = 1e-3) -> bool:
        """Oscillations de ``Cl`` d'amplitude supérieure à ``threshold`` sur le dernier quart."""
        # Dernier quart de l'enregistrement.
        tail = self.time >= self.time[0] + 0.75 * (self.time[-1] - self.time[0])
        # Demi-étendue de Cl comparée au seuil.
        return bool(0.5 * np.ptp(self.cl[tail]) > threshold)

    def spectrum(self, name: str = "cl", t_start: float | None = None) -> Spectrum:
        """Spectre d'amplitude de la série ``name`` (régime établi détecté sur ``Cl`` par défaut)."""
        if t_start is None:
            t_start = stationary_start(self.time, self.cl)
        # getattr(self, "cd") : série demandée par son nom.
        return amplitude_spectrum(self.time, getattr(self, name), t_start)

    def summary(self, t_start: float | None = None) -> ForceSummary:
        """Moyennes, fluctuations et Strouhal sur le régime établi (ou à partir de ``t_start``)."""
        t = self.time
        if t.size < 8:
            raise ValueError("Historique trop court pour une synthèse.")
        # Écoulement instationnaire ?
        unsteady = self.is_unsteady()
        # Début de la fenêtre : régime établi détecté, ou dernier quart si stationnaire.
        if t_start is None:
            t_start = stationary_start(t, self.cl) if unsteady else float(t[0] + 0.75 * (t[-1] - t[0]))
        # Échantillons de la fenêtre.
        sel = t >= t_start
        ts = t[sel]

        def mean(y: np.ndarray) -> float:
            # Moyenne temporelle sur la fenêtre.
            return _time_mean(ts, y[sel])

        def rms(y: np.ndarray) -> float:
            # Écart-type temporel ; max(..., 0) protège contre un arrondi négatif.
            return float(np.sqrt(max(_time_mean(ts, (y[sel] - mean(y)) ** 2), 0.0)))

        # Strouhal seulement pour un écoulement instationnaire.
        strouhal = n_periods = float("nan")
        if unsteady:
            strouhal, spec = strouhal_number(t, self.cl, self.reference_length, self.reference_velocity, t_start)
            n_periods = spec.n_periods
        # Moyenne de la traînée par volume de contrôle (valeurs finies seulement).
        cd_cv_mean = float("nan")
        if self.cd_cv is not None:
            ok = sel & np.isfinite(self.cd_cv)
            cd_cv_mean = _time_mean(t[ok], self.cd_cv[ok]) if ok.sum() > 1 else float("nan")
        return ForceSummary(
            t_start=float(t_start),
            t_end=float(t[-1]),
            unsteady=unsteady,
            cd_mean=mean(self.cd),
            cd_rms=rms(self.cd),
            cd_pressure_mean=mean(self.cd_pressure),
            cd_viscous_mean=mean(self.cd_viscous),
            cd_convective_mean=mean(self.cd_convective),
            cl_mean=mean(self.cl),
            cl_rms=rms(self.cl),
            cl_amplitude=float(0.5 * np.ptp(self.cl[sel])),
            cm_mean=mean(self.cm),
            strouhal=float(strouhal),
            n_periods=float(n_periods),
            cd_cv_mean=cd_cv_mean,
        )

    def save(self, path: str | Path) -> Path:
        # Colonnes principales.
        columns = {name: getattr(self, name) for name in _FORCE_COLUMNS}
        # Colonnes du volume de contrôle si elles existent (dict.update ajoute des clés).
        if self.cd_cv is not None:
            columns.update(cd_cv=self.cd_cv, cl_cv=self.cl_cv)
        return _save_table(
            path, columns, {"reference_length": self.reference_length, "reference_velocity": self.reference_velocity}
        )


class ForceMonitor:
    """Enregistre ``Cd(t)``, ``Cl(t)`` (pression + frottement) et ``Cm(t)`` d'un obstacle.

    Le moniteur s'inscrit lui-même auprès du solveur (appel tous les ``every`` pas) ::

        forces = ForceMonitor(solver)
        solver.run()
        print(forces.history().summary())

    Coefficients : ``C = F / (½ ρ U∞² L)``, et ``Cm = -M / (½ ρ U∞² L²)`` moment de
    tangage (positif à cabrer) autour de ``moment_center`` : quart de corde d'un profil
    NACA, centre géométrique sinon. ``control_volume=(xmin, xmax, ymin, ymax)`` ajoute
    ``cd_cv`` et ``cl_cv``, issus du bilan de quantité de mouvement (vérification).
    """

    def __init__(
        self,
        solver: NavierStokesSolver,
        obstacle: int | Obstacle = 0,
        *,
        every: int = 1,
        reference_length: float | None = None,
        moment_center: tuple[float, float] | None = None,
        control_volume: tuple[float, float, float, float] | None = None,
        attach: bool = True,
    ) -> None:
        # Obstacle suivi et son masque (même sous-échantillonnage que le solveur).
        ob = _obstacle(solver, obstacle)
        grid = solver.grid
        body = ob.mask(grid, solver.config.numerics.mask_supersampling)
        self.obstacle = ob
        # Contour en escalier et échanges discrets (schéma d'advection du solveur).
        self.surface = BodySurface(grid, solver.solid, body, scheme=solver.config.numerics.advection)
        # Longueur et vitesse de référence des coefficients.
        self.reference_length = ob.reference_length if reference_length is None else reference_length
        self.reference_velocity = solver.U_ref
        # Centre du moment : imposé, sinon pivot du profil (getattr avec valeur par défaut :
        # attribut « pivot_point » s'il existe, centre géométrique sinon).
        self.moment_center = moment_center if moment_center is not None else getattr(ob, "pivot_point", ob.center)
        # Volume de contrôle optionnel.
        self.control_volume = None if control_volume is None else ControlVolume(grid, solver.solid, control_volume)
        # Masse volumique, viscosité dynamique μ = ρ ν et pression dynamique ½ ρ U².
        self._rho, self._mu = solver.rho, solver.rho * solver.nu
        self._q = 0.5 * solver.rho * solver.U_ref**2
        # Noms des séries (plus celles du volume de contrôle s'il est actif).
        names = _FORCE_COLUMNS + (("cd_cv", "cl_cv") if self.control_volume else ())
        # Une liste vide par série.
        self._data: dict[str, list[float]] = {name: [] for name in names}
        # (instant, quantité de mouvement) du dernier appel (dérivée temporelle du bilan).
        self._previous_momentum: tuple[float, tuple[float, float]] | None = None
        # Inscription auprès du solveur : il appellera self(solver) tous les ``every`` pas.
        if attach:
            solver.add_callback(self, every)

    def coefficients(self, solver: NavierStokesSolver) -> dict[str, float]:
        """Coefficients instantanés de l'état courant."""
        st = solver.state
        # Efforts intégrés sur le contour.
        f = self.surface.forces(st.u, st.v, st.p, self._mu, self._rho, self.moment_center)
        # Normalisation des forces : ½ ρ U² L.
        qL = self._q * self.reference_length
        return {
            "time": st.t,
            "cd": f.fx / qL,
            "cl": f.fy / qL,
            "cd_pressure": f.fx_pressure / qL,
            "cd_viscous": f.fx_viscous / qL,
            "cd_convective": f.fx_convective / qL,
            "cl_pressure": f.fy_pressure / qL,
            "cl_viscous": f.fy_viscous / qL,
            "cl_convective": f.fy_convective / qL,
            # Moment : ½ ρ U² L² ; signe - : positif à cabrer (convention aéronautique).
            "cm": -f.moment / (qL * self.reference_length),
        }

    def __call__(self, solver: NavierStokesSolver) -> None:
        """Appelé par le solveur : enregistre les coefficients du pas courant."""
        # Ajout de chaque coefficient à sa série.
        for name, value in self.coefficients(solver).items():
            self._data[name].append(value)
        if self.control_volume is not None:
            st = solver.state
            # Flux et contraintes à travers le contour du volume de contrôle.
            fx, fy = self.control_volume.surface_forces(st.u, st.v, st.p, self._rho, self._mu)
            # Quantité de mouvement contenue.
            momentum = self.control_volume.momentum(st.u, st.v, self._rho)
            # Dérivée temporelle par différence avec l'appel précédent (NaN au premier appel).
            dmx = dmy = float("nan")
            if self._previous_momentum is not None and st.t > self._previous_momentum[0]:
                t0, (mx0, my0) = self._previous_momentum
                dmx, dmy = (momentum[0] - mx0) / (st.t - t0), (momentum[1] - my0) / (st.t - t0)
            self._previous_momentum = (st.t, momentum)
            qL = self._q * self.reference_length
            # F = flux - dM/dt.
            self._data["cd_cv"].append((fx - dmx) / qL)
            self._data["cl_cv"].append((fy - dmy) / qL)

    def history(self) -> ForceHistory:
        """Séries enregistrées sous forme de tableaux NumPy."""
        d = {name: np.asarray(values) for name, values in self._data.items()}
        # ** dépaquette le dictionnaire en arguments nommés ; d.get renvoie None si absent.
        return ForceHistory(
            **{name: d[name] for name in _FORCE_COLUMNS},
            reference_length=self.reference_length,
            reference_velocity=self.reference_velocity,
            cd_cv=d.get("cd_cv"),
            cl_cv=d.get("cl_cv"),
        )


class FieldAverager:
    """Moyennes temporelles des champs à partir de ``t_start`` (pondérées par le temps écoulé).

    Les vitesses sont moyennées aux faces : le champ moyen reste à divergence nulle et
    se prête aux mêmes opérateurs que le champ instantané. Les produits ``uu``, ``vv``,
    ``uv`` au centre des cellules donnent les tensions de Reynolds. Le moniteur s'inscrit
    lui-même auprès du solveur.
    """

    def __init__(self, solver: NavierStokesSolver, t_start: float = 0.0, *, every: int = 1) -> None:
        g = solver.grid
        # Grille, masque, début de la moyenne.
        self.grid, self.solid, self.t_start = g, solver.solid, float(t_start)
        # Sommes pondérées des champs (mêmes formes que les tableaux du solveur).
        self._sum_u, self._sum_v, self._sum_p = np.zeros(g.shape_u), np.zeros(g.shape_v), np.zeros(g.shape_p)
        # Sommes pondérées des produits (tensions de Reynolds).
        self._sum_uu, self._sum_vv, self._sum_uv = np.zeros(g.shape_p), np.zeros(g.shape_p), np.zeros(g.shape_p)
        # Durée totale moyennée et nombre d'échantillons.
        self.duration = 0.0
        self.n_samples = 0
        # Instant du dernier appel.
        self._t_last = solver.state.t
        # Inscription auprès du solveur.
        solver.add_callback(self, every)

    def __call__(self, solver: NavierStokesSolver) -> None:
        st = solver.state
        # Poids = temps écoulé depuis le dernier échantillon (ou depuis t_start).
        weight = st.t - max(self._t_last, self.t_start)
        self._t_last = st.t
        # Avant t_start : rien à accumuler.
        if weight <= 0.0:
            return
        # Vitesses aux centres (produits de Reynolds).
        uc, vc = cell_centered_velocity(st.u, st.v)
        # Accumulation pondérée (+= : en place, sans créer de nouveau tableau).
        self._sum_u += weight * st.u
        self._sum_v += weight * st.v
        self._sum_p += weight * st.p
        self._sum_uu += weight * uc * uc
        self._sum_vv += weight * vc * vc
        self._sum_uv += weight * uc * vc
        self.duration += weight
        self.n_samples += 1

    def _check(self) -> None:
        # Aucune moyenne possible avant le premier échantillon.
        if self.duration <= 0.0:
            raise RuntimeError("Aucun échantillon moyenné (t_start non atteint ?).")

    def mean_state(self) -> FlowState:
        """État moyen (sommes pondérées divisées par la durée)."""
        self._check()
        d = self.duration
        return FlowState(self._sum_u / d, self._sum_v / d, self._sum_p / d, t=self._t_last)

    def fields(self) -> FlowFields:
        """Champs moyens (vitesse, pression, vorticité et critère Q du champ moyen)."""
        st = self.mean_state()
        return fields_from_arrays(self.grid, self.solid, st.u, st.v, st.p, st.t)

    def reynolds_stresses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tensions de Reynolds ``(<u'u'>, <v'v'>, <u'v'>)`` au centre des cellules."""
        # Champ moyen aux centres.
        st = self.mean_state()
        uc, vc = cell_centered_velocity(st.u, st.v)
        d = self.duration
        # <u'u'> = <uu> - <u><u>, etc.
        return self._sum_uu / d - uc * uc, self._sum_vv / d - vc * vc, self._sum_uv / d - uc * vc


# ============================================================= 6. incompressibilité
@dataclass(frozen=True)
class DivergenceReport:
    """Bilan de la contrainte d'incompressibilité ``∇·u = 0``."""

    # Maximum de |∇·u| sur tous les pas enregistrés.
    history_max: float
    # Maximum et moyenne quadratique pour le champ courant.
    current_max: float
    current_rms: float
    # Position (x, y) du maximum courant.
    location: tuple[float, float]

    def __str__(self) -> str:
        return (
            f"max|div u| : {self.history_max:.2e} sur toute la simulation, {self.current_max:.2e} "
            f"actuellement (rms {self.current_rms:.2e}, en x = {self.location[0]:.3f}, y = {self.location[1]:.3f})"
        )


def divergence_report(solver: NavierStokesSolver) -> DivergenceReport:
    """Divergence maximale (historique et champ courant) et sa localisation."""
    # Divergence du champ courant.
    div = solver.divergence()
    fluid = solver.fluid
    # |∇·u| dans le fluide (0 dans le solide).
    magnitude = np.where(fluid, np.abs(div), 0.0)
    # np.argmax donne l'indice aplati du maximum ; np.unravel_index le convertit en (i, j).
    i, j = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    # Historique enregistré par le solveur à chaque pas.
    history = solver.diagnostics.divergence_max
    return DivergenceReport(
        history_max=float(max(history)) if history else float(magnitude.max()),
        current_max=float(magnitude.max()),
        # rms = racine de la moyenne des carrés, sur les cellules fluides.
        current_rms=float(np.sqrt(np.mean(div[fluid] ** 2))),
        location=(float(solver.grid.x_c[i]), float(solver.grid.y_c[j])),
    )

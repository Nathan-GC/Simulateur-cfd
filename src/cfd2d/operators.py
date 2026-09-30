"""Opérateurs discrets sur grille MAC : advection, diffusion, divergence.

Toutes les fonctions sont vectorisées (NumPy) et opèrent sur les tableaux avec
cellules fantômes décrits dans :class:`~cfd2d.grid.StaggeredGrid` :
``u`` de forme ``(Nx+1, Ny+2)`` et ``v`` de forme ``(Nx+2, Ny+1)``.

Rappel sur le découpage NumPy utilisé partout : ``a[1:]`` = tous les éléments sauf le
premier, ``a[:-1]`` = tous sauf le dernier ; ``a[1:] - a[:-1]`` est donc la différence entre
chaque élément et son prédécesseur, calculée d'un bloc sur tout le tableau (sans boucle).
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

import numpy as np

# La grille fournit les pas d'espace dx, dy.
from .grid import StaggeredGrid

# Noms des schémas d'advection acceptés (utilisé dans le message d'erreur).
ADVECTION_SCHEMES = ("upwind", "central", "quick", "tvd")


def reconstruct(q: np.ndarray, a: np.ndarray, axis: int, scheme: str) -> np.ndarray:
    """Valeur de ``q`` aux faces situées entre deux nœuds consécutifs selon ``axis``.

    ``q`` a ``n`` nœuds selon ``axis`` et ``a`` (vitesse advectante aux faces) en a ``n-1`` :
    le signe de ``a`` désigne le nœud amont ``k`` de chaque face.

    * ``upwind`` : ``q_k``, décentré amont d'ordre 1 (robuste, très diffusif) ;
    * ``central`` : ``(q_k + q_k+1)/2``, ordre 2 (oscillant si Re de maille > 2) ;
    * ``quick`` : centré ``- (q_k-1 - 2 q_k + q_k+1)/8``, interpolation quadratique amont
      de Leonard (ordre 3) ;
    * ``tvd`` : ``q_k ± h_k`` avec ``h_k`` la demi-pente limitée de van Leer (MUSCL,
      ordre 2 sans oscillations).

    Les corrections nodales (courbure, pente) sont nulles aux deux nœuds extrêmes, où
    le stencil est incomplet : on y retombe sur le schéma centré (``quick``) ou amont (``tvd``).
    """
    # np.moveaxis place l'axe de travail en première position (vue, sans copie) : le même code
    # traite ainsi la reconstruction selon x (axis=0) comme selon y (axis=1).
    qm = np.moveaxis(q, axis, 0)
    # Même opération pour la vitesse advectante.
    am = np.moveaxis(a, axis, 0)
    if scheme == "central":
        # Moyenne arithmétique des deux nœuds encadrant chaque face.
        face = 0.5 * (qm[:-1] + qm[1:])
    elif scheme == "upwind":
        # np.where(condition, A, B) choisit élément par élément : nœud de gauche (k) si la
        # vitesse est positive (l'information vient de la gauche), nœud de droite sinon.
        face = np.where(am > 0.0, qm[:-1], qm[1:])
    elif scheme == "quick":
        # np.diff : différences entre nœuds consécutifs, d_k = q_{k+1} - q_k (n-1 valeurs).
        d = np.diff(qm, axis=0)
        # Courbure discrète aux nœuds, nulle par défaut (nœuds extrêmes : stencil incomplet).
        curv = np.zeros_like(qm)
        # Nœuds intérieurs : q_{k-1} - 2 q_k + q_{k+1} = d_k - d_{k-1}.
        curv[1:-1] = d[1:] - d[:-1]
        # Partie centrée (ordre 2) de la valeur à la face.
        central = 0.5 * (qm[:-1] + qm[1:])
        # Courbure prise au nœud AMONT de chaque face (nœud k si a > 0, nœud k+1 sinon).
        upwind_curvature = np.where(am > 0.0, curv[:-1], curv[1:])
        # QUICK = centré - courbure amont / 8 (interpolation parabolique sur 3 nœuds amont).
        face = central - 0.125 * upwind_curvature
    elif scheme == "tvd":
        # Différences entre nœuds consécutifs.
        d = np.diff(qm, axis=0)
        # Produit des pentes gauche et droite de chaque nœud intérieur (> 0 si q est monotone).
        prod = d[:-1] * d[1:]
        # Demi-pente limitée aux nœuds (nulle aux extrémités et aux extrema locaux).
        half_slope = np.zeros_like(qm)
        # Limiteur de van Leer sous forme de moyenne harmonique : d_g d_d / (d_g + d_d) ;
        # np.divide(..., out=..., where=...) n'effectue la division que là où prod > 0, ce qui
        # évite les divisions par zéro et laisse 0 ailleurs (extremum => pas de pente).
        np.divide(prod, d[:-1] + d[1:], out=half_slope[1:-1], where=prod > 0.0)
        # Valeur à la face extrapolée depuis le nœud amont : q_k + h_k si a > 0, q_{k+1} - h_{k+1} sinon.
        face = np.where(am > 0.0, qm[:-1] + half_slope[:-1], qm[1:] - half_slope[1:])
    else:
        # Nom de schéma inconnu.
        raise ValueError(f"Schéma d'advection inconnu : {scheme!r} (choix : {ADVECTION_SCHEMES}).")
    # Remet l'axe de travail à sa place d'origine.
    return np.moveaxis(face, 0, axis)


def advective_fluxes(
    u: np.ndarray, v: np.ndarray, scheme: str = "quick"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flux convectifs aux faces des volumes de contrôle décalés.

    Retourne ``(Fux, Fuy, Fvx, Fvy)`` : flux de quantité de mouvement selon x
    (``u·u`` aux centres des cellules, forme ``(Nx, Ny)`` ; ``v·u`` aux coins des faces u
    intérieures, forme ``(Nx-1, Ny+1)``) et selon y (``u·v`` aux coins des faces v
    intérieures, forme ``(Nx+1, Ny-1)`` ; ``v·v`` aux centres, forme ``(Nx, Ny)``). La
    vitesse advectante est interpolée linéairement, la quantité transportée reconstruite
    par ``scheme``. Les flux ne dépendent que des tableaux : on peut les évaluer sur une
    sous-fenêtre (les deux rangées de bord y sont alors approchées).
    """
    # --- Volumes de contrôle de u (centrés sur les faces verticales) ---------------------
    # Lignes physiques de u (sans les lignes fantômes) : forme (Nx+1, Ny).
    uu = u[:, 1:-1]                                   # (Nx+1, Ny)
    # Vitesse advectante sur les faces est/ouest de ces volumes = centres des cellules :
    # moyenne des deux faces u voisines.
    a = 0.5 * (uu[:-1] + uu[1:])                      # u aux centres de cellules (Nx, Ny)
    # Flux de u transporté selon x : vitesse advectante x valeur reconstruite de u.
    fux = a * reconstruct(uu, a, 0, scheme)
    # Faces nord/sud des volumes de u = coins des cellules : moyenne des deux faces v voisines
    # (colonnes v d'indices i et i+1 pour les faces u intérieures i = 1..Nx-1).
    a = 0.5 * (v[1:-2, :] + v[2:-1, :])               # v aux coins (Nx-1, Ny+1)
    # Flux de u transporté selon y (lignes fantômes incluses pour la reconstruction).
    fuy = a * reconstruct(u[1:-1, :], a, 1, scheme)
    # --- Volumes de contrôle de v (centrés sur les faces horizontales) -------------------
    # Faces est/ouest des volumes de v = coins : moyenne des deux faces u voisines
    # (lignes u d'indices j et j+1 pour les faces v intérieures j = 1..Ny-1).
    a = 0.5 * (u[:, 1:-2] + u[:, 2:-1])               # u aux coins (Nx+1, Ny-1)
    # Flux de v transporté selon x (colonnes fantômes incluses).
    fvx = a * reconstruct(v[:, 1:-1], a, 0, scheme)
    # Colonnes physiques de v : forme (Nx, Ny+1).
    vv = v[1:-1, :]                                   # (Nx, Ny+1)
    # Faces nord/sud des volumes de v = centres des cellules.
    a = 0.5 * (vv[:, :-1] + vv[:, 1:])                # v aux centres de cellules (Nx, Ny)
    # Flux de v transporté selon y.
    fvy = a * reconstruct(vv, a, 1, scheme)
    return fux, fuy, fvx, fvy


def advection(
    u: np.ndarray, v: np.ndarray, grid: StaggeredGrid, scheme: str = "quick"
) -> tuple[np.ndarray, np.ndarray]:
    """Terme convectif conservatif ``∇·(u ⊗ u)`` aux faces intérieures.

    Retourne ``(Au, Av)`` de formes ``(Nx-1, Ny)`` (faces u ``i = 1..Nx-1``) et
    ``(Nx, Ny-1)`` (faces v ``j = 1..Ny-1``), divergences des flux de
    :func:`advective_fluxes`.
    """
    # Flux aux faces de tous les volumes de contrôle.
    fux, fuy, fvx, fvy = advective_fluxes(u, v, scheme)
    # Forme conservative : (flux sortant - flux entrant) / taille du volume, dans chaque
    # direction ; np.diff(..., axis=0) = flux est - flux ouest, axis=1 = nord - sud.
    adv_u = np.diff(fux, axis=0) / grid.dx + np.diff(fuy, axis=1) / grid.dy
    # Idem pour la quantité de mouvement selon y.
    adv_v = np.diff(fvx, axis=0) / grid.dx + np.diff(fvy, axis=1) / grid.dy
    return adv_u, adv_v


def laplacian_u(
    u: np.ndarray,
    grid: StaggeredGrid,
    wall_north: np.ndarray | None = None,
    wall_south: np.ndarray | None = None,
) -> np.ndarray:
    """Laplacien de ``u`` aux faces intérieures, forme ``(Nx-1, Ny)``.

    ``wall_north`` / ``wall_south`` (booléens de même forme) signalent un voisin enfoui
    dans un obstacle : il est remplacé par la valeur miroir ``-u_P`` afin d'imposer
    l'adhérence sur la paroi située à mi-distance.
    """
    # Valeur au point P : faces intérieures i = 1..Nx-1, lignes physiques.
    up = u[1:-1, 1:-1]
    # Voisins nord (ligne suivante) et sud (ligne précédente), lignes fantômes comprises.
    un, us = u[1:-1, 2:], u[1:-1, :-2]
    # Voisin nord enfoui dans un obstacle : valeur miroir -u_P (u = 0 à mi-distance = paroi).
    if wall_north is not None:
        un = np.where(wall_north, -up, un)
    # Idem pour le voisin sud.
    if wall_south is not None:
        us = np.where(wall_south, -up, us)
    # Dérivée seconde selon x : (u_E - 2 u_P + u_W) / dx² (voisins est u[2:], ouest u[:-2]).
    d2u_dx2 = (u[2:, 1:-1] - 2.0 * up + u[:-2, 1:-1]) / grid.dx**2
    # Dérivée seconde selon y : (u_N - 2 u_P + u_S) / dy².
    d2u_dy2 = (un - 2.0 * up + us) / grid.dy**2
    # Laplacien = somme des deux dérivées secondes (schéma centré à 5 points).
    return d2u_dx2 + d2u_dy2


def laplacian_v(
    v: np.ndarray,
    grid: StaggeredGrid,
    wall_east: np.ndarray | None = None,
    wall_west: np.ndarray | None = None,
) -> np.ndarray:
    """Laplacien de ``v`` aux faces intérieures, forme ``(Nx, Ny-1)`` (voir :func:`laplacian_u`)."""
    # Valeur au point P : colonnes physiques, faces intérieures j = 1..Ny-1.
    vp = v[1:-1, 1:-1]
    # Voisins est et ouest (colonnes fantômes comprises).
    ve, vw = v[2:, 1:-1], v[:-2, 1:-1]
    # Voisin est enfoui dans un obstacle : valeur miroir (adhérence).
    if wall_east is not None:
        ve = np.where(wall_east, -vp, ve)
    # Idem pour le voisin ouest.
    if wall_west is not None:
        vw = np.where(wall_west, -vp, vw)
    # Dérivée seconde selon x.
    d2v_dx2 = (ve - 2.0 * vp + vw) / grid.dx**2
    # Dérivée seconde selon y (voisins nord v[:, 2:] et sud v[:, :-2]).
    d2v_dy2 = (v[1:-1, 2:] - 2.0 * vp + v[1:-1, :-2]) / grid.dy**2
    return d2v_dx2 + d2v_dy2


def divergence(u: np.ndarray, v: np.ndarray, grid: StaggeredGrid) -> np.ndarray:
    """Divergence discrète ``∇·u`` au centre des cellules, forme ``(Nx, Ny)``."""
    # (u_est - u_ouest)/dx : faces i+1 et i de chaque cellule, lignes physiques.
    du_dx = (u[1:, 1:-1] - u[:-1, 1:-1]) / grid.dx
    # (v_nord - v_sud)/dy : faces j+1 et j de chaque cellule, colonnes physiques.
    dv_dy = (v[1:-1, 1:] - v[1:-1, :-1]) / grid.dy
    # Bilan de masse de chaque cellule (nul pour un écoulement incompressible).
    return du_dx + dv_dy


def cell_centered_velocity(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vitesses interpolées au centre des cellules, formes ``(Nx, Ny)``."""
    # u au centre = moyenne des faces ouest et est ; v au centre = moyenne des faces sud et nord.
    return 0.5 * (u[:-1, 1:-1] + u[1:, 1:-1]), 0.5 * (v[1:-1, :-1] + v[1:-1, 1:])

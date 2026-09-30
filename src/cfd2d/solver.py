"""Solveur Navier–Stokes 2D incompressible par projection (Chorin) sur grille MAC.

Équations résolues (ρ et ν constants) ::

    ∂u/∂t + ∇·(u ⊗ u) = -∇p/ρ + ν ∇²u,        ∇·u = 0

Un pas de temps (méthode à pas fractionnaires de Chorin) :

1. prédiction : ``u* = uⁿ + Δt · H(u)`` avec ``H = -∇·(u ⊗ u) + ν ∇²u`` intégré
   explicitement (Euler, Adams–Bashforth 2 ou Runge–Kutta SSP 3) ;
2. Poisson : ``∇²p = (ρ/Δt) ∇·u*`` sur les cellules fluides ;
3. correction : ``uⁿ⁺¹ = u* - (Δt/ρ) ∇p``, de divergence discrète nulle.

Les obstacles sont représentés par un masque binaire (frontière immergée en marches
d'escalier) : toute face touchant une cellule solide a une vitesse nulle (imperméabilité
et adhérence), la pression y vérifie une condition de Neumann, et l'adhérence
tangentielle est imposée dans les termes visqueux par des valeurs miroirs.
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# Journalisation (progression du calcul, avertissements).
import logging

# ``time.perf_counter`` : chronomètre haute résolution (mesure des performances).
import time

# ``Callable`` : type des fonctions de rappel (callbacks).
from collections.abc import Callable

# ``dataclass`` / ``field`` : conteneurs de l'état et des diagnostics.
from dataclasses import dataclass, field

import numpy as np

# Conditions aux limites : gestionnaire, et types d'entrée/sortie testés par isinstance.
from .boundary import BoundaryHandler, Inlet, Outlet

# Configuration complète de la simulation.
from .config import SimulationConfig

# Construction du masque binaire des obstacles.
from .geometry import build_mask

# Grille décalée.
from .grid import StaggeredGrid

# Opérateurs discrets (advection, vitesses aux centres, divergence, laplaciens).
from .operators import advection, cell_centered_velocity, divergence, laplacian_u, laplacian_v

# Opérateur de Poisson, bilan de résolution, fabrique de solveurs de pression.
from .pressure import PoissonOperator, SolveInfo, make_pressure_solver

# Journal du module.
logger = logging.getLogger(__name__)

# Type d'une fonction de rappel : reçoit le solveur, ne renvoie rien (le nom de la classe est
# écrit entre guillemets car elle n'est pas encore définie à cet endroit du fichier).
Callback = Callable[["NavierStokesSolver"], None]
# Spécification d'un champ initial : constante ou fonction f(x, y).
FieldSpec = float | Callable[[np.ndarray, np.ndarray], np.ndarray]

#: Nombre de Courant par défaut de chaque schéma temporel.
DEFAULT_CFL = {"euler": 0.2, "ab2": 0.4, "rk3": 0.8}
#: Nombre de Fourier 2D maximal ``ν Δt (1/Δx² + 1/Δy²)`` assurant la stabilité de la diffusion.
FOURIER_LIMIT = {"euler": 0.5, "ab2": 0.25, "rk3": 0.628}
# Coefficient de sécurité appliqué aux limites théoriques de stabilité.
_SAFETY = 0.9


class SimulationDivergedError(RuntimeError):
    """Levée lorsque la solution devient non finie (instabilité numérique)."""


@dataclass
class FlowState:
    """État instantané : vitesses aux faces (avec fantômes) et pression aux centres."""

    # Vitesse horizontale aux faces verticales, forme (Nx+1, Ny+2).
    u: np.ndarray
    # Vitesse verticale aux faces horizontales, forme (Nx+2, Ny+1).
    v: np.ndarray
    # Pression au centre des cellules, forme (Nx, Ny).
    p: np.ndarray
    # Temps physique courant.
    t: float = 0.0
    # Numéro du pas de temps.
    step: int = 0

    @classmethod
    def zeros(cls, grid: StaggeredGrid) -> FlowState:
        """État au repos (vitesses et pression nulles)."""
        # np.zeros(forme) : tableau rempli de 0.0 de la forme demandée.
        return cls(np.zeros(grid.shape_u), np.zeros(grid.shape_v), np.zeros(grid.shape_p))

    def copy(self) -> FlowState:
        """Copie indépendante de l'état (les tableaux sont dupliqués)."""
        return FlowState(self.u.copy(), self.v.copy(), self.p.copy(), self.t, self.step)


@dataclass
class Diagnostics:
    """Historique des indicateurs enregistrés à chaque pas de temps."""

    # Une liste par indicateur (field(default_factory=list) : liste neuve pour chaque objet).
    time: list[float] = field(default_factory=list)
    dt: list[float] = field(default_factory=list)
    cfl: list[float] = field(default_factory=list)
    divergence_max: list[float] = field(default_factory=list)
    kinetic_energy: list[float] = field(default_factory=list)
    pressure_iterations: list[int] = field(default_factory=list)
    pressure_residual: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        # len(diagnostics) = nombre de pas enregistrés.
        return len(self.time)

    def as_arrays(self) -> dict[str, np.ndarray]:
        # vars(self) : dictionnaire {nom d'attribut: valeur} ; chaque liste devient un tableau NumPy.
        return {name: np.asarray(values) for name, values in vars(self).items()}


class NavierStokesSolver:
    """Solveur incompressible 2D piloté par une :class:`~cfd2d.config.SimulationConfig`.

    Exemple ::

        solver = NavierStokesSolver(presets.cylinder_flow(Re=100))
        solver.add_callback(lambda s: print(s.state.t), every=100)
        solver.run(t_end=50.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        # Configuration complète.
        self.config = config
        # Raccourci vers la configuration du domaine.
        d = config.domain
        # Grille décalée (coordonnées, pas d'espace, formes des tableaux).
        self.grid = StaggeredGrid(d.Lx, d.Ly, d.Nx, d.Ny)
        # Copie de la liste des obstacles.
        self.obstacles = list(config.obstacles)
        # Masque binaire (Nx, Ny) : True = cellule solide.
        self.solid = build_mask(self.grid, self.obstacles, config.numerics.mask_supersampling)
        # Masque complémentaire : cellules fluides.
        self.fluid = ~self.solid
        # Masse volumique.
        self.rho = config.flow.rho
        # Vitesse de référence U_inf.
        self.U_ref = config.flow.U_inf
        # Longueur de référence (diamètre, corde, hauteur...).
        self.L_ref = config.reference_length
        # Viscosité cinématique effective (déduite de Re si nécessaire).
        self.nu = config.flow.viscosity(self.L_ref)
        # Gestionnaire des conditions aux limites (vérifie aussi leur cohérence).
        self.boundary = BoundaryHandler(self.grid, config.boundaries, self.U_ref)
        # Classement des faces (actives, figées, sorties) : voir _build_topology.
        self._build_topology()

        # Raccourci vers les options numériques.
        num = config.numerics
        # Début du chronométrage de la préparation de l'opérateur de pression.
        tic = time.perf_counter()
        # Assemblage de la matrice de Poisson (une seule fois : masque et CL sont fixes).
        self.poisson = PoissonOperator(
            self.grid, self.solid, self.u_active, self.v_active, self.u_dirichlet, self.v_dirichlet
        )
        # Création du solveur linéaire choisi (factorisation LU pour le solveur direct).
        self.pressure_solver = make_pressure_solver(
            num.pressure_solver,
            self.poisson,
            tol=num.pressure_tol,
            maxiter=num.pressure_maxiter,
            preconditioner=num.preconditioner,
            omega=num.sor_omega,
        )
        # Durée de préparation (message sans accents pour les consoles Windows).
        logger.info("Operateur de pression pret en %.2f s (%s).", time.perf_counter() - tic, self.pressure_solver.name)

        # État initial au repos.
        self.state = FlowState.zeros(self.grid)
        # Historique des indicateurs.
        self.diagnostics = Diagnostics()
        # Liste des rappels (fréquence, fonction).
        self._callbacks: list[tuple[int, Callback]] = []
        # Tendance du pas précédent (schéma Adams-Bashforth 2).
        self._rhs_prev: tuple[np.ndarray, np.ndarray] | None = None
        # Pas de temps précédent.
        self._dt_prev: float | None = None
        # Compteurs d'avertissements (pour ne pas inonder le journal).
        self._n_dt_reduced = 0
        self._n_unconverged = 0
        # Vitesses maximales imposées aux bords (plancher du critère CFL).
        self._u_bc_max, self._v_bc_max = self.boundary.max_speeds()
        # Champ initial (projeté pour être à divergence nulle).
        self.set_initial_velocity()

    # ================================================================== topologie
    def _build_topology(self) -> None:
        """Classe les faces : actives (inconnues), figées à 0 (solide), sorties (Dirichlet p)."""
        # Raccourcis : grille et masque solide.
        g, s = self.grid, self.solid
        # Conditions aux limites.
        bcs = self.config.boundaries

        # Faces u touchant au moins une cellule solide (vitesse figée à 0) ; forme (Nx+1, Ny).
        u_touch = np.zeros((g.Nx + 1, g.Ny), dtype=bool)  # face u touchant une cellule solide
        # Face intérieure i : cellules voisines i-1 (s[:-1]) et i (s[1:]) ; | = OU logique.
        u_touch[1:-1] = s[:-1] | s[1:]
        # Faces du bord : une seule cellule voisine.
        u_touch[0], u_touch[-1] = s[0], s[-1]
        # Idem pour les faces v, forme (Nx, Ny+1).
        v_touch = np.zeros((g.Nx, g.Ny + 1), dtype=bool)
        v_touch[:, 1:-1] = s[:, :-1] | s[:, 1:]
        v_touch[:, 0], v_touch[:, -1] = s[:, 0], s[:, -1]

        # Faces de sortie (pression de Dirichlet p = 0) : tableaux booléens initialement faux.
        u_dir = np.zeros_like(u_touch)
        v_dir = np.zeros_like(v_touch)
        # Sortie à l'ouest : toutes les faces du bord x = 0 adjacentes à une cellule fluide.
        if isinstance(bcs.west, Outlet):
            u_dir[0] = ~s[0]
        # Sortie à l'est.
        if isinstance(bcs.east, Outlet):
            u_dir[-1] = ~s[-1]
        # Sortie au sud.
        if isinstance(bcs.south, Outlet):
            v_dir[:, 0] = ~s[:, 0]
        # Sortie au nord.
        if isinstance(bcs.north, Outlet):
            v_dir[:, -1] = ~s[:, -1]

        # Faces actives (corrigées par la projection) : faces de sortie + faces intérieures
        # entre deux cellules fluides.
        u_act = u_dir.copy()
        u_act[1:-1] = ~u_touch[1:-1]
        v_act = v_dir.copy()
        v_act[:, 1:-1] = ~v_touch[:, 1:-1]

        # Attributs publics (utilisés par l'opérateur de Poisson et l'analyse).
        self.u_active, self.v_active = u_act, v_act
        self.u_dirichlet, self.v_dirichlet = u_dir, v_dir
        # Faces figées à zéro.
        self._u_fixed, self._v_fixed = u_touch, v_touch
        # Masques en flottants (0.0 / 1.0) : multiplier par ces masques annule la mise à jour
        # des faces figées sans instruction conditionnelle (plus rapide).
        self._u_interior = u_act[1:-1].astype(float)  # faces u intérieures mises à jour
        self._v_interior = v_act[:, 1:-1].astype(float)
        # Masques des faces de sortie de chaque côté.
        self._u_out_w, self._u_out_e = u_dir[0].astype(float), u_dir[-1].astype(float)
        self._v_out_s, self._v_out_n = v_dir[:, 0].astype(float), v_dir[:, -1].astype(float)

        # Adhérence tangentielle sur les obstacles : voisin « enfoui » (deux cellules solides).
        # None = pas d'obstacle : le laplacien standard s'applique partout.
        self._u_wall_n = self._u_wall_s = self._v_wall_e = self._v_wall_w = None
        if s.any():
            # Face u intérieure enfouie : ses deux cellules voisines sont solides (& = ET logique).
            u_buried = s[:-1] & s[1:]  # faces u intérieures (Nx-1, Ny)
            # Face v intérieure enfouie.
            v_buried = s[:, :-1] & s[:, 1:]  # faces v intérieures (Nx, Ny-1)
            # Pour chaque face u : le voisin NORD (ligne j+1) est-il enfoui ? (dernière ligne :
            # voisin = ligne fantôme gérée par les CL, donc False).
            self._u_wall_n = np.zeros_like(u_buried)
            self._u_wall_n[:, :-1] = u_buried[:, 1:]
            # Voisin SUD (ligne j-1) enfoui ?
            self._u_wall_s = np.zeros_like(u_buried)
            self._u_wall_s[:, 1:] = u_buried[:, :-1]
            # Pour chaque face v : voisin EST (colonne i+1) enfoui ?
            self._v_wall_e = np.zeros_like(v_buried)
            self._v_wall_e[:-1, :] = v_buried[1:, :]
            # Voisin OUEST (colonne i-1) enfoui ?
            self._v_wall_w = np.zeros_like(v_buried)
            self._v_wall_w[1:, :] = v_buried[:-1, :]

    def _zero_solid_faces(self, u: np.ndarray, v: np.ndarray) -> None:
        """Annule la vitesse sur toutes les faces touchant un solide (adhérence, imperméabilité)."""
        # u[:, 1:-1] est une vue ; l'indexation booléenne [masque] = 0 écrit dans u.
        u[:, 1:-1][self._u_fixed] = 0.0
        v[1:-1, :][self._v_fixed] = 0.0

    # ============================================================ propriétés utiles
    @property
    def reynolds(self) -> float:
        # Nombre de Reynolds effectif.
        return self.U_ref * self.L_ref / self.nu

    @property
    def time(self) -> float:
        # Temps physique courant.
        return self.state.t

    def cell_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Vitesses ``(u, v)`` au centre des cellules, formes ``(Nx, Ny)`` (nulles dans le solide)."""
        return cell_centered_velocity(self.state.u, self.state.v)

    def divergence(self) -> np.ndarray:
        """Divergence discrète du champ courant, forme ``(Nx, Ny)``."""
        return divergence(self.state.u, self.state.v, self.grid)

    def summary(self) -> str:
        """Résumé d'une ligne de la simulation (grille, Re, schémas)."""
        g, cfg = self.grid, self.config
        # f-strings : {valeur:.4g} = 4 chiffres significatifs ; self.solid.mean() = fraction
        # de cellules solides (moyenne d'un tableau booléen).
        return (
            f"{cfg.name} | grille {g.Nx}x{g.Ny} (dx={g.dx:.4g}, dy={g.dy:.4g}) | "
            f"Re={self.reynolds:.4g} (nu={self.nu:.4g}, L_ref={self.L_ref:.4g}) | "
            f"solide {100.0 * self.solid.mean():.2f} % | advection={cfg.numerics.advection}, "
            f"temps={cfg.time.scheme}, pression={self.pressure_solver.name}"
        )

    # ======================================================== condition initiale
    def set_initial_velocity(
        self, u0: FieldSpec | None = None, v0: FieldSpec = 0.0, perturbation: float | None = None
    ) -> None:
        """Initialise la vitesse puis la projette sur les champs à divergence nulle.

        ``u0``, ``v0`` : constantes ou fonctions ``f(x, y)``. Par défaut ``u0`` reproduit le
        profil de l'entrée ouest (écoulement établi) ou vaut 0 (cavité, domaine fermé).
        ``perturbation`` (défaut : ``flow.perturbation``) ajoute un mode transverse
        ``ε U sin(πx/Lx) sin(πy/Ly)`` qui brise la symétrie haut/bas.
        """
        g, st = self.grid, self.state
        # Champ u par défaut : profil d'entrée ou repos.
        if u0 is None:
            u0 = self._default_initial_u()
        # Coordonnées des nœuds physiques de u et de v.
        xu, yu = g.u_nodes()
        xv, yv = g.v_nodes()
        # callable(f) : vrai si f est une fonction ; sinon la constante est diffusée sur le tableau.
        st.u[:, 1:-1] = u0(xu, yu) if callable(u0) else u0
        st.v[1:-1, :] = v0(xv, yv) if callable(v0) else v0
        # Amplitude de la perturbation.
        eps = self.config.flow.perturbation if perturbation is None else perturbation
        if eps:
            # Mode sinueux (v de même signe au-dessus et au-dessous de l'axe) : brise la symétrie
            # et déclenche plus tôt le lâcher tourbillonnaire ; nul sur les bords du domaine.
            st.v[1:-1, :] += eps * self.U_ref * np.sin(np.pi * xv / g.Lx) * np.sin(np.pi * yv / g.Ly)

        # Conditions aux limites du champ initial.
        self.boundary.apply_normal(st.u, st.v, st.t)
        self.boundary.extrapolate_outlets(st.u, st.v)
        self._zero_solid_faces(st.u, st.v)
        # Projection : le champ devient exactement à divergence nulle (dt_eff arbitraire).
        self._project(st.u, st.v, dt_eff=1.0)
        # Cellules fantômes cohérentes avec le champ projeté.
        self.boundary.apply_ghosts(st.u, st.v)
        st.p[:] = 0.0  # le potentiel de la projection initiale n'est pas une pression
        # Réinitialisation de l'historique Adams-Bashforth.
        self._rhs_prev, self._dt_prev = None, None

    def restore_state(self, state: FlowState) -> None:
        """Remplace l'état courant (reprise d'un calcul sauvegardé).

        L'historique d'Adams-Bashforth est réinitialisé : le premier pas après la reprise
        est un pas d'Euler.
        """
        g = self.grid
        # Les tableaux doivent correspondre exactement à la grille du solveur.
        if state.u.shape != g.shape_u or state.v.shape != g.shape_v or state.p.shape != g.shape_p:
            raise ValueError("Etat incompatible avec la grille du solveur.")
        # Copie indépendante (l'appelant peut réutiliser ses tableaux).
        self.state = state.copy()
        # Cellules fantômes cohérentes avec les conditions aux limites courantes.
        self.boundary.apply_ghosts(self.state.u, self.state.v)
        # Pas de tendance précédente : redémarrage d'AB2.
        self._rhs_prev, self._dt_prev = None, None

    def _default_initial_u(self) -> FieldSpec:
        """u initial par défaut : profil de l'entrée ouest si elle existe, repos sinon."""
        west = self.config.boundaries.west
        # Pas d'entrée à l'ouest (cavité...) : fluide au repos.
        if not isinstance(west, Inlet):
            return 0.0
        # Vitesse débitante de l'entrée.
        U = self.boundary.inlet_velocity(west)
        # Fonction (x, y) -> profil d'entrée évalué à l'ordonnée y (identique pour tout x).
        return lambda x, y: west.profile_values(y, self.grid.Ly, U, self.state.t)

    # ============================================================== pas de temps
    def stable_dt(self) -> float:
        """Pas de temps maximal stable pour le champ courant.

        Critère combiné advection-diffusion ``1/Δt = C_adv/CFL + C_diff/Fo`` avec
        ``C_adv = |u|max/Δx + |v|max/Δy`` et ``C_diff = ν (1/Δx² + 1/Δy²)``.
        """
        g, st, tc = self.grid, self.state, self.config.time
        # Vitesses maximales (np.abs(...).max()) : lignes physiques, avec le plancher des vitesses
        # imposées aux bords (utile au démarrage d'une cavité au repos).
        umax = max(float(np.abs(st.u[:, 1:-1]).max()), self._u_bc_max)
        vmax = max(float(np.abs(st.v[1:-1, :]).max()), self._v_bc_max)
        # np.isfinite : faux pour NaN ou infini (calcul divergé).
        if not np.isfinite(umax + vmax):
            raise SimulationDivergedError(f"Vitesse non finie a t = {st.t:.6g} (pas {st.step}).")
        # Courant visé et Fourier maximal (valeurs utilisateur ou valeurs sûres du schéma).
        cfl = tc.cfl if tc.cfl is not None else DEFAULT_CFL[tc.scheme]
        fourier = tc.fourier if tc.fourier is not None else _SAFETY * FOURIER_LIMIT[tc.scheme]
        # Taux imposé par l'advection : (|u|/dx + |v|/dy) / CFL.
        advective_rate = (umax / g.dx + vmax / g.dy) / cfl
        # Taux imposé par la diffusion : nu (1/dx² + 1/dy²) / Fo.
        diffusive_rate = self.nu * (1.0 / g.dx**2 + 1.0 / g.dy**2) / fourier
        # Les deux contraintes se cumulent (critère combiné, plus sûr que le minimum).
        rate = advective_rate + diffusive_rate
        dt = 1.0 / rate
        # Euler explicite avec un schéma non décentré : stable seulement si dt < 2 nu / |u|².
        if tc.scheme == "euler" and self.config.numerics.advection != "upwind" and umax + vmax > 0:
            dt = min(dt, _SAFETY * 2.0 * self.nu / (umax**2 + vmax**2))  # Euler + schéma centré
        return dt

    def next_dt(self) -> float:
        """Pas de temps du prochain pas : ``dt`` imposé s'il est stable, sinon réduit."""
        # Limite de stabilité pour l'état courant.
        dt_stable = self.stable_dt()
        # Pas de temps demandé par l'utilisateur.
        dt_user = self.config.time.dt
        # Mode adaptatif.
        if dt_user is None:
            return dt_stable
        # dt imposé trop grand : réduction, avec un avertissement tous les 1000 cas.
        if dt_user > dt_stable:
            if self._n_dt_reduced % 1000 == 0:
                logger.warning(
                    "dt impose (%.3g) > dt stable (%.3g) a t = %.4g : dt reduit.", dt_user, dt_stable, self.state.t
                )
            self._n_dt_reduced += 1
            return dt_stable
        # dt imposé stable : utilisé tel quel (échantillonnage temporel régulier).
        return dt_user

    # ================================================================== physique
    def _rhs(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tendance explicite ``H = -∇·(u ⊗ u) + ν ∇²u`` sur les faces actives."""
        # Cellules fantômes à jour avant d'évaluer les stencils.
        self.boundary.apply_ghosts(u, v)
        # Terme convectif ∇·(u⊗u) aux faces intérieures.
        adv_u, adv_v = advection(u, v, self.grid, self.config.numerics.advection)
        # Laplaciens avec condition miroir au voisinage des obstacles.
        lap_u = laplacian_u(u, self.grid, self._u_wall_n, self._u_wall_s)
        lap_v = laplacian_v(v, self.grid, self._v_wall_e, self._v_wall_w)
        # Tendances aux dimensions des tableaux complets (zéro sur les faces non calculées).
        Fu = np.zeros_like(u)
        Fv = np.zeros_like(v)
        # H = nu ∇²u - ∇·(u⊗u) sur les faces intérieures, annulée sur les faces figées.
        Fu[1:-1, 1:-1] = (self.nu * lap_u - adv_u) * self._u_interior
        Fv[1:-1, 1:-1] = (self.nu * lap_v - adv_v) * self._v_interior
        # Sorties convectives : équation de transport de la vitesse de sortie.
        self.boundary.add_outlet_tendency(Fu, Fv, u, v)
        return Fu, Fv

    def _project(self, u: np.ndarray, v: np.ndarray, dt_eff: float) -> tuple[np.ndarray, SolveInfo]:
        """Rend ``(u, v)`` à divergence nulle (en place) et retourne la pression."""
        g = self.grid
        # Second membre b = -(rho/dt) ∇·u* de l'équation A p = b (A = -∇²).
        b = self.poisson.rhs(divergence(u, v, g), self.rho / dt_eff)
        # Estimation initiale (solveurs itératifs) : pression du pas précédent, même jauge.
        x0 = self.poisson.gauge(self.state.p)
        # Résolution du système linéaire.
        phi, info = self.pressure_solver.solve(b, x0)
        # Vecteur solution remis en forme (Nx, Ny).
        p = phi.reshape(g.shape_p)
        # Coefficient de correction dt/rho.
        c = dt_eff / self.rho
        # Correction des faces u intérieures : u -= (dt/rho) (p_droite - p_gauche)/dx,
        # multipliée par le masque des faces actives (les faces figées restent nulles).
        u[1:-1, 1:-1] -= c * (p[1:] - p[:-1]) / g.dx * self._u_interior
        # Correction des faces v intérieures.
        v[1:-1, 1:-1] -= c * (p[:, 1:] - p[:, :-1]) / g.dy * self._v_interior
        # Faces de sortie : p = 0 sur le bord, à une demi-maille du centre de la cellule.
        u[0, 1:-1] -= c * (p[0] - 0.0) / (0.5 * g.dx) * self._u_out_w
        u[-1, 1:-1] -= c * (0.0 - p[-1]) / (0.5 * g.dx) * self._u_out_e
        v[1:-1, 0] -= c * (p[:, 0] - 0.0) / (0.5 * g.dy) * self._v_out_s
        v[1:-1, -1] -= c * (0.0 - p[:, -1]) / (0.5 * g.dy) * self._v_out_n
        # Domaine fermé : pression définie à une constante près -> moyenne nulle sur le fluide.
        if self.poisson.singular:
            p = p - p[self.fluid].mean()
            p[self.solid] = 0.0
        return p, info

    def _finish_stage(self, u: np.ndarray, v: np.ndarray, t: float, dt_eff: float) -> tuple[np.ndarray, SolveInfo]:
        """Conditions aux limites du champ prédit, projection, cellules fantômes."""
        # Vitesses normales imposées (entrées, parois) à l'instant t.
        self.boundary.apply_normal(u, v, t)
        # Sorties de type Neumann : extrapolation de la vitesse prédite.
        self.boundary.extrapolate_outlets(u, v)
        # Faces des obstacles remises à zéro.
        self._zero_solid_faces(u, v)
        # Projection (u, v modifiés en place).
        p, info = self._project(u, v, dt_eff)
        # Fantômes cohérents avec le champ projeté.
        self.boundary.apply_ghosts(u, v)
        return p, info

    def _advance_multistep(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, SolveInfo]:
        """Euler explicite ou Adams–Bashforth 2 (pas variable), une projection par pas."""
        st = self.state
        # Tendance au temps n.
        Fu, Fv = self._rhs(st.u, st.v)
        # Tendance et pas du pas précédent (Adams-Bashforth).
        prev, dt_prev = self._rhs_prev, self._dt_prev
        # AB2 possible si l'historique existe et si le pas n'a pas plus que doublé (sinon les
        # coefficients deviennent grands et le schéma perd sa stabilité).
        if self.config.time.scheme == "ab2" and prev is not None and dt_prev is not None and dt <= 2.0 * dt_prev:
            # Coefficients d'Adams-Bashforth 2 à pas variable : (1 + w) Fⁿ - w Fⁿ⁻¹, w = dt/(2 dt_prev).
            w = 0.5 * dt / dt_prev
            Gu, Gv = (1.0 + w) * Fu - w * prev[0], (1.0 + w) * Fv - w * prev[1]
        else:  # Euler (ou démarrage d'AB2)
            Gu, Gv = Fu, Fv
        # Mémorisation de la tendance pour le pas suivant.
        self._rhs_prev = (Fu, Fv)
        # Prédiction u* = uⁿ + dt G (nouveaux tableaux : l'état courant n'est pas modifié).
        u, v = st.u + dt * Gu, st.v + dt * Gv
        # Conditions aux limites à t + dt, projection.
        p, info = self._finish_stage(u, v, st.t + dt, dt)
        return u, v, p, info

    def _advance_rk3(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, SolveInfo]:
        """Runge–Kutta SSP à 3 étages (Shu–Osher), projection à chaque étage."""
        st = self.state
        # État de départ uⁿ et temps tⁿ.
        u0, v0, t = st.u, st.v, st.t
        # Étage 1 : u⁽¹⁾ = P[uⁿ + dt H(uⁿ)] (P = projection), évalué à t + dt.
        Fu, Fv = self._rhs(u0, v0)
        u1, v1 = u0 + dt * Fu, v0 + dt * Fv
        self._finish_stage(u1, v1, t + dt, dt)
        # Étage 2 : u⁽²⁾ = P[¾ uⁿ + ¼ (u⁽¹⁾ + dt H(u⁽¹⁾))], évalué à t + dt/2 (poids dt/4).
        Fu, Fv = self._rhs(u1, v1)
        u2, v2 = 0.75 * u0 + 0.25 * (u1 + dt * Fu), 0.75 * v0 + 0.25 * (v1 + dt * Fv)
        self._finish_stage(u2, v2, t + 0.5 * dt, 0.25 * dt)
        # Étage 3 : uⁿ⁺¹ = P[⅓ uⁿ + ⅔ (u⁽²⁾ + dt H(u⁽²⁾))], évalué à t + dt (poids 2dt/3).
        Fu, Fv = self._rhs(u2, v2)
        u3, v3 = (u0 + 2.0 * (u2 + dt * Fu)) / 3.0, (v0 + 2.0 * (v2 + dt * Fv)) / 3.0
        # La pression retenue est celle du dernier étage.
        p, info = self._finish_stage(u3, v3, t + dt, 2.0 * dt / 3.0)
        return u3, v3, p, info

    # ============================================================ boucle en temps
    def step(self, dt: float | None = None) -> float:
        """Avance d'un pas de temps (``dt=None`` : :meth:`next_dt`) et retourne ``dt``."""
        # Pas de temps : calculé ou imposé par l'appelant.
        dt = self.next_dt() if dt is None else float(dt)
        # ``not dt > 0`` rejette aussi NaN.
        if not dt > 0:
            raise ValueError(f"Pas de temps invalide : {dt!r}.")
        # Avancement selon le schéma choisi.
        if self.config.time.scheme == "rk3":
            u, v, p, info = self._advance_rk3(dt)
        else:
            u, v, p, info = self._advance_multistep(dt)
        # Mise à jour de l'état.
        st = self.state
        st.u, st.v, st.p = u, v, p
        st.t += dt
        st.step += 1
        # Pas mémorisé pour Adams-Bashforth.
        self._dt_prev = dt
        # Solveur de pression itératif non convergé : avertissement (tous les 100 cas).
        if not info.converged:
            if self._n_unconverged % 100 == 0:
                logger.warning(
                    "Solveur de pression non converge au pas %d (residu %.2e) : augmenter pressure_maxiter.",
                    st.step, info.residual,
                )
            self._n_unconverged += 1
        # Enregistrement des diagnostics (et détection d'une divergence numérique).
        self._record(dt, info)
        # Appel des rappels dont la fréquence divise le numéro de pas (% = reste de la division).
        for every, callback in self._callbacks:
            if st.step % every == 0:
                callback(self)
        return dt

    def run(self, t_end: float | None = None, *, max_steps: int | None = None, log_every: int = 500) -> Diagnostics:
        """Intègre jusqu'à ``t_end`` (défaut : ``config.time.t_end``) ou ``max_steps`` pas."""
        # Instant final.
        t_end = self.config.time.t_end if t_end is None else float(t_end)
        # Pas et heure de départ (vitesse de calcul).
        start_step, tic = self.state.step, time.perf_counter()
        # Résumé de la simulation dans le journal.
        logger.info(self.summary())
        # Boucle jusqu'à t_end (petite tolérance relative pour les arrondis de somme des dt).
        while self.state.t < t_end - 1e-9 * max(1.0, abs(t_end)):
            # Arrêt anticipé après max_steps pas.
            if max_steps is not None and self.state.step - start_step >= max_steps:
                break
            # Dernier pas raccourci pour tomber exactement sur t_end.
            self.step(min(self.next_dt(), t_end - self.state.t))
            # Progression périodique dans le journal.
            if log_every and self.state.step % log_every == 0:
                self._log_progress(start_step, tic)
        # Bilan final.
        self._log_progress(start_step, tic)
        return self.diagnostics

    def add_callback(self, callback: Callback, every: int = 1) -> None:
        """Appelle ``callback(solver)`` tous les ``every`` pas (enregistrement, sondes, images...)."""
        if every < 1:
            raise ValueError("every doit être >= 1.")
        # Ajout du couple (fréquence, fonction).
        self._callbacks.append((int(every), callback))

    # =============================================================== diagnostics
    def _record(self, dt: float, info: SolveInfo) -> None:
        """Enregistre les indicateurs du pas qui vient d'être calculé."""
        g, st = self.grid, self.state
        # Vitesses maximales (lignes physiques).
        umax = float(np.abs(st.u[:, 1:-1]).max())
        vmax = float(np.abs(st.v[1:-1, :]).max())
        # Solution non finie : arrêt avec un message d'aide.
        if not np.isfinite(umax + vmax):
            raise SimulationDivergedError(
                f"Solution non finie a t = {st.t:.6g} (pas {st.step}) : reduire le CFL ou dt, "
                "raffiner la grille ou utiliser un schema d'advection plus dissipatif."
            )
        # Vitesses aux centres (énergie cinétique).
        uc, vc = cell_centered_velocity(st.u, st.v)
        d = self.diagnostics
        # Temps et pas de temps.
        d.time.append(st.t)
        d.dt.append(dt)
        # Nombre de Courant effectif du pas.
        d.cfl.append(dt * (umax / g.dx + vmax / g.dy))
        # Divergence maximale (contrôle de l'incompressibilité, ~1e-15 attendu).
        d.divergence_max.append(float(np.abs(divergence(st.u, st.v, g)).max()))
        # Énergie cinétique totale ½ rho Σ (u² + v²) dA.
        d.kinetic_energy.append(0.5 * self.rho * float(np.sum(uc * uc + vc * vc)) * g.cell_area)
        # Coût et qualité de la résolution de pression.
        d.pressure_iterations.append(info.iterations)
        d.pressure_residual.append(info.residual)

    def _log_progress(self, start_step: int, tic: float) -> None:
        """Écrit une ligne de progression dans le journal."""
        d = self.diagnostics
        # Rien à afficher avant le premier pas.
        if not len(d):
            return
        # Temps écoulé depuis le début de run().
        elapsed = time.perf_counter() - tic
        # Vitesse de calcul en pas par seconde.
        rate = (self.state.step - start_step) / elapsed if elapsed > 0 else float("nan")
        # Formatage : %7d entier sur 7 caractères, %.3e notation scientifique, etc.
        logger.info(
            "pas %7d | t = %9.4f | dt = %.3e | CFL = %.3f | max|div| = %.2e | Ec = %.5g | %.0f pas/s",
            self.state.step, d.time[-1], d.dt[-1], d.cfl[-1], d.divergence_max[-1], d.kinetic_energy[-1], rate,
        )

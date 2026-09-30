"""Configuration d'une simulation : domaine, fluide, temps, schémas numériques."""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``dataclass`` : classes de configuration ; ``field(default_factory=...)`` crée une valeur par
# défaut NOUVELLE pour chaque instance (indispensable pour les objets mutables comme les listes).
from dataclasses import dataclass, field

# ``Literal`` : type restreint à des chaînes précises ; ``get_args`` extrait ces chaînes
# (ex. get_args(Literal["a", "b"]) == ("a", "b")) pour valider les choix de l'utilisateur.
from typing import Literal, get_args

# Conditions aux limites des quatre côtés.
from .boundary import BoundaryConfig

# Classe de base des obstacles (annotation de la liste d'obstacles).
from .geometry import Obstacle

# Schémas d'advection disponibles (voir operators.reconstruct).
AdvectionScheme = Literal["upwind", "central", "quick", "tvd"]
# Schémas d'intégration en temps (Euler explicite, Adams-Bashforth 2, Runge-Kutta SSP 3).
TimeScheme = Literal["euler", "ab2", "rk3"]
# Solveurs de l'équation de Poisson.
PressureMethod = Literal["direct", "cg", "bicgstab", "sor", "jacobi"]
# Préconditionneurs des méthodes de Krylov (cg, bicgstab).
Preconditioner = Literal["none", "jacobi", "ilu", "amg"]


def _check_choice(value: str, choices_type: object, name: str) -> None:
    """Lève ValueError si ``value`` n'est pas l'une des chaînes autorisées par le Literal."""
    # Tuple des valeurs autorisées.
    choices = get_args(choices_type)
    # Valeur inconnue : message listant les choix possibles (", ".join assemble les chaînes).
    if value not in choices:
        raise ValueError(f"{name} = {value!r} invalide ; choix possibles : {', '.join(choices)}.")


def _check_positive(value: float | None, name: str) -> None:
    """Lève ValueError si ``value`` est fourni (non None) et n'est pas strictement positif."""
    # ``not value > 0`` rejette aussi NaN (toute comparaison avec NaN est fausse).
    if value is not None and not value > 0:
        raise ValueError(f"{name} doit être strictement positif (reçu {value!r}).")


@dataclass
class DomainConfig:
    """Domaine rectangulaire ``[0, Lx] x [0, Ly]`` discrétisé en ``Nx x Ny`` cellules."""

    # Dimensions physiques du domaine.
    Lx: float = 1.0
    Ly: float = 1.0
    # Nombre de cellules dans chaque direction.
    Nx: int = 64
    Ny: int = 64


@dataclass
class FlowConfig:
    """Propriétés du fluide et de l'écoulement de référence.

    Spécifier soit ``nu`` (viscosité cinématique), soit ``Re`` ; dans ce second cas
    ``nu = U_inf · L_ref / Re`` où ``L_ref`` est la longueur de référence (celle du
    premier obstacle, ou ``Ly`` sans obstacle, sauf si ``L_ref`` est fourni).
    ``perturbation`` est l'amplitude relative (en fraction de ``U_inf``) d'une
    perturbation transverse initiale qui accélère le déclenchement des instabilités
    (lâcher tourbillonnaire) dans une configuration symétrique.
    """

    # Vitesse de référence (vitesse d'entrée par défaut).
    U_inf: float = 1.0
    # Masse volumique (constante : écoulement incompressible).
    rho: float = 1.0
    # Viscosité cinématique (exclusif avec Re).
    nu: float | None = None
    # Nombre de Reynolds U_inf L_ref / nu (exclusif avec nu).
    Re: float | None = None
    # Longueur de référence imposée (sinon déduite des obstacles ou du domaine).
    L_ref: float | None = None
    # Amplitude de la perturbation initiale (0 = aucune).
    perturbation: float = 0.0

    def __post_init__(self) -> None:
        # (nu is None) == (Re is None) est vrai si les deux sont fournis ou si aucun ne l'est.
        if (self.nu is None) == (self.Re is None):
            raise ValueError("Spécifier exactement un paramètre parmi 'nu' et 'Re'.")
        # Toutes les grandeurs physiques fournies doivent être positives ; getattr lit
        # l'attribut dont le nom est donné par la chaîne.
        for name in ("U_inf", "rho", "nu", "Re", "L_ref"):
            _check_positive(getattr(self, name), name)

    def viscosity(self, L_ref: float) -> float:
        """Viscosité cinématique effective."""
        # Viscosité donnée explicitement.
        if self.nu is not None:
            return self.nu
        # Sinon Re est forcément défini (garanti par __post_init__) ; assert sert au typage.
        assert self.Re is not None
        # nu = U L / Re.
        return self.U_inf * L_ref / self.Re


@dataclass
class TimeConfig:
    """Intégration temporelle.

    * ``dt=None`` : pas de temps adaptatif, recalculé à chaque itération à partir des
      critères de stabilité (advection : nombre de Courant ``cfl`` ; diffusion : nombre de
      Fourier ``fourier``) ;
    * ``dt`` fixé : utilisé tel quel tant qu'il respecte ces critères, réduit sinon
      (avec avertissement).

    ``cfl=None`` et ``fourier=None`` prennent des valeurs sûres propres au schéma.
    """

    # Instant final de la simulation.
    t_end: float = 10.0
    # Pas de temps imposé (None = adaptatif).
    dt: float | None = None
    # Nombre de Courant visé (None = valeur par défaut du schéma).
    cfl: float | None = None
    # Nombre de Fourier maximal (None = valeur sûre du schéma).
    fourier: float | None = None
    # Schéma d'intégration temporelle.
    scheme: TimeScheme = "ab2"

    def __post_init__(self) -> None:
        # Le schéma doit être l'un des trois proposés.
        _check_choice(self.scheme, TimeScheme, "scheme")
        # Grandeurs temporelles positives.
        for name in ("t_end", "dt", "cfl", "fourier"):
            _check_positive(getattr(self, name), name)


@dataclass
class NumericsConfig:
    """Choix numériques : schéma d'advection, solveur de pression, masque."""

    # Schéma de reconstruction des flux convectifs.
    advection: AdvectionScheme = "quick"
    # Méthode de résolution de l'équation de Poisson.
    pressure_solver: PressureMethod = "direct"
    # Tolérance relative des solveurs itératifs.
    pressure_tol: float = 1e-8
    # Nombre maximal d'itérations des solveurs itératifs (10_000 : séparateur de milliers).
    pressure_maxiter: int = 10_000
    # Préconditionneur des méthodes de Krylov.
    preconditioner: Preconditioner = "ilu"
    # Paramètre de relaxation du SOR (None = valeur optimale estimée).
    sor_omega: float | None = None
    # Sous-échantillonnage des cellules pour la construction du masque (4 = 4x4 points).
    mask_supersampling: int = 4

    def __post_init__(self) -> None:
        # Validation des choix textuels.
        _check_choice(self.advection, AdvectionScheme, "advection")
        _check_choice(self.pressure_solver, PressureMethod, "pressure_solver")
        _check_choice(self.preconditioner, Preconditioner, "preconditioner")
        # Tolérance strictement positive.
        _check_positive(self.pressure_tol, "pressure_tol")
        # SOR ne converge que pour 0 < omega < 2.
        if self.sor_omega is not None and not 0.0 < self.sor_omega < 2.0:
            raise ValueError("sor_omega doit appartenir à ]0, 2[.")
        # Au moins un point par cellule.
        if self.mask_supersampling < 1:
            raise ValueError("mask_supersampling doit être >= 1.")


@dataclass
class SimulationConfig:
    """Configuration complète d'une simulation."""

    # default_factory : chaque SimulationConfig reçoit ses propres objets (pas de partage).
    domain: DomainConfig = field(default_factory=DomainConfig)
    # lambda: FlowConfig(Re=100.0) : fonction sans argument créant la configuration par défaut.
    flow: FlowConfig = field(default_factory=lambda: FlowConfig(Re=100.0))
    boundaries: BoundaryConfig = field(default_factory=BoundaryConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    # Liste des obstacles immergés (vide = domaine sans obstacle).
    obstacles: list[Obstacle] = field(default_factory=list)
    # Nom de la simulation (journaux, fichiers de sortie).
    name: str = "simulation"

    def __post_init__(self) -> None:
        # Vérifie la cohérence des conditions aux limites dès la création.
        self.boundaries.validate()

    @property
    def reference_length(self) -> float:
        """Longueur de référence : ``flow.L_ref``, sinon premier obstacle, sinon ``Ly``."""
        # Priorité 1 : longueur imposée explicitement.
        if self.flow.L_ref is not None:
            return self.flow.L_ref
        # Priorité 2 : longueur caractéristique du premier obstacle (diamètre, corde...).
        if self.obstacles:
            return self.obstacles[0].reference_length
        # Priorité 3 : hauteur du domaine (canal, cavité).
        return self.domain.Ly

    @property
    def nu(self) -> float:
        # Viscosité effective (calculée depuis Re si besoin).
        return self.flow.viscosity(self.reference_length)

    @property
    def reynolds(self) -> float:
        # Nombre de Reynolds effectif Re = U L / nu.
        return self.flow.U_inf * self.reference_length / self.nu

"""Conditions aux limites sur les quatre côtés du domaine rectangulaire.

Sur un côté, la composante *normale* de la vitesse est portée par les faces du bord
(``u`` à l'ouest/est, ``v`` au sud/nord) ; la composante *tangentielle* est imposée via
la rangée de cellules fantômes adjacente (valeur miroir). La pression vérifie une
condition de Neumann homogène partout, sauf aux sorties où ``p = 0`` (Dirichlet).
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``Iterator`` : type des générateurs (fonctions utilisant ``yield``).
from collections.abc import Iterator

# ``dataclass`` : génération automatique du constructeur des conditions aux limites.
from dataclasses import dataclass

# ``Enum`` : énumération de valeurs nommées (les quatre côtés du domaine).
from enum import Enum

# ``Literal`` : type restreint à quelques chaînes précises (ex. "uniform" | "parabolic").
from typing import Literal

import numpy as np

# Grille décalée (coordonnées des faces, pas d'espace).
from .grid import StaggeredGrid


# Hériter de ``str`` permet de comparer un membre à sa chaîne : Side.WEST == "west" est vrai.
class Side(str, Enum):
    """Côtés du domaine."""

    # Côté x = 0.
    WEST = "west"
    # Côté x = Lx.
    EAST = "east"
    # Côté y = 0.
    SOUTH = "south"
    # Côté y = Ly.
    NORTH = "north"

    @property
    def inward_sign(self) -> float:
        """+1 si la normale entrante est orientée selon +x ou +y (ouest, sud), -1 sinon."""
        # Une vitesse entrante vaut +U à l'ouest et au sud, -U à l'est et au nord.
        return 1.0 if self in (Side.WEST, Side.SOUTH) else -1.0

    @property
    def is_vertical(self) -> bool:
        """True pour les côtés ouest et est (normale selon x)."""
        # Les côtés ouest et est sont des segments verticaux (normale horizontale).
        return self in (Side.WEST, Side.EAST)


# Classe de base vide : sert à reconnaître toutes les conditions (isinstance) ;
# frozen=True rend les conditions immuables et hachables.
@dataclass(frozen=True)
class BoundaryCondition:
    """Classe de base des conditions aux limites."""


@dataclass(frozen=True)
class NoSlipWall(BoundaryCondition):
    """Paroi adhérente : vitesse normale nulle, vitesse tangentielle imposée.

    ``velocity`` est la vitesse tangentielle de la paroi (selon +x pour les côtés sud et
    nord, +y pour ouest et est) ; non nulle pour une cavité entraînée.
    """

    # Vitesse de glissement imposée de la paroi (0 = paroi fixe).
    velocity: float = 0.0


@dataclass(frozen=True)
class SlipWall(BoundaryCondition):
    """Paroi glissante (ou plan de symétrie) : vitesse normale nulle, cisaillement nul."""


@dataclass(frozen=True)
class Inlet(BoundaryCondition):
    """Entrée à vitesse normale imposée, orientée vers l'intérieur du domaine.

    ``profile='uniform'`` impose ``U`` ; ``profile='parabolic'`` impose un profil de
    Poiseuille de même débit (maximum ``1.5 U``). ``velocity=None`` utilise ``U_inf``.
    ``ramp_time > 0`` fait croître la vitesse de 0 à ``U`` selon une loi en ``1 - cos``.
    """

    # Vitesse débitante (None -> U_inf de la configuration de l'écoulement).
    velocity: float | None = None
    # Forme du profil d'entrée.
    profile: Literal["uniform", "parabolic"] = "uniform"
    # Durée de la montée progressive en vitesse (0 = démarrage impulsif).
    ramp_time: float = 0.0

    def __post_init__(self) -> None:
        # Seuls deux profils sont implémentés.
        if self.profile not in ("uniform", "parabolic"):
            raise ValueError(f"Profil d'entrée inconnu : {self.profile!r}.")
        # Une durée négative n'a pas de sens.
        if self.ramp_time < 0:
            raise ValueError("ramp_time doit être positif ou nul.")

    def ramp(self, t: float) -> float:
        """Facteur de montée en vitesse dans [0, 1]."""
        # Pas de rampe, ou rampe terminée : pleine vitesse.
        if self.ramp_time <= 0.0 or t >= self.ramp_time:
            return 1.0
        # Loi en ½ (1 - cos(π t / T)) : dérivée nulle au départ et à l'arrivée (démarrage doux).
        # max(t, 0) protège contre un temps négatif.
        return 0.5 * (1.0 - np.cos(np.pi * max(t, 0.0) / self.ramp_time))

    def profile_values(self, s: np.ndarray, length: float, U: float, t: float) -> np.ndarray:
        """Vitesse entrante aux abscisses ``s`` (le long du côté, de longueur ``length``)."""
        # Vitesse débitante à l'instant t (rampe comprise).
        scale = U * self.ramp(t)
        # Profil uniforme : np.full crée un tableau de la forme de s rempli de la valeur scale.
        if self.profile == "uniform":
            return np.full(np.shape(s), scale, dtype=float)
        # Coordonnée réduite η = s / L dans [0, 1].
        eta = np.asarray(s) / length
        # Profil de Poiseuille u = 6 U η (1 - η) : débit identique au profil uniforme
        # (moyenne de 6η(1-η) sur [0, 1] = 1) et maximum 1.5 U au centre.
        return 6.0 * scale * eta * (1.0 - eta)

    def peak_factor(self) -> float:
        """Rapport vitesse maximale / vitesse débitante du profil."""
        # 1.5 pour la parabole, 1 pour le profil uniforme (utilisé par le critère CFL).
        return 1.5 if self.profile == "parabolic" else 1.0


@dataclass(frozen=True)
class Outlet(BoundaryCondition):
    """Sortie libre : pression nulle (Dirichlet) et vitesse extrapolée.

    * ``kind='neumann'`` : gradient normal nul ``∂u/∂n = 0`` ;
    * ``kind='convective'`` : ``∂u/∂t + U_c ∂u/∂n = 0``, qui laisse sortir les tourbillons
      avec moins de réflexions (``convective_velocity=None`` utilise ``U_inf``).

    Dans les deux cas la vitesse normale de sortie est ensuite corrigée par la
    projection, ce qui garantit la conservation globale de la masse.
    """

    # Type d'extrapolation de la vitesse de sortie.
    kind: Literal["neumann", "convective"] = "neumann"
    # Vitesse de convection U_c de la condition convective (None -> U_inf).
    convective_velocity: float | None = None

    def __post_init__(self) -> None:
        # Validation du type de sortie.
        if self.kind not in ("neumann", "convective"):
            raise ValueError(f"Type de sortie inconnu : {self.kind!r}.")


@dataclass(frozen=True)
class BoundaryConfig:
    """Conditions aux limites des quatre côtés (par défaut : soufflerie à parois glissantes)."""

    # Valeurs par défaut : entrée à l'ouest, sortie à l'est, parois glissantes en haut et en bas.
    # (Les instances frozen étant immuables, les partager comme valeurs par défaut est sûr.)
    west: BoundaryCondition = Inlet()
    east: BoundaryCondition = Outlet()
    south: BoundaryCondition = SlipWall()
    north: BoundaryCondition = SlipWall()

    def items(self) -> Iterator[tuple[Side, BoundaryCondition]]:
        """Itère sur les couples (côté, condition) dans l'ordre ouest, est, sud, nord."""
        # Itérer sur une Enum parcourt ses membres dans l'ordre de déclaration.
        for side in Side:
            # getattr(self, "west") lit l'attribut dont le nom est la valeur du membre ;
            # ``yield`` fait de la méthode un générateur (valeurs produites à la demande).
            yield side, getattr(self, side.value)

    def __getitem__(self, side: Side) -> BoundaryCondition:
        # Permet config[Side.WEST] ou config["west"] (Side(...) convertit la chaîne en membre).
        return getattr(self, Side(side).value)

    @property
    def has_outlet(self) -> bool:
        # any(...) : vrai si au moins un côté est une sortie.
        return any(isinstance(bc, Outlet) for _, bc in self.items())

    def validate(self) -> None:
        """Vérifie la cohérence de la configuration (lève une exception sinon)."""
        # Chaque côté doit porter un objet condition aux limites.
        for side, bc in self.items():
            if not isinstance(bc, BoundaryCondition):
                raise TypeError(f"Condition invalide sur le côté {side.value} : {bc!r}.")
        # Un débit entrant sans sortie est incompatible avec l'incompressibilité (∇·u = 0).
        if any(isinstance(bc, Inlet) for _, bc in self.items()) and not self.has_outlet:
            raise ValueError("Une entrée (Inlet) nécessite au moins une sortie (Outlet).")

    # Constructeur alternatif : BoundaryConfig.channel(...).
    @classmethod
    def channel(
        cls,
        walls: Literal["slip", "no-slip"] = "slip",
        profile: Literal["uniform", "parabolic"] = "uniform",
        outlet: Literal["neumann", "convective"] = "neumann",
        ramp_time: float = 0.0,
    ) -> BoundaryConfig:
        """Canal / soufflerie : entrée à l'ouest, sortie à l'est, parois au sud et au nord."""
        # Seuls deux types de parois sont proposés.
        if walls not in ("slip", "no-slip"):
            raise ValueError("walls doit valoir 'slip' ou 'no-slip'.")
        # Paroi glissante (soufflerie idéale) ou adhérente (canal réel).
        wall = SlipWall() if walls == "slip" else NoSlipWall()
        # cls(...) appelle le constructeur de la classe (BoundaryConfig ou une sous-classe).
        return cls(
            west=Inlet(profile=profile, ramp_time=ramp_time),
            east=Outlet(kind=outlet),
            south=wall,
            north=wall,
        )

    @classmethod
    def lid_driven_cavity(cls, lid_velocity: float = 1.0) -> BoundaryConfig:
        """Cavité fermée entraînée par la paroi nord."""
        # Quatre parois adhérentes ; seule la paroi nord glisse à la vitesse lid_velocity.
        return cls(
            west=NoSlipWall(),
            east=NoSlipWall(),
            south=NoSlipWall(),
            north=NoSlipWall(velocity=lid_velocity),
        )


class BoundaryHandler:
    """Applique une :class:`BoundaryConfig` aux tableaux ``u``, ``v`` (avec fantômes)."""

    def __init__(self, grid: StaggeredGrid, config: BoundaryConfig, U_ref: float) -> None:
        # Contrôle de cohérence dès la construction.
        config.validate()
        # Grille (coordonnées des faces, pas d'espace).
        self.grid = grid
        # Conditions des quatre côtés.
        self.config = config
        # Vitesse de référence (valeur par défaut des entrées et des sorties convectives).
        self.U_ref = U_ref

    # staticmethod : méthode sans accès à self (fonction utilitaire rangée dans la classe).
    @staticmethod
    def views(
        side: Side, u: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vues ``(normale au bord, normale intérieure voisine, fantôme tangentiel, tangentielle intérieure)``."""
        # Le découpage NumPy de base (a[0, 1:-1]) renvoie des VUES : écrire dedans (vue[...] = x)
        # modifie directement u ou v. Les rangées 1:-1 excluent les lignes fantômes de u.
        if side is Side.WEST:
            # u sur la face x = 0, u sur la première face intérieure, colonne fantôme de v
            # (x = -dx/2) et première colonne intérieure de v (x = +dx/2).
            return u[0, 1:-1], u[1, 1:-1], v[0, :], v[1, :]
        if side is Side.EAST:
            # Symétrique à l'est : indices négatifs = en partant de la fin.
            return u[-1, 1:-1], u[-2, 1:-1], v[-1, :], v[-2, :]
        if side is Side.SOUTH:
            # v sur la face y = 0, v intérieur, ligne fantôme de u (y = -dy/2), ligne intérieure.
            return v[1:-1, 0], v[1:-1, 1], u[:, 0], u[:, 1]
        # NORTH : symétrique au sud.
        return v[1:-1, -1], v[1:-1, -2], u[:, -1], u[:, -2]

    def _along(self, side: Side) -> tuple[np.ndarray, float, float]:
        """(abscisses des faces normales le long du côté, longueur du côté, pas normal)."""
        # Raccourci vers la grille.
        g = self.grid
        # Côté vertical : les faces u sont à mi-hauteur des cellules (y_c), longueur Ly, pas dx.
        if side.is_vertical:
            return g.y_c, g.Ly, g.dx
        # Côté horizontal : faces v à mi-largeur (x_c), longueur Lx, pas normal dy.
        return g.x_c, g.Lx, g.dy

    def inlet_velocity(self, bc: Inlet) -> float:
        # Vitesse explicite de l'entrée, sinon vitesse de référence U_inf.
        return self.U_ref if bc.velocity is None else bc.velocity

    def apply_normal(self, u: np.ndarray, v: np.ndarray, t: float) -> None:
        """Impose les vitesses normales des entrées et des parois à l'instant ``t``."""
        # Parcours des quatre côtés.
        for side, bc in self.config.items():
            # [0] : vue sur la composante normale portée par les faces du bord.
            normal = self.views(side, u, v)[0]
            if isinstance(bc, Inlet):
                # Abscisses des faces le long du côté et longueur du côté.
                s, length, _ = self._along(side)
                # normal[...] = ... écrit dans la vue (donc dans u ou v) ; le signe oriente la
                # vitesse vers l'intérieur du domaine.
                normal[...] = side.inward_sign * bc.profile_values(s, length, self.inlet_velocity(bc), t)
            elif isinstance(bc, (NoSlipWall, SlipWall)):
                # Paroi imperméable : vitesse normale nulle.
                normal[...] = 0.0
            # Sortie : la vitesse normale est une inconnue (prédite puis projetée).

    def apply_ghosts(self, u: np.ndarray, v: np.ndarray) -> None:
        """Remplit les cellules fantômes tangentielles à partir des valeurs intérieures."""
        for side, bc in self.config.items():
            # Vues sur la rangée fantôme et la première rangée intérieure (composante tangentielle).
            _, _, ghost, inner = self.views(side, u, v)
            if isinstance(bc, NoSlipWall):
                # La paroi est à mi-distance entre fantôme et intérieur : (ghost + inner)/2 = V_paroi
                # donc ghost = 2 V_paroi - inner (valeur miroir).
                ghost[...] = 2.0 * bc.velocity - inner  # valeur bc.velocity sur la paroi
            elif isinstance(bc, Inlet):
                # Vitesse tangentielle nulle à l'entrée : miroir antisymétrique.
                ghost[...] = -inner  # vitesse tangentielle nulle à l'entrée
            else:  # paroi glissante ou sortie : dérivée normale nulle
                # Copie symétrique : ∂u_t/∂n = 0 (pas de cisaillement).
                ghost[...] = inner

    def apply(self, u: np.ndarray, v: np.ndarray, t: float) -> None:
        """Vitesses normales imposées puis cellules fantômes."""
        # L'ordre compte : les fantômes utilisent des valeurs intérieures déjà à jour.
        self.apply_normal(u, v, t)
        self.apply_ghosts(u, v)

    def extrapolate_outlets(self, u: np.ndarray, v: np.ndarray) -> None:
        """Sorties 'neumann' : ``∂u_n/∂n = 0`` appliqué au champ prédit."""
        for side, bc in self.config.items():
            # Seules les sorties de type Neumann sont concernées.
            if isinstance(bc, Outlet) and bc.kind == "neumann":
                # Vues sur la face de sortie et la face intérieure voisine.
                normal, inner, _, _ = self.views(side, u, v)
                # Gradient normal nul : la face de sortie prend la valeur de sa voisine.
                normal[...] = inner

    def add_outlet_tendency(self, Fu: np.ndarray, Fv: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
        """Sorties 'convective' : écrit ``-U_c ∂u_n/∂n`` dans la tendance des faces de sortie."""
        for side, bc in self.config.items():
            # Seules les sorties convectives sont concernées.
            if isinstance(bc, Outlet) and bc.kind == "convective":
                # Vitesse de convection (explicite ou U_inf).
                Uc = self.U_ref if bc.convective_velocity is None else bc.convective_velocity
                # Vitesse normale sur la face de sortie et sur la face intérieure voisine.
                normal, inner, _, _ = self.views(side, u, v)
                # Distance entre ces deux faces (pas normal au côté).
                h = self._along(side)[2]
                # ∂u/∂t = -U_c (u_bord - u_intérieur)/h, écrit dans la vue de la tendance Fu/Fv
                # (décentrement amont : l'information sort du domaine).
                self.views(side, Fu, Fv)[0][...] = -Uc * (normal - inner) / h

    def max_speeds(self) -> tuple[float, float]:
        """Vitesses maximales imposées aux bords (|u|, |v|), pour le critère CFL."""
        # Maxima initiaux.
        umax = vmax = 0.0
        for side, bc in self.config.items():
            if isinstance(bc, Inlet):
                # Vitesse maximale du profil d'entrée (1.5 U pour une parabole).
                speed = abs(self.inlet_velocity(bc)) * bc.peak_factor()
                # Entrée verticale -> composante u ; horizontale -> composante v.
                if side.is_vertical:
                    umax = max(umax, speed)
                else:
                    vmax = max(vmax, speed)
            elif isinstance(bc, NoSlipWall):
                # Paroi mobile : sa vitesse tangentielle entre dans le critère.
                if side.is_vertical:
                    vmax = max(vmax, abs(bc.velocity))
                else:
                    umax = max(umax, abs(bc.velocity))
        return umax, vmax

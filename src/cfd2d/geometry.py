"""Obstacles immergés et construction du masque binaire solide.

Chaque obstacle fournit une fonction d'appartenance ``contains(x, y)``. Le masque
``M(i, j) ∈ {0, 1}`` (1 = solide) s'obtient en sur-échantillonnant chaque cellule :
elle est déclarée solide si au moins la moitié de sa surface est dans l'obstacle.
"""

# Annotations de type évaluées paresseusement (permet ``np.ndarray | None`` partout).
from __future__ import annotations

# Journalisation (messages d'avertissement : obstacle non résolu, poches fluides comblées...).
import logging

# ``ABC`` / ``abstractmethod`` : définition d'une classe de base abstraite (interface commune
# que chaque forme d'obstacle doit implémenter).
from abc import ABC, abstractmethod

# Types génériques pour les annotations : fonction appelable, objet itérable.
from collections.abc import Callable, Iterable

# ``dataclass`` génère le constructeur ; ``field`` règle un attribut (ici : exclu du __init__).
from dataclasses import dataclass, field

# ``Path`` : manipulation portable des chemins de fichiers (images).
from pathlib import Path

import numpy as np

# ``matplotlib.image.imread`` : lecture d'images (PNG natif, JPEG... via Pillow).
from matplotlib import image as mpimg

# ``matplotlib.path.Path`` : polygone avec test rapide « point dans le polygone » (écrit en C++).
from matplotlib.path import Path as MplPath

# ``scipy.ndimage`` : traitement d'images (ici étiquetage des zones fluides connexes).
from scipy import ndimage

# Import relatif : la grille décalée définie dans grid.py du même paquet.
from .grid import StaggeredGrid

# Journal propre au module (nom "cfd2d.geometry"), configurable par l'utilisateur.
logger = logging.getLogger(__name__)

# Alias de type : boîte englobante (xmin, xmax, ymin, ymax).
Bounds = tuple[float, float, float, float]
# Alias de type : fonction (x, y) -> tableau booléen d'appartenance au solide.
ContainsFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class Obstacle(ABC):
    """Corps solide immergé dans l'écoulement."""

    # Méthode abstraite : chaque sous-classe DOIT la définir, sinon elle n'est pas instanciable.
    @abstractmethod
    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tableau booléen de même forme que ``x`` et ``y`` : ``True`` dans le solide."""

    # Propriété abstraite : longueur caractéristique utilisée pour adimensionner.
    @property
    @abstractmethod
    def reference_length(self) -> float:
        """Longueur de référence pour Re, Cd, Cl et Sr (diamètre, hauteur frontale, corde)."""

    # Propriété abstraite : point de référence (origine des stations de sillage x/D, angle θ).
    @property
    @abstractmethod
    def center(self) -> tuple[float, float]:
        """Point de référence de l'obstacle (centre géométrique)."""

    # Propriété abstraite : boîte englobante (utile pour l'arrière de l'obstacle, les tests rapides).
    @property
    @abstractmethod
    def bounds(self) -> Bounds:
        """Boîte englobante ``(xmin, xmax, ymin, ymax)``."""

    def outline(self, n: int = 256) -> np.ndarray | None:
        """Contour fermé ``(m, 2)`` parcouru dans le sens trigonométrique (None si inconnu)."""
        # Comportement par défaut : contour inconnu ; les sous-classes le redéfinissent.
        return None

    def mask(self, grid: StaggeredGrid, supersample: int = 1) -> np.ndarray:
        """Masque booléen ``(Nx, Ny)`` des cellules solides."""
        # Délègue à la fonction générique ``rasterize`` en lui passant la méthode ``contains``.
        return rasterize(self.contains, grid, supersample)


def rasterize(contains: ContainsFn, grid: StaggeredGrid, supersample: int = 1) -> np.ndarray:
    """Discrétise une fonction d'appartenance sur la grille.

    Chaque cellule est échantillonnée en ``supersample²`` points ; elle est solide si la
    fraction de points intérieurs est au moins 1/2 (``supersample=1`` : test au centre).
    """
    # Conversion en entier (accepte 4.0 par exemple).
    s = int(supersample)
    # Au moins un point d'échantillonnage par cellule et par direction.
    if s < 1:
        raise ValueError("supersample doit être un entier >= 1.")
    # Positions relatives des sous-points dans une cellule, en fraction de maille :
    # pour s = 2 -> [0.25, 0.75] ; pour s = 1 -> [0.5] (centre de la cellule).
    frac = (np.arange(s) + 0.5) / s
    # np.arange(Nx)[:, None] est une colonne (Nx, 1) ; + frac (s,) donne par diffusion
    # (broadcasting) une matrice (Nx, s) des positions ; .ravel() l'aplatit en (Nx*s,),
    # dans l'ordre cellule par cellule ; * dx convertit en coordonnées physiques.
    xs = (np.arange(grid.Nx)[:, None] + frac).ravel() * grid.dx
    # Même construction selon y : (Ny*s,) ordonnées des sous-points.
    ys = (np.arange(grid.Ny)[:, None] + frac).ravel() * grid.dy
    # Grille 2D de tous les sous-points, forme (Nx*s, Ny*s), indexation [x, y].
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    # Test d'appartenance vectorisé ; np.asarray(..., dtype=bool) garantit un tableau booléen ;
    # .reshape(Nx, s, Ny, s) regroupe les sous-points de chaque cellule (axes 1 et 3).
    inside = np.asarray(contains(X, Y), dtype=bool).reshape(grid.Nx, s, grid.Ny, s)
    # .mean(axis=(1, 3)) : fraction de sous-points solides par cellule ; seuil à 1/2.
    return inside.mean(axis=(1, 3)) >= 0.5


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotation de ``angle`` radians (sens trigonométrique) de points ``(n, 2)`` autour de l'origine."""
    # Cosinus et sinus de l'angle de rotation.
    c, s = np.cos(angle), np.sin(angle)
    # Les points sont des lignes (x, y) : le produit matriciel ``@`` par la transposée de la
    # matrice de rotation [[c, -s], [s, c]] donne x' = x c - y s et y' = x s + y c.
    return points @ np.array([[c, s], [-s, c]])


def _signed_area(vertices: np.ndarray) -> float:
    """Aire algébrique d'un polygone (formule du lacet) : > 0 si parcouru dans le sens trigo."""
    # Colonnes des abscisses et des ordonnées des sommets.
    x, y = vertices[:, 0], vertices[:, 1]
    # np.roll(y, -1) décale le tableau d'un cran (sommet suivant, avec bouclage) ;
    # np.dot calcule la somme des produits : A = ½ Σ (x_k y_{k+1} - x_{k+1} y_k).
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


# ============================================================================ formes
# @dataclass génère __init__(self, xc, yc, diameter) et __repr__ à partir des annotations.
@dataclass
class Cylinder(Obstacle):
    """Cylindre circulaire de diamètre ``diameter`` centré en ``(xc, yc)``."""

    # Abscisse du centre.
    xc: float
    # Ordonnée du centre.
    yc: float
    # Diamètre du cylindre.
    diameter: float

    def __post_init__(self) -> None:
        # Validation : un diamètre nul ou négatif n'a pas de sens.
        if self.diameter <= 0:
            raise ValueError("Le diamètre du cylindre doit être strictement positif.")

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Rayon du cylindre.
        r = 0.5 * self.diameter
        # Test (x - xc)² + (y - yc)² <= r², vectorisé sur tous les points ; np.asarray accepte
        # aussi des scalaires ou des listes.
        return (np.asarray(x) - self.xc) ** 2 + (np.asarray(y) - self.yc) ** 2 <= r * r

    @property
    def reference_length(self) -> float:
        # Le diamètre est la longueur de référence usuelle d'un cylindre (Re = U D / nu).
        return self.diameter

    @property
    def center(self) -> tuple[float, float]:
        # Centre du cercle.
        return (self.xc, self.yc)

    @property
    def bounds(self) -> Bounds:
        # Rayon.
        r = 0.5 * self.diameter
        # Carré circonscrit au cercle.
        return (self.xc - r, self.xc + r, self.yc - r, self.yc + r)

    def outline(self, n: int = 256) -> np.ndarray:
        # n angles régulièrement répartis sur [0, 2π[ (endpoint=False évite de doubler 0 et 2π).
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        # Rayon.
        r = 0.5 * self.diameter
        # np.column_stack assemble les colonnes x et y en un tableau (n, 2) de points du cercle.
        return np.column_stack([self.xc + r * np.cos(theta), self.yc + r * np.sin(theta)])


# eq=False : pas de comparaison automatique (les tableaux NumPy ne se comparent pas avec ==).
@dataclass(eq=False)
class PolygonObstacle(Obstacle):
    """Obstacle polygonal quelconque défini par ses sommets ``(n, 2)``.

    ``ref_length`` : longueur de référence ; par défaut la hauteur frontale (étendue en y).
    """

    # Sommets du polygone, tableau (n, 2).
    vertices: np.ndarray
    # Longueur de référence imposée (None -> hauteur frontale).
    ref_length: float | None = None
    # Objet matplotlib servant au test « point dans le polygone » ; init=False : calculé dans
    # __post_init__, pas passé au constructeur ; repr=False : non affiché.
    _path: MplPath = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Conversion en tableau de flottants (accepte listes, tuples...).
        verts = np.asarray(self.vertices, dtype=float)
        # Il faut un tableau 2D à 2 colonnes et au moins 3 sommets.
        if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
            raise ValueError("Un polygone nécessite au moins 3 sommets de forme (n, 2).")
        # Si le dernier sommet répète le premier (polygone « fermé »), on le retire.
        if np.allclose(verts[0], verts[-1]):
            verts = verts[:-1]
        # Aire négative = sens horaire : on inverse l'ordre ([::-1]) pour le sens trigonométrique.
        if _signed_area(verts) < 0:
            verts = verts[::-1]
        # Stockage des sommets normalisés.
        self.vertices = verts
        # MplPath exige un polygone explicitement refermé : np.vstack ajoute le premier sommet
        # à la fin ; closed=True indique à matplotlib qu'il s'agit d'un contour fermé.
        self._path = MplPath(np.vstack([verts, verts[:1]]), closed=True)

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # np.broadcast_arrays donne à x et y une forme commune (ex. scalaire + tableau).
        xb, yb = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        # Boîte englobante du polygone.
        xmin, xmax, ymin, ymax = self.bounds
        # Résultat initialisé à False (fluide) partout.
        inside = np.zeros(xb.shape, dtype=bool)
        # Pré-filtre rapide : seuls les points dans la boîte englobante peuvent être intérieurs.
        candidates = (xb >= xmin) & (xb <= xmax) & (yb >= ymin) & (yb <= ymax)
        # .any() : au moins un candidat ? (évite un appel inutile pour un domaine sans recouvrement).
        if candidates.any():
            # Points candidats rangés en tableau (m, 2) attendu par matplotlib.
            points = np.column_stack([xb[candidates], yb[candidates]])
            # contains_points : test point-dans-polygone vectorisé ; on écrit le résultat aux
            # positions des candidats grâce à l'indexation booléenne.
            inside[candidates] = self._path.contains_points(points)
        # Tableau booléen de même forme que les entrées.
        return inside

    @property
    def reference_length(self) -> float:
        # Longueur imposée par l'utilisateur, si fournie.
        if self.ref_length is not None:
            return self.ref_length
        # Sinon : np.ptp (« peak to peak ») = max - min des ordonnées = hauteur frontale.
        return float(np.ptp(self.vertices[:, 1]))

    @property
    def center(self) -> tuple[float, float]:
        """Centre de gravité de la surface du polygone."""
        # Coordonnées des sommets.
        x, y = self.vertices[:, 0], self.vertices[:, 1]
        # Produit vectoriel de sommets consécutifs : x_k y_{k+1} - x_{k+1} y_k.
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        # Aire algébrique du polygone.
        area = 0.5 * cross.sum()
        # Formule du centre de gravité : cx = Σ (x_k + x_{k+1}) cross_k / (6 A).
        cx = float(((x + np.roll(x, -1)) * cross).sum() / (6.0 * area))
        # Idem pour cy.
        cy = float(((y + np.roll(y, -1)) * cross).sum() / (6.0 * area))
        return (cx, cy)

    @property
    def bounds(self) -> Bounds:
        # .min(axis=0) / .max(axis=0) : minimum et maximum de chaque colonne (x puis y).
        (xmin, ymin), (xmax, ymax) = self.vertices.min(axis=0), self.vertices.max(axis=0)
        # Conversion en flottants Python.
        return (float(xmin), float(xmax), float(ymin), float(ymax))

    @property
    def area(self) -> float:
        # Aire (positive, les sommets étant rangés dans le sens trigonométrique).
        return _signed_area(self.vertices)

    def outline(self, n: int = 256) -> np.ndarray:
        # Le contour d'un polygone est la liste de ses sommets (copie pour éviter toute
        # modification accidentelle de l'objet ; n est ignoré).
        return self.vertices.copy()


class _PolygonShape(Obstacle):
    """Base des formes décrites par un polygone interne ``self._polygon``."""

    # Annotation seulement (pas un champ de dataclass) : le polygone est créé par la sous-classe.
    _polygon: PolygonObstacle

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Délégation au polygone interne.
        return self._polygon.contains(x, y)

    @property
    def center(self) -> tuple[float, float]:
        # Centre de gravité du polygone.
        return self._polygon.center

    @property
    def bounds(self) -> Bounds:
        # Boîte englobante du polygone.
        return self._polygon.bounds

    @property
    def area(self) -> float:
        # Aire du polygone.
        return self._polygon.area

    def outline(self, n: int = 256) -> np.ndarray:
        # Sommets du polygone.
        return self._polygon.outline(n)


@dataclass
class Rectangle(_PolygonShape):
    """Rectangle ``width x height`` centré en ``(xc, yc)``, tourné de ``angle_deg`` (sens trigo).

    La longueur de référence est la hauteur frontale (projetée normalement à l'axe x).
    """

    # Centre du rectangle.
    xc: float
    yc: float
    # Largeur (selon x avant rotation) et hauteur (selon y avant rotation).
    width: float
    height: float
    # Angle de rotation en degrés, sens trigonométrique.
    angle_deg: float = 0.0

    def __post_init__(self) -> None:
        # Dimensions strictement positives.
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Les dimensions du rectangle doivent être strictement positives.")
        # Demi-largeur et demi-hauteur.
        hw, hh = 0.5 * self.width, 0.5 * self.height
        # Les 4 coins autour de l'origine, dans le sens trigonométrique.
        corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        # np.radians convertit les degrés en radians ; rotation des coins puis translation au
        # centre (l'ajout de la liste [xc, yc] est diffusé sur chaque ligne).
        vertices = _rotate(corners, np.radians(self.angle_deg)) + [self.xc, self.yc]
        # Polygone interne utilisé par les méthodes héritées de _PolygonShape.
        self._polygon = PolygonObstacle(vertices)

    # classmethod : constructeur alternatif appelé sur la classe (Rectangle.square(...)).
    @classmethod
    def square(cls, xc: float, yc: float, side: float, angle_deg: float = 0.0) -> Rectangle:
        """Carré de côté ``side``."""
        # Un carré est un rectangle de largeur = hauteur = side.
        return cls(xc, yc, side, side, angle_deg)

    @property
    def reference_length(self) -> float:
        # Angle en radians.
        a = np.radians(self.angle_deg)
        # Hauteur projetée sur l'axe y (perpendiculaire à l'écoulement selon x) :
        # |w sin a| + |h cos a| ; vaut h sans rotation et la diagonale-projection sinon.
        return float(abs(self.width * np.sin(a)) + abs(self.height * np.cos(a)))

    @property
    def center(self) -> tuple[float, float]:
        # Centre exact (égal au centre de gravité, mais sans erreur d'arrondi).
        return (self.xc, self.yc)

    @property
    def aspect_ratio(self) -> float:
        """Allongement : grande dimension / petite dimension (1 pour un carré)."""
        return max(self.width, self.height) / min(self.width, self.height)

    def _chord_ends(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Milieux des deux petits côtés, de l'amont vers l'aval : extrémités de la « corde »."""
        # Direction de la grande dimension (axe de la largeur, tourné de angle_deg, ou axe de la
        # hauteur s'il est plus long) et demi-longueur correspondante.
        a = np.radians(self.angle_deg)
        if self.width >= self.height:
            direction, half = np.array([np.cos(a), np.sin(a)]), 0.5 * self.width
        else:
            direction, half = np.array([-np.sin(a), np.cos(a)]), 0.5 * self.height
        # Les deux extrémités, rangées par abscisse croissante (amont d'abord pour un écoulement selon +x).
        center = np.array([self.xc, self.yc])
        ends = sorted((center - half * direction, center + half * direction), key=lambda p: p[0])
        return (float(ends[0][0]), float(ends[0][1])), (float(ends[1][0]), float(ends[1][1]))

    @property
    def leading_edge(self) -> tuple[float, float]:
        """Bord d'attaque d'une plaque : milieu du petit côté amont."""
        return self._chord_ends()[0]

    @property
    def trailing_edge(self) -> tuple[float, float]:
        """Bord de fuite d'une plaque : milieu du petit côté aval."""
        return self._chord_ends()[1]


def naca4_coordinates(code: str, n: int = 200, closed_te: bool = True) -> np.ndarray:
    """Contour (sens trigonométrique) d'un profil NACA 4 chiffres de corde unité.

    Répartition en cosinus de ``n`` points par face ; le contour part du bord de fuite,
    longe l'extrados jusqu'au bord d'attaque puis revient par l'intrados.
    """
    # Le code doit comporter exactement 4 chiffres (ex. "2412").
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Code NACA 4 chiffres invalide : {code!r}.")
    # 1er chiffre : cambrure maximale m (% de corde) ; 2e : position de la cambrure p (dixièmes
    # de corde) ; 2 derniers : épaisseur relative maximale t (% de corde).
    m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0
    # Un profil d'épaisseur nulle n'a pas d'aire (ligne) : refusé.
    if t <= 0:
        raise ValueError("L'épaisseur relative du profil doit être non nulle.")
    # Répartition en cosinus des abscisses : np.linspace(0, π, n) puis x = (1 - cos β)/2 ;
    # les points se resserrent aux bords d'attaque et de fuite (fortes courbures).
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    # Dernier coefficient de la loi d'épaisseur : -0.1036 ferme le bord de fuite (épaisseur
    # nulle en x = 1), -0.1015 est la définition originale (bord de fuite épais).
    a4 = -0.1036 if closed_te else -0.1015
    # Demi-épaisseur NACA : yt = 5 t (0.2969 √x - 0.1260 x - 0.3516 x² + 0.2843 x³ + a4 x⁴).
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4)
    # Ligne moyenne (cambrure) yc et sa pente dyc, nulles pour un profil symétrique (00xx).
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    # Profil cambré : deux arcs de parabole de part et d'autre de x = p.
    if m > 0 and p > 0:
        # Masque booléen des points en avant de la cambrure maximale.
        front = x < p
        # Complément : points en arrière.
        back = ~front
        # Ligne moyenne avant : yc = m/p² (2 p x - x²).
        yc[front] = m / p**2 * (2 * p * x[front] - x[front] ** 2)
        # Pente avant : dyc/dx = 2 m / p² (p - x).
        dyc[front] = 2 * m / p**2 * (p - x[front])
        # Ligne moyenne arrière : yc = m/(1-p)² ((1 - 2p) + 2 p x - x²).
        yc[back] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[back] - x[back] ** 2)
        # Pente arrière : dyc/dx = 2 m/(1-p)² (p - x).
        dyc[back] = 2 * m / (1 - p) ** 2 * (p - x[back])
    # Angle local de la ligne moyenne : l'épaisseur est portée perpendiculairement à celle-ci.
    theta = np.arctan(dyc)
    # Extrados (upper) : point de la ligne moyenne décalé de yt le long de la normale.
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    # Intrados (lower) : décalage opposé.
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)
    # Contour : extrados parcouru à l'envers ([::-1], du bord de fuite au bord d'attaque) puis
    # intrados sans son premier point (xl[1:], le bord d'attaque déjà présent) ;
    # np.concatenate enchaîne les tableaux, np.column_stack forme les couples (x, y).
    return np.column_stack([np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])])


@dataclass
class NACA4(_PolygonShape):
    """Profil d'aile NACA 4 chiffres.

    Le bord d'attaque est placé en ``(x_le, y_le)`` avant rotation ; le profil est
    ensuite incliné de ``alpha_deg`` (incidence positive = cabré, bord d'attaque vers
    le haut pour un écoulement selon +x) autour du point situé à ``pivot`` x corde.
    """

    # Code du profil (4 chiffres).
    code: str = "0012"
    # Corde (longueur du profil).
    chord: float = 1.0
    # Position du bord d'attaque avant rotation.
    x_le: float = 0.0
    y_le: float = 0.0
    # Angle d'incidence en degrés (positif = nez vers le haut).
    alpha_deg: float = 0.0
    # Point de rotation, en fraction de corde (0.25 = quart de corde, usuel en aérodynamique).
    pivot: float = 0.25
    # Nombre de points par face du contour.
    n_points: int = 200
    # Bord de fuite fermé (épaisseur nulle) ou non.
    closed_te: bool = True

    def __post_init__(self) -> None:
        # Validation de la corde.
        if self.chord <= 0:
            raise ValueError("La corde doit être strictement positive.")
        # Contour du profil de corde unité, mis à l'échelle de la corde réelle.
        pts = naca4_coordinates(self.code, self.n_points, self.closed_te) * self.chord
        # Point de rotation dans le repère du profil (sur la corde, à pivot x corde).
        pivot = np.array([self.pivot * self.chord, 0.0])
        # Rotation autour du pivot d'un angle -alpha (sens horaire => bord d'attaque vers le
        # haut pour alpha > 0), puis translation du bord d'attaque en (x_le, y_le).
        pts = _rotate(pts - pivot, -np.radians(self.alpha_deg)) + pivot + [self.x_le, self.y_le]
        # Polygone interne (tests d'appartenance, contour, centre, aire).
        self._polygon = PolygonObstacle(pts)

    @property
    def reference_length(self) -> float:
        # Les coefficients d'un profil sont rapportés à la corde.
        return self.chord

    def _transform(self, xi: float) -> tuple[float, float]:
        """Position du point de corde ``xi`` (fraction de corde) après rotation."""
        # Vecteur pivot -> point de corde, dans le repère du profil (ligne (1, 2) pour _rotate).
        d = np.array([[(xi - self.pivot) * self.chord, 0.0]])
        # Même rotation que le contour ; [0] extrait l'unique point.
        x, y = _rotate(d, -np.radians(self.alpha_deg))[0]
        # Retour dans le repère du domaine (translation du pivot).
        return (float(x + self.x_le + self.pivot * self.chord), float(y + self.y_le))

    @property
    def pivot_point(self) -> tuple[float, float]:
        """Centre de rotation (quart de corde par défaut), centre du moment de tangage."""
        # Le pivot est invariant par la rotation.
        return self._transform(self.pivot)

    @property
    def leading_edge(self) -> tuple[float, float]:
        # Bord d'attaque : point de corde xi = 0.
        return self._transform(0.0)

    @property
    def trailing_edge(self) -> tuple[float, float]:
        # Bord de fuite : point de corde xi = 1.
        return self._transform(1.0)


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convertit une image (niveaux de gris, RGB ou RGBA) en niveaux de gris dans [0, 1]."""
    # Conversion en tableau NumPy (sans copie si c'en est déjà un).
    arr = np.asarray(img)
    # Image entière (ex. uint8 0..255) : np.iinfo(dtype).max donne la valeur maximale du type,
    # on normalise dans [0, 1].
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr / np.iinfo(arr.dtype).max
    # Masque booléen : True -> 1.0, False -> 0.0.
    elif arr.dtype == bool:
        arr = arr.astype(float)
    # Garantit des flottants pour les calculs suivants.
    arr = arr.astype(float)
    # Image déjà en niveaux de gris (tableau 2D) : rien d'autre à faire.
    if arr.ndim == 2:
        return arr
    # Sinon il faut 3 (RGB) ou 4 (RGBA) canaux sur le dernier axe.
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("Image attendue en niveaux de gris, RGB ou RGBA.")
    # Luminance (norme ITU-R BT.601) : produit matriciel des canaux R, G, B par les poids.
    gray = arr[..., :3] @ np.array([0.299, 0.587, 0.114])
    # Image RGBA : les zones transparentes sont composées sur un fond blanc (= fluide).
    if arr.shape[2] == 4:  # zones transparentes = fond blanc (fluide)
        # Canal alpha (opacité, 0 = transparent).
        alpha = arr[..., 3]
        # Mélange : gris x alpha + blanc x (1 - alpha).
        gray = alpha * gray + (1.0 - alpha)
    return gray


@dataclass(eq=False)
class ImageObstacle(Obstacle):
    """Obstacle défini par une image noir et blanc (pixels sombres = solide).

    ``image`` est un chemin de fichier (PNG, JPEG...) ou un tableau 2D/3D. L'image est
    centrée en ``(xc, yc)`` avec une largeur physique ``width`` (la hauteur suit le rapport
    d'aspect). Un pixel est solide si son niveau de gris est inférieur à ``threshold``
    (supérieur si ``invert=True``). ``ref_length`` vaut par défaut la hauteur frontale.
    """

    # Chemin du fichier image ou tableau de pixels.
    image: str | Path | np.ndarray
    # Position du centre de l'image dans le domaine.
    xc: float
    yc: float
    # Largeur physique de l'image.
    width: float
    # Seuil de niveau de gris séparant solide et fluide.
    threshold: float = 0.5
    # Inversion de la convention (pixels clairs = solide).
    invert: bool = False
    # Longueur de référence imposée (None -> hauteur frontale).
    ref_length: float | None = None
    # Attributs internes calculés dans __post_init__ (exclus du constructeur et de repr) :
    # masque des pixels solides, taille physique d'un pixel, coin inférieur gauche de l'image.
    _solid_px: np.ndarray = field(init=False, repr=False)
    _pixel: float = field(init=False, repr=False)
    _origin: tuple[float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Tableau fourni directement, sinon lecture du fichier avec matplotlib (mpimg.imread).
        raw = self.image if isinstance(self.image, np.ndarray) else mpimg.imread(Path(self.image))
        # Conversion en niveaux de gris dans [0, 1].
        gray = _to_grayscale(raw)
        # Seuillage : pixels sombres (ou clairs si invert) = solide.
        self._solid_px = gray > self.threshold if self.invert else gray < self.threshold
        # Une image sans pixel solide est probablement une erreur de seuil.
        if not self._solid_px.any():
            raise ValueError("L'image ne contient aucun pixel solide (vérifier threshold/invert).")
        # Nombre de lignes (hauteur) et de colonnes (largeur) de l'image, en pixels.
        n_rows, n_cols = self._solid_px.shape
        # Taille physique d'un pixel carré.
        self._pixel = self.width / n_cols
        # Hauteur physique (rapport d'aspect conservé).
        height = n_rows * self._pixel
        # Coin inférieur gauche de l'image dans le domaine.
        self._origin = (self.xc - 0.5 * self.width, self.yc - 0.5 * height)

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Formes compatibles pour x et y.
        xb, yb = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        # Dimensions de l'image en pixels.
        n_rows, n_cols = self._solid_px.shape
        # Numéro de colonne du pixel contenant chaque point (np.floor puis conversion entière).
        col = np.floor((xb - self._origin[0]) / self._pixel).astype(np.int64)
        # Numéro de ligne compté depuis le BAS de l'image (axe y physique vers le haut).
        row = np.floor((yb - self._origin[1]) / self._pixel).astype(np.int64)  # depuis le bas
        # Points situés dans l'emprise de l'image.
        valid = (col >= 0) & (col < n_cols) & (row >= 0) & (row < n_rows)
        # Par défaut : fluide.
        inside = np.zeros(xb.shape, dtype=bool)
        # Dans une image, la ligne 0 est en HAUT : ligne image = n_rows - 1 - ligne depuis le bas.
        inside[valid] = self._solid_px[n_rows - 1 - row[valid], col[valid]]
        return inside

    @property
    def bounds(self) -> Bounds:
        # np.nonzero : indices (lignes, colonnes) de tous les pixels solides.
        rows, cols = np.nonzero(self._solid_px)
        # Nombre de lignes de l'image.
        n_rows = self._solid_px.shape[0]
        # Coin inférieur gauche.
        x0, y0 = self._origin
        # Boîte englobante des pixels solides, convertie en coordonnées physiques (les lignes
        # sont retournées car la ligne 0 de l'image est en haut).
        return (
            x0 + cols.min() * self._pixel,
            x0 + (cols.max() + 1) * self._pixel,
            y0 + (n_rows - 1 - rows.max()) * self._pixel,
            y0 + (n_rows - rows.min()) * self._pixel,
        )

    @property
    def reference_length(self) -> float:
        # Longueur imposée par l'utilisateur, si fournie.
        if self.ref_length is not None:
            return self.ref_length
        # Sinon hauteur frontale : étendue verticale de la partie solide.
        _, _, ymin, ymax = self.bounds
        return ymax - ymin

    @property
    def center(self) -> tuple[float, float]:
        # Centre de la boîte englobante des pixels solides.
        xmin, xmax, ymin, ymax = self.bounds
        return (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))

    def outline(self, n: int = 256) -> np.ndarray | None:
        """Plus grand contour fermé des pixels solides (isovaleur 1/2), sens trigonométrique."""
        # contourpy : moteur de tracé de lignes de niveau utilisé par matplotlib (import local
        # pour ne le charger que si nécessaire).
        import contourpy

        # np.pad(..., 1) ajoute une bordure de pixels fluides (0) autour de l'image : les contours
        # touchant le bord de l'image sont ainsi fermés.
        padded = np.pad(self._solid_px.astype(float), 1)
        # Générateur de contours ; line_type="Separate" : une liste de tableaux (m, 2), un par ligne.
        generator = contourpy.contour_generator(z=padded, line_type="Separate")
        # Lignes de niveau 0.5 (frontière entre pixels 0 et 1), en coordonnées (colonne, ligne).
        lines = generator.lines(0.5)
        # On écarte les lignes dégénérées (moins de 4 points).
        closed = [line for line in lines if len(line) > 3]
        # Aucun contour exploitable.
        if not closed:
            return None
        # Nombre de lignes de l'image (pour retourner l'axe vertical).
        n_rows = self._solid_px.shape[0]
        # Liste des polygones convertis en coordonnées physiques.
        polygons = []
        for line in closed:
            # Colonne et ligne (réelles) de chaque point, corrigées de la bordure ajoutée par np.pad.
            col, row = line[:, 0] - 1.0, line[:, 1] - 1.0  # retrait du bord ajouté
            # Abscisse physique (centre du pixel = col + 0.5).
            x = self._origin[0] + (col + 0.5) * self._pixel
            # Ordonnée physique (ligne 0 en haut de l'image => inversion).
            y = self._origin[1] + (n_rows - 1.0 - row + 0.5) * self._pixel
            # Assemblage (m, 2) ; [:-1] retire le dernier point, qui répète le premier.
            polygons.append(np.column_stack([x, y])[:-1])  # le dernier point répète le premier
        # Le contour extérieur est celui d'aire (en valeur absolue) maximale.
        largest = max(polygons, key=lambda poly: abs(_signed_area(poly)))
        # Orientation trigonométrique garantie (inversion si l'aire est négative).
        return largest if _signed_area(largest) > 0 else largest[::-1]


# ====================================================================== masque global
def fill_isolated_fluid(solid: np.ndarray) -> np.ndarray:
    """Convertit en solide les poches fluides non connectées à l'écoulement principal.

    La connexité est celle des faces (4-voisinage) : une poche fermée (intérieur d'un
    anneau, cellule piégée au bord de fuite...) rendrait l'équation de Poisson singulière.
    """
    # ndimage.label numérote les composantes connexes des cellules fluides (~solid) ; la
    # connexité par défaut en 2D est le 4-voisinage (liaison par une face). labels vaut 0
    # dans le solide et 1..n dans les n zones fluides.
    labels, n = ndimage.label(~solid)
    # Une seule zone fluide : rien à combler.
    if n <= 1:
        return solid
    # np.bincount compte les cellules de chaque étiquette (taille de chaque zone).
    sizes = np.bincount(labels.ravel())
    # L'étiquette 0 (solide) ne doit pas être retenue comme zone principale.
    sizes[0] = 0
    # np.argmax : étiquette de la plus grande zone fluide (l'écoulement principal) ; toutes
    # les autres zones fluides sont isolées.
    isolated = (labels > 0) & (labels != int(np.argmax(sizes)))
    # Message à l'utilisateur (sans accents : consoles Windows en cp1252).
    logger.warning(
        "%d cellule(s) fluide(s) isolee(s) de l'ecoulement principal convertie(s) en solide.",
        int(isolated.sum()),
    )
    # Union (| = OU logique) : les poches isolées deviennent solides.
    return solid | isolated


def build_mask(
    grid: StaggeredGrid,
    obstacles: Iterable[Obstacle],
    supersample: int = 1,
    fill_isolated: bool = True,
) -> np.ndarray:
    """Masque binaire ``(Nx, Ny)`` (``True`` = solide) de l'union des obstacles."""
    # Masque initial entièrement fluide (False partout).
    solid = np.zeros(grid.shape_p, dtype=bool)
    # Parcours de tous les obstacles.
    for obstacle in obstacles:
        # Masque de cet obstacle seul.
        m = obstacle.mask(grid, supersample)
        # Obstacle plus petit qu'une maille : il disparaît de la grille.
        if not m.any():
            logger.warning("Obstacle %r non resolu par la grille (aucune cellule solide).", obstacle)
        # Moins de 8 mailles par longueur de référence : résultats peu fiables.
        elif obstacle.reference_length < 8 * max(grid.dx, grid.dy):
            logger.warning(
                "Obstacle %r peu resolu : %.1f cellules par longueur de reference.",
                obstacle,
                obstacle.reference_length / max(grid.dx, grid.dy),
            )
        # Union avec les obstacles précédents (|= : OU logique en place).
        solid |= m
    # Un domaine sans fluide ne peut pas être simulé.
    if solid.all():
        raise ValueError("Le domaine est entièrement solide.")
    # Comblement éventuel des poches fluides isolées (sinon Poisson singulier).
    return fill_isolated_fluid(solid) if fill_isolated else solid

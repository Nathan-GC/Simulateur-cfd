"""Tests des obstacles (cylindre, rectangle, NACA, polygone, image) et du masque binaire."""

# ``logging`` : niveau des messages capturés par la fixture ``caplog``.
import logging

# ``plt.imsave`` : écriture d'une image PNG de test.
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cfd2d import NACA4, Cylinder, ImageObstacle, PolygonObstacle, Rectangle, StaggeredGrid, build_mask
from cfd2d.geometry import Obstacle, naca4_coordinates

# Grille commune : domaine 4 x 2, maille de 0.01 (fine, pour des aires précises).
GRID = StaggeredGrid(Lx=4.0, Ly=2.0, Nx=400, Ny=200)


def solid_area(mask: np.ndarray, grid: StaggeredGrid) -> float:
    """Aire solide = nombre de cellules solides x aire d'une cellule."""
    # mask.sum() compte les True.
    return mask.sum() * grid.cell_area


def test_cylinder_mask_area_and_symmetry():
    # Cylindre de diamètre 0.5 centré en (1, 1).
    cyl = Cylinder(1.0, 1.0, 0.5)
    # Masque avec 4 x 4 sous-points par cellule.
    mask = cyl.mask(GRID, supersample=4)
    # Aire discrète ≈ π R² à 1 % près.
    assert solid_area(mask, GRID) == pytest.approx(np.pi * 0.25**2, rel=0.01)
    # Centre sur une ligne de faces : le masque est symétrique haut/bas ([:, ::-1] inverse l'axe y).
    np.testing.assert_array_equal(mask, mask[:, ::-1])  # centre sur une ligne de faces
    # Longueur de référence = diamètre.
    assert cyl.reference_length == 0.5


def test_rotated_rectangle_area_and_frontal_height():
    # Rectangle 0.6 x 0.3 tourné de 30°.
    rect = Rectangle(2.0, 1.0, 0.6, 0.3, angle_deg=30.0)
    # L'aire ne dépend pas de la rotation : 0.18.
    assert solid_area(rect.mask(GRID, 4), GRID) == pytest.approx(0.18, rel=0.02)
    a = np.radians(30.0)
    # Hauteur frontale = projection sur l'axe y.
    assert rect.reference_length == pytest.approx(0.6 * np.sin(a) + 0.3 * np.cos(a))
    # Carré non tourné : hauteur frontale = côté.
    assert Rectangle.square(1.0, 1.0, 0.4).reference_length == pytest.approx(0.4)


def test_naca_symmetric_profile_geometry():
    # Contour d'un NACA 0012 (symétrique, épaisseur 12 %).
    pts = naca4_coordinates("0012", n=401)
    # Bord d'attaque en x = 0, bord de fuite en x = 1.
    assert pts[:, 0].min() == pytest.approx(0.0, abs=1e-12)
    assert pts[:, 0].max() == pytest.approx(1.0)
    # Épaisseur maximale : étendue des ordonnées = 0.12.
    assert np.ptp(pts[:, 1]) == pytest.approx(0.12, rel=0.01)  # épaisseur maximale 12 %
    # Symétrie : les ordonnées triées sont opposées à celles triées en sens inverse.
    np.testing.assert_allclose(np.sort(pts[:, 1]), -np.sort(pts[:, 1])[::-1], atol=1e-12)


def test_naca_mask_area_is_rotation_invariant():
    # Aires du NACA 2412 à 0° et à 15° d'incidence.
    areas = []
    for alpha in (0.0, 15.0):
        foil = NACA4("2412", chord=1.0, x_le=1.0, y_le=1.0, alpha_deg=alpha)
        areas.append(solid_area(foil.mask(GRID, 4), GRID))
    # aire d'un profil de 12 % d'épaisseur ≈ 0.685 t c²
    assert areas[0] == pytest.approx(0.685 * 0.12, rel=0.03)
    # La rotation ne change pas l'aire.
    assert areas[1] == pytest.approx(areas[0], rel=0.03)


def test_naca_positive_incidence_raises_the_nose():
    foil = NACA4("0012", chord=1.0, x_le=1.0, y_le=1.0, alpha_deg=10.0)
    # Positions du bord d'attaque et du bord de fuite après rotation.
    le, te = foil.leading_edge, foil.trailing_edge
    # Incidence positive : bord d'attaque au-dessus de l'axe, bord de fuite au-dessous.
    assert le[1] > 1.0 > te[1]
    # Un point juste derrière le bord d'attaque est dans le profil.
    assert foil.contains(np.array(le[0] + 0.02), np.array(le[1]))


def test_invalid_naca_code():
    # Un code à 2 chiffres est refusé.
    with pytest.raises(ValueError):
        NACA4("12")


def test_polygon_orientation_and_centroid():
    # Carré unité donné dans le sens horaire : il doit être réorienté (aire positive).
    square = PolygonObstacle(np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float))  # sens horaire
    assert square.area == pytest.approx(1.0)
    # Centre de gravité au milieu.
    assert square.center == pytest.approx((0.5, 0.5))


# tmp_path : fixture pytest fournissant un dossier temporaire propre à chaque test.
def test_image_obstacle_from_png(tmp_path):
    n = 200
    # np.mgrid : grilles d'indices de lignes (yy) et de colonnes (xx).
    yy, xx = np.mgrid[:n, :n]
    # Disque de rayon 0.3 n pixels au centre de l'image.
    disk = (xx - n / 2 + 0.5) ** 2 + (yy - n / 2 + 0.5) ** 2 <= (0.3 * n) ** 2
    img = np.where(disk, 0.0, 1.0)  # disque noir sur fond blanc
    path = tmp_path / "disk.png"
    # Écriture du PNG en niveaux de gris.
    plt.imsave(path, img, cmap="gray", vmin=0.0, vmax=1.0)
    # Image de largeur physique 1 centrée en (2, 1) : le disque a un rayon physique de 0.3.
    obstacle = ImageObstacle(path, xc=2.0, yc=1.0, width=1.0)
    assert solid_area(obstacle.mask(GRID, 2), GRID) == pytest.approx(np.pi * 0.3**2, rel=0.02)
    # Hauteur frontale = diamètre du disque.
    assert obstacle.reference_length == pytest.approx(0.6, rel=0.02)
    assert obstacle.center == pytest.approx((2.0, 1.0), abs=0.01)


def test_image_obstacle_outline_is_a_closed_ccw_contour():
    n = 100
    yy, xx = np.mgrid[:n, :n]
    # Disque de 30 pixels de rayon.
    img = np.where((xx - 49.5) ** 2 + (yy - 49.5) ** 2 <= 30**2, 0.0, 1.0)
    obstacle = ImageObstacle(img, xc=2.0, yc=1.0, width=1.0)
    # Le contour extrait est converti en polygone (aire, centre).
    polygon = PolygonObstacle(obstacle.outline())
    assert polygon.area == pytest.approx(np.pi * 0.3**2, rel=0.02)
    assert polygon.center == pytest.approx((2.0, 1.0), abs=0.01)


def test_image_obstacle_invert_and_array_input():
    # Image blanche avec un carré noir au centre, passée directement en tableau.
    img = np.ones((10, 10))
    img[3:7, 3:7] = 0.0
    # Convention par défaut : sombre = solide ; invert=True : clair = solide.
    dark = ImageObstacle(img, xc=1.0, yc=1.0, width=1.0)
    light = ImageObstacle(img, xc=1.0, yc=1.0, width=1.0, invert=True)
    # Le centre de l'image (carré noir) est solide pour « dark », fluide pour « light ».
    assert dark.contains(np.array(1.0), np.array(1.0))
    assert not light.contains(np.array(1.0), np.array(1.0))


class _Ring(Obstacle):
    """Anneau : son intérieur est une poche fluide fermée."""

    def contains(self, x, y):
        # Distance au centre (2, 1).
        r = np.hypot(x - 2.0, y - 1.0)
        # Solide entre les rayons 0.3 et 0.5.
        return (r <= 0.5) & (r >= 0.3)

    # Attributs simples remplaçant les propriétés abstraites de la classe de base.
    reference_length = 1.0
    center = (2.0, 1.0)
    bounds = (1.5, 2.5, 0.5, 1.5)


# caplog : fixture pytest qui capture les messages de journalisation.
def test_build_mask_fills_enclosed_fluid(caplog):
    with caplog.at_level(logging.WARNING):
        mask = build_mask(GRID, [_Ring()], supersample=2)
    # Un avertissement signale le comblement de la poche fluide.
    assert "isolee" in caplog.text
    # Le centre de l'anneau est devenu solide.
    assert mask[GRID.locate(2.0, 1.0)]
    # Aire solide = disque plein de rayon 0.5.
    assert solid_area(mask, GRID) == pytest.approx(np.pi * 0.25, rel=0.02)


def test_build_mask_union_of_obstacles():
    # Deux cylindres disjoints.
    a, b = Cylinder(1.0, 1.0, 0.4), Cylinder(3.0, 1.0, 0.4)
    mask = build_mask(GRID, [a, b])
    # Le masque global est l'union : ses cellules solides sont la somme des deux.
    assert mask.sum() == a.mask(GRID).sum() + b.mask(GRID).sum()

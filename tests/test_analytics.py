"""Tests du module d'analyse : champs dérivés, efforts, spectres, profils, moniteurs."""

import numpy as np
import pytest

from cfd2d import NACA4, Cylinder, NavierStokesSolver, Rectangle, StaggeredGrid, presets
from cfd2d.analytics import (
    BodySurface,
    ControlVolume,
    FieldAverager,
    ForceHistory,
    ForceMonitor,
    chord_line,
    chordwise_pressure,
    compute_fields,
    crossing_frequency,
    divergence_report,
    q_criterion,
    recirculation_length,
    save_wake_profiles,
    stationary_start,
    stream_function,
    strouhal_number,
    surface_pressure,
    vorticity_nodes,
    wake_profiles,
)


def analytic_faces(grid: StaggeredGrid, fu, fv) -> tuple[np.ndarray, np.ndarray]:
    """Tableaux u, v (cellules fantômes comprises) échantillonnant des fonctions analytiques."""
    # Positions des nœuds de u : faces x_f (colonne) et ordonnées des lignes, fantômes compris (ligne).
    xu, yu = grid.x_f[:, None], ((np.arange(grid.Ny + 2) - 0.5) * grid.dy)[None, :]
    # Positions des nœuds de v : abscisses des colonnes, fantômes compris, et faces y_f.
    xv, yv = ((np.arange(grid.Nx + 2) - 0.5) * grid.dx)[:, None], grid.y_f[None, :]
    # np.broadcast_to étend le résultat à la forme complète ; .copy() le rend modifiable.
    return np.broadcast_to(fu(xu, yu), grid.shape_u).copy(), np.broadcast_to(fv(xv, yv), grid.shape_v).copy()


def small_cylinder(**kwargs) -> NavierStokesSolver:
    """Petit cas « cylindre » rapide (8 cellules par diamètre)."""
    # Options par défaut, remplaçables par les arguments nommés du test ({**a, **b} fusionne).
    options = dict(cells_per_diameter=8, length=10.0, height=5.0, x_center=3.0)
    return NavierStokesSolver(presets.cylinder_flow(**{**options, **kwargs}))


# --------------------------------------------------------------------------- champs
def test_vorticity_and_q_criterion_of_solid_body_rotation():
    g = StaggeredGrid(1.0, 1.0, 16, 16)
    omega = 3.0
    # Rotation solide de vitesse angulaire Ω autour de (0.5, 0.5) : u = -Ω y', v = Ω x'.
    u, v = analytic_faces(g, lambda x, y: -omega * (y - 0.5), lambda x, y: omega * (x - 0.5))
    # Vorticité uniforme 2Ω.
    np.testing.assert_allclose(vorticity_nodes(u, v, g), 2.0 * omega)
    # Q = Ω² (rotation pure).
    np.testing.assert_allclose(q_criterion(u, v, g), omega**2)


def test_q_criterion_is_negative_in_pure_strain():
    g = StaggeredGrid(1.0, 1.0, 16, 16)
    # Déformation pure (point d'arrêt) : u = 2x, v = -2y ; « + 0 * y » donne la bonne forme 2D.
    u, v = analytic_faces(g, lambda x, y: 2.0 * x + 0 * y, lambda x, y: -2.0 * y + 0 * x)
    # Aucune rotation.
    np.testing.assert_allclose(vorticity_nodes(u, v, g), 0.0, atol=1e-12)
    # Q = -½(4 + 4) = -4 : la déformation domine.
    np.testing.assert_allclose(q_criterion(u, v, g), -4.0)


def test_wall_vorticity_uses_mirror_condition():
    g = StaggeredGrid(1.0, 1.0, 16, 16)
    # Paroi horizontale : les 4 premières lignes de cellules sont solides.
    solid = np.zeros(g.shape_p, dtype=bool)
    solid[:, :4] = True  # paroi en y = 0.25
    gamma = 2.0
    # Cisaillement linéaire u = γ (y - 0.25) au-dessus de la paroi, 0 dans le solide.
    u, v = analytic_faces(g, lambda x, y: gamma * np.maximum(y - 0.25, 0.0) + 0 * x, lambda x, y: 0 * x + 0 * y)
    # Avec le masque (miroir) : ω = -γ jusque sur la paroi.
    np.testing.assert_allclose(vorticity_nodes(u, v, g, solid)[:, 4:], -gamma)
    # Sans masque : la vorticité pariétale ne vaut que la moitié.
    np.testing.assert_allclose(vorticity_nodes(u, v, g)[:, 4], -gamma / 2)  # sans miroir : moitié


def test_stream_function_reproduces_discrete_velocities():
    s = small_cylinder()
    s.run(max_steps=20)
    u, v, g = s.state.u, s.state.v, s.grid
    psi = stream_function(u, v, g)
    # u = ∂ψ/∂y par construction.
    np.testing.assert_allclose(np.diff(psi, axis=1) / g.dy, u[:, 1:-1], atol=1e-10)
    # v = -∂ψ/∂x : vrai seulement si le champ est à divergence nulle (chemin d'intégration indifférent).
    np.testing.assert_allclose(-np.diff(psi, axis=0) / g.dx, v[1:-1, :], atol=1e-9)  # chemin indifférent


def test_compute_fields_masks_the_solid():
    s = small_cylinder()
    f = compute_fields(s)
    # Champs aux centres des cellules.
    assert f.speed.shape == s.grid.shape_p
    # Vorticité nulle dans le solide.
    assert np.all(f.vorticity[s.solid] == 0.0)
    # Version masquée : autant d'éléments masqués que de cellules solides.
    assert f.masked("speed").mask.sum() == s.solid.sum()
    # Champ initial projeté : divergence nulle.
    assert np.abs(f.divergence).max() < 1e-10


# -------------------------------------------------------------------------- efforts
def test_pressure_force_on_cylinder_matches_contour_integral():
    # Cylindre de diamètre 1 résolu par 40 cellules.
    g = StaggeredGrid(4.0, 4.0, 160, 160)
    solid = Cylinder(2.0, 2.0, 1.0).mask(g, 4)
    X, Y = g.cell_centers()
    # Pression p = cos φ (φ = angle polaire) : -∮ p n dS = (-πR, 0) analytiquement.
    p = np.cos(np.arctan2(Y - 2.0, X - 2.0))  # -∮ p n dS = (-πR, 0)
    # Vitesses nulles : seule la pression contribue.
    f = BodySurface(g, solid).forces(np.zeros(g.shape_u), np.zeros(g.shape_v), p, mu=0.0, center=(2.0, 2.0))
    assert f.fx_pressure == pytest.approx(-0.5 * np.pi, rel=0.02)
    # Symétrie haut/bas : pas de force selon y ni de moment.
    assert abs(f.fy_pressure) < 1e-10 and abs(f.moment) < 1e-10
    assert f.fx_viscous == 0.0 and f.fx_convective == 0.0


def test_viscous_force_from_linear_shear():
    g = StaggeredGrid(4.0, 2.0, 80, 40)
    # Rectangle aligné sur les faces de la grille.
    solid = Rectangle(2.0, 0.75, 1.0, 0.5).mask(g)  # x ∈ [1.5, 2.5], y ∈ [0.5, 1.0]
    gamma, mu = 3.0, 0.2
    # Cisaillement u = γ (y - 1) au-dessus de la face supérieure du rectangle.
    u, v = analytic_faces(g, lambda x, y: gamma * np.maximum(y - 1.0, 0.0) + 0 * x, lambda x, y: 0 * x + 0 * y)
    f = BodySurface(g, solid).forces(u, v, np.zeros(g.shape_p), mu)
    # Force de frottement = τ x largeur = μ γ x 1.
    assert f.fx_viscous == pytest.approx(mu * gamma * 1.0, rel=1e-10)  # τ = μγ sur la face supérieure
    assert f.fy_viscous == 0.0 and f.fx_pressure == 0.0 and f.fx_convective == 0.0


def test_local_window_gives_the_same_forces_as_the_full_domain():
    s = small_cylinder()
    s.run(t_end=2.0)
    st, mu = s.state, s.rho * s.nu
    # Calcul sur la fenêtre locale (marge par défaut) ...
    local = BodySurface(s.grid, s.solid).forces(st.u, st.v, st.p, mu, s.rho, (3.0, 2.5))
    # ... et sur tout le domaine (marge immense).
    full = BodySurface(s.grid, s.solid, margin=10**6).forces(st.u, st.v, st.p, mu, s.rho, (3.0, 2.5))
    # Résultats identiques (pytest.approx compare aussi les dataclasses champ par champ).
    assert local == pytest.approx(full, rel=1e-12, abs=1e-14)


def test_control_volume_sees_no_force_in_uniform_flow():
    g = StaggeredGrid(4.0, 2.0, 40, 20)
    # Volume de contrôle dans un domaine sans obstacle.
    cv = ControlVolume(g, np.zeros(g.shape_p, dtype=bool), (1.0, 3.0, 0.5, 1.5))
    # Écoulement uniforme u = 1, pression uniforme : flux entrant = flux sortant.
    fx, fy = cv.surface_forces(np.ones(g.shape_u), np.zeros(g.shape_v), np.full(g.shape_p, 2.0), rho=1.0, mu=0.1)
    assert abs(fx) < 1e-12 and abs(fy) < 1e-12
    # Quantité de mouvement contenue = ρ u x aire (2 x 1).
    assert cv.momentum(np.ones(g.shape_u), np.zeros(g.shape_v), rho=1.0)[0] == pytest.approx(2.0)


def test_control_volume_must_enclose_the_body():
    s = small_cylinder()
    # Un rectangle qui coupe le cylindre est refusé.
    with pytest.raises(ValueError):
        ControlVolume(s.grid, s.solid, (2.8, 6.0, 1.0, 4.0))  # coupe le cylindre (x = 3 ± 0.5)


def test_force_methods_agree_on_steady_cylinder_flow():
    # Écoulement stationnaire (Re = 20) sans perturbation.
    s = NavierStokesSolver(
        presets.cylinder_flow(
            Re=20.0, cells_per_diameter=10, length=12.0, height=8.0, x_center=4.0, perturbation=0.0, t_end=30.0
        )
    )
    # Moniteur d'efforts avec vérification par volume de contrôle.
    forces = ForceMonitor(s, control_volume=(2.5, 6.5, 2.0, 6.0))
    s.run()
    summary = forces.history().summary()
    assert not summary.unsteady
    # Cd = 2.05 en milieu infini ; 2.58 avec 12.5 % de blocage (valeur convergée en maillage)
    assert 2.45 < summary.cd_mean < 2.7
    # l'intégration sur le contour vérifie le bilan de quantité de mouvement discret
    assert summary.cd_cv_mean == pytest.approx(summary.cd_mean, rel=5e-3)
    # Écoulement symétrique : portance nulle.
    assert abs(summary.cl_mean) < 1e-3
    assert 0.8 < recirculation_length(s) < 1.0  # 0.93 D en milieu infini (Coutanceau & Bouard)


def test_naca_moment_center_is_quarter_chord():
    cfg = presets.naca_airfoil(cells_per_chord=16, alpha_deg=10.0)
    s = NavierStokesSolver(cfg)
    # attach=False : moniteur non inscrit auprès du solveur (calcul à la demande).
    monitor = ForceMonitor(s, attach=False)
    foil = s.obstacles[0]
    assert isinstance(foil, NACA4)
    # Le moment est pris au quart de corde (pivot du profil).
    assert monitor.moment_center == pytest.approx(foil.pivot_point)
    # set(dict) = ensemble des clés ; >= : inclusion.
    assert set(monitor.coefficients(s)) >= {"cd", "cl", "cm"}


# -------------------------------------------------------------------------- spectres
def test_strouhal_of_synthetic_signal_with_transient():
    rng = np.random.default_rng(0)
    # Instants à pas variable (comme avec le pas de temps adaptatif).
    t = np.cumsum(rng.uniform(0.005, 0.015, 20_000))  # pas variable, t ≈ 0..200
    f = 0.1732
    # Sinusoïde dont l'amplitude croît puis sature (tanh⁴), décalée de 0.01.
    cl = np.tanh(t / 30.0) ** 4 * 0.33 * np.sin(2.0 * np.pi * f * t) + 0.01
    # Détection du régime établi.
    t0 = stationary_start(t, cl)
    assert 70.0 < t0 < 110.0  # amplitude à 5 % près de sa valeur finale dès t ≈ 76
    # Strouhal (L = U = 1 : St = f).
    st, spectrum = strouhal_number(t, cl, length=1.0, velocity=1.0)
    assert st == pytest.approx(f, rel=2e-3)
    assert spectrum.peak_amplitude == pytest.approx(0.33, rel=0.03)
    # Méthode indépendante par passages à zéro.
    assert crossing_frequency(t, cl, t0) == pytest.approx(f, rel=2e-3)


def test_steady_history_has_no_strouhal():
    t = np.linspace(0.0, 10.0, 500)
    ones, zeros = np.ones_like(t), np.zeros_like(t)
    # Historique stationnaire : Cd constant, Cl quasi nul.
    history = ForceHistory(
        time=t, cd=2.0 * ones, cl=1e-9 * np.sin(t), cd_pressure=ones, cd_viscous=ones, cd_convective=zeros,
        cl_pressure=zeros, cl_viscous=zeros, cl_convective=zeros, cm=zeros, reference_length=1.0,
        reference_velocity=1.0,
    )
    summary = history.summary()
    # Pas d'oscillation : pas de Strouhal (NaN).
    assert not summary.unsteady and np.isnan(summary.strouhal)
    assert summary.cd_mean == pytest.approx(2.0)
    assert "stationnaire" in str(summary)


# --------------------------------------------------------------------------- profils
def test_surface_pressure_follows_the_contour():
    s = NavierStokesSolver(presets.cylinder_flow(cells_per_diameter=20, length=8.0, height=6.0, x_center=3.0))
    X, Y = s.grid.cell_centers()
    # Pression imposée = cos θ (θ depuis l'amont) : Cp exact le long de la paroi.
    s.state.p[:] = 0.5 * np.cos(np.arctan2(Y - 3.0, -(X - 3.0)))  # ½ρU² = 0.5 : Cp = cos θ
    dist = surface_pressure(s, p_ref=0.0, n=180)
    # Départ au point d'arrêt amont.
    assert dist.theta[0] == pytest.approx(0.0, abs=1.0)
    # Angle strictement croissant.
    assert np.all(np.diff(dist.theta) > 0.0)
    # Le premier quart du parcours est sur le dessus du cylindre (y > 3).
    assert dist.y[dist.theta.size // 4] > 3.0  # le parcours longe d'abord le dessus
    np.testing.assert_allclose(dist.cp, np.cos(np.radians(dist.theta)), atol=0.05)


def test_chord_line_only_for_slender_bodies():
    # Corps non profilés : pas de corde.
    assert chord_line(Cylinder(0.0, 0.0, 1.0)) is None
    assert chord_line(Rectangle.square(0.0, 0.0, 1.0)) is None
    # Plaque à 20° d'incidence (rotation horaire) : bord d'attaque en amont et au-dessus du bord de fuite.
    le, te = chord_line(Rectangle(0.0, 0.0, 2.0, 0.2, angle_deg=-20.0))
    assert le[0] < te[0] and le[1] > te[1]
    # Les deux extrémités sont distantes de la longueur de la plaque.
    assert np.hypot(te[0] - le[0], te[1] - le[1]) == pytest.approx(2.0)


def test_chordwise_pressure_separates_upper_and_lower_surfaces():
    # Profil NACA 0012 à 5° d'incidence (le champ n'est pas calculé : pression imposée).
    s = NavierStokesSolver(presets.naca_airfoil(code="0012", alpha_deg=5.0, cells_per_chord=24, length=4.0,
                                                height=2.0, x_le=1.0))
    ob = s.obstacles[0]
    # Pression égale à l'ordonnée relative au bord d'attaque (½ρU² = 0.5, d'où Cp = y - y_le).
    _, Y = s.grid.cell_centers()
    s.state.p[:] = 0.5 * (Y - 1.0)
    chord = chord_line(ob)
    assert chord == (ob.leading_edge, ob.trailing_edge)
    cp = chordwise_pressure(surface_pressure(s, p_ref=0.0, n=240), *chord)
    # Sur chaque face : abscisses réduites croissantes, du bord d'attaque au bord de fuite.
    for x in (cp.x_upper, cp.x_lower):
        assert np.all(np.diff(x) >= 0.0)
        assert x[0] == pytest.approx(0.0, abs=0.03) and x[-1] == pytest.approx(1.0, abs=0.03)
    # L'extrados est au-dessus de la corde : sa « pression » (son ordonnée) est la plus grande.
    assert cp.cp_upper.mean() > cp.cp_lower.mean()


def test_wake_profile_metrics_on_gaussian_wake(tmp_path):
    s = NavierStokesSolver(presets.cylinder_flow(cells_per_diameter=10, length=16.0, height=8.0, x_center=3.0))
    f = compute_fields(s)
    # Sillage gaussien synthétique : u = 1 - 0.5 exp(-η²/0.25), identique à toutes les abscisses.
    eta = f.y[None, :] - 4.0
    f.u = np.broadcast_to(1.0 - 0.5 * np.exp(-(eta**2) / 0.25), f.u.shape).copy()
    profiles = wake_profiles(s, (1, 2, 5, 10, 50), fields=f)
    # La station à 50 D, hors du domaine, est ignorée.
    assert [p.station for p in profiles] == [1, 2, 5, 10]  # 50 D : hors domaine
    p = profiles[1]
    # Station x/D = 2 derrière le centre (x = 3) : x = 5.
    assert p.x == pytest.approx(5.0)
    assert p.centerline_velocity == pytest.approx(0.5, abs=0.01)
    assert p.deficit == pytest.approx(0.5, abs=0.01)
    # Demi-largeur analytique : exp(-η²/0.25) = 1/2 => η = 0.5 √(ln 2).
    assert p.half_width == pytest.approx(0.5 * np.sqrt(np.log(2.0)), rel=0.02)
    # Export CSV relu avec numpy (names=True : la première ligne donne les noms de colonnes).
    path = save_wake_profiles(tmp_path / "wake.csv", profiles)
    table = np.genfromtxt(path, delimiter=",", names=True)
    # 4 profils de Ny points.
    assert table.size == 4 * s.grid.Ny


def test_recirculation_length_on_synthetic_bubble():
    s = NavierStokesSolver(presets.cylinder_flow(cells_per_diameter=20, length=10.0, height=6.0, x_center=3.0))
    f = compute_fields(s)
    # Écoulement de retour (u = -0.2) jusqu'à x = 4.8, soit 1.3 D derrière le culot (x = 3.5).
    f.u = np.where(f.x[:, None] + 0 * f.u < 4.8, -0.2, 0.8)  # retour de l'écoulement jusqu'à 1.3 D du culot
    assert recirculation_length(s, fields=f) == pytest.approx(1.3, abs=0.05)
    # Sans écoulement de retour : longueur nulle.
    f.u[:] = 1.0
    assert recirculation_length(s, fields=f) == 0.0


# ------------------------------------------------------------------------ moniteurs
def test_monitors_record_average_and_export(tmp_path):
    s = small_cylinder()
    # Efforts tous les 2 pas ; moyennes à partir de t = 0.5.
    forces = ForceMonitor(s, every=2)
    averager = FieldAverager(s, t_start=0.5)
    s.run(t_end=1.0)
    history = forces.history()
    # Un enregistrement tous les deux pas (// = division entière).
    assert history.time.size == len(s.diagnostics) // 2
    # Traînée positive.
    assert np.all(history.cd > 0.0)
    # La décomposition somme bien au total.
    np.testing.assert_allclose(history.cd, history.cd_pressure + history.cd_viscous + history.cd_convective)
    # Durée moyennée : de 0.5 à 1.0.
    assert averager.duration == pytest.approx(0.5)
    # Le champ moyen reste à divergence nulle.
    mean = averager.fields()
    assert np.abs(mean.divergence).max() < 1e-10
    # Variances positives (aux arrondis près).
    uu, vv, _ = averager.reynolds_stresses()
    assert uu.min() > -1e-10 and vv.min() > -1e-10
    # Exports CSV et NPZ, puis relecture.
    history.save(tmp_path / "forces.csv")
    history.save(tmp_path / "forces.npz")
    table = np.genfromtxt(tmp_path / "forces.csv", delimiter=",", names=True)
    np.testing.assert_allclose(table["cd"], history.cd, rtol=1e-9)
    # Les métadonnées sont conservées dans le .npz.
    assert float(np.load(tmp_path / "forces.npz")["reference_length"]) == 1.0


def test_divergence_report():
    s = NavierStokesSolver(presets.lid_driven_cavity(N=16))
    s.run(max_steps=10)
    report = divergence_report(s)
    # Divergence au niveau des erreurs d'arrondi, sur l'historique et le champ courant.
    assert report.history_max < 1e-12 and report.current_max < 1e-12
    # Texte du rapport.
    assert "max|div u|" in str(report)

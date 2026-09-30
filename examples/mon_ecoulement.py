"""Gabarit de simulation : tous les paramètres de l'écoulement réunis en un seul endroit.

Mode d'emploi : modifier les valeurs de la section « PARAMÈTRES » ci-dessous, puis lancer ::

    .venv\\Scripts\\python examples\\mon_ecoulement.py

Les résultats (tableau de bord, animation, efforts) sont écrits dans ``outputs/<NOM>/``.
Avec les valeurs par défaut (cylindre, Re = 100, 16 cellules par diamètre, t = 80), le
calcul dure environ 1 à 2 minutes.
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# Journalisation de la progression.
import logging

# ``shutil.which`` : détection de l'exécutable ffmpeg (vidéo MP4).
import shutil

# Chemins de fichiers portables.
from pathlib import Path

# Backend matplotlib sans fenêtre (à choisir avant d'importer le paquet).
import matplotlib

matplotlib.use("Agg")

# Briques de configuration, obstacles, conditions aux limites, solveur et moniteurs.
from cfd2d import (  # noqa: E402
    NACA4,
    BoundaryConfig,
    Cylinder,
    DomainConfig,
    FieldAverager,
    FlowConfig,
    ForceMonitor,
    Inlet,
    NavierStokesSolver,
    NoSlipWall,
    NumericsConfig,
    Outlet,
    Rectangle,
    SimulationConfig,
    SlipWall,
    TimeConfig,
)

# Visualisation (tableau de bord, animation).
from cfd2d import visualization as viz  # noqa: E402

# ==================================================================================
#                                   PARAMÈTRES
# ==================================================================================
# --- Nom du cas : les résultats vont dans outputs/<NOM>/
NOM = "mon_ecoulement"

# --- Écoulement
# Nombre de Reynolds Re = U_INF * D / nu (D = longueur de référence de l'obstacle).
# Repères pour un cylindre : Re < 47 écoulement stationnaire (deux tourbillons attachés),
# 47 < Re < ~190 allée de Von Kármán périodique (régime où un calcul 2D est réaliste).
REYNOLDS = 100.0
# Vitesse de l'écoulement amont (la viscosité s'en déduit : nu = U_INF * D / REYNOLDS).
U_INF = 1.0
# Amplitude de la perturbation initiale (fraction de U_INF) : déclenche plus tôt le lâcher
# tourbillonnaire d'une géométrie symétrique ; 0 = écoulement parfaitement symétrique
# (à choisir pour Re < 47 : la perturbation s'y amortit lentement).
PERTURBATION = 0.05

# --- Domaine (longueurs exprimées en tailles d'obstacle D)
D = 1.0
LONGUEUR = 16.0 * D
HAUTEUR = 8.0 * D
# Résolution : nombre de cellules par D (même maille en x et en y).
# 12-16 : essais rapides ; 20 : bon compromis ; 30+ : précis mais ~3,4x plus long qu'à 20.
CELLULES_PAR_D = 16

# --- Obstacle : garder UNE des lignes « OBSTACLE = ... » (commenter les autres avec #)
X_OBSTACLE, Y_OBSTACLE = 4.0 * D, 0.5 * HAUTEUR
OBSTACLE = Cylinder(X_OBSTACLE, Y_OBSTACLE, D)
# OBSTACLE = Rectangle.square(X_OBSTACLE, Y_OBSTACLE, D)                                # carré
# OBSTACLE = Rectangle(X_OBSTACLE, Y_OBSTACLE, 2.0 * D, 0.4 * D, angle_deg=-20.0)       # plaque inclinée
# OBSTACLE = NACA4("2412", chord=D, x_le=X_OBSTACLE - 0.25 * D, y_le=Y_OBSTACLE, alpha_deg=8.0)  # profil

# --- Conditions aux limites
# Entrée (côté gauche) : "uniform" (U constant) ou "parabolic" (Poiseuille, max 1.5 U) ;
# ramp_time > 0 : montée progressive de la vitesse sur cette durée.
ENTREE = Inlet(profile="uniform", ramp_time=0.0)
# Sortie (côté droit) : "convective" (recommandé pour les sillages) ou "neumann".
SORTIE = Outlet(kind="convective")
# Parois haut et bas : SlipWall() = soufflerie idéale (pas de frottement),
# NoSlipWall() = canal à parois adhérentes.
PAROIS = SlipWall()

# --- Temps
# Durée simulée (en unités D/U) : ~50 pour établir l'allée à Re = 100, puis au moins
# 10 périodes (~60) pour un Strouhal fiable.
T_FINAL = 80.0
# Schéma temporel : "ab2" (défaut, économique), "rk3" (plus robuste), "euler" (pédagogique).
SCHEMA_TEMPS = "ab2"
# Nombre de Courant visé (None = valeur sûre du schéma : 0.4 pour ab2, 0.8 pour rk3).
CFL = None

# --- Méthodes numériques
# Advection : "quick" (défaut, précis), "tvd" (sans oscillation), "upwind" (robuste mais
# très diffusif), "central" (oscille si la maille est trop grossière).
ADVECTION = "quick"
# Pression : "direct" (le plus rapide) ; "cg", "bicgstab", "sor", "jacobi" (itératifs, plus lents).
SOLVEUR_PRESSION = "direct"

# --- Sorties
# Début des moyennes temporelles (Cp moyen, profils de sillage moyens).
DEBUT_MOYENNES = 0.5 * T_FINAL
# Animation de la vorticité sur les dernières unités de temps (0 = pas d'animation).
DUREE_ANIMATION = 20.0
# ==================================================================================

# Dossier de sortie.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / NOM


def build_config() -> SimulationConfig:
    """Assemble la configuration complète à partir des paramètres ci-dessus."""
    return SimulationConfig(
        name=NOM,
        # Nombre de cellules = longueur / taille de maille (round : entier le plus proche).
        domain=DomainConfig(
            Lx=LONGUEUR, Ly=HAUTEUR, Nx=round(LONGUEUR / D * CELLULES_PAR_D), Ny=round(HAUTEUR / D * CELLULES_PAR_D)
        ),
        # Re donné : la viscosité est calculée avec la longueur de référence de l'obstacle.
        flow=FlowConfig(U_inf=U_INF, Re=REYNOLDS, perturbation=PERTURBATION),
        # Entrée à gauche, sortie à droite, parois en haut et en bas.
        boundaries=BoundaryConfig(west=ENTREE, east=SORTIE, south=PAROIS, north=PAROIS),
        obstacles=[OBSTACLE],
        time=TimeConfig(t_end=T_FINAL, scheme=SCHEMA_TEMPS, cfl=CFL),
        numerics=NumericsConfig(advection=ADVECTION, pressure_solver=SOLVEUR_PRESSION),
    )


def main() -> None:
    # Messages de progression avec l'heure.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # Création du solveur (grille, masque, matrice de pression, champ initial).
    solver = NavierStokesSolver(build_config())
    # Moniteurs, appelés automatiquement pendant le calcul : efforts à chaque pas, moyennes
    # temporelles tous les 5 pas, images d'animation tous les 25 pas en fin de calcul.
    forces = ForceMonitor(solver)
    averager = FieldAverager(solver, t_start=DEBUT_MOYENNES, every=5)
    recorder = viz.FrameRecorder(solver, every=25, t_start=T_FINAL - DUREE_ANIMATION) if DUREE_ANIMATION else None

    # Simulation.
    solver.run(log_every=1000)

    # Synthèse : Cd, Cl, Strouhal...
    history = forces.history()
    print("\n" + str(history.summary()))
    # Exports et figures.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history.save(OUTPUT_DIR / "forces.csv")
    viz.plot_dashboard(solver, history, mean_fields=averager.fields(), path=OUTPUT_DIR / "dashboard.png")
    if recorder is not None and len(recorder):
        # MP4 si ffmpeg est installé, GIF sinon.
        extension = ".mp4" if shutil.which("ffmpeg") else ".gif"
        viz.animate(recorder, OUTPUT_DIR / f"vorticity{extension}", view=viz.default_view(solver))
    print(f"\nResultats dans {OUTPUT_DIR}")


# Exécuté seulement quand le fichier est lancé comme script.
if __name__ == "__main__":
    main()

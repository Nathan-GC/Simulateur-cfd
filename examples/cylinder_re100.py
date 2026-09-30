"""Cylindre à Re = 100 : allée de Von Kármán, indicateurs aérodynamiques et visualisations.

Usage ::

    python examples/cylinder_re100.py [cellules_par_diametre] [t_final]

Produit dans ``outputs/cylinder_Re100/`` :

* ``dashboard.png`` (vorticité) et ``dashboard_streamlines.png`` (lignes de courant) ;
* ``vorticity.mp4`` (ou ``vorticity.gif`` sans ffmpeg) : les 30 dernières unités de temps ;
* ``report.html`` : rapport interactif (Plotly) ;
* ``forces.csv``, ``spectrum_cl.csv``, ``cp_mean.csv``, ``wake_profiles.csv`` : données tabulées ;
* ``fields_final.npz``, ``fields_mean.npz``, ``flow_final.vtk``, ``flow_mean.vtk`` (ParaView) ;
* ``checkpoint.npz`` : état final, pour prolonger le calcul avec :func:`cfd2d.io.load_checkpoint`.
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# Journalisation de la progression du calcul.
import logging

# ``shutil.which`` : détection de l'exécutable ffmpeg (vidéo MP4).
import shutil

# ``sys.argv`` : arguments de la ligne de commande.
import sys

# Chemins de fichiers portables.
from pathlib import Path

# Backend matplotlib non interactif (aucune fenêtre ouverte) : à choisir avant tout import de pyplot.
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

# API du solveur : moniteurs, solveur, champs, configurations types.
from cfd2d import FieldAverager, ForceMonitor, NavierStokesSolver, compute_fields, presets  # noqa: E402

# Modules d'analyse, d'entrées/sorties et de visualisation (alias courts).
from cfd2d import analytics as an  # noqa: E402
from cfd2d import io as cio  # noqa: E402
from cfd2d import visualization as viz  # noqa: E402

# Dossier de sortie : <projet>/outputs/cylinder_Re100 (parents[1] = dossier parent de examples/).
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "cylinder_Re100"


def main(cells_per_diameter: int = 20, t_end: float = 150.0) -> None:
    # Messages de progression avec l'heure.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # Configuration type : cylindre D = 1 dans un domaine 20 D x 10 D, sortie convective.
    solver = NavierStokesSolver(presets.cylinder_flow(Re=100.0, cells_per_diameter=cells_per_diameter, t_end=t_end))
    # Centre du cylindre (placement du volume de contrôle).
    xc, yc = solver.obstacles[0].center

    # --- Moniteurs (appelés automatiquement par le solveur pendant run)
    # Efforts à chaque pas, avec vérification par un volume de contrôle de 4 D x 4 D.
    forces = ForceMonitor(solver, control_volume=(xc - 1.5, xc + 2.5, yc - 2.0, yc + 2.0))
    # Moyennes temporelles sur la seconde moitié du calcul (régime établi), tous les 5 pas.
    averager = FieldAverager(solver, t_start=0.5 * t_end, every=5)
    # Images de vorticité pour l'animation : les 30 dernières unités de temps, tous les 25 pas.
    recorder = viz.FrameRecorder(solver, every=25, t_start=max(0.0, t_end - 30.0))

    # --- Simulation
    solver.run(log_every=2000)

    # --- Analyse
    # Séries temporelles des coefficients et leur synthèse (moyennes, Strouhal...).
    history = forces.history()
    summary = history.summary()
    # Champs moyens (vitesse, pression, vorticité du champ moyen).
    mean = averager.fields()
    # Cp moyen le long de la paroi et profils moyens de sillage à x/D = 1, 2, 5, 10.
    cp = an.surface_pressure(solver, fields=mean)
    profiles = an.wake_profiles(solver, (1.0, 2.0, 5.0, 10.0), fields=mean)
    # Angle du minimum de Cp sur l'extrados (première moitié du parcours).
    theta_min = cp.theta[np.argmin(cp.cp[: cp.theta.size // 2])]

    print("\n=== Efforts ===")
    # print appelle ForceSummary.__str__ (texte multiligne).
    print(summary)
    # Vérification indépendante de la fréquence par les passages à zéro.
    print(f"Frequence par passages a zero : St = {an.crossing_frequency(history.time, history.cl, summary.t_start):.4f}")
    print("\n=== Paroi et sillage (champs moyens) ===")
    # np.interp(180, theta, cp) : Cp au culot (θ = 180°).
    print(f"Cp au point d'arret = {cp.cp[0]:.3f} | Cp min = {cp.cp.min():.3f} (theta = {theta_min:.0f} deg) | "
          f"Cp culot = {np.interp(180.0, cp.theta, cp.cp):.3f}")
    print(f"Longueur de recirculation moyenne : Lr/D = {an.recirculation_length(solver, fields=mean):.3f}")
    for p in profiles:
        print(f"x/D = {p.station:4.1f} : u axe/U = {p.centerline_velocity:6.3f}, deficit = {p.deficit:.3f}, "
              f"demi-largeur = {p.half_width:.3f} D")
    print("\n=== Incompressibilite ===")
    print(an.divergence_report(solver))

    # --- Exports de données
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Tables CSV (ouvrables dans un tableur).
    history.save(OUTPUT_DIR / "forces.csv")
    history.spectrum().save(OUTPUT_DIR / "spectrum_cl.csv")
    cp.save(OUTPUT_DIR / "cp_mean.csv")
    an.save_wake_profiles(OUTPUT_DIR / "wake_profiles.csv", profiles)
    # Champs 2D : archives NumPy et fichiers VTK pour ParaView.
    final = compute_fields(solver)
    cio.save_fields_npz(OUTPUT_DIR / "fields_final.npz", final, Re=solver.reynolds)
    cio.save_fields_npz(OUTPUT_DIR / "fields_mean.npz", mean, Re=solver.reynolds)
    cio.save_vtk(OUTPUT_DIR / "flow_final.vtk", final, solver.grid)
    cio.save_vtk(OUTPUT_DIR / "flow_mean.vtk", mean, solver.grid, title="cfd2d champ moyen")
    # Point de reprise.
    cio.save_checkpoint(OUTPUT_DIR / "checkpoint.npz", solver)

    # --- Visualisations
    # Tableaux de bord (vorticité, puis lignes de courant).
    viz.plot_dashboard(solver, history, mean_fields=mean, path=OUTPUT_DIR / "dashboard.png")
    viz.plot_dashboard(solver, history, mean_fields=mean, main="streamlines", path=OUTPUT_DIR / "dashboard_streamlines.png")
    # Animation : MP4 si ffmpeg est disponible, GIF (plus lourd, résolution réduite) sinon.
    view = viz.default_view(solver)
    if shutil.which("ffmpeg"):
        viz.animate(recorder, OUTPUT_DIR / "vorticity.mp4", fps=20, view=view)
    else:
        viz.animate(recorder, OUTPUT_DIR / "vorticity.gif", fps=20, view=view, dpi=60)
    # Rapport interactif HTML (autonome, lisible hors ligne).
    viz.interactive_report(history, OUTPUT_DIR / "report.html", solver=solver, mean_fields=mean)
    print(f"\nResultats exportes dans {OUTPUT_DIR}")


# Exécuté seulement quand le fichier est lancé comme script (pas lors d'un import).
if __name__ == "__main__":
    # Arguments optionnels : résolution puis durée.
    args = sys.argv[1:]
    main(int(args[0]) if args else 20, float(args[1]) if len(args) > 1 else 150.0)

"""Cavité entraînée à Re = 100 : validation du solveur contre Ghia, Ghia & Shin (1982).

Usage ::

    python examples/lid_driven_cavity.py [N]

Produit ``outputs/cavity_Re100_N<N>.png`` : profils médians comparés à la référence et
lignes de courant de l'écoulement stationnaire.
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# Journalisation de la progression.
import logging

# ``sys.argv`` : argument optionnel (résolution N).
import sys

# Chemins de fichiers portables.
from pathlib import Path

# Contexte de style matplotlib et figure indépendante de pyplot.
from matplotlib import rc_context
from matplotlib.figure import Figure

# Solveur et configurations types.
from cfd2d import NavierStokesSolver, presets

# Données de référence et extraction des profils médians.
from cfd2d.validation import GHIA_RE100, cavity_centerlines, ghia_errors

# Charte graphique commune et tracé des lignes de courant.
from cfd2d.visualization import INK, SERIES, STYLE, SURFACE, plot_streamlines

# Dossier de sortie : <projet>/outputs.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main(N: int = 64) -> None:
    # Messages de progression avec l'heure.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # Cavité N x N à Re = 100, intégrée jusqu'à t = 25 (régime stationnaire atteint).
    solver = NavierStokesSolver(presets.lid_driven_cavity(Re=100.0, N=N, t_end=25.0))
    solver.run(log_every=2000)
    # Écarts maximaux aux données de Ghia et al.
    err_u, err_v = ghia_errors(solver)
    print(f"Ecart maximal a Ghia et al. (1982) : u = {err_u:.4f}, v = {err_v:.4f}")

    # Profils u(y) sur l'axe vertical et v(x) sur l'axe horizontal.
    y, u, x, v = cavity_centerlines(solver)
    # Figure construite avec la charte graphique du paquet.
    with rc_context(STYLE):
        # Trois panneaux côte à côte ; layout="constrained" ajuste les marges.
        fig = Figure(figsize=(15, 4.8), layout="constrained")
        ax_u, ax_v, ax_s = fig.subplots(1, 3)
        # Simulation (trait, couleur 1) et référence (cercles creux, couleur 2).
        ax_u.plot(u, y, color=SERIES[0], label=f"cfd2d ({N}×{N})")
        ax_u.plot(GHIA_RE100["u"], GHIA_RE100["y"], "o", color=SERIES[1], markersize=6, mfc="none", mew=1.5,
                  label="Ghia et al. (1982)")
        # ax.set(...) règle plusieurs propriétés à la fois.
        ax_u.set(xlabel="u / U", ylabel="y / L", title="u sur l'axe vertical x = L/2")
        # Même comparaison pour v(x).
        ax_v.plot(x, v, color=SERIES[0], label=f"cfd2d ({N}×{N})")
        ax_v.plot(GHIA_RE100["x"], GHIA_RE100["v"], "o", color=SERIES[1], markersize=6, mfc="none", mew=1.5,
                  label="Ghia et al. (1982)")
        ax_v.set(xlabel="x / L", ylabel="v / U", title="v sur l'axe horizontal y = L/2")
        # Légendes (deux séries par panneau).
        for ax in (ax_u, ax_v):
            ax.legend()
        # Lignes de courant colorées par la vitesse.
        plot_streamlines(ax_s, solver, density=1.4)
        ax_s.set_title("Lignes de courant")
        # Titre général.
        fig.suptitle(f"Cavité entraînée, Re = 100 : écart max à Ghia u {err_u:.3f}, v {err_v:.3f}", color=INK)
        # Enregistrement (fond de la charte).
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"cavity_Re100_N{N}.png"
        fig.savefig(path, dpi=130, facecolor=SURFACE)
    print(f"Figure : {path}")


# Exécuté seulement quand le fichier est lancé comme script.
if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 64)

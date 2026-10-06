# cfd2d — Solveur Navier–Stokes 2D incompressible

Solveur CFD 2D paramétrable : différences finies sur **grille MAC décalée**, méthode de
**projection de Chorin**, obstacles arbitraires par **masque binaire** (frontières immergées).

**Première visite ?** Suivre le [tutoriel pas à pas](TUTORIEL.md) : installation, lancement,
modification des paramètres, puis lecture guidée de la théorie et du code. Pour paramétrer
un écoulement sans écrire de code : l'[application web](#application-web) (réglages à la
souris) ou le gabarit [examples/mon_ecoulement.py](examples/mon_ecoulement.py).

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
```

## Application web

```bash
python -m streamlit run app.py
```

L'application s'ouvre dans le navigateur (http://localhost:8501), en thème clair ou sombre
(menu ⋮ > *Settings*), et s'utilise aussi sur téléphone. Elle n'implémente aucun calcul : elle
assemble une `SimulationConfig`, exécute le solveur et présente les résultats du paquet.

- **📚 Théorie & Documentation** : théorie illustrée (équations, grille MAC, projection de
  Chorin, obstacles en marches d'escalier, calculateur de stabilité, régimes de sillage et loi
  St(Re)), ce README et le tutoriel, exercices corrigés, code source commenté.
- **🚀 Simulation Interactive** : trois cas (obstacle en soufflerie, cavité entraînée, canal),
  huit scénarios guidés (régimes de sillage, diffusion numérique, stabilité, blocage,
  convergence en maillage, cavité de Re = 100 à 1000, canal, profil d'aile), récapitulatif
  avant calcul (blocage, épaisseur de l'obstacle en mailles, durée estimée, régime attendu) et
  script Python équivalent aux réglages.
- **Résultats** : indicateurs adaptés à l'obstacle (corps non profilé : Cd, Cl rms, St et sa
  fiabilité, Lr ; profil : Cl, Cd, L/D, Cm), lecture commentée (comparaison à la littérature,
  correction de blocage), champs interactifs, efforts et spectre, Cp et sillage, animations à
  la demande, anatomie d'un pas de projection, diagnostics et exports.
- **Historique et études** : les calculs tournent en arrière-plan (page utilisable, aperçu en
  direct, bouton d'arrêt) et sont enregistrés dans `outputs/app_runs/` : réaffichage,
  rechargement des réglages, superposition des courbes, prolongation. Une étude paramétrique
  enchaîne une série de calculs (balayage en Re, en résolution...), avec extrapolation de
  Richardson et indice de convergence (GCI) pour la convergence en maillage.

## Démarrage rapide

```python
import logging
from cfd2d import NavierStokesSolver, presets

logging.basicConfig(level=logging.INFO)
solver = NavierStokesSolver(presets.cylinder_flow(Re=100, cells_per_diameter=20))
solver.run(t_end=150)               # allée de Von Kármán périodique dès t ≈ 50

u, v = solver.cell_velocity()       # vitesses au centre des cellules (Nx, Ny)
p = solver.state.p                  # pression (Nx, Ny)
diag = solver.diagnostics.as_arrays()   # t, dt, CFL, max|div u|, énergie cinétique...
```

Analyse (efforts, spectres, champs dérivés, profils) :

```python
from cfd2d import FieldAverager, ForceMonitor, NavierStokesSolver, presets
from cfd2d import analytics as an

solver = NavierStokesSolver(presets.cylinder_flow(Re=100))
forces = ForceMonitor(solver)                  # Cd(t), Cl(t), Cm(t) à chaque pas
averager = FieldAverager(solver, t_start=75)   # moyennes temporelles sur t ≥ 75
solver.run(t_end=150)

history = forces.history()
print(history.summary())                       # Cd moyen, Cl rms, St (FFT du régime établi détecté)
fields = an.compute_fields(solver)             # ‖u‖, p, ω, critère Q, ∇·u (champs 2D)
mean = averager.fields()
cp = an.surface_pressure(solver, fields=mean)  # <Cp>(θ) le long de la paroi
wake = an.wake_profiles(solver, (1, 2, 5, 10), fields=mean)   # u(y)/U∞ à x/D = 1, 2, 5, 10
print(an.divergence_report(solver))            # suivi de max|∇·u|
history.save("outputs/forces.csv")             # .csv ou .npz
```

Visualisation et exports :

```python
from cfd2d import io, visualization as viz

recorder = viz.FrameRecorder(solver, every=25, t_start=120)   # à créer avant solver.run()
# ... solver.run(...)
viz.plot_dashboard(solver, history, mean_fields=mean, path="outputs/dashboard.png")
viz.plot_dashboard(solver, history, mean_fields=mean, main="streamlines", path="outputs/lignes.png")
viz.animate(recorder, "outputs/vorticite.mp4")                 # .mp4 (ffmpeg) ou .gif (Pillow)
viz.interactive_report(history, "outputs/rapport.html", solver=solver, mean_fields=mean)
io.save_vtk("outputs/champ.vtk", fields, solver.grid)          # à ouvrir dans ParaView
io.VTKSeriesWriter(solver, "outputs/vtk", every=200)           # série temporelle .vtk.series
io.save_checkpoint("outputs/reprise.npz", solver)              # reprise : io.load_checkpoint
```

Exemple complet (cylindre à Re = 100, ~8 min) : `python examples/cylinder_re100.py 20 150`
produit dans `outputs/cylinder_Re100/` le tableau de bord (vorticité et lignes de courant), la
vidéo de l'allée de Von Kármán, le rapport interactif, les tables CSV (efforts, spectre, Cp,
profils), les champs `.npz` / `.vtk` et un point de reprise.

Configuration complète (tout est paramétrable) :

```python
from cfd2d import *

config = SimulationConfig(
    domain=DomainConfig(Lx=20, Ly=10, Nx=400, Ny=200),
    flow=FlowConfig(U_inf=1.0, Re=100, perturbation=0.05),       # ou nu=...
    boundaries=BoundaryConfig(
        west=Inlet(profile="uniform"),                            # ou "parabolic"
        east=Outlet(kind="convective"),                           # ou "neumann"
        south=SlipWall(), north=SlipWall(),                       # ou NoSlipWall(velocity=...)
    ),
    obstacles=[NACA4("2412", chord=1, x_le=4, y_le=5, alpha_deg=8)],
    time=TimeConfig(t_end=50, scheme="ab2", cfl=0.4),             # dt=None : pas adaptatif
    numerics=NumericsConfig(advection="quick", pressure_solver="direct"),
)
NavierStokesSolver(config).run()
```

## Structure du projet

```
src/cfd2d/
├── grid.py          grille MAC : p aux centres, u/v aux faces, cellules fantômes
├── geometry.py      obstacles (cylindre, rectangle/carré, NACA 4 chiffres, image N&B, polygone) → masque binaire
├── boundary.py      conditions aux limites : entrée uniforme/parabolique, sortie Neumann/convective,
│                    parois adhérentes (éventuellement mobiles : cavité entraînée) ou glissantes
├── config.py        dataclasses de configuration (domaine, fluide, temps, schémas)
├── operators.py     advection conservative (upwind, centré, QUICK, TVD van Leer), laplacien, divergence
├── pressure.py      opérateur de Poisson + solveurs : LU direct, CG/BiCGSTAB, SOR rouge-noir, Jacobi
├── solver.py        NavierStokesSolver : projection de Chorin, Euler/AB2/RK3, CFL automatique
├── presets.py       cas types : cylindre, cavité, canal de Poiseuille, profil NACA
├── validation.py    données de référence (Ghia et al. 1982)
├── analytics.py     champs dérivés (‖u‖, ω, critère Q, ψ), efforts Cd/Cl/Cm, Strouhal (FFT),
│                    Cp(θ), profils de sillage, recirculation, moyennes temporelles, ∇·u
├── visualization.py tableau de bord PNG, animations MP4/GIF, rapport HTML interactif (Plotly)
└── io.py            export .npz / .vtk (ParaView), séries temporelles, points de reprise
examples/            mon_ecoulement.py (gabarit : tous les paramètres), cylinder_re100.py
                     (exemple complet), lid_driven_cavity.py (validation)
app.py               application web Streamlit (point d'entrée)
webapp/              code de l'interface :
├── params.py        réglages → SimulationConfig, diagnostic avant calcul (blocage, épaisseur, coût)
├── runner.py        calcul dans un fil d'arrière-plan : moniteurs, progression, arrêt, prolongation
├── analysis.py      post-traitement (appels à cfd2d.analytics) et lecture commentée des résultats
├── results.py       affichage des résultats (indicateurs, sous-onglets) ; figures.py : graphiques
├── history.py       historique, comparaison, études paramétriques (Richardson, GCI)
├── store.py         enregistrement des calculs (outputs/app_runs/), calibration de la durée
├── scenarios.py     scénarios guidés ; codegen.py : script Python équivalent aux réglages
├── docs.py          onglet Théorie & Documentation ; preview.py : schéma du domaine, aperçu en direct
└── common.py        chemins, libellés, mise en forme, charte des figures
.streamlit/          configuration de l'application (thèmes clair et sombre)
tests/               tests unitaires et validations physiques (pytest), tests de l'application
outputs/             résultats
```

Le code est commenté ligne à ligne (rôle de chaque instruction et de chaque fonction appelée),
tests compris.

## Méthode numérique

Un pas de temps :

1. **Prédiction** `u* = uⁿ + Δt·H(u)`, `H = -∇·(u⊗u) + ν∇²u`, intégré explicitement
   (Euler, Adams–Bashforth 2 à pas variable, ou Runge–Kutta SSP-3) ;
2. **Poisson** `∇²p = (ρ/Δt) ∇·u*` sur les cellules fluides ;
3. **Correction** `uⁿ⁺¹ = u* - (Δt/ρ)∇p` : divergence discrète nulle à la précision machine.

- **Obstacles** : toute face touchant une cellule solide a une vitesse nulle ; la pression y
  vérifie une condition de Neumann (l'opérateur de Poisson n'est assemblé que sur le fluide) ;
  l'adhérence tangentielle est imposée dans les termes visqueux par des valeurs miroirs. Les
  poches fluides fermées (intérieur d'un anneau...) sont automatiquement comblées.
- **Pression** : l'opérateur est symétrique défini positif et constant ; le solveur direct le
  factorise une seule fois (ordonnancement symétrique MMD) puis ne fait qu'une descente-remontée
  par pas. Sans sortie (cavité), le problème de Neumann pur est régularisé en fixant une cellule.
- **Pas de temps** : critère combiné `1/Δt = (|u|/Δx + |v|/Δy)/CFL + ν(1/Δx² + 1/Δy²)/Fo`,
  avec des limites propres à chaque schéma. Un `dt` imposé est conservé s'il est stable, réduit
  (avec avertissement) sinon.

## Analyse

- **Efforts** : intégration sur le contour en marches d'escalier de la pression (cellule fluide
  adjacente) et des flux visqueux et convectifs que le solveur échange avec les faces figées du
  corps. Cette somme vérifie exactement le bilan de quantité de mouvement discret : elle coïncide
  avec un bilan indépendant sur un volume de contrôle (`ForceMonitor(..., control_volume=box)`).
  L'intégration « continue » du seul frottement `μ ∂u_t/∂n` sous-estime la traînée d'environ
  10 % sur un contour en escalier, sans converger avec le maillage.
- **Strouhal** : le régime établi est détecté sur les extrema de `Cl(t)` (amplitude stable à 5 %
  près), le signal (pas de temps variable) est rééchantillonné, fenêtré (Hann) et complété de
  zéros avant la FFT ; le pic est localisé par interpolation parabolique. Les passages à zéro
  fournissent une estimation indépendante.
- **Paroi et sillage** : `Cp` échantillonné le long du contour exact de l'obstacle (interpolation
  restreinte au fluide), profils `u(y)/U∞` aux stations `x/D`, demi-largeur et déficit du sillage,
  longueur de recirculation. Utiliser les champs moyens (`FieldAverager`) pour un sillage
  instationnaire.

## Visualisation

- **Tableau de bord** (`plot_dashboard`) : carte de vorticité (ou lignes de courant colorées par
  la vitesse) avec l'obstacle, `Cd(t)` et `Cl(t)` en petits multiples (zone teintée : régime
  analysé), spectre de `Cl` avec le pic de Strouhal, `Cp(θ)`, profils de sillage et indicateurs.
- **Animations** (`FrameRecorder` + `animate`) : MP4 H.264 encodé par ffmpeg (via ffmpeg-python,
  l'exécutable `ffmpeg` doit être dans le PATH) ou GIF (Pillow).
- **Rapport interactif** (`interactive_report`) : les mêmes graphiques en HTML (Plotly), avec
  survol des valeurs ; `report_figure` renvoie la figure pour un notebook Jupyter.
- **ParaView** : `save_vtk` écrit les champs (pression, vitesse, vorticité, critère Q, divergence,
  masque solide) au format VTK legacy ; `VTKSeriesWriter` produit une série temporelle.

## Validation

```bash
pytest                    # suite complète : 121 tests (~1 min 30), application comprise
python examples/lid_driven_cavity.py 64
```

| Cas | Résultat |
|---|---|
| Opérateur de Poisson (solution analytique) | convergence d'ordre 2 (Neumann et Dirichlet) |
| Projection (5 solveurs de pression) | `max\|∇·u\| < 1e-8 × initial`, pressions identiques |
| Canal de Poiseuille, Re = 10 | profil à 0,4 % près, `dp/dx` à 0,8 % |
| Cavité entraînée, Re = 100 (Ghia 1982) | écart max 0,005 (u) et 0,009 (v) sur les axes médians |
| Efforts, Re = 20 stationnaire | contour = bilan de quantité de mouvement à 4 chiffres (Cd = 2,5829) ; Lr/D = 0,93 |
| Cylindre, Re = 100, 20 cellules/D, blocage 10 % | St = 0,178 (FFT et passages à zéro), Cd = 1,496 (bilan : 1,4956), amplitude Cl = 0,357, Cp arrêt 1,06, Cp culot −0,91, Lr/D = 1,42 (milieu infini : St 0,165, Cd 1,33) |

Coût indicatif : ~20 ms par pas sur une grille 400 × 200 (80 000 cellules), soit ~7 min pour
150 unités de temps du cylindre (~8 min avec les moniteurs, les exports et les figures).

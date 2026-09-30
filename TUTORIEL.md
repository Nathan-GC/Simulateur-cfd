# Tutoriel cfd2d : lancer, paramétrer, comprendre

Ce tutoriel se suit dans l'ordre, en trois parties :

| Partie | Objectif | Durée |
|---|---|---|
| **A. Lancer** | installer, vérifier, faire tourner les exemples | 20 min |
| **B. Paramétrer** | changer le Reynolds, l'obstacle, les parois, la résolution... | 30 min |
| **C. Comprendre** | lire le code dans le bon ordre, avec la théorie à chaque étape | quelques heures |

Toutes les commandes se tapent dans un terminal PowerShell ouvert **à la racine du projet**
(le dossier `Simulateur cfd`). Dans VS Code, ce fichier s'affiche mis en forme avec
`Ctrl+Maj+V`.

---

## Partie A : lancer le programme depuis zéro

### A.1 Installer les outils (une fois par ordinateur)

| Outil | À quoi il sert | Installation | Vérification |
|---|---|---|---|
| Python ≥ 3.10 (3.12 conseillé) | exécuter le code | `winget install --id Python.Python.3.12 -e --scope user`, ou [python.org](https://www.python.org/downloads/) en cochant *Add python.exe to PATH* | `py --version` |
| VS Code + extension *Python* (Microsoft) | lire et modifier le code | [code.visualstudio.com](https://code.visualstudio.com/) | `code --version` |
| ffmpeg (optionnel) | vidéos MP4 (sinon : GIF) | `winget install --id Gyan.FFmpeg -e` | `ffmpeg -version` |
| ParaView (optionnel) | explorer les champs `.vtk` en 3D/2D | [paraview.org/download](https://www.paraview.org/download/) | — |

> Sur ton PC, Python 3.12, VS Code et ffmpeg sont déjà installés : passe directement à A.2.
> Après une installation, ferme et rouvre le terminal pour que les nouvelles commandes soient trouvées.

### A.2 Ouvrir le projet

1. VS Code : *Fichier > Ouvrir le dossier...*, choisir `Simulateur cfd`.
2. Ouvrir un terminal : *Terminal > Nouveau terminal*. L'invite doit se terminer par `Simulateur cfd>`.

### A.3 Créer l'environnement Python du projet (une fois par projet)

Un *environnement virtuel* (`.venv`) est une copie de Python propre au projet, avec ses
propres bibliothèques, isolée du reste de l'ordinateur.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```

1. crée le dossier `.venv` ;
2. met à jour l'installateur de paquets `pip` ;
3. installe les dépendances listées dans [requirements.txt](requirements.txt) : numpy (tableaux),
   scipy (matrices creuses, FFT), matplotlib (figures), plotly (rapport interactif),
   ffmpeg-python (vidéos), pytest (tests) ;
4. installe le paquet `cfd2d` lui-même en mode « éditable » (`-e`) : toute modification des
   fichiers de `src/cfd2d/` est prise en compte immédiatement, sans réinstaller.

> Sur ton PC, c'est déjà fait (le dossier `.venv` existe).

**Pourquoi écrire `.venv\Scripts\python` ?** C'est le Python de l'environnement virtuel :
l'appeler directement fonctionne toujours. Optionnellement, on peut « activer »
l'environnement pour taper simplement `python` :

```powershell
.venv\Scripts\Activate.ps1
```

Si PowerShell répond *« l'exécution de scripts est désactivée sur ce système »*, autoriser
une fois pour toutes les scripts locaux de ton compte, puis réessayer :

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Dans VS Code, `Ctrl+Maj+P` > *Python: Select Interpreter* > `.venv` fait utiliser cet
environnement par l'éditeur (autocomplétion, débogueur, bouton ▶).

### A.4 Vérifier l'installation : lancer les tests

```powershell
.venv\Scripts\python -m pytest
```

Résultat attendu, en une minute environ : `95 passed`. Ces tests vérifient le code sur des cas
dont on connaît la réponse exacte : parabole de Poiseuille dans un canal, cavité entraînée
comparée aux données de Ghia et al. (1982), convergence d'ordre 2 de l'équation de Poisson,
divergence nulle à 10⁻¹⁵ près...

Pour n'exécuter qu'un fichier, avec le nom de chaque test : `.venv\Scripts\python -m pytest tests/test_grid.py -v`.

### A.5 Premier calcul (moins d'une minute) : la cavité entraînée

Un fluide dans une boîte carrée, mis en mouvement par le couvercle qui glisse :

```powershell
.venv\Scripts\python examples\lid_driven_cavity.py 64
```

Pendant le calcul, une ligne de progression s'affiche régulièrement :

```
pas    2000 | t =  14.1815 | dt = 7.289e-03 | CFL = 0.296 | max|div| = 5.00e-15 | Ec = 104.5 | 50 pas/s
```

| Champ | Signification |
|---|---|
| `pas` | numéro du pas de temps |
| `t` | temps physique simulé |
| `dt` | pas de temps, recalculé à chaque pas pour rester stable |
| `CFL` | nombre de Courant : distance parcourue par le fluide en un pas, en mailles (< 0.4 en AB2) |
| `max\|div\|` | écart maximal à l'incompressibilité ∇·u = 0 (10⁻¹⁵ = parfait) |
| `Ec` | énergie cinétique totale : se stabilise quand l'écoulement est établi |
| `pas/s` | vitesse de calcul |

Résultat : `outputs/cavity_Re100_N64.png`, profils de vitesse superposés aux points de
référence de Ghia et al., et lignes de courant.

### A.6 Le cas complet : allée de Von Kármán derrière un cylindre

```powershell
.venv\Scripts\python examples\cylinder_re100.py 20 150
```

Les deux arguments sont la résolution (cellules par diamètre) et la durée simulée. Compter
environ 8 minutes ; pour un essai rapide (≈ 1 min, moins précis) : `... cylinder_re100.py 12 80`.

À la fin, la console affiche la synthèse (traînée moyenne Cd, portance Cl, nombre de
Strouhal St, pression pariétale, sillage, divergence) et tout est écrit dans
`outputs/cylinder_Re100/` :

| Fichier | Contenu | Comment l'ouvrir |
|---|---|---|
| `dashboard.png` | tableau de bord : vorticité, Cd(t), Cl(t), spectre, Cp(θ), profils de sillage | double-clic (ou dans VS Code) |
| `dashboard_streamlines.png` | idem avec les lignes de courant | idem |
| `vorticity.mp4` | animation des 30 dernières unités de temps | lecteur vidéo |
| `report.html` | rapport interactif : survol des valeurs, zoom | double-clic → navigateur |
| `forces.csv`, `spectrum_cl.csv`, `cp_mean.csv`, `wake_profiles.csv` | données tabulées | Excel/LibreOffice (séparateur : virgule), pandas |
| `fields_final.npz`, `fields_mean.npz` | champs 2D instantanés et moyens | `numpy.load(...)` |
| `flow_final.vtk`, `flow_mean.vtk` | mêmes champs pour ParaView | ParaView (voir A.7) |
| `checkpoint.npz` | état final, pour prolonger le calcul | voir B.5 |

### A.7 (Optionnel) Explorer les champs dans ParaView

1. *File > Open* > `outputs/cylinder_Re100/flow_final.vtk`, puis **Apply**.
2. Dans la liste de coloration (barre d'outils), choisir `vorticity` (ou `pressure`, `speed`...).
3. Masquer l'obstacle : *Filters > Alphabetical > Threshold*, champ `solid`, bornes 0 à 0, **Apply**.
4. Flèches de vitesse : *Filters > Glyph* sur `velocity`.

Pour une animation dans ParaView, enregistrer une série pendant le calcul avec
`cfd2d.io.VTKSeriesWriter(solver, "outputs/vtk", every=200)` puis ouvrir le fichier
`flow.vtk.series` produit.

---

## Partie B : modifier les paramètres de l'écoulement

### B.1 Le plus simple : le gabarit `examples/mon_ecoulement.py`

Tous les paramètres y sont réunis dans une section commentée ; on modifie les valeurs puis on
relance :

```powershell
.venv\Scripts\python examples\mon_ecoulement.py
```

Avec les valeurs par défaut (cylindre, Re = 100, 16 cellules par diamètre, t = 80), le calcul
prend environ 1 min 30. Les résultats vont dans `outputs/<NOM>/`. Change `NOM` à chaque
expérience pour ne pas écraser la précédente.

| Paramètre du gabarit | Rôle | Valeurs typiques |
|---|---|---|
| `REYNOLDS` | nombre de Reynolds Re = U D / ν | 20 à 200 |
| `U_INF` | vitesse amont | 1 |
| `PERTURBATION` | déclencheur du lâcher tourbillonnaire | 0.05 (0 = symétrique) |
| `LONGUEUR`, `HAUTEUR` | taille du domaine (en D) | 16 D x 8 D |
| `CELLULES_PAR_D` | résolution | 12 (rapide) à 30 (précis) |
| `X_OBSTACLE`, `Y_OBSTACLE`, `OBSTACLE` | forme et position de l'obstacle | cylindre, carré, plaque, NACA |
| `ENTREE`, `SORTIE`, `PAROIS` | conditions aux limites | voir recette 4 |
| `T_FINAL` | durée simulée (unités D/U) | 80 à 200 |
| `SCHEMA_TEMPS`, `CFL` | intégration en temps | `"ab2"`, `None` |
| `ADVECTION`, `SOLVEUR_PRESSION` | méthodes numériques | `"quick"`, `"direct"` |
| `DEBUT_MOYENNES`, `DUREE_ANIMATION` | post-traitement | moitié de T_FINAL, 20 |

### B.2 Recettes

**Recette 1 : changer le nombre de Reynolds.** Pour un cylindre :

- `REYNOLDS = 40` avec `PERTURBATION = 0` : écoulement **stationnaire**, deux tourbillons
  attachés derrière le cylindre (longueur ≈ 2,2 D, Cd ≈ 1,9). La portance est nulle et le
  tableau de bord affiche « écoulement stationnaire » à la place du spectre. Si on garde
  `PERTURBATION = 0.05`, la perturbation s'amortit lentement sous le seuil critique : à t = 80,
  une faible oscillation subsiste et est encore détectée.
- `REYNOLDS = 100` : **allée de Von Kármán** périodique, St ≈ 0,18 dans le domaine du gabarit
  (0,165 en milieu infini : l'écart vient du confinement, voir recette 5).
- `REYNOLDS = 150` : lâcher plus intense, St ≈ 0,19.

Le seuil d'apparition du lâcher est Re ≈ 47. Au-delà de Re ≈ 190, l'écoulement réel devient
tridimensionnel : un calcul 2D reste possible mais n'est plus réaliste. Plus Re est grand, plus
les couches limites sont fines (épaisseur ∝ D/√Re) : prévoir ≥ 20 cellules/D à Re = 150 et
≥ 30 au-delà de 200.

**Recette 2 : changer la vitesse.** À `REYNOLDS` fixé, la viscosité s'ajuste
(ν = U D / Re) : l'écoulement adimensionné est le même (même St, même Cd), seule l'échelle de
temps change. Avec `U_INF = 2`, tout va deux fois plus vite : `T_FINAL` peut être divisé par 2.
Pour imposer la viscosité du fluide plutôt que Re, écrire `FlowConfig(U_inf=U_INF, nu=0.01)`
dans `build_config()` : Re devient une conséquence.

**Recette 3 : changer l'obstacle.** Garder une seule ligne `OBSTACLE = ...` :

```python
OBSTACLE = Rectangle.square(X_OBSTACLE, Y_OBSTACLE, D)                       # carré
OBSTACLE = Rectangle(X_OBSTACLE, Y_OBSTACLE, 2 * D, 0.4 * D, angle_deg=-20)  # plaque inclinée
OBSTACLE = NACA4("2412", chord=D, x_le=X_OBSTACLE - 0.25 * D, y_le=Y_OBSTACLE, alpha_deg=8)
```

- Forme libre : dessiner une forme noire sur fond blanc (PNG), puis
  `OBSTACLE = ImageObstacle("ma_forme.png", xc=4, yc=4, width=1.5)`.
- Polygone : `OBSTACLE = PolygonObstacle([[x1, y1], [x2, y2], ...])`.
- Pour ces deux formes, ajouter `ImageObstacle` ou `PolygonObstacle` à la liste des imports
  `from cfd2d import (...)` en haut du fichier.
- Plusieurs obstacles : `obstacles=[OBSTACLE_1, OBSTACLE_2]` dans `build_config()`. Les efforts
  sont calculés pour le premier ; `ForceMonitor(solver, obstacle=1)` suit le second.

La longueur de référence (Re, Cd, St) est le diamètre pour un cylindre, la hauteur frontale
pour un rectangle ou une image, la corde pour un profil NACA.

**Recette 4 : soufflerie ou canal.** Soufflerie idéale (défaut) : `PAROIS = SlipWall()`.
Canal à parois adhérentes avec profil d'entrée de Poiseuille (proche du cas-test de Schäfer
et Turek) :

```python
ENTREE = Inlet(profile="parabolic")
PAROIS = NoSlipWall()
```

- Sortie : `Outlet(kind="convective")` laisse sortir les tourbillons sans réflexion ;
  `"neumann"` impose simplement un gradient de vitesse nul.
- Démarrage en douceur : `Inlet(ramp_time=2.0)` fait monter la vitesse en 2 unités de temps.

**Recette 5 : taille du domaine.** Le rapport D/HAUTEUR est le *taux de blocage* : 1/8 =
12,5 % par défaut, ce qui augmente St et Cd de quelques %. Pour s'en approcher du milieu infini
(St ≈ 0,165 à Re = 100), augmenter `HAUTEUR` (16 D ou plus). Garder au moins 4 D entre
l'entrée et l'obstacle et 10 D derrière lui.

**Recette 6 : résolution et temps de calcul.** Le coût croît comme le cube de
`CELLULES_PAR_D` : deux fois plus de cellules dans chaque direction, et des pas de temps deux
fois plus petits. Ordres de grandeur pour le gabarit (T = 80) :

| `CELLULES_PAR_D` | 12 | 16 | 20 | 30 |
|---|---|---|---|---|
| durée | ≈ 45 s | ≈ 1 min 30 | ≈ 3 min | ≈ 10 min |

Une bonne pratique : refaire le même calcul à deux résolutions et vérifier que St et Cd
bougent peu (étude de convergence).

**Recette 7 : méthodes numériques.** À explorer pour voir leur effet :

- `ADVECTION = "upwind"` : très robuste mais **diffusif**. La diffusion numérique
  (≈ U Δx / 2) s'ajoute à la viscosité, et à Re = 100 le lâcher tourbillonnaire est affaibli :
  une expérience très parlante.
- `ADVECTION = "central"` : précis, mais oscille si la maille est trop grossière
  (quand U Δx / ν > 2).
- `SCHEMA_TEMPS = "rk3"` : plus robuste (CFL par défaut 0.8) ; `"euler"` : ordre 1, pédagogique.
- `SOLVEUR_PRESSION = "sor"` ou `"jacobi"` : méthodes itératives classiques, beaucoup plus
  lentes que `"direct"`, utiles pour comparer.

**Recette 8 : durée et moyennes.** À Re = 100, l'allée s'établit vers t ≈ 50 D/U. Le nombre de
Strouhal est fiable à partir d'une dizaine de périodes analysées : T_FINAL ≥ 120. La synthèse
affiche le nombre de périodes utilisées (« 5.7 periodes »). `DEBUT_MOYENNES` doit tomber dans le
régime établi.

### B.3 Écrire sa propre configuration

Le gabarit ne fait qu'assembler une `SimulationConfig` ([src/cfd2d/config.py](src/cfd2d/config.py)).
Voici tous les blocs et leurs paramètres :

```python
from cfd2d import *

config = SimulationConfig(
    name="mon_cas",
    domain=DomainConfig(Lx=20, Ly=10, Nx=400, Ny=200),      # domaine et grille
    flow=FlowConfig(U_inf=1.0, rho=1.0, Re=100, perturbation=0.05),   # ou nu=... à la place de Re
    boundaries=BoundaryConfig(
        west=Inlet(velocity=None, profile="uniform", ramp_time=0.0),   # velocity=None -> U_inf
        east=Outlet(kind="convective"),
        south=SlipWall(),
        north=SlipWall(),                                   # NoSlipWall(velocity=1.0) = paroi mobile
    ),
    obstacles=[Cylinder(5, 5, 1)],
    time=TimeConfig(t_end=150, dt=None, cfl=None, scheme="ab2"),       # dt=None : pas adaptatif
    numerics=NumericsConfig(advection="quick", pressure_solver="direct", mask_supersampling=4),
)
solver = NavierStokesSolver(config)
solver.run()
```

| Bloc | Paramètre | Défaut | Rôle |
|---|---|---|---|
| `DomainConfig` | `Lx`, `Ly`, `Nx`, `Ny` | 1, 1, 64, 64 | dimensions et nombre de cellules (garder Lx/Nx = Ly/Ny) |
| `FlowConfig` | `U_inf`, `rho` | 1, 1 | vitesse de référence, masse volumique |
| | `Re` *ou* `nu` | — | exactement l'un des deux |
| | `L_ref` | obstacle | longueur de référence imposée |
| | `perturbation` | 0 | perturbation initiale |
| `TimeConfig` | `t_end` | 10 | durée simulée |
| | `dt` | None | pas imposé (réduit automatiquement s'il est instable) |
| | `cfl`, `fourier` | None | limites de stabilité (None : valeurs sûres du schéma) |
| | `scheme` | `"ab2"` | `"euler"`, `"ab2"`, `"rk3"` |
| `NumericsConfig` | `advection` | `"quick"` | `"upwind"`, `"central"`, `"quick"`, `"tvd"` |
| | `pressure_solver` | `"direct"` | `"direct"`, `"cg"`, `"bicgstab"`, `"sor"`, `"jacobi"` |
| | `pressure_tol`, `pressure_maxiter` | 1e-8, 10000 | solveurs itératifs |
| | `preconditioner` | `"ilu"` | `"none"`, `"jacobi"`, `"ilu"`, `"amg"` (paquet pyamg) |
| | `mask_supersampling` | 4 | sous-points par cellule pour construire le masque |

Configurations toutes faites dans [src/cfd2d/presets.py](src/cfd2d/presets.py), réglables par
leurs arguments :

```python
from cfd2d import NavierStokesSolver, presets

presets.cylinder_flow(Re=150, cells_per_diameter=24, height=16, t_end=200)
presets.naca_airfoil(code="0012", alpha_deg=10, Re=1000, cells_per_chord=64)
presets.channel_flow(Re=10, profile="uniform")
presets.lid_driven_cavity(Re=400, N=96)
```

### B.4 Moniteurs : ce qui est enregistré pendant le calcul

Les moniteurs sont appelés automatiquement à chaque pas (ou tous les `every` pas). Ils doivent
être créés **avant** `solver.run()` :

```python
from cfd2d import FieldAverager, ForceMonitor
from cfd2d import io, visualization as viz

forces = ForceMonitor(solver)                              # Cd(t), Cl(t), Cm(t)
averager = FieldAverager(solver, t_start=75, every=5)      # champs moyens
recorder = viz.FrameRecorder(solver, every=25, t_start=120)   # images d'animation
series = io.VTKSeriesWriter(solver, "outputs/vtk", every=200, t_start=100)   # série ParaView
solver.run()
```

### B.5 Reprendre un calcul

```python
from cfd2d import NavierStokesSolver, io, presets

solver = NavierStokesSolver(presets.cylinder_flow(Re=100, cells_per_diameter=20))  # même configuration
io.load_checkpoint("outputs/cylinder_Re100/checkpoint.npz", solver)
solver.run(t_end=250)          # repart de t = 150
```

### B.6 Messages d'alerte

| Message | Cause | Que faire |
|---|---|---|
| `Obstacle ... peu resolu` | moins de 8 mailles par longueur de référence | augmenter la résolution |
| `dt impose ... > dt stable ... dt reduit` | `dt` imposé trop grand | laisser `dt=None` |
| `Seulement X periodes analysees` | calcul trop court pour le spectre | augmenter `T_FINAL` |
| `Regime etabli non detecte` | oscillations pas encore stabilisées | augmenter `T_FINAL` |
| `Solveur de pression non converge` | solveur itératif limité par `pressure_maxiter` | `"direct"`, ou augmenter `pressure_maxiter` |
| `cellule(s) fluide(s) isolee(s)` | poche de fluide fermée dans un obstacle, comblée | information seulement |
| `Station x/L = ... hors du domaine` | profil de sillage au-delà de la sortie | allonger le domaine |
| `SimulationDivergedError` | instabilité numérique | réduire `cfl`, raffiner, `advection="tvd"` ou `"upwind"` |

---

## Partie C : comprendre la théorie et le code

### C.0 Vue d'ensemble

```
 SimulationConfig ....................... config.py      ← les paramètres (Partie B)
        │
        ▼
 NavierStokesSolver(config) ............. solver.py      préparation (une seule fois)
   ├─ StaggeredGrid ..................... grid.py        la grille décalée MAC
   ├─ build_mask ........................ geometry.py    obstacles → masque binaire
   ├─ BoundaryHandler ................... boundary.py    conditions aux limites
   ├─ _build_topology ................... solver.py      faces actives / figées / de sortie
   ├─ PoissonOperator + factorisation ... pressure.py    matrice de pression
   └─ set_initial_velocity .............. solver.py      champ initial projeté (∇·u = 0)
        │
        ▼
 solver.run() : boucle en temps ......... solver.py
   ┌──► step(dt)
   │      ├─ _rhs ....................... operators.py   advection + diffusion
   │      ├─ prédiction u* .............. solver.py      Euler / AB2 / RK3
   │      ├─ _finish_stage .............. boundary.py    conditions aux limites, obstacle
   │      ├─ _project ................... pressure.py    Poisson puis correction
   │      ├─ _record .................... solver.py      dt, CFL, max|div|, énergie
   │      └─ moniteurs .................. analytics.py, visualization.py, io.py
   └─── tant que t < t_end
        │
        ▼
 Post-traitement ........................ analytics.py → visualization.py, io.py
```

Le code est commenté **ligne à ligne**. Pour chaque fichier : lire d'abord le docstring en
tête (le « pourquoi »), puis les fonctions dans l'ordre indiqué ci-dessous.

### C.1 Étape 1 : les équations et la méthode

**À lire :** [README.md](README.md) (section *Méthode numérique*) et le docstring en tête de
[src/cfd2d/solver.py](src/cfd2d/solver.py).

**Théorie.** Écoulement incompressible, ρ et ν constants :

$$
\frac{\partial \mathbf{u}}{\partial t} + \nabla\cdot(\mathbf{u}\otimes\mathbf{u})
= -\frac{1}{\rho}\nabla p + \nu\,\nabla^2 \mathbf{u},
\qquad \nabla\cdot\mathbf{u} = 0
$$

- **Nombre de Reynolds.** Rapport des effets d'inertie aux effets visqueux :
  Re = U L / ν. Deux écoulements de même Re et de même géométrie sont semblables.
- **Rôle de la pression.** Il n'y a pas d'équation d'évolution pour p. La pression est la
  grandeur qui s'ajuste à chaque instant pour que le champ reste à divergence nulle.
- **Méthode de projection** (Chorin, 1968), un pas de temps en trois temps :
  1. *prédiction*, sans pression : u* = uⁿ + Δt [−∇·(u⊗u) + ν∇²u] ;
  2. *Poisson* : on cherche p tel que la vitesse corrigée soit à divergence nulle ;
  3. *correction* : uⁿ⁺¹ = u* − (Δt/ρ) ∇p.

En prenant la divergence de la correction et en imposant ∇·uⁿ⁺¹ = 0, on obtient
l'équation de l'étape 2 :

$$
\nabla^2 p = \frac{\rho}{\Delta t}\,\nabla\cdot\mathbf{u}^*
$$

### C.2 Étape 2 : la grille décalée (MAC)

**À lire :** [src/cfd2d/grid.py](src/cfd2d/grid.py) (court). **Test associé :**
[tests/test_grid.py](tests/test_grid.py).

**Théorie.** Si p, u et v étaient stockés au même point, un champ de pression en damier
(+1, −1, +1...) aurait un gradient discret nul : il serait invisible pour le solveur et
polluerait la solution. La grille décalée MAC (Harlow et Welch, 1965) l'évite :

```
                 v[i, j+1]
             ┌───────↑───────┐
             │               │
    u[i, j] →│    p[i, j]    │→ u[i+1, j]
             │               │
             └───────↑───────┘
                  v[i, j]
```

- la pression est au centre de chaque cellule, u sur les faces verticales, v sur les faces
  horizontales ;
- le bilan de masse d'une cellule, (u_est − u_ouest)/Δx + (v_nord − v_sud)/Δy, utilise
  exactement ses quatre faces ;
- dans le code, les tableaux u et v ont une rangée de **cellules fantômes** de chaque côté
  (d'où les formes `(Nx+1, Ny+2)` et `(Nx+2, Ny+1)`) : elles servent à imposer les conditions
  aux limites. Le tableau en tête du docstring de `StaggeredGrid` résume les positions.

### C.3 Étape 3 : les obstacles (frontière immergée)

**À lire :** [src/cfd2d/geometry.py](src/cfd2d/geometry.py), dans l'ordre : `Obstacle`,
`rasterize`, `Cylinder`, `naca4_coordinates` et `NACA4`, `ImageObstacle`, `fill_isolated_fluid`,
`build_mask`. **Test associé :** [tests/test_geometry.py](tests/test_geometry.py).

**Théorie.** La grille reste cartésienne : l'obstacle n'est pas maillé mais « immergé ».
Un **masque binaire** marque les cellules solides ; la paroi devient un escalier qui suit les
faces de la grille (précision géométrique d'ordre Δx). Chaque cellule est échantillonnée en
4x4 points et déclarée solide si au moins la moitié l'est. Une poche de fluide enfermée dans
un obstacle est comblée : elle rendrait l'équation de Poisson insoluble (sa pression ne serait
définie qu'à une constante près).

### C.4 Étape 4 : les conditions aux limites

**À lire :** [src/cfd2d/boundary.py](src/cfd2d/boundary.py) : `Side`, les classes de
conditions (`Inlet`, `Outlet`, `NoSlipWall`, `SlipWall`), puis `BoundaryHandler` (`views`,
`apply_normal`, `apply_ghosts`, `extrapolate_outlets`, `add_outlet_tendency`).

**Théorie.**

- **Dirichlet** : on impose la valeur (vitesse d'entrée, vitesse nulle sur une paroi).
  **Neumann** : on impose la dérivée normale (dérivée nulle en sortie, sur une paroi
  glissante).
- La composante **normale** de la vitesse est portée par les faces du bord : elle s'impose
  directement.
- La composante **tangentielle** s'impose par la cellule fantôme, à une demi-maille hors du
  domaine. Pour que la paroi, située au milieu, ait la vitesse V :
  fantôme = 2V − intérieur (**valeur miroir**) ; pour un gradient nul : fantôme = intérieur.
- **Sortie** : pression imposée p = 0 et vitesse extrapolée. La variante *convective*
  (∂u/∂t + U ∂u/∂n = 0) transporte les tourbillons hors du domaine sans qu'ils rebondissent.

### C.5 Étape 5 : les opérateurs discrets

**À lire :** [src/cfd2d/operators.py](src/cfd2d/operators.py) dans l'ordre `reconstruct` →
`advective_fluxes` → `advection` → `laplacian_u`/`laplacian_v` → `divergence`. **Test
associé :** [tests/test_operators.py](tests/test_operators.py).

**Théorie (volumes finis).**

- Chaque inconnue u ou v possède son propre petit volume de contrôle, centré sur sa face.
- **Advection** sous forme conservative : ∇·(u⊗u) = somme des flux sortants par les faces.
  Toute la difficulté est d'estimer la quantité transportée sur une face :
  - `upwind` : valeur amont, ordre 1. Stable, mais ajoute une viscosité numérique ≈ U Δx / 2 ;
  - `central` : moyenne, ordre 2, mais oscillations quand le *Reynolds de maille*
    U Δx / ν dépasse 2 ;
  - `quick` : parabole passant par 3 nœuds pris côté amont (Leonard, 1979), ordre 3 ;
  - `tvd` : pente limitée (van Leer), ordre 2 sans oscillation.
- **Diffusion** : laplacien à 5 points (u_E − 2u_P + u_W)/Δx² + ... Près d'un obstacle, le
  voisin situé dans le solide est remplacé par −u_P (valeur miroir) pour que la vitesse
  s'annule sur la paroi.

### C.6 Étape 6 : l'équation de pression

**À lire :** [src/cfd2d/pressure.py](src/cfd2d/pressure.py) : `PoissonOperator.__init__`
(assemblage face par face), `rhs`, puis les solveurs `DirectPressureSolver`,
`KrylovPressureSolver`, `SORPressureSolver`. **Test associé :**
[tests/test_pressure.py](tests/test_pressure.py) (convergence d'ordre 2, divergence ramenée à
10⁻¹⁵).

**Théorie.**

- Après discrétisation, l'équation de Poisson devient un système linéaire **A p = b** :
  une ligne par cellule, au plus 5 coefficients non nuls par ligne (matrice creuse).
- A est construite comme « divergence du gradient » restreinte aux faces qui peuvent bouger.
  Elle est donc **symétrique définie positive**, et la vitesse projetée est exactement à
  divergence nulle.
- Conditions aux limites de la pression :
  - flux nul (Neumann) sur les parois et les obstacles ;
  - p = 0 (Dirichlet) en sortie ;
  - sans sortie (cavité), le problème n'est défini qu'à une constante près : on fixe la
    pression d'une cellule.
- Résolution :
  - **LU directe** (défaut) : factorisation A = LU faite une seule fois (A ne change pas),
    puis deux remontées triangulaires par pas ;
  - **gradient conjugué** préconditionné ;
  - **Jacobi** et **SOR** : méthodes itératives classiques, lentes (nombre d'itérations qui
    croît avec la taille de la grille), présentes pour comparer.

### C.7 Étape 7 : le solveur (assemblage de tout)

**À lire :** [src/cfd2d/config.py](src/cfd2d/config.py) (les paramètres), puis
[src/cfd2d/solver.py](src/cfd2d/solver.py) dans cet ordre :

1. `step` : un pas de temps complet ;
2. `_advance_multistep` : prédiction Euler / Adams–Bashforth 2 ;
3. `_rhs` : tendance −∇·(u⊗u) + ν∇²u ;
4. `_finish_stage` : conditions aux limites puis projection ;
5. `_project` : Poisson et correction ;
6. `stable_dt`, `next_dt` : choix du pas de temps ;
7. `__init__`, `_build_topology` : préparation ;
8. `set_initial_velocity`, `run`, `_record`.

**Test associé :** [tests/test_solver.py](tests/test_solver.py) (écoulement uniforme conservé,
Poiseuille, Ghia).

**Théorie.**

- **Schémas en temps :**
  - Euler : ordre 1 ;
  - Adams–Bashforth 2 : ordre 2, réutilise la tendance du pas précédent ; coefficients
    adaptés si le pas varie ;
  - Runge–Kutta SSP-3 : trois étages, une projection par étage.
- **Stabilité** d'un schéma explicite :
  - *nombre de Courant* (U Δt / Δx) : le fluide ne doit pas traverser plus d'une fraction de
    maille par pas ;
  - *nombre de Fourier* (ν Δt / Δx²) : la diffusion ne doit pas aller plus vite que ce que le
    pas permet.
  - Le pas de temps respecte les deux, recalculé à chaque itération.
- **Obstacle dans la boucle :** les faces qui touchent le solide sont remises à zéro
  (adhérence et imperméabilité), et elles sont exclues de la correction de pression.
- **Condition initiale :** elle est projetée pour être à divergence nulle. Une petite
  perturbation non symétrique déclenche plus tôt l'instabilité de sillage.

### C.8 Étape 8 : l'analyse des résultats

**À lire :** [src/cfd2d/analytics.py](src/cfd2d/analytics.py), section par section (le
docstring en tête donne le plan). **Test associé :**
[tests/test_analytics.py](tests/test_analytics.py).

**Théorie.**

1. **Champs dérivés :**
   - vorticité ω = ∂v/∂x − ∂u/∂y, qui mesure la rotation locale ;
   - critère Q = ½(‖Ω‖² − ‖S‖²) : Q > 0 là où la rotation l'emporte sur la déformation
     (cœurs de tourbillons) ;
   - fonction de courant ψ : ses lignes de niveau sont les lignes de courant.
2. **Efforts.** La force exercée sur le corps est l'intégrale de la pression et du frottement
   sur sa paroi. Sur un contour en escalier, il faut sommer **exactement** les échanges
   discrets du solveur, sinon la traînée est sous-estimée d'environ 10 %. Le bilan de
   quantité de mouvement sur un volume de contrôle vérifie ce calcul :
   F = ∮σ·n dS − ∮ρu(u·n) dS − d/dt ∫ρu dV. Coefficients : C = F / (½ ρ U² L).
3. **Nombre de Strouhal**, St = f D / U, avec f la fréquence du lâcher tourbillonnaire,
   mesurée par FFT sur Cl(t) :
   - on détecte d'abord le régime établi ;
   - on rééchantillonne le signal (le pas de temps varie) ;
   - on applique une fenêtre de Hann et on complète de zéros ;
   - on interpole paraboliquement autour du pic.
4. **Profils :** Cp(θ) le long de la paroi, profils de vitesse du sillage, longueur de la
   bulle de recirculation. Pour un sillage instationnaire, on les calcule sur le champ
   **moyenné en temps** (`FieldAverager`).
5. **Moniteurs :** objets appelés à chaque pas par le solveur (mécanisme de *callback*).
6. **Incompressibilité :** suivi de max|∇·u|.

### C.9 Étape 9 : visualisation et fichiers

**À lire :** [src/cfd2d/visualization.py](src/cfd2d/visualization.py) (`plot_dashboard`, puis
`FrameRecorder`/`animate`, puis `report_figure`) et [src/cfd2d/io.py](src/cfd2d/io.py)
(`save_vtk`, `VTKSeriesWriter`, points de reprise). **Tests :**
[tests/test_visualization.py](tests/test_visualization.py), [tests/test_io.py](tests/test_io.py).

### C.10 Étape 10 : les exemples, de bout en bout

1. [examples/lid_driven_cavity.py](examples/lid_driven_cavity.py) : un calcul et sa validation ;
2. [examples/mon_ecoulement.py](examples/mon_ecoulement.py) : la configuration explicite ;
3. [examples/cylinder_re100.py](examples/cylinder_re100.py) : toute la chaîne (moniteurs,
   analyse, figures, exports).

### C.11 Outils pour explorer le code

**VS Code :**
- `F12` : aller à la définition d'une fonction ;
- `Maj+F12` : trouver toutes ses utilisations ;
- `Ctrl+Maj+O` : liste des fonctions du fichier ;
- survol à la souris : affiche le docstring.

**Débogueur.** Cliquer dans la marge à gauche d'une ligne de `solver.py` (par exemple dans
`_project`) pour poser un point d'arrêt. Ouvrir ensuite un exemple et lancer
*Exécuter > Démarrer le débogage* (`F5`). À l'arrêt, les tableaux `u`, `v`, `p` sont
inspectables.

**Console interactive** (`.venv\Scripts\python`) pour manipuler les objets :

```python
>>> from cfd2d import NavierStokesSolver, presets
>>> s = NavierStokesSolver(presets.cylinder_flow(cells_per_diameter=8))
>>> s.grid.shape_u, s.grid.shape_v, s.grid.shape_p   # formes des tableaux MAC
>>> s.solid.sum()                                      # nombre de cellules solides
>>> s.poisson.matrix.shape, s.poisson.matrix.nnz       # taille et remplissage de A
>>> s.step()                                           # un pas de temps
>>> s.diagnostics.divergence_max[-1]                   # ≈ 1e-15
```

### C.12 Exercices pour vérifier sa compréhension

1. **Diffusion numérique.** Cavité N = 32 avec `advection="upwind"` puis `"quick"`, en
   comparant `validation.ghia_errors(solver)`. Refaire le cylindre à Re = 100 en `"upwind"` :
   que devient le lâcher tourbillonnaire ?
2. **Transition.** Cylindre à Re = 30, 40, 50, 60, avec la perturbation par défaut et un
   T_FINAL long (150). Pour quels Re l'oscillation de la portance s'amortit-elle, et pour
   lesquels s'amplifie-t-elle ? Mesurer la longueur de recirculation à Re = 40
   (≈ 2,2 D en milieu infini).
3. **Loi St(Re).** Tracer St pour Re = 60, 80, 100, 150 et comparer à la loi de Williamson
   (1988) : St = 0,1816 − 3,3265/Re + 1,6·10⁻⁴ Re (milieu infini ; le confinement augmente St
   de quelques %).
4. **Convergence en maillage.** Cd et St pour 12, 16, 20 et 24 cellules/D.
5. **Solveurs de pression.** Chronométrer une cavité N = 48 avec `"direct"`, `"cg"` et
   `"sor"` ; comparer `solver.diagnostics.pressure_iterations`.
6. **Profil d'aile.** NACA 0012 à Re = 1000 pour α = 0, 4 et 8° : Cl augmente-t-il
   linéairement avec α ?
7. **Nouvel obstacle.** Écrire une classe `Ellipse(Obstacle)` sur le modèle de `Cylinder`
   (méthode `contains` et propriétés `reference_length`, `center`, `bounds`), puis un test.

### C.13 Pour aller plus loin (références)

- F. H. Harlow et J. E. Welch (1965), *Phys. Fluids* 8, 2182 : la grille MAC.
- A. J. Chorin (1968), *Math. Comp.* 22, 745 : la méthode de projection.
- M. Griebel, T. Dornseifer et T. Neunhoeffer (1998), *Numerical Simulation in Fluid
  Dynamics: A Practical Introduction*, SIAM. Le livre le plus proche de ce code (MAC, Chorin,
  obstacles marqués sur la grille).
- J. H. Ferziger, M. Perić et R. L. Street, *Computational Methods for Fluid Dynamics*,
  Springer : volumes finis et méthodes de projection en détail.
- B. P. Leonard (1979), *Comput. Methods Appl. Mech. Eng.* 19, 59 : le schéma QUICK.
- R. Mittal et G. Iaccarino (2005), *Annu. Rev. Fluid Mech.* 37, 239 : les frontières immergées.
- U. Ghia, K. N. Ghia et C. T. Shin (1982), *J. Comput. Phys.* 48, 387 : la cavité de référence.
- C. H. K. Williamson (1996), *Annu. Rev. Fluid Mech.* 28, 477 : le sillage du cylindre.

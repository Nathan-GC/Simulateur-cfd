"""Du bouton au code : script Python équivalent aux réglages, extraits du code source du paquet."""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``inspect.getsource`` : code source d'une fonction ou d'une classe.
import inspect

# ``Any`` : valeurs de types variés.
from typing import Any

# Streamlit : fenêtres surgissantes et affichage du code.
import streamlit as st

# Fonctions et classes du paquet montrées dans les fenêtres « Code ».
from cfd2d import BoundaryConfig, FlowConfig, NavierStokesSolver, SimulationConfig
from cfd2d import boundary, geometry, operators, pressure

from .params import build_config

# Sujets des sections de paramètres -> fonctions du paquet à lire (libellé, objet).
CODE_TOPICS: dict[str, list[tuple[str, Any]]] = {
    "flow": [
        ("FlowConfig.viscosity : ν = U L / Re", FlowConfig.viscosity),
        ("SimulationConfig.reference_length : longueur de référence", SimulationConfig.reference_length.fget),
    ],
    "geometry": [
        ("geometry.rasterize : masque par sur-échantillonnage", geometry.rasterize),
        ("geometry.Cylinder.contains : point dans le cylindre", geometry.Cylinder.contains),
        ("geometry.build_mask : union des obstacles, poches comblées", geometry.build_mask),
    ],
    "boundary": [
        ("BoundaryHandler.apply_normal : vitesses normales imposées", boundary.BoundaryHandler.apply_normal),
        ("BoundaryHandler.apply_ghosts : cellules fantômes (miroir)", boundary.BoundaryHandler.apply_ghosts),
        ("BoundaryConfig.channel : soufflerie ou canal", BoundaryConfig.channel),
    ],
    "time": [
        ("NavierStokesSolver.stability_rates : critère CFL + Fourier", NavierStokesSolver.stability_rates),
        ("NavierStokesSolver._advance_multistep : Euler / AB2", NavierStokesSolver._advance_multistep),
        ("NavierStokesSolver._advance_rk3 : Runge-Kutta SSP 3", NavierStokesSolver._advance_rk3),
    ],
    "numerics": [
        ("operators.reconstruct : schémas d'advection", operators.reconstruct),
        ("NavierStokesSolver._project : Poisson et correction", NavierStokesSolver._project),
        ("pressure.make_pressure_solver : choix du solveur", pressure.make_pressure_solver),
    ],
}


def code_popover(topic: str) -> None:
    """Fenêtre surgissante « Code » : source des fonctions du paquet liées à une section."""
    entries = CODE_TOPICS.get(topic)
    if not entries:
        return
    with st.popover("Code", icon=":material/code:", help="Les fonctions du paquet qui réalisent ces réglages."):
        index = st.selectbox("Fonction", range(len(entries)), format_func=lambda k: entries[k][0],
                             key=f"code_topic_{topic}")
        obj = entries[index][1]
        # inspect.getsourcefile : fichier du paquet ; getsource : texte de la fonction.
        path = inspect.getsourcefile(obj) or ""
        st.caption(f"Fichier : `src/cfd2d/{path.replace(chr(92), '/').split('/cfd2d/')[-1]}`")
        st.code(inspect.getsource(obj), language="python", line_numbers=True, height=420)


def python_script(p: dict[str, Any], label: str) -> str:
    """Script Python autonome reproduisant les réglages ``p`` (configuration, calcul, analyse)."""
    config = build_config(p)
    b = config.boundaries
    obstacles = ""
    if config.obstacles:
        ob = config.obstacles[0]
        # Image : le fichier doit être enregistré à côté du script (le tableau de pixels n'est pas recopié).
        if p.get("shape") == "image":
            xc, yc = p["upstream"], 0.5 * p["height"] + p["y_offset"]
            text = (f"ImageObstacle('mon_obstacle.png', xc={xc!r}, yc={yc!r}, width={p['image_width']!r}, "
                    f"threshold={p['threshold']!r}, invert={p['invert']!r})")
        else:
            # repr d'une dataclass : constructeur Python valide (champs nommés).
            text = repr(ob)
        obstacles = f"    obstacles=[{text}],\n"
    # Classes à importer : configuration, conditions aux limites, obstacle, moniteurs (obstacle).
    names = {"BoundaryConfig", "DomainConfig", "FlowConfig", "NavierStokesSolver", "NumericsConfig", "SimulationConfig",
             "TimeConfig"} | {type(x).__name__ for x in (b.west, b.east, b.south, b.north)}
    if config.obstacles:
        names |= {type(config.obstacles[0]).__name__, "FieldAverager", "ForceMonitor"}
    imports = ", ".join(sorted(names))
    lines = [
        f'"""Calcul généré par l\'application web cfd2d : {label}.',
        "",
        "Lancement depuis la racine du projet :",
        "",
        "    .venv\\\\Scripts\\\\python mon_calcul.py",
        '"""',
        "",
        "import logging",
        "",
        "import matplotlib",
        "",
        "matplotlib.use(\"Agg\")",
        "",
        f"from cfd2d import {imports}  # noqa: E402",
    ]
    # Imports propres au cas : tableau de bord (obstacle) ou comparaison à Ghia (cavité).
    if p["case"] == "obstacle":
        lines.append("from cfd2d import visualization as viz  # noqa: E402")
    elif p["case"] == "cavity":
        lines.append("from cfd2d.validation import ghia_errors  # noqa: E402")
    lines += [
        "",
        "# Configuration complète (identique aux réglages de l'application).",
        "config = SimulationConfig(",
        f"    name={config.name!r},",
        f"    domain={config.domain!r},",
        f"    flow={config.flow!r},",
        "    boundaries=BoundaryConfig(",
        f"        west={b.west!r},",
        f"        east={b.east!r},",
        f"        south={b.south!r},",
        f"        north={b.north!r},",
        "    ),",
        obstacles.rstrip("\n") if obstacles else "    obstacles=[],",
        f"    time={config.time!r},",
        f"    numerics={config.numerics!r},",
        ")",
        "",
        "# Messages de progression avec l'heure.",
        'logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")',
        "solver = NavierStokesSolver(config)",
    ]
    if p["case"] == "obstacle":
        t_avg = 0.5 * p["t_end"] if p["avg_start"] is None else p["avg_start"] * p["t_end"]
        lines += [
            "# Moniteurs : efforts à chaque pas, moyennes temporelles sur la seconde partie du calcul.",
            "forces = ForceMonitor(solver)",
            f"averager = FieldAverager(solver, t_start={t_avg!r}, every=5)",
            "solver.run(log_every=1000)",
            "",
            "# Synthèse (Cd, Cl, Strouhal) et tableau de bord.",
            "history = forces.history()",
            "print(history.summary())",
            'viz.plot_dashboard(solver, history, mean_fields=averager.fields(), path="outputs/mon_calcul/dashboard.png")',
        ]
    elif p["case"] == "cavity":
        lines += [
            "solver.run(log_every=1000)",
            "",
            "# Comparaison aux profils de Ghia et al. (1982) (tables disponibles à Re = 100, 400 et 1000).",
            f"print('Ecarts a Ghia (u, v) :', ghia_errors(solver, Re={int(round(config.reynolds))}))"
            if round(config.reynolds) in (100, 400, 1000) else "# (pas de table de Ghia à ce Reynolds)",
        ]
    else:
        lines += [
            "solver.run(log_every=1000)",
            "",
            "# Vitesse maximale en fin de canal (Poiseuille : 1,5 fois la vitesse débitante).",
            "print('u max en sortie / U =', solver.state.u[-2, 1:-1].max() / config.flow.U_inf)",
        ]
    return "\n".join(lines) + "\n"

"""Paramètres d'un calcul : contrôles de l'interface, configuration du solveur, diagnostic avant calcul.

Chaque contrôle (widget) est repéré par une clé de l'état de session (``st.session_state``),
initialisée une fois par :func:`init_state` à partir de :data:`DEFAULTS`. Les widgets sont
créés sans valeur par défaut explicite : un scénario, l'historique ou une étude peuvent ainsi
modifier les réglages en écrivant directement dans l'état de session (:func:`apply_params`).

Un calcul est entièrement décrit par un dictionnaire de paramètres simples (nombres, textes,
octets d'image) produit par :func:`read_params` : il se compare, se met en cache, s'enregistre
en JSON et se convertit en :class:`~cfd2d.config.SimulationConfig` par :func:`build_config`.
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``math.ceil`` : arrondi à l'entier supérieur (nombre de pas estimé).
import math

# Tampon mémoire : lecture des images téléversées.
from io import BytesIO

# ``Any`` : valeurs de types variés.
from typing import Any

import numpy as np

# Streamlit : widgets et état de session.
import streamlit as st

# Pillow : lecture et réduction des images d'obstacles.
from PIL import Image

# ``ndimage.distance_transform_edt`` : épaisseur de l'obstacle en mailles.
from scipy import ndimage

# Briques de configuration, obstacles et conditions aux limites du paquet.
from cfd2d import (
    NACA4,
    BoundaryConfig,
    Cylinder,
    DomainConfig,
    FlowConfig,
    ImageObstacle,
    Inlet,
    NoSlipWall,
    NumericsConfig,
    Obstacle,
    Outlet,
    Rectangle,
    SimulationConfig,
    SlipWall,
    StaggeredGrid,
    TimeConfig,
)

# Corps élancés (profil, plaque) : leur résolution se juge sur l'épaisseur.
from cfd2d.analytics import chord_line

# Nombres de Courant par défaut et limites de Fourier de chaque schéma temporel.
from cfd2d.solver import DEFAULT_CFL, FOURIER_LIMIT

from .common import (
    ADVECTION_SCHEMES,
    CASE_HELP,
    CASES,
    OUTLETS,
    OUTLETS_HELP,
    PRECONDITIONERS,
    PRESSURE_SOLVERS,
    PROFILES,
    PROFILES_HELP,
    SHAPES,
    TIME_SCHEMES,
    WAKE_STATIONS,
    WALLS,
    WALLS_HELP,
    format_duration,
    keep,
    thousands,
)

# Préfixe des clés de widgets de chaque cas.
PREFIX = {"obstacle": "obs", "cavity": "cav", "channel": "chn"}
# Durée par défaut, durée maximale et unité de temps de chaque cas.
TIME_SETTINGS = {
    "obstacle": (80.0, 400.0, "unités convectives D/U∞"),
    "cavity": (25.0, 200.0, "unités L/U"),
    "channel": (15.0, 200.0, "unités H/U"),
}
# Coût indicatif d'un pas de temps (s) sur une machine au repos : part fixe + part proportionnelle
# au nombre de cellules ; facteurs multiplicatifs des solveurs itératifs. Corrigé ensuite par la
# vitesse mesurée des calculs précédents (voir :func:`estimate_cost`).
STEP_COST, CELL_COST = 1.0e-3, 0.25e-6
SOLVER_COST = {"direct": 1.0, "cg": 2.0, "bicgstab": 2.5, "sor": 6.0, "jacobi": 25.0}
# Coefficient de sécurité appliqué par le solveur aux limites de stabilité (solver._SAFETY).
SAFETY = 0.9


def _defaults() -> dict[str, Any]:
    """Valeurs initiales de tous les widgets (clé de l'état de session -> valeur)."""
    d: dict[str, Any] = {"case": "obstacle", "scenario": "none", "study_on": False}
    # Réglages communs aux trois cas (Reynolds, temps, méthodes numériques).
    for case, prefix in PREFIX.items():
        re_default = 30.0 if case == "channel" else 100.0
        t_end = TIME_SETTINGS[case][0]
        d.update({
            f"{prefix}_visc_mode": "re", f"{prefix}_re": re_default, f"{prefix}_nu": 1.0 / re_default,
            f"{prefix}_U": 1.0, f"{prefix}_rho": 1.0, f"{prefix}_t_end": t_end, f"{prefix}_scheme": "ab2",
            f"{prefix}_dt_mode": "auto", f"{prefix}_cfl_auto": True, f"{prefix}_cfl": 0.4, f"{prefix}_dt": 0.01,
            f"{prefix}_advection": "quick", f"{prefix}_psolver": "direct", f"{prefix}_ptol": 1e-8,
            f"{prefix}_pmaxiter": 10_000, f"{prefix}_precond": "ilu", f"{prefix}_omega_auto": True,
            f"{prefix}_omega": 1.8, f"{prefix}_fo_auto": True, f"{prefix}_fo": 0.2, f"{prefix}_live": True,
        })
    # Obstacle en soufflerie.
    d.update({
        "obs_shape": "cylinder", "obs_rotation": 0.0, "obs_rect_w": 2.0, "obs_rect_h": 0.4, "obs_rect_alpha": 20.0,
        "obs_naca": "2412", "obs_naca_alpha": 8.0, "obs_image_w": 2.0, "obs_threshold": 0.5, "obs_invert": False,
        "obs_upstream": 4.0, "obs_downstream": 12.0, "obs_height": 8.0, "obs_y_offset": 0.0, "obs_cells": 12,
        "obs_perturbation": 0.05, "obs_start": "impulsive", "obs_ramp": 5.0, "obs_inlet": "uniform",
        "obs_outlet": "convective", "obs_north": "slip", "obs_south": "slip", "obs_supersampling": 4,
        "obs_avg_mode": "auto", "obs_avg": 0.5, "obs_stations": [1.0, 2.0, 5.0, 10.0], "obs_cv": False,
        "obs_anim": True, "obs_anim_duration": 20.0, "obs_fps": 20,
    })
    # Cavité et canal.
    d.update({"cav_N": 48})
    d.update({"chn_length": 6.0, "chn_Ny": 20, "chn_inlet": "uniform", "chn_outlet": "neumann",
              "chn_start": "impulsive", "chn_ramp": 2.0})
    return d


#: Valeurs initiales de tous les widgets de paramètres.
DEFAULTS = _defaults()


def init_state() -> None:
    """Initialise l'état de session (sans écraser les réglages déjà présents)."""
    for key, value in DEFAULTS.items():
        # setdefault : n'écrit que si la clé est absente ; les listes sont copiées (pas de partage).
        st.session_state.setdefault(key, list(value) if isinstance(value, list) else value)


# ============================================================== paramètres <-> état de session
def read_params(case: str) -> dict[str, Any]:
    """Paramètres du cas ``case`` lus dans l'état de session (dictionnaire de valeurs simples)."""
    s = st.session_state
    prefix = PREFIX[case]

    def get(name: str) -> Any:
        # Valeur du widget « préfixe_nom ».
        return s[f"{prefix}_{name}"]

    visc = get("visc_mode")
    p: dict[str, Any] = {
        "case": case,
        # Viscosité : par Re ou par ν (l'autre vaut None).
        "visc_mode": visc, "Re": float(get("re")) if visc == "re" else None,
        "nu": float(get("nu")) if visc == "nu" else None, "U": float(get("U")), "rho": float(get("rho")),
        # Temps : pas adaptatif (dt None) ou imposé ; CFL et Fourier par défaut (None) ou choisis.
        "t_end": float(get("t_end")), "scheme": get("scheme"),
        "dt": float(get("dt")) if get("dt_mode") == "fixed" else None,
        "cfl": float(get("cfl")) if get("dt_mode") == "auto" and not get("cfl_auto") else None,
        "fourier": None if get("fo_auto") else float(get("fo")),
        # Méthodes numériques.
        "advection": get("advection"), "pressure_solver": get("psolver"), "pressure_tol": float(get("ptol")),
        "pressure_maxiter": int(get("pmaxiter")), "preconditioner": get("precond"),
        "sor_omega": None if get("omega_auto") else float(get("omega")),
        "live_preview": bool(get("live")),
    }
    if case == "obstacle":
        shape = s.obs_shape
        upload = s.get("obs_image")
        p.update(
            shape=shape, rotation=float(s.obs_rotation),
            # Incidence : celle de la plaque ou celle du profil selon la forme.
            incidence=float(s.obs_rect_alpha if shape == "rectangle" else s.obs_naca_alpha if shape == "naca" else 0.0),
            rect_width=float(s.obs_rect_w), rect_height=float(s.obs_rect_h), naca_code=str(s.obs_naca).strip(),
            # getvalue() : contenu du fichier téléversé, en octets (comparable et mis en cache).
            image=upload.getvalue() if upload is not None else None, image_width=float(s.obs_image_w),
            threshold=float(s.obs_threshold), invert=bool(s.obs_invert),
            upstream=float(s.obs_upstream), downstream=float(s.obs_downstream), height=float(s.obs_height),
            y_offset=float(s.obs_y_offset), cells=int(s.obs_cells), supersampling=int(s.obs_supersampling),
            perturbation=float(s.obs_perturbation), ramp_time=float(s.obs_ramp) if s.obs_start == "ramp" else 0.0,
            inlet_profile=s.obs_inlet, outlet=s.obs_outlet, north=s.obs_north, south=s.obs_south,
            # Moyennes : début automatique (None, régime établi détecté) ou fraction de T choisie.
            avg_start=None if s.obs_avg_mode == "auto" else float(s.obs_avg),
            stations=tuple(sorted(float(x) for x in s.obs_stations)), control_volume=bool(s.obs_cv),
            animation=bool(s.obs_anim), anim_duration=float(s.obs_anim_duration), fps=int(s.obs_fps),
        )
    elif case == "cavity":
        p.update(N=int(s.cav_N))
    else:
        p.update(length=float(s.chn_length), Ny=int(s.chn_Ny), inlet_profile=s.chn_inlet, outlet=s.chn_outlet,
                 ramp_time=float(s.chn_ramp) if s.chn_start == "ramp" else 0.0)
    return p


def state_values(case: str, params: dict[str, Any]) -> dict[str, Any]:
    """Clés de widgets et valeurs correspondant à des paramètres (complets ou partiels).

    Inverse de :func:`read_params` : sert à recharger les réglages d'un calcul de l'historique,
    d'un scénario ou d'une étude.
    """
    prefix = PREFIX[case]
    out: dict[str, Any] = {"case": case}
    # Correspondances directes nom de paramètre -> suffixe de clé.
    direct = {"U": "U", "rho": "rho", "t_end": "t_end", "scheme": "scheme", "advection": "advection",
              "pressure_solver": "psolver", "pressure_tol": "ptol", "pressure_maxiter": "pmaxiter",
              "preconditioner": "precond", "live_preview": "live"}
    obstacle = {"shape": "shape", "rotation": "rotation", "rect_width": "rect_w", "rect_height": "rect_h",
                "naca_code": "naca", "image_width": "image_w", "threshold": "threshold", "invert": "invert",
                "upstream": "upstream", "downstream": "downstream", "height": "height", "y_offset": "y_offset",
                "cells": "cells", "supersampling": "supersampling", "perturbation": "perturbation",
                "inlet_profile": "inlet", "outlet": "outlet", "north": "north", "south": "south",
                "control_volume": "cv", "animation": "anim", "anim_duration": "anim_duration", "fps": "fps"}
    other = {"N": "N", "length": "length", "Ny": "Ny", "inlet_profile": "inlet", "outlet": "outlet"}
    mapping = direct | (obstacle if case == "obstacle" else other)
    for name, value in params.items():
        if name in mapping:
            out[f"{prefix}_{mapping[name]}"] = value
        elif name == "Re" and value is not None:
            out.update({f"{prefix}_visc_mode": "re", f"{prefix}_re": float(value)})
        elif name == "nu" and value is not None:
            out.update({f"{prefix}_visc_mode": "nu", f"{prefix}_nu": float(value)})
        elif name == "dt":
            out[f"{prefix}_dt_mode"] = "auto" if value is None else "fixed"
            if value is not None:
                out[f"{prefix}_dt"] = float(value)
        elif name == "cfl":
            out[f"{prefix}_cfl_auto"] = value is None
            if value is not None:
                out.update({f"{prefix}_cfl": float(value), f"{prefix}_dt_mode": "auto"})
        elif name == "fourier":
            out[f"{prefix}_fo_auto"] = value is None
            if value is not None:
                out[f"{prefix}_fo"] = float(value)
        elif name == "sor_omega":
            out[f"{prefix}_omega_auto"] = value is None
            if value is not None:
                out[f"{prefix}_omega"] = float(value)
        elif name == "ramp_time":
            out[f"{prefix}_start"] = "ramp" if value else "impulsive"
            if value:
                out[f"{prefix}_ramp"] = float(value)
        elif name == "incidence" and case == "obstacle":
            # La même incidence s'applique à la plaque et au profil.
            out.update({"obs_rect_alpha": float(value), "obs_naca_alpha": float(value)})
        elif name == "avg_start" and case == "obstacle":
            out["obs_avg_mode"] = "auto" if value is None else "manual"
            if value is not None:
                out["obs_avg"] = float(value)
        elif name == "stations" and case == "obstacle":
            out["obs_stations"] = [float(x) for x in value]
    return out


def apply_params(case: str, params: dict[str, Any]) -> None:
    """Écrit des paramètres dans l'état de session (les widgets les afficheront au prochain affichage).

    À appeler depuis une fonction de rappel (``on_click``), exécutée avant la création des widgets.
    """
    for key, value in state_values(case, params).items():
        st.session_state[key] = value


# ======================================================================= configuration
def flow_config(p: dict[str, Any], *, L_ref: float | None = None, perturbation: float = 0.0) -> FlowConfig:
    """Fluide et écoulement de référence ; la viscosité est donnée par Re ou par ν."""
    # Exactement un des deux paramètres (exigence de FlowConfig) ; ** dépaquette le dictionnaire.
    viscosity = {"Re": p["Re"]} if p["visc_mode"] == "re" else {"nu": p["nu"]}
    return FlowConfig(U_inf=p["U"], rho=p["rho"], L_ref=L_ref, perturbation=perturbation, **viscosity)


def time_config(p: dict[str, Any]) -> TimeConfig:
    """Intégration temporelle (durée, schéma, pas de temps adaptatif ou imposé)."""
    return TimeConfig(t_end=p["t_end"], dt=p["dt"], cfl=p["cfl"], fourier=p["fourier"], scheme=p["scheme"])


def numerics_config(p: dict[str, Any]) -> NumericsConfig:
    """Schéma d'advection, solveur de pression et masque des obstacles."""
    return NumericsConfig(
        advection=p["advection"], pressure_solver=p["pressure_solver"], pressure_tol=p["pressure_tol"],
        pressure_maxiter=p["pressure_maxiter"], preconditioner=p["preconditioner"], sor_omega=p["sor_omega"],
        mask_supersampling=p.get("supersampling", 4),
    )


@st.cache_data(show_spinner=False, max_entries=8)
def image_array(data: bytes, max_size: int = 400) -> np.ndarray:
    """Image téléversée -> tableau RGBA, réduite à ``max_size`` pixels de côté au plus."""
    with Image.open(BytesIO(data)) as image:
        # convert("RGBA") : format unique (la transparence est traitée comme du fluide).
        image = image.convert("RGBA")
        # thumbnail réduit l'image en conservant ses proportions (jamais agrandie).
        image.thumbnail((max_size, max_size))
        return np.asarray(image)


def make_obstacle(p: dict[str, Any]) -> Obstacle:
    """Obstacle centré en ``(x, y)`` ; la longueur unité D est sa taille caractéristique."""
    # Centre : distance amont depuis l'entrée ; mi-hauteur décalée de y_offset.
    xc, yc = p["upstream"], 0.5 * p["height"] + p["y_offset"]
    shape = p["shape"]
    if shape == "cylinder":
        return Cylinder(xc, yc, 1.0)
    if shape == "square":
        return Rectangle.square(xc, yc, 1.0, angle_deg=p["rotation"])
    if shape == "rectangle":
        # Incidence positive = bord d'attaque (côté amont) vers le haut : rotation de sens horaire.
        return Rectangle(xc, yc, p["rect_width"], p["rect_height"], angle_deg=-p["incidence"])
    if shape == "naca":
        # Bord d'attaque à un quart de corde en amont : le profil pivote autour de (xc, yc).
        return NACA4(p["naca_code"], chord=1.0, x_le=xc - 0.25, y_le=yc, alpha_deg=p["incidence"])
    if shape == "image":
        if p["image"] is None:
            raise ValueError("Aucune image chargée.")
        return ImageObstacle(image_array(p["image"]), xc, yc, p["image_width"], threshold=p["threshold"],
                             invert=p["invert"])
    raise ValueError(f"Forme inconnue : {shape!r}.")


def obstacle_config(p: dict[str, Any]) -> SimulationConfig:
    """Obstacle dans une soufflerie : entrée à gauche, sortie à droite, parois en haut et en bas."""
    # Longueurs en unités D (taille de l'obstacle) : domaine = amont + aval.
    Lx, Ly, n = p["upstream"] + p["downstream"], p["height"], p["cells"]
    walls = {"slip": SlipWall(), "noslip": NoSlipWall()}
    return SimulationConfig(
        name=f"{p['shape']}_Re{p['Re']:g}" if p["Re"] is not None else f"{p['shape']}_nu{p['nu']:g}",
        # Nombre de cellules = longueur x cellules par unité de longueur.
        domain=DomainConfig(Lx=Lx, Ly=Ly, Nx=round(Lx * n), Ny=round(Ly * n)),
        # Re basé sur la longueur de référence de l'obstacle (L_ref=None : déduite de l'obstacle).
        flow=flow_config(p, perturbation=p["perturbation"]),
        boundaries=BoundaryConfig(west=Inlet(profile=p["inlet_profile"], ramp_time=p["ramp_time"]),
                                  east=Outlet(kind=p["outlet"]), south=walls[p["south"]], north=walls[p["north"]]),
        obstacles=[make_obstacle(p)],
        time=time_config(p),
        numerics=numerics_config(p),
    )


def cavity_config(p: dict[str, Any]) -> SimulationConfig:
    """Cavité carrée de côté 1 entraînée par sa paroi supérieure (vitesse U)."""
    N = p["N"]
    return SimulationConfig(
        name="cavity",
        domain=DomainConfig(Lx=1.0, Ly=1.0, Nx=N, Ny=N),
        # Re basé sur la vitesse du couvercle et le côté de la cavité.
        flow=flow_config(p, L_ref=1.0),
        boundaries=BoundaryConfig.lid_driven_cavity(lid_velocity=p["U"]),
        time=time_config(p),
        numerics=numerics_config(p),
    )


def channel_config(p: dict[str, Any]) -> SimulationConfig:
    """Canal plan de hauteur 1 à parois adhérentes (Re basé sur la hauteur et la vitesse débitante)."""
    Ny = p["Ny"]
    return SimulationConfig(
        name="channel",
        domain=DomainConfig(Lx=p["length"], Ly=1.0, Nx=round(p["length"] * Ny), Ny=Ny),
        flow=flow_config(p, L_ref=1.0),
        boundaries=BoundaryConfig(west=Inlet(profile=p["inlet_profile"], ramp_time=p["ramp_time"]),
                                  east=Outlet(kind=p["outlet"]), south=NoSlipWall(), north=NoSlipWall()),
        time=time_config(p),
        numerics=numerics_config(p),
    )


def build_config(p: dict[str, Any]) -> SimulationConfig:
    """Configuration complète correspondant aux paramètres ``p`` (selon ``p['case']``)."""
    builders = {"obstacle": obstacle_config, "cavity": cavity_config, "channel": channel_config}
    return builders[p["case"]](p)


# ======================================================================= diagnostic
def geometry_key(p: dict[str, Any]) -> dict[str, Any]:
    """Paramètres dont dépendent la grille et le masque (clé de cache des calculs géométriques)."""
    names = {
        "obstacle": ("case", "shape", "rotation", "incidence", "rect_width", "rect_height", "naca_code", "image",
                     "image_width", "threshold", "invert", "upstream", "downstream", "height", "y_offset", "cells",
                     "supersampling", "inlet_profile", "outlet", "north", "south", "ramp_time"),
        "cavity": ("case", "N", "U"),
        "channel": ("case", "length", "Ny", "inlet_profile", "outlet", "ramp_time"),
    }[p["case"]]
    # Les autres paramètres (Re, temps...) n'influent pas sur la géométrie : valeurs fictives valides
    # (la vitesse n'intervient que pour la cavité, dont elle fixe la paroi mobile).
    fixed = {"visc_mode": "re", "Re": 100.0, "nu": None, "U": 1.0, "rho": 1.0, "t_end": 1.0,
             "scheme": "ab2", "dt": None, "cfl": None, "fourier": None, "advection": "quick",
             "pressure_solver": "direct", "pressure_tol": 1e-8, "pressure_maxiter": 10, "preconditioner": "ilu",
             "sor_omega": None, "perturbation": 0.0}
    return fixed | {name: p[name] for name in names if name in p}


@st.cache_data(show_spinner=False, max_entries=64)
def geometry_info(key: dict[str, Any]) -> dict[str, float]:
    """Grandeurs géométriques de l'obstacle sur la grille : cellules solides, épaisseur en mailles."""
    config = build_config(key)
    if not config.obstacles:
        return {}
    d = config.domain
    grid = StaggeredGrid(d.Lx, d.Ly, d.Nx, d.Ny)
    ob = config.obstacles[0]
    # Masque de l'obstacle seul (même sur-échantillonnage que le solveur).
    body = ob.mask(grid, config.numerics.mask_supersampling)
    # distance_transform_edt : distance (en mailles) de chaque cellule solide au fluide le plus
    # proche ; deux fois la distance maximale moins une maille = épaisseur maximale en mailles.
    thickness = 2.0 * float(ndimage.distance_transform_edt(body).max()) - 1.0 if body.any() else 0.0
    xmin, xmax, ymin, ymax = ob.bounds
    return {"solid_cells": float(body.sum()), "thickness_cells": max(thickness, 0.0),
            "frontal_height": ymax - ymin, "x_rear": xmax}


def check_params(p: dict[str, Any]) -> tuple[SimulationConfig | None, list[str], list[str]]:
    """Construit la configuration et la contrôle : (configuration ou None, erreurs, conseils)."""
    errors: list[str] = []
    # Vérifications préalables propres aux obstacles.
    if p["case"] == "obstacle":
        code = p["naca_code"]
        if p["shape"] == "naca" and not (len(code) == 4 and code.isdigit() and int(code[2:]) > 0):
            errors.append("Code NACA invalide : 4 chiffres attendus, épaisseur (2 derniers) non nulle, ex. 2412.")
        if p["shape"] == "image" and p["image"] is None:
            errors.append("Chargez une image (forme sombre sur fond clair) pour définir l'obstacle.")
        if p["animation"] and p["anim_duration"] >= p["t_end"]:
            errors.append("La durée animée doit être inférieure à la durée simulée.")
    if errors:
        return None, errors, []
    # Construction (les classes du paquet valident leurs propres paramètres : ValueError).
    try:
        config = build_config(p)
    except ValueError as exc:
        return None, [str(exc)], []
    d = config.domain
    dx, dy = d.Lx / d.Nx, d.Ly / d.Ny
    advice: list[str] = []
    if config.obstacles:
        ob = config.obstacles[0]
        xmin, xmax, ymin, ymax = ob.bounds
        # L'obstacle doit rester à l'intérieur du domaine, à au moins deux mailles des bords.
        if xmin < 2 * dx or xmax > d.Lx - 2 * dx or ymin < 2 * dy or ymax > d.Ly - 2 * dy:
            errors.append("L'obstacle touche ou dépasse le bord du domaine : agrandir le domaine ou le recentrer.")
        info = geometry_info(geometry_key(p))
        # Blocage : hauteur frontale (étendue verticale) rapportée à la hauteur du domaine.
        blockage = (ymax - ymin) / d.Ly
        if blockage > 0.2:
            advice.append(f"Blocage élevé ({100 * blockage:.0f} %) : les parois accélèrent l'écoulement "
                          "(Cd et St surestimés).")
        # Résolution : mailles par longueur de référence et, pour un corps mince, dans l'épaisseur.
        per_length = ob.reference_length / max(dx, dy)
        if per_length < 8:
            advice.append(f"Obstacle peu résolu ({per_length:.1f} mailles par longueur de référence) : "
                          "augmenter la résolution.")
        thickness = info.get("thickness_cells", per_length)
        if chord_line(ob) is not None or p["shape"] == "image":
            if thickness < 4:
                advice.append(f"Épaisseur résolue par seulement {thickness:.0f} maille(s) : le corps est mal "
                              "représenté (8 mailles au moins conseillées ; pour un profil, ≥ 64 mailles par corde).")
            elif thickness < 8:
                advice.append(f"Épaisseur résolue par {thickness:.0f} mailles : résultats indicatifs "
                              "(8 mailles au moins conseillées).")
        # Aval court : le sillage est perturbé par la sortie.
        if d.Lx - xmax < 4.0 * ob.reference_length:
            advice.append("Sortie proche de l'obstacle (< 4 L) : le sillage risque d'être perturbé.")
        if config.reynolds < 47 and p["perturbation"] > 0 and p["shape"] == "cylinder":
            advice.append("Re < 47 : l'écoulement est stationnaire ; mettre la perturbation à 0 (elle s'amortit "
                          "lentement).")
        if config.reynolds > 190 and p["shape"] == "cylinder":
            advice.append("Re > 190 : en réalité le sillage devient tridimensionnel ; le calcul 2D reste qualitatif.")
    # Reynolds de maille : rapport convection / diffusion à l'échelle d'une maille.
    cell_reynolds = config.flow.U_inf * max(dx, dy) / config.nu
    if cell_reynolds > 10:
        advice.append(f"Reynolds de maille élevé (Re_Δ = {cell_reynolds:.0f}) : maille trop grossière pour ce Reynolds.")
    if p["advection"] == "central" and cell_reynolds > 2:
        advice.append("Schéma centré avec Re_Δ > 2 : oscillations probables (préférer QUICK ou TVD).")
    if p["pressure_solver"] == "jacobi" and d.Nx * d.Ny > 5000:
        advice.append("Solveur de Jacobi sur une grande grille : calcul extrêmement lent.")
    return (None if errors else config), errors, advice


def estimate_cost(config: SimulationConfig, peak: float, calibration: float = 1.0) -> dict[str, float]:
    """Ordre de grandeur du pas de temps, du nombre de pas et de la durée du calcul.

    Reprend le critère de stabilité du solveur (``NavierStokesSolver.stability_rates``) avec une
    vitesse maximale estimée ``peak x U`` ; ``calibration`` corrige la durée d'après la vitesse
    mesurée des calculs précédents. Renvoie ``dt``, ``steps``, ``seconds``, ``base_seconds``
    (durée non calibrée) et ``advective_share`` (part de l'advection dans la contrainte de pas).
    """
    d, tc, num = config.domain, config.time, config.numerics
    dx, dy = d.Lx / d.Nx, d.Ly / d.Ny
    umax = peak * config.flow.U_inf
    # Nombre de Courant et de Fourier effectifs (valeurs du schéma si non imposées).
    cfl = tc.cfl if tc.cfl is not None else DEFAULT_CFL[tc.scheme]
    fourier = tc.fourier if tc.fourier is not None else SAFETY * FOURIER_LIMIT[tc.scheme]
    # Taux advectif (|v| max estimée à la moitié de |u| max) et taux diffusif.
    advective = umax * (1.0 / dx + 0.5 / dy) / cfl
    diffusive = config.nu * (1.0 / dx**2 + 1.0 / dy**2) / fourier
    dt = 1.0 / (advective + diffusive)
    # Euler explicite avec un schéma non décentré : contrainte supplémentaire dt < 2 nu / |u|².
    if tc.scheme == "euler" and num.advection != "upwind":
        dt = min(dt, SAFETY * 2.0 * config.nu / (1.25 * umax**2))
    # Pas imposé : utilisé s'il est plus petit que la limite de stabilité.
    if tc.dt is not None:
        dt = min(dt, tc.dt)
    steps = math.ceil(tc.t_end / dt)
    # Coût d'un pas : trois étages pour RK3, facteur propre au solveur de pression.
    stages = 3 if tc.scheme == "rk3" else 1
    base = steps * stages * (STEP_COST + CELL_COST * d.Nx * d.Ny) * SOLVER_COST[num.pressure_solver]
    return {"dt": dt, "steps": steps, "seconds": base * calibration, "base_seconds": base,
            "advective_share": advective / (advective + diffusive)}


#: Vitesse maximale attendue de chaque cas (accélération autour d'un obstacle, pic parabolique).
PEAK_SPEED = {"obstacle": 1.6, "cavity": 1.0, "channel": 1.5}


def run_summary(p: dict[str, Any], config: SimulationConfig, calibration: tuple[float, int]) -> tuple[list[tuple[str, str]], dict]:
    """Récapitulatif avant calcul : lignes (libellé, valeur) et estimation du coût.

    ``calibration`` : (facteur correctif, nombre de calculs sur lesquels il a été mesuré).
    """
    d = config.domain
    dx = d.Lx / d.Nx
    factor, n_runs = calibration
    cost = estimate_cost(config, PEAK_SPEED[p["case"]], factor)
    share = cost["advective_share"]
    rows = [
        ("Nombre de Reynolds", f"Re = {config.reynolds:.4g}"),
        ("Viscosité cinématique", f"ν = {config.nu:.4g}"),
        ("Longueur de référence", f"L = {config.reference_length:.4g}"),
        ("Grille", f"{d.Nx} × {d.Ny} = {thousands(d.Nx * d.Ny)} cellules"),
        ("Reynolds de maille", f"Re_Δ = U Δx / ν = {config.flow.U_inf * dx / config.nu:.3g}"),
        ("Pas de temps estimé", f"Δt ≈ {cost['dt']:.2e} → ≈ {thousands(cost['steps'])} pas"),
        ("Contrainte limitante", f"advection {100 * share:.0f} % · diffusion {100 * (1 - share):.0f} %"),
        ("Durée estimée", f"≈ {format_duration(cost['seconds'])}"
         + (f" (calibrée sur {n_runs} calcul{'s' if n_runs > 1 else ''})" if n_runs else " (machine au repos)")),
    ]
    if config.obstacles:
        ob = config.obstacles[0]
        info = geometry_info(geometry_key(p))
        rows.insert(3, ("Blocage (hauteur frontale / H)", f"{100 * info['frontal_height'] / d.Ly:.1f} %"))
        if chord_line(ob) is not None or p["shape"] == "image":
            rows.insert(4, ("Mailles dans l'épaisseur", f"{info['thickness_cells']:.0f}"))
        rows.insert(1, ("Durée en temps convectifs", f"T U∞ / L = {p['t_end'] * config.flow.U_inf / ob.reference_length:.4g}"))
    return rows, cost


def regime_caption(p: dict[str, Any], config: SimulationConfig) -> str | None:
    """Régime attendu pour un cylindre, selon Re (repères classiques)."""
    if p["case"] != "obstacle" or p["shape"] != "cylinder":
        return None
    Re = config.reynolds
    if Re < 5:
        return "Re < 5 : écoulement rampant, sans décollement."
    if Re < 47:
        return "5 < Re < 47 : deux tourbillons attachés, écoulement stationnaire."
    if Re < 190:
        return "47 < Re < 190 : allée de Von Kármán périodique (régime laminaire 2D)."
    return "Re > 190 : transition vers un sillage tridimensionnel (le calcul 2D reste qualitatif)."


# ============================================================================= widgets
#: Libellés courts des cas (contrôle segmenté de la colonne de paramètres).
CASE_SHORT = {"obstacle": "Obstacle", "cavity": "Cavité", "channel": "Canal"}


def case_selector() -> str:
    """Choix du cas d'étude (contrôle segmenté) ; renvoie son identifiant."""
    case = st.segmented_control("Cas d'étude", list(CASES), format_func=CASE_SHORT.get, required=True, width="stretch",
                                **keep("case"))
    st.caption(f"**{CASES[case]}** : {CASE_HELP[case]}")
    return case


def flow_widgets(prefix: str, *, velocity_label: str, re_help: str) -> None:
    """Viscosité (par Re ou par ν), vitesse de référence et masse volumique."""
    st.radio("Viscosité définie par", ["re", "nu"], horizontal=True,
             format_func={"re": "le nombre de Reynolds", "nu": "la viscosité ν"}.get, **keep(f"{prefix}_visc_mode"))
    if st.session_state[f"{prefix}_visc_mode"] == "re":
        st.number_input("Nombre de Reynolds Re", min_value=0.1, max_value=10000.0, step=10.0, format="%g", help=re_help,
                        **keep(f"{prefix}_re"))
    else:
        st.number_input("Viscosité cinématique ν", min_value=1e-6, max_value=10.0, step=0.001, format="%.5f",
                        help="Re = U L / ν est alors calculé (voir le récapitulatif).", **keep(f"{prefix}_nu"))
    c1, c2 = st.columns(2)
    c1.number_input(velocity_label, min_value=0.05, max_value=20.0, step=0.1, format="%g",
                    help="À Re fixé, changer la vitesse change la viscosité (ν = U L / Re) : l'écoulement adimensionné "
                    "est identique, seule l'échelle de temps change.", **keep(f"{prefix}_U"))
    c2.number_input("Masse volumique ρ", min_value=0.01, max_value=5000.0, step=0.1, format="%g",
                    help="N'influe que sur l'échelle de la pression : les coefficients (Cd, Cl, Cp) n'en dépendent pas.",
                    **keep(f"{prefix}_rho"))


def time_widgets(case: str) -> None:
    """Durée simulée, schéma temporel et pas de temps (adaptatif ou imposé)."""
    prefix = PREFIX[case]
    _, t_max, unit = TIME_SETTINGS[case]
    st.slider("Durée simulée T", min_value=1.0, max_value=t_max, step=1.0, **keep(f"{prefix}_t_end"),
              help=f"Temps physique simulé ; avec une vitesse de 1, il s'exprime en {unit}.")
    st.selectbox("Schéma temporel", list(TIME_SCHEMES), format_func=TIME_SCHEMES.get, **keep(f"{prefix}_scheme"))
    st.radio("Pas de temps Δt", ["auto", "fixed"], horizontal=True,
             format_func={"auto": "adaptatif (CFL)", "fixed": "imposé"}.get, **keep(f"{prefix}_dt_mode"))
    if st.session_state[f"{prefix}_dt_mode"] == "auto":
        default = DEFAULT_CFL[st.session_state[f"{prefix}_scheme"]]
        st.checkbox(f"CFL par défaut du schéma ({default:g})", **keep(f"{prefix}_cfl_auto"))
        if not st.session_state[f"{prefix}_cfl_auto"]:
            # Jusqu'à 5 : au-delà de la limite de stabilité, pour observer une divergence.
            st.slider("Nombre de Courant visé", min_value=0.05, max_value=5.0, step=0.05, **keep(f"{prefix}_cfl"),
                      help="Valeurs sûres : Euler 0,2 ; AB2 0,4 ; RK3 0,8. Le Courant effectif reste inférieur à la "
                      "valeur visée (la diffusion entre aussi dans le critère) ; au-delà de ~3-4 en AB2, le calcul diverge.")
    else:
        st.number_input("Δt imposé", min_value=1e-5, max_value=1.0, step=0.001, format="%.4f", **keep(f"{prefix}_dt"),
                        help="Conservé s'il respecte les critères de stabilité, réduit automatiquement sinon.")


def numerics_widgets(prefix: str, *, obstacle: bool) -> None:
    """Options numériques avancées : advection, solveur de pression, Fourier, masque."""
    s = st.session_state
    st.selectbox("Schéma d'advection", list(ADVECTION_SCHEMES), format_func=ADVECTION_SCHEMES.get,
                 **keep(f"{prefix}_advection"))
    st.selectbox("Solveur de l'équation de Poisson", list(PRESSURE_SOLVERS), format_func=PRESSURE_SOLVERS.get,
                 **keep(f"{prefix}_psolver"))
    if s[f"{prefix}_psolver"] != "direct":
        c1, c2 = st.columns(2)
        c1.select_slider("Tolérance relative", options=[1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
                         format_func=lambda v: f"{v:.0e}", **keep(f"{prefix}_ptol"))
        c2.number_input("Itérations maximales", min_value=100, max_value=200_000, step=1000, **keep(f"{prefix}_pmaxiter"))
    if s[f"{prefix}_psolver"] in ("cg", "bicgstab"):
        st.selectbox("Préconditionneur", list(PRECONDITIONERS), format_func=PRECONDITIONERS.get,
                     **keep(f"{prefix}_precond"))
    if s[f"{prefix}_psolver"] == "sor":
        st.checkbox("ω optimal estimé", **keep(f"{prefix}_omega_auto"))
        if not s[f"{prefix}_omega_auto"]:
            st.slider("Relaxation ω", min_value=1.0, max_value=1.99, step=0.01, **keep(f"{prefix}_omega"))
    st.checkbox("Nombre de Fourier sûr du schéma", help="Limite de stabilité de la diffusion ν Δt (1/Δx² + 1/Δy²).",
                **keep(f"{prefix}_fo_auto"))
    if not s[f"{prefix}_fo_auto"]:
        st.slider("Nombre de Fourier maximal", min_value=0.05, max_value=2.0, step=0.05, **keep(f"{prefix}_fo"),
                  help="Limites théoriques (diffusion pure) : Euler 0,5 ; AB2 0,25 ; RK3 0,63.")
    if obstacle:
        st.slider("Sur-échantillonnage du masque", min_value=1, max_value=8, **keep("obs_supersampling"),
                  help="Points testés par cellule et par direction : une cellule est solide si au moins la moitié "
                  "de ses points sont dans l'obstacle.")


def section(title: str, *, expanded: bool = False, icon: str | None = None):
    """Section repliable du panneau de paramètres."""
    return st.expander(title, expanded=expanded, icon=icon)


def obstacle_widgets(extras) -> None:
    """Paramètres du cas « obstacle en soufflerie ».

    ``extras(topic)`` affiche, en tête de section, les fenêtres « Théorie » et « Code » associées.
    """
    s = st.session_state
    with section("Fluide et écoulement", expanded=True, icon=":material/water:"):
        extras("flow")
        flow_widgets("obs", velocity_label="Vitesse amont U∞",
                     re_help="Re = U∞ L / ν, L = longueur de référence de l'obstacle (diamètre, côté, hauteur frontale "
                     "ou corde). Cylindre : Re < 47 stationnaire, 47 < Re < 190 allée de Von Kármán.")
    with section("Obstacle et domaine", expanded=True, icon=":material/interests:"):
        extras("geometry")
        st.selectbox("Forme de l'obstacle", list(SHAPES), format_func=SHAPES.get, **keep("obs_shape"))
        shape = s.obs_shape
        if shape == "square":
            st.slider("Rotation (°)", min_value=0.0, max_value=45.0, step=1.0, help="45° : losange.", **keep("obs_rotation"))
        elif shape == "rectangle":
            c1, c2 = st.columns(2)
            c1.number_input("Longueur", min_value=0.1, max_value=10.0, step=0.1, format="%g", **keep("obs_rect_w"))
            c2.number_input("Épaisseur", min_value=0.05, max_value=5.0, step=0.05, format="%g", **keep("obs_rect_h"))
            st.slider("Incidence α (°)", min_value=-45.0, max_value=45.0, step=1.0, **keep("obs_rect_alpha"),
                      help="Positive : bord d'attaque (amont) vers le haut.")
        elif shape == "naca":
            c1, c2 = st.columns([1, 2])
            c1.text_input("Code NACA", max_chars=4, **keep("obs_naca"),
                          help="Cambrure (%), position de la cambrure (dixièmes), épaisseur (%).")
            c2.slider("Incidence α (°)", min_value=-20.0, max_value=20.0, step=0.5, **keep("obs_naca_alpha"))
            st.caption("Corde = 1 D ; rotation autour du quart de corde. Visez au moins 64 mailles par corde.")
        elif shape == "image":
            st.file_uploader("Image de l'obstacle (PNG, JPG)", type=["png", "jpg", "jpeg", "bmp", "gif"], key="obs_image",
                             help="Pixels sombres = solide ; transparence = fluide.")
            c1, c2 = st.columns(2)
            c1.number_input("Largeur de l'image", min_value=0.2, max_value=10.0, step=0.1, format="%g", **keep("obs_image_w"))
            c2.slider("Seuil de gris", min_value=0.05, max_value=0.95, step=0.05, **keep("obs_threshold"))
            st.checkbox("Inverser (pixels clairs = solide)", **keep("obs_invert"))
            if s.get("obs_image") is not None:
                st.image(s.obs_image.getvalue(), width=160, caption="Image chargée", alt="Image de l'obstacle chargée")
        c1, c2 = st.columns(2)
        c1.slider("Distance amont", min_value=1.5, max_value=15.0, step=0.5, **keep("obs_upstream"),
                  help="De l'entrée au centre de l'obstacle (en D).")
        c2.slider("Longueur aval", min_value=3.0, max_value=40.0, step=1.0, **keep("obs_downstream"),
                  help="Du centre de l'obstacle à la sortie (en D) : longueur du sillage simulé.")
        c1, c2 = st.columns(2)
        c1.slider("Hauteur du domaine", min_value=2.0, max_value=20.0, step=0.5, **keep("obs_height"))
        c2.slider("Décalage vertical", min_value=-5.0, max_value=5.0, step=0.25, **keep("obs_y_offset"),
                  help="Position du centre par rapport à mi-hauteur (en D).")
        st.slider("Résolution (cellules par D)", min_value=6, max_value=80, step=1, **keep("obs_cells"),
                  help="12 : essai rapide ; 20 : bon compromis ; 30 et plus : précis mais long (coût ∝ résolution³). "
                  "Profils et plaques minces : 40 à 80.")
    with section("Conditions aux limites et initiales", icon=":material/border_style:"):
        extras("boundary")
        c1, c2 = st.columns(2)
        c1.selectbox("Entrée (gauche)", list(PROFILES), format_func=PROFILES.get, help=PROFILES_HELP, **keep("obs_inlet"))
        c2.selectbox("Sortie (droite)", list(OUTLETS), format_func=OUTLETS.get, help=OUTLETS_HELP, **keep("obs_outlet"))
        c1, c2 = st.columns(2)
        c1.selectbox("Paroi haute", list(WALLS), format_func=WALLS.get, help=WALLS_HELP, **keep("obs_north"))
        c2.selectbox("Paroi basse", list(WALLS), format_func=WALLS.get, help=WALLS_HELP, **keep("obs_south"))
        st.slider("Perturbation initiale (fraction de U∞)", min_value=0.0, max_value=0.2, step=0.01,
                  **keep("obs_perturbation"),
                  help="Mode transverse ε U sin(πx/Lx) sin(πy/Ly) qui brise la symétrie et déclenche plus tôt le lâcher "
                  "tourbillonnaire (0 pour un écoulement stationnaire, Re < 47).")
        st.radio("Démarrage", ["impulsive", "ramp"], horizontal=True, **keep("obs_start"),
                 format_func={"impulsive": "impulsif", "ramp": "progressif"}.get,
                 help="Impulsif : écoulement uniforme dès t = 0. Progressif : fluide au repos et vitesse d'entrée "
                 "croissant de 0 à U∞ (loi en 1 − cos).")
        if s.obs_start == "ramp":
            st.slider("Durée de la montée en vitesse", min_value=0.5, max_value=20.0, step=0.5, **keep("obs_ramp"))
    with section("Temps", icon=":material/schedule:"):
        extras("time")
        time_widgets("obstacle")
    with section("Méthodes numériques (avancé)", icon=":material/tune:"):
        extras("numerics")
        numerics_widgets("obs", obstacle=True)
    with section("Sorties et post-traitement", icon=":material/insights:"):
        st.radio("Début des moyennes temporelles", ["auto", "manual"], horizontal=True, **keep("obs_avg_mode"),
                 format_func={"auto": "automatique", "manual": "fraction de T"}.get,
                 help="Automatique : moyennes sur le régime établi détecté après le calcul (Cp, sillage, recirculation).")
        if s.obs_avg_mode == "manual":
            st.slider("Début des moyennes (fraction de T)", min_value=0.0, max_value=0.9, step=0.05, **keep("obs_avg"))
        st.multiselect("Stations du sillage x/L", WAKE_STATIONS, **keep("obs_stations"),
                       help="Mesurées depuis le centre d'un corps non profilé, depuis l'arrière d'un profil ou d'une plaque.")
        st.toggle("Vérifier Cd par un bilan de quantité de mouvement", **keep("obs_cv"),
                  help="Volume de contrôle de 4 L × 4 L autour de l'obstacle (vérification indépendante).")
        st.toggle("Enregistrer la fin du calcul pour l'animer", **keep("obs_anim"),
                  help="Instantanés de vitesse et de pression : animations de n'importe quelle grandeur après le calcul.")
        if s.obs_anim:
            c1, c2 = st.columns(2)
            c1.slider("Durée enregistrée", min_value=2.0, max_value=60.0, step=1.0, **keep("obs_anim_duration"),
                      help="Dernières unités de temps du calcul.")
            c2.slider("Images par seconde", min_value=5, max_value=30, **keep("obs_fps"))
        st.toggle("Aperçu en direct pendant le calcul", **keep("obs_live"))


def cavity_widgets(extras) -> None:
    """Paramètres de la cavité entraînée."""
    with section("Fluide et couvercle", expanded=True, icon=":material/water:"):
        extras("flow")
        flow_widgets("cav", velocity_label="Vitesse du couvercle U",
                     re_help="Re = U L / ν (L = côté de la cavité). Données de Ghia et al. à Re = 100, 400 et 1000.")
    with section("Maillage", expanded=True, icon=":material/grid_on:"):
        extras("geometry")
        st.slider("Cellules par côté N", min_value=16, max_value=192, step=8, **keep("cav_N"),
                  help="N pair : les faces de la grille tombent sur les axes médians. Re = 1000 : N ≥ 96 conseillé.")
        st.caption("Cavité carrée de côté L = 1, quatre parois adhérentes, couvercle mobile (paroi haute). "
                   "Le fluide part du repos.")
    with section("Temps", icon=":material/schedule:"):
        extras("time")
        time_widgets("cavity")
    with section("Méthodes numériques (avancé)", icon=":material/tune:"):
        extras("numerics")
        numerics_widgets("cav", obstacle=False)
    with section("Sorties", icon=":material/insights:"):
        st.toggle("Aperçu en direct pendant le calcul", **keep("cav_live"))


def channel_widgets(extras) -> None:
    """Paramètres du canal plan."""
    s = st.session_state
    with section("Fluide et écoulement", expanded=True, icon=":material/water:"):
        extras("flow")
        flow_widgets("chn", velocity_label="Vitesse débitante U",
                     re_help="Re = U H / ν (H = hauteur du canal, U = vitesse moyenne).")
    with section("Canal et maillage (longueurs en H)", expanded=True, icon=":material/straighten:"):
        extras("geometry")
        st.slider("Longueur du canal", min_value=2.0, max_value=30.0, step=0.5, **keep("chn_length"))
        st.slider("Cellules sur la hauteur", min_value=8, max_value=64, step=2, **keep("chn_Ny"))
    with section("Conditions aux limites et initiales", icon=":material/border_style:"):
        extras("boundary")
        c1, c2 = st.columns(2)
        c1.selectbox("Profil d'entrée", list(PROFILES), format_func=PROFILES.get, **keep("chn_inlet"),
                     help="Uniforme : le profil se développe vers la parabole. Parabolique : déjà établi.")
        c2.selectbox("Sortie", list(OUTLETS), format_func=OUTLETS.get, help=OUTLETS_HELP, **keep("chn_outlet"))
        st.radio("Démarrage", ["impulsive", "ramp"], horizontal=True, **keep("chn_start"),
                 format_func={"impulsive": "impulsif", "ramp": "progressif"}.get)
        if s.chn_start == "ramp":
            st.slider("Durée de la montée en vitesse", min_value=0.5, max_value=10.0, step=0.5, **keep("chn_ramp"))
    with section("Temps", icon=":material/schedule:"):
        extras("time")
        time_widgets("channel")
    with section("Méthodes numériques (avancé)", icon=":material/tune:"):
        extras("numerics")
        numerics_widgets("chn", obstacle=False)
    with section("Sorties", icon=":material/insights:"):
        st.toggle("Aperçu en direct pendant le calcul", **keep("chn_live"))


#: Panneau de paramètres de chaque cas.
WIDGETS = {"obstacle": obstacle_widgets, "cavity": cavity_widgets, "channel": channel_widgets}

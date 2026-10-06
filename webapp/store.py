"""Enregistrement des calculs sur disque (``outputs/app_runs/``) : historique, relecture, calibration.

Chaque calcul occupe un dossier ``<identifiant>/`` contenant :

* ``params.json`` (paramètres ; l'image d'un obstacle est à part, ``image.bin``) ;
* ``meta.json`` (libellé, date, durée, grille, indicateurs principaux, étude paramétrique) ;
* ``state.npz`` (état final, point de reprise), ``diagnostics.npz`` (indicateurs de chaque pas) ;
* ``forces.npz`` (efforts Cd(t), Cl(t)...) et ``mean_state.npz`` (état moyen) pour un obstacle ;
* les animations déjà rendues (``animation_<grandeur>_<ips>.mp4`` ou ``.gif``).

``meta.json`` est écrit en dernier : un dossier sans ce fichier (calcul interrompu pendant
l'écriture) est ignoré.
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``json`` : paramètres et métadonnées lisibles ; ``shutil.rmtree`` : suppression d'un dossier.
import json
import shutil

# ``Any`` : valeurs de types variés.
from typing import Any

import numpy as np

# Paquet : solveur, état, diagnostics, efforts, points de reprise.
from cfd2d import Diagnostics, FlowState, NavierStokesSolver
from cfd2d import analytics as an
from cfd2d import io as cio

from .analysis import SimulationResult, analyze
from .common import RUNS_DIR
from .params import build_config

# Version du format d'enregistrement (lecture des anciens dossiers si le format évolue).
FORMAT_VERSION = 1


def _jsonable(value: Any) -> Any:
    """Convertit une valeur en type JSON (tuples -> listes, flottants NumPy -> float, NaN -> None)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def key_metrics(result: SimulationResult) -> dict[str, float | None]:
    """Indicateurs principaux d'un calcul (tableau de l'historique, études paramétriques)."""
    m: dict[str, float | None] = {}
    s, summary, ex = result.solver, result.summary, result.extra
    m["Re"] = s.reynolds
    if result.case == "obstacle" and summary is not None:
        m.update(Cd=summary.cd_mean, Cl=summary.cl_mean, Cl_rms=summary.cl_rms, Cl_amplitude=summary.cl_amplitude,
                 Cm=summary.cm_mean, St=summary.strouhal if summary.unsteady else None,
                 n_periods=summary.n_periods if summary.unsteady else None, Lr=result.recirculation,
                 blockage=ex.get("blockage"), L_D=ex.get("lift_to_drag"), St_blockage=ex.get("st_blockage"))
    elif result.case == "cavity":
        m.update(vortex_x=ex["vortex"][0], vortex_y=ex["vortex"][1], psi_min=ex["vortex"][2],
                 steadiness=ex["steadiness"])
        if "ghia_errors" in ex:
            m.update(ghia_u=ex["ghia_errors"][0], ghia_v=ex["ghia_errors"][1])
    elif result.case == "channel":
        m.update(entry_length=ex["entry_length"], dpdx_error=(ex["dpdx"] - ex["dpdx_analytic"]) / abs(ex["dpdx_analytic"]),
                 profile_error=ex["profile_error"])
    m["divergence"] = an.divergence_report(s).history_max
    return _jsonable(m)


def save_run(result: SimulationResult) -> None:
    """Enregistre (ou remplace) le dossier du calcul ``result.run_id``."""
    folder = RUNS_DIR / result.run_id
    folder.mkdir(parents=True, exist_ok=True)
    s = result.solver
    # Paramètres ; l'image éventuelle (octets) est écrite à part.
    params = dict(result.params)
    image = params.pop("image", None)
    if image is not None:
        (folder / "image.bin").write_bytes(image)
    (folder / "params.json").write_text(json.dumps(_jsonable(params), indent=1), encoding="utf-8")
    # État final (point de reprise), diagnostics de chaque pas, efforts et moyenne.
    cio.save_checkpoint(folder / "state.npz", s)
    np.savez_compressed(folder / "diagnostics.npz", **s.diagnostics.as_arrays())
    if result.history is not None:
        result.history.save(folder / "forces.npz")
    if result.mean_state is not None:
        ms = result.mean_state
        t0, t1 = result.mean_window or (np.nan, np.nan)
        np.savez_compressed(folder / "mean_state.npz", u=ms.u, v=ms.v, p=ms.p, t=ms.t, t0=t0, t1=t1,
                            duration=result.mean_duration)
    # Animations rendues (l'extension suit le type MIME).
    for (quantity, fps), (data, mime) in result.animations.items():
        (folder / f"animation_{quantity}_{fps}.{mime.split('/')[1]}").write_bytes(data)
    g = s.grid
    meta = {
        "format": FORMAT_VERSION, "run_id": result.run_id, "label": result.label, "created": result.created,
        "case": result.case, "shape": result.params.get("shape"), "grid": [g.Nx, g.Ny], "cells": g.Nx * g.Ny,
        "t": s.time, "steps": s.state.step, "run_time": result.run_time, "predicted_time": result.predicted_time,
        "stages": 3 if s.config.time.scheme == "rk3" else 1, "pressure_solver": s.config.numerics.pressure_solver,
        "stopped": result.stopped, "study": result.study, "warnings": result.warnings,
        "metrics": key_metrics(result),
    }
    # Écrit en dernier : marque un dossier complet.
    (folder / "meta.json").write_text(json.dumps(_jsonable(meta), indent=1), encoding="utf-8")


def list_runs() -> list[dict[str, Any]]:
    """Métadonnées de tous les calculs enregistrés, du plus récent au plus ancien."""
    if not RUNS_DIR.is_dir():
        return []
    runs = []
    for folder in RUNS_DIR.iterdir():
        meta_path = folder / "meta.json"
        if meta_path.is_file():
            try:
                runs.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                # Fichier illisible (écriture interrompue) : ignoré.
                continue
    # Tri par date de création décroissante (format ISO : l'ordre alphabétique est chronologique).
    return sorted(runs, key=lambda m: m.get("created", ""), reverse=True)


def load_params(run_id: str) -> dict[str, Any]:
    """Paramètres d'un calcul enregistré (image de l'obstacle comprise)."""
    folder = RUNS_DIR / run_id
    params = json.loads((folder / "params.json").read_text(encoding="utf-8"))
    # Les listes JSON redeviennent des tuples là où le paramètre en est un.
    if "stations" in params:
        params["stations"] = tuple(params["stations"])
    # Image de l'obstacle (cas « obstacle » seulement ; None pour les autres formes).
    if params.get("case") == "obstacle":
        image = folder / "image.bin"
        params["image"] = image.read_bytes() if image.is_file() else None
    return params


def load_history(run_id: str) -> an.ForceHistory | None:
    """Séries d'efforts d'un calcul enregistré (None pour une cavité ou un canal)."""
    path = RUNS_DIR / run_id / "forces.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        columns = {name: data[name] for name in data.files}
    # Grandeurs de référence (tableaux 0-D) et colonnes du volume de contrôle (facultatives).
    reference = {name: float(columns.pop(name)) for name in ("reference_length", "reference_velocity")}
    return an.ForceHistory(**columns, **reference)


def load_run(run_id: str) -> SimulationResult:
    """Relit un calcul enregistré : solveur reconstruit à son état final, analyse refaite."""
    folder = RUNS_DIR / run_id
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    params = load_params(run_id)
    # Solveur reconstruit à partir des paramètres, puis remis dans son état final.
    solver = NavierStokesSolver(build_config(params))
    cio.load_checkpoint(folder / "state.npz", solver)
    with np.load(folder / "diagnostics.npz") as data:
        # Diagnostics : une liste Python par indicateur (tolist convertit les tableaux).
        solver.diagnostics = Diagnostics(**{name: data[name].tolist() for name in data.files})
    result = SimulationResult(
        run_id=run_id, params=params, solver=solver, run_time=float(meta["run_time"]), warnings=list(meta["warnings"]),
        label=meta.get("label", ""), created=meta.get("created", ""), study=meta.get("study"),
        stopped=bool(meta.get("stopped")), from_disk=True, predicted_time=float(meta.get("predicted_time") or np.nan),
        history=load_history(run_id),
    )
    mean_path = folder / "mean_state.npz"
    if mean_path.is_file():
        with np.load(mean_path) as data:
            result.mean_state = FlowState(data["u"], data["v"], data["p"], t=float(data["t"]))
            result.mean_window = (float(data["t0"]), float(data["t1"]))
            result.mean_duration = float(data["duration"])
    # Animations déjà rendues (nom : animation_<grandeur>_<ips>.<ext>).
    for path in folder.glob("animation_*_*.*"):
        _, quantity, fps = path.stem.split("_")
        mime = "video/mp4" if path.suffix == ".mp4" else "image/gif"
        result.animations[(quantity, int(fps))] = (path.read_bytes(), mime)
    return analyze(result)


def delete_run(run_id: str) -> None:
    """Supprime le dossier d'un calcul enregistré."""
    folder = RUNS_DIR / run_id
    # Garde-fou : on ne supprime qu'un sous-dossier direct du dossier des calculs.
    if folder.parent == RUNS_DIR and folder.is_dir():
        shutil.rmtree(folder)


def calibration(pressure_solver: str, max_runs: int = 10) -> tuple[float, int]:
    """Facteur correctif des durées estimées : médiane de (durée mesurée / durée prévue).

    Calculée sur les ``max_runs`` derniers calculs complets du même solveur de pression ;
    (1, 0) s'il n'y en a pas encore.
    """
    ratios = []
    for meta in list_runs():
        predicted, measured = meta.get("predicted_time"), meta.get("run_time")
        if meta.get("pressure_solver") == pressure_solver and not meta.get("stopped") and predicted and measured:
            ratios.append(measured / predicted)
        if len(ratios) >= max_runs:
            break
    return (float(np.median(ratios)), len(ratios)) if ratios else (1.0, 0)

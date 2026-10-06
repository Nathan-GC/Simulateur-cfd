"""Exécution des calculs dans un fil d'arrière-plan : moniteurs, progression, arrêt, prolongation.

Un :class:`Job` (tâche) enchaîne une ou plusieurs demandes de calcul (:class:`RunRequest`) :
un calcul unique, les calculs d'une étude paramétrique, ou la prolongation d'un calcul
terminé. Le fil de calcul ne fait aucun appel à Streamlit : il met à jour les attributs de la
tâche (avancement, aperçu), que la page lit et affiche périodiquement (fragment Streamlit).
La page reste ainsi utilisable pendant le calcul, et un clic sur « Arrêter » interrompt le
calcul proprement (les résultats déjà calculés sont analysés).
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# Journalisation : capture des avertissements émis par le paquet pendant un calcul.
import logging

# Fil d'exécution du calcul et événement d'arrêt.
import threading

# Chronomètre (durée du calcul, cadence de l'affichage).
import time

# ``shutil.which`` : présence de l'exécutable ffmpeg (animation MP4 plutôt que GIF).
import shutil

# Dossier temporaire : l'animation est écrite dans un fichier puis relue en octets.
import tempfile

# ``uuid4`` : identifiant unique de chaque calcul.
import uuid

# ``dataclass`` : demande de calcul et tâche ; ``field`` : valeurs par défaut mutables.
from dataclasses import dataclass, field

# Date de création des calculs (identifiant et historique).
from datetime import datetime

# Chemins de fichiers portables.
from pathlib import Path

# ``Any`` : valeurs de types variés.
from typing import Any

import numpy as np

# Paquet : configuration, solveur, moniteurs, état, exception de divergence, visualisation.
from cfd2d import (
    FieldAverager,
    FlowState,
    ForceMonitor,
    NavierStokesSolver,
    SimulationConfig,
    SimulationDivergedError,
)
from cfd2d import analytics as an
from cfd2d import visualization as viz

from .analysis import SimulationResult, analyze
from .common import SHAPES, format_duration, friendly_message
from .preview import live_caption, live_frame
from .store import save_run

# Fractions de la durée simulée auxquelles démarrent les moyennes temporelles « automatiques ».
AVERAGING_FRACTIONS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# Mémoire maximale des instantanés conservés pour les animations et nombre maximal d'instantanés.
SNAPSHOT_MEMORY, MAX_SNAPSHOTS = 300e6, 90
# Intervalles de rafraîchissement de l'avancement et de l'aperçu en direct (secondes).
PROGRESS_EVERY, PREVIEW_EVERY = 0.3, 1.5
# Verrou des rendus matplotlib (les paramètres de style sont globaux : un rendu à la fois).
MATPLOTLIB_LOCK = threading.RLock()


class RunCancelled(Exception):
    """Levée dans le fil de calcul quand l'utilisateur demande l'arrêt."""


class WarningCollector(logging.Handler):
    """Récupère les avertissements du paquet émis par un fil d'exécution donné."""

    def __init__(self, thread_id: int) -> None:
        super().__init__(level=logging.WARNING)
        # Fil du calcul : les messages d'autres calculs (autres fils) sont ignorés.
        self.thread_id = thread_id
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        # Message traduit en français ; pas de doublon.
        message = friendly_message(record.getMessage())
        if record.thread == self.thread_id and message not in self.messages:
            self.messages.append(message)


class SnapshotRecorder:
    """Moniteur : conserve des instantanés (u, v, p) à intervalles réguliers de temps physique.

    Les instantanés, en simple précision, servent à animer après coup n'importe quelle grandeur
    (vorticité, vitesse, pression) et à parcourir la fin du calcul avec un curseur. Au-delà de
    ``max_frames``, les plus anciens sont oubliés (fenêtre glissante, utile en prolongation).
    """

    def __init__(self, t_start: float, interval: float, max_frames: int) -> None:
        self.interval, self.max_frames, self.next_time = interval, max_frames, t_start
        self.states: list[FlowState] = []

    def __call__(self, solver: NavierStokesSolver) -> None:
        t = solver.time
        if t + 1e-12 < self.next_time:
            return
        st = solver.state
        # Copie en float32 (moitié de la mémoire d'un float64).
        self.states.append(FlowState(st.u.astype(np.float32), st.v.astype(np.float32), st.p.astype(np.float32),
                                     t=t, step=st.step))
        if len(self.states) > self.max_frames:
            del self.states[0]
        # Instant suivant (on saute les instants dépassés si le pas de temps est grand).
        while self.next_time <= t + 1e-12:
            self.next_time += self.interval


@dataclass
class RunRequest:
    """Demande de calcul : paramètres et configuration (construite dans le fil de la page).

    ``base`` : résultat à prolonger de ``extra_time`` (sinon nouveau calcul).
    """

    params: dict[str, Any]
    config: SimulationConfig | None
    label: str
    predicted_time: float = float("nan")
    study: dict[str, Any] | None = None
    base: SimulationResult | None = None
    extra_time: float = 0.0


@dataclass
class Job:
    """Tâche de calcul suivie par la page (attributs mis à jour par le fil de calcul)."""

    requests: list[RunRequest]
    # État : "pending", "running", "done" ou "stopped" ; indice du calcul en cours.
    status: str = "pending"
    index: int = 0
    # Avancement (0 à 1, toutes demandes confondues), texte d'état, aperçu en direct.
    fraction: float = 0.0
    text: str = "Préparation du solveur (assemblage et factorisation de la matrice de pression)…"
    preview: np.ndarray | None = None
    preview_caption: str = ""
    # Résultats obtenus et erreurs rencontrées.
    results: list[SimulationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Demande d'arrêt (positionnée par le bouton « Arrêter »).
    cancel: threading.Event = field(default_factory=threading.Event)
    started: float = field(default_factory=time.perf_counter)

    @property
    def active(self) -> bool:
        return self.status in ("pending", "running")

    @property
    def total(self) -> int:
        return len(self.requests)


class JobProgress:
    """Moniteur d'avancement : met à jour la tâche et vérifie la demande d'arrêt à chaque pas."""

    def __init__(self, job: Job, solver: NavierStokesSolver, t_end: float, live: bool, forces: ForceMonitor | None) -> None:
        self.job, self.t_end, self.live, self.forces = job, t_end, live, forces
        # Grandeur de l'aperçu : vorticité autour d'un obstacle, vitesse sinon.
        self.quantity = "vorticity" if solver.obstacles else "speed"
        self.t0, self.tic = solver.time, time.perf_counter()
        self.step0 = solver.state.step
        # Dernières mises à jour (premier aperçu après 1 s).
        self.last_text, self.last_preview = -np.inf, self.tic - PREVIEW_EVERY + 1.0

    def __call__(self, solver: NavierStokesSolver) -> None:
        job = self.job
        # Arrêt demandé : interrompt solver.run() (l'exception remonte jusqu'à execute).
        if job.cancel.is_set():
            raise RunCancelled
        now = time.perf_counter()
        if now - self.last_text >= PROGRESS_EVERY:
            self.last_text = now
            d = solver.diagnostics
            fraction = (solver.time - self.t0) / max(self.t_end - self.t0, 1e-12)
            elapsed = now - self.tic
            remaining = elapsed * (1.0 - fraction) / fraction if fraction > 0 else float("nan")
            # Avancement global : calculs déjà terminés + fraction du calcul courant.
            job.fraction = float(np.clip((job.index + fraction) / job.total, 0.0, 1.0))
            prefix = f"Calcul {job.index + 1}/{job.total} · " if job.total > 1 else ""
            job.text = (
                f"{prefix}t = {solver.time:.2f} / {self.t_end:.4g} · pas {solver.state.step} · Δt = {d.dt[-1]:.2e} · "
                f"CFL = {d.cfl[-1]:.2f} · {(solver.state.step - self.step0) / max(elapsed, 1e-9):.0f} pas/s · "
                f"reste ≈ {format_duration(remaining)}"
            )
        if self.live and now - self.last_preview >= PREVIEW_EVERY:
            self.last_preview = now
            caption = f"Aperçu en direct à t = {solver.time:.2f} : {live_caption(self.quantity)}"
            if self.forces is not None:
                # Coefficients instantanés (calcul local, ~1 ms).
                c = self.forces.coefficients(solver)
                caption += f" · Cd = {c['cd']:.3f} · Cl = {c['cl']:+.3f}"
            job.preview, job.preview_caption = live_frame(solver, self.quantity), caption


def control_box(solver: NavierStokesSolver) -> tuple[float, float, float, float]:
    """Volume de contrôle de 4 L x 4 L autour de l'obstacle, ramené à l'intérieur du domaine."""
    g, L = solver.grid, solver.L_ref
    xc, yc = solver.obstacles[0].center
    # np.clip borne chaque côté à deux mailles des bords du domaine.
    return (
        float(np.clip(xc - 1.5 * L, 2 * g.dx, g.Lx - 2 * g.dx)),
        float(np.clip(xc + 2.5 * L, 2 * g.dx, g.Lx - 2 * g.dx)),
        float(np.clip(yc - 2.0 * L, 2 * g.dy, g.Ly - 2 * g.dy)),
        float(np.clip(yc + 2.0 * L, 2 * g.dy, g.Ly - 2 * g.dy)),
    )


def attach_monitors(p: dict[str, Any], solver: NavierStokesSolver, t_end: float, messages: list[str]) -> dict[str, Any]:
    """Moniteurs d'un calcul d'obstacle : efforts, moyennes, instantanés (à partir de l'instant courant)."""
    monitors: dict[str, Any] = {}
    if p["case"] != "obstacle":
        return monitors
    # Efforts à chaque pas (avec vérification optionnelle par volume de contrôle).
    try:
        monitors["forces"] = ForceMonitor(solver, control_volume=control_box(solver) if p["control_volume"] else None)
    except ValueError as exc:
        messages.append(f"Volume de contrôle ignoré : {exc}")
        monitors["forces"] = ForceMonitor(solver)
    # Moyennes temporelles tous les 5 pas : plusieurs départs (automatique) ou départ imposé.
    t0 = solver.time
    fractions = AVERAGING_FRACTIONS if p["avg_start"] is None else (p["avg_start"],)
    monitors["averagers"] = [FieldAverager(solver, t_start=t0 + f * (t_end - t0), every=5) for f in fractions]
    if p["animation"]:
        # Nombre d'instantanés limité par la mémoire (u, v, p en float32 : ~12 octets par cellule).
        cells = solver.grid.Nx * solver.grid.Ny
        n_frames = int(np.clip(SNAPSHOT_MEMORY / (12 * cells), 10, MAX_SNAPSHOTS))
        duration = min(p["anim_duration"], t_end - t0)
        recorder = SnapshotRecorder(t_end - duration, duration / n_frames, n_frames)
        solver.add_callback(recorder)
        monitors["snapshots"] = recorder
    return monitors


def merge_histories(old: an.ForceHistory | None, new: an.ForceHistory | None) -> an.ForceHistory | None:
    """Concatène deux séries d'efforts consécutives (prolongation d'un calcul relu sur disque)."""
    if old is None or new is None:
        return new or old
    names = ("time", "cd", "cl", "cd_pressure", "cd_viscous", "cd_convective", "cl_pressure", "cl_viscous",
             "cl_convective", "cm")
    columns = {name: np.concatenate([getattr(old, name), getattr(new, name)]) for name in names}
    return an.ForceHistory(**columns, reference_length=new.reference_length, reference_velocity=new.reference_velocity)


def render_animation(result: SimulationResult, quantity: str, fps: int) -> tuple[bytes, str] | None:
    """Animation MP4 (ffmpeg) ou GIF (Pillow) des instantanés conservés : (octets, type MIME)."""
    states = result.snapshots
    if len(states) < 2:
        return None
    solver = result.solver
    # Images de la grandeur demandée, calculées à partir des instantanés.
    recorder = viz.FrameRecorder(solver, quantity=quantity, attach=False)
    for state in states:
        recorder.record(state, solver)
    view = viz.default_view(solver)
    # MP4 si ffmpeg est disponible, GIF en secours (résolution réduite).
    formats = ([(".mp4", "video/mp4", 80)] if shutil.which("ffmpeg") else []) + [(".gif", "image/gif", 60)]
    for suffix, mime, dpi in formats:
        try:
            with MATPLOTLIB_LOCK, tempfile.TemporaryDirectory() as tmp:
                path = viz.animate(recorder, Path(tmp) / f"animation{suffix}", fps=fps, view=view, dpi=dpi)
                return path.read_bytes(), mime
        except (RuntimeError, OSError):
            # Échec d'ffmpeg (ou tube interrompu) : format suivant.
            continue
    return None


def run_label(p: dict[str, Any]) -> str:
    """Libellé court d'un calcul (historique, comparaisons)."""
    re_text = f"Re = {p['Re']:g}" if p["Re"] is not None else f"ν = {p['nu']:g}"
    if p["case"] == "obstacle":
        shape = SHAPES[p["shape"]].split(" ")[0].lower()
        if p["shape"] == "naca":
            shape = f"NACA {p['naca_code']} α = {p['incidence']:g}°"
        elif p["shape"] == "rectangle":
            shape = f"plaque α = {p['incidence']:g}°"
        return f"{shape}, {re_text}, {p['cells']} mailles/D, T = {p['t_end']:g}"
    if p["case"] == "cavity":
        return f"cavité, {re_text}, N = {p['N']}, T = {p['t_end']:g}"
    return f"canal, {re_text}, {p['Ny']} mailles/H, L = {p['length']:g} H"


def execute(request: RunRequest, job: Job) -> SimulationResult | None:
    """Exécute une demande (nouveau calcul ou prolongation) puis l'analyse ; None si rien à analyser."""
    p = request.params
    collector = WarningCollector(threading.get_ident())
    package_logger = logging.getLogger("cfd2d")
    package_logger.addHandler(collector)
    base = request.base
    try:
        if base is None:
            solver = NavierStokesSolver(request.config)
            t_end = p["t_end"]
            monitors = attach_monitors(p, solver, t_end, collector.messages)
            previous_time, history_before = 0.0, None
        else:
            solver = base.solver
            t_end = solver.time + request.extra_time
            previous_time = base.run_time
            if base.monitors:
                # Calcul de la session : les moniteurs déjà attachés continuent d'enregistrer.
                monitors, history_before = base.monitors, None
            else:
                # Calcul relu sur disque : nouveaux moniteurs, efforts concaténés à l'historique relu.
                monitors = attach_monitors(p, solver, t_end, collector.messages)
                history_before = base.history
        progress = JobProgress(job, solver, t_end, p["live_preview"], monitors.get("forces"))
        solver.add_callback(progress)
        tic, stopped = time.perf_counter(), False
        try:
            # log_every=0 : pas de lignes de progression dans le terminal (la page les affiche).
            solver.run(t_end=t_end, log_every=0)
        except RunCancelled:
            stopped = True
        finally:
            solver.remove_callback(progress)
        run_time = previous_time + time.perf_counter() - tic
        # Nouveau calcul arrêté avant le moindre pas utile : rien à analyser.
        if base is None and solver.state.step < 10:
            return None
        job.text = "Post-traitement : analyse des résultats…"
        forces = monitors.get("forces")
        history = forces.history() if forces is not None else None
        if base is None:
            run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{p['case']}_{uuid.uuid4().hex[:6]}"
            label, created = request.label, f"{datetime.now():%Y-%m-%d %H:%M:%S}"
        else:
            # Prolongation : même identifiant (le dossier enregistré est remplacé).
            run_id, created = base.run_id, base.created
            label = base.label if "(prolongé)" in base.label else base.label + " (prolongé)"
        result = SimulationResult(
            run_id=run_id,
            # Durée simulée effective (prolongation, arrêt) enregistrée dans les paramètres.
            params=dict(p, t_end=float(solver.time)) if base is not None or stopped else p,
            solver=solver, run_time=run_time, warnings=collector.messages, label=label, created=created,
            study=request.study if base is None else base.study, stopped=stopped,
            predicted_time=request.predicted_time if base is None else float("nan"),
            history=merge_histories(history_before, history) if history_before is not None else history,
            averagers=monitors.get("averagers", []),
            snapshots=monitors["snapshots"].states if "snapshots" in monitors else [],
            monitors=monitors,
        )
        if stopped:
            result.warnings.insert(0, f"Calcul arrêté à t = {solver.time:.2f} (demande de l'utilisateur).")
        analyze(result)
        # Animation par défaut (vorticité) rendue tout de suite : prête à l'affichage et enregistrée.
        if result.snapshots:
            job.text = f"Post-traitement : rendu de l'animation ({len(result.snapshots)} images)…"
            animation = render_animation(result, "vorticity", p["fps"])
            if animation is not None:
                result.animations[("vorticity", p["fps"])] = animation
        job.text = "Enregistrement des résultats…"
        save_run(result)
        return result
    finally:
        # Toujours retirer le collecteur (même en cas d'erreur).
        package_logger.removeHandler(collector)


def run_job(job: Job) -> None:
    """Exécute les demandes de la tâche l'une après l'autre (corps du fil de calcul)."""
    job.status = "running"
    for k, request in enumerate(job.requests):
        if job.cancel.is_set():
            break
        job.index = k
        try:
            result = execute(request, job)
            if result is not None:
                job.results.append(result)
        except SimulationDivergedError as exc:
            # Instabilité numérique : message du solveur et pistes de correction.
            job.errors.append(
                f"{request.label} : le calcul a divergé ({exc}). Pistes : réduire le CFL ou Δt, raffiner la grille, "
                "choisir un schéma d'advection plus dissipatif (TVD, décentré) ou RK3."
            )
        except (ValueError, RuntimeError, ImportError, MemoryError) as exc:
            job.errors.append(f"{request.label} : le calcul n'a pas pu aboutir ({exc}).")
    job.fraction = 1.0
    job.status = "stopped" if job.cancel.is_set() else "done"


def start_in_background(job: Job) -> None:
    """Lance la tâche dans un fil d'arrière-plan (la page reste utilisable pendant le calcul)."""
    # État « running » posé avant le démarrage : une nouvelle exécution de la page ne relance pas la tâche.
    job.status = "running"
    # daemon=True : le fil ne bloque pas l'arrêt du serveur.
    threading.Thread(target=run_job, args=(job,), daemon=True, name="cfd2d-calcul").start()

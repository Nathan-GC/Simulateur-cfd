"""Résultat d'un calcul et post-traitement : efforts, moyennes, sillage, validation, lecture commentée."""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``dataclass`` : conteneur des résultats ; ``field`` : valeurs par défaut mutables.
from dataclasses import dataclass, field

# ``Any`` : valeurs de types variés.
from typing import Any

import numpy as np

# Paquet : solveur, état, analyse et données de référence.
from cfd2d import FieldAverager, FlowState, NavierStokesSolver
from cfd2d import analytics as an
from cfd2d.validation import (
    GHIA,
    GHIA_PRIMARY_VORTEX,
    cavity_centerlines,
    cylinder_reference,
    ghia_errors,
    williamson_strouhal,
)


@dataclass
class SimulationResult:
    """Résultats d'un calcul, conservés en session (et sur disque, voir :mod:`webapp.store`)."""

    # Identifiant, paramètres, solveur (état final, grille, diagnostics), durée de calcul, messages.
    run_id: str
    params: dict[str, Any]
    solver: NavierStokesSolver
    run_time: float
    warnings: list[str]
    # Libellé lisible, date de création, appartenance à une étude paramétrique.
    label: str = ""
    created: str = ""
    study: dict[str, Any] | None = None
    # Calcul arrêté par l'utilisateur, relu sur disque ; durée prévue (calibration des estimations).
    stopped: bool = False
    from_disk: bool = False
    predicted_time: float = float("nan")
    # Données accumulées pendant le calcul : efforts, moyennes, instantanés, moniteurs attachés.
    history: an.ForceHistory | None = None
    averagers: list[FieldAverager] = field(default_factory=list)
    snapshots: list[FlowState] = field(default_factory=list)
    monitors: dict[str, Any] = field(default_factory=dict)
    # Moyenne retenue : état moyen (faces), fenêtre et durée.
    mean_state: FlowState | None = None
    mean_window: tuple[float, float] | None = None
    mean_duration: float = 0.0
    # Résultats de l'analyse (voir :func:`analyze`).
    fields: an.FlowFields | None = None
    mean: an.FlowFields | None = None
    summary: an.ForceSummary | None = None
    spectrum: an.Spectrum | None = None
    cp: an.SurfaceDistribution | None = None
    cp_chord: an.ChordwiseDistribution | None = None
    profiles: list[an.WakeProfile] = field(default_factory=list)
    recirculation: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)
    comments: list[str] = field(default_factory=list)
    # Animations déjà rendues : (grandeur, images par seconde) -> (octets, type MIME).
    animations: dict[tuple[str, int], tuple[bytes, str]] = field(default_factory=dict)
    # Figures déjà calculées (évite de les recalculer à chaque interaction).
    cache: dict[Any, Any] = field(default_factory=dict)

    @property
    def case(self) -> str:
        return self.params["case"]

    @property
    def slender(self) -> bool:
        """Corps élancé (profil, plaque) : Cp le long de la corde, sillage depuis l'arrière."""
        return bool(self.solver.obstacles) and an.chord_line(self.solver.obstacles[0]) is not None


def memo(result: SimulationResult, key: Any, factory) -> Any:
    """Valeur mémorisée dans ``result.cache`` (calculée par ``factory()`` au premier appel)."""
    if key not in result.cache:
        result.cache[key] = factory()
    return result.cache[key]


# ============================================================================ analyse
def analyze(result: SimulationResult) -> SimulationResult:
    """Calcule toutes les grandeurs d'analyse du résultat (appelée après le calcul ou la relecture)."""
    result.fields = an.compute_fields(result.solver)
    result.extra = {}
    if result.case == "obstacle":
        _analyze_obstacle(result)
    elif result.case == "cavity":
        _analyze_cavity(result)
    else:
        _analyze_channel(result)
    result.comments = interpret(result)
    # Les figures mémorisées ne correspondent plus aux nouveaux résultats.
    result.cache.clear()
    return result


def _choose_mean(result: SimulationResult) -> None:
    """Retient la moyenne temporelle commençant dans le régime établi (point de départ automatique).

    Plusieurs moyennes démarrent pendant le calcul (30 %, 40 %... 80 % de la durée) : on garde
    la première qui commence après le début du régime établi détecté sur Cl. Avec un début
    imposé, la moyenne unique est gardée et un avertissement signale un départ trop précoce.
    """
    s, summary = result.solver, result.summary
    candidates = [a for a in result.averagers if a.duration > 0.0]
    if not candidates:
        # Résultat relu sur disque : la moyenne enregistrée est conservée telle quelle.
        return
    # Début souhaité : régime établi détecté (instationnaire) ou dernier quart (stationnaire).
    target = summary.t_start if summary is not None else 0.5 * s.time
    if result.params.get("avg_start") is None:
        after = [a for a in candidates if a.t_start >= target - 1e-9]
        chosen = min(after, key=lambda a: a.t_start) if after else max(candidates, key=lambda a: a.t_start)
        if not after:
            result.warnings.append(f"Moyennes : le régime établi n'est détecté qu'à t = {target:.1f} ; la moyenne "
                                   f"retenue commence avant (t = {chosen.t_start:.1f}). Prolongez le calcul.")
    else:
        chosen = candidates[0]
        if summary is not None and summary.unsteady and chosen.t_start < target - 1e-6:
            result.warnings.append(f"Les moyennes ont commencé à t = {chosen.t_start:.1f}, avant le régime établi "
                                   f"(t = {target:.1f}) : Cp moyen et sillage incluent du transitoire.")
    result.mean_state = chosen.mean_state()
    result.mean_window = (chosen.t_start, s.time)
    result.mean_duration = chosen.duration


def _analyze_obstacle(result: SimulationResult) -> None:
    """Efforts, spectre, moyennes, Cp pariétal, sillage, blocage et références."""
    s, p, ex = result.solver, result.params, result.extra
    ob = s.obstacles[0]
    L, U = s.L_ref, s.U_ref
    history = result.history
    result.summary = result.spectrum = None
    # Synthèse (au moins 8 échantillons) et spectre si l'écoulement oscille.
    if history is not None and history.time.size >= 8:
        result.summary = history.summary()
        if result.summary.unsteady:
            result.spectrum = history.spectrum()
            # Vérification indépendante de la fréquence par les passages à zéro de Cl.
            f0 = an.crossing_frequency(history.time, history.cl, result.summary.t_start)
            ex["st_crossing"] = f0 * history.reference_length / history.reference_velocity
    # Moyenne retenue et champs moyens correspondants.
    _choose_mean(result)
    ms = result.mean_state
    result.mean = an.fields_from_arrays(s.grid, s.solid, ms.u, ms.v, ms.p, ms.t) if ms is not None else None
    # Cp le long du contour exact ; le long de la corde pour un corps élancé.
    chord = an.chord_line(ob)
    result.cp = an.surface_pressure(s, fields=result.mean) if ob.outline() is not None else None
    result.cp_chord = an.chordwise_pressure(result.cp, *chord) if result.cp is not None and chord is not None else None
    # Stations du sillage (depuis l'arrière d'un corps élancé, le centre sinon), dans le domaine.
    origin = "rear" if chord is not None else "center"
    x0 = ob.bounds[1] if chord is not None else ob.center[0]
    stations = tuple(st for st in p["stations"] if x0 + st * L <= s.grid.x_c[-1])
    result.profiles = an.wake_profiles(s, stations, fields=result.mean, origin=origin) if stations else []
    ex["wake_origin"] = origin
    # Recirculation sur l'axe : seulement pour un corps non profilé (le sillage d'un corps incliné dévie).
    result.recirculation = an.recirculation_length(s, fields=result.mean) if chord is None else float("nan")
    # Blocage (hauteur frontale / hauteur du domaine) et Strouhal corrigé par conservation du débit :
    # au droit de l'obstacle, la vitesse moyenne entre les parois vaut U / (1 - β).
    xmin, xmax, ymin, ymax = ob.bounds
    beta = (ymax - ymin) / s.grid.Ly
    ex["blockage"] = beta
    summary = result.summary
    if summary is not None and summary.unsteady:
        ex["st_blockage"] = summary.strouhal * (1.0 - beta)
    # Références du cylindre : loi de Williamson et fourchettes de la littérature (milieu infini).
    if p["shape"] == "cylinder":
        st_w = williamson_strouhal(s.reynolds)
        if np.isfinite(st_w):
            ex["st_williamson"] = st_w
        ex["references"] = cylinder_reference(s.reynolds)
    # Corps élancé : finesse et comparaison à la théorie des profils minces (fluide parfait).
    if chord is not None and summary is not None and abs(summary.cd_mean) > 1e-12:
        ex["lift_to_drag"] = summary.cl_mean / summary.cd_mean
        if p["shape"] == "naca":
            ex["cl_thin_airfoil"] = thin_airfoil_cl(p["naca_code"], p["incidence"])


def thin_airfoil_cl(code: str, alpha_deg: float) -> float:
    """Portance d'un profil NACA 4 chiffres selon la théorie des profils minces (fluide parfait).

    ``Cl = 2π (α - α₀)``, où l'incidence de portance nulle ``α₀`` dépend de la ligne moyenne :
    ``α₀ = -(1/π) ∫₀^π (dy_c/dx)(cos θ - 1) dθ`` avec ``x = (1 - cos θ)/2``.
    """
    m, p = int(code[0]) / 100.0, int(code[1]) / 10.0
    theta = np.linspace(0.0, np.pi, 4001)
    x = 0.5 * (1.0 - np.cos(theta))
    # Pente de la ligne moyenne NACA (deux arcs de parabole de part et d'autre de x = p).
    slope = np.zeros_like(x)
    if m > 0 and p > 0:
        slope = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    # np.trapezoid : intégrale par la méthode des trapèzes.
    alpha0 = -np.trapezoid(slope * (np.cos(theta) - 1.0), theta) / np.pi
    return float(2.0 * np.pi * (np.radians(alpha_deg) - alpha0))


def _analyze_cavity(result: SimulationResult) -> None:
    """Profils médians (comparaison à Ghia), tourbillon principal, stationnarité."""
    s, ex = result.solver, result.extra
    U, L = s.U_ref, s.L_ref
    # Profils u(y) sur l'axe vertical et v(x) sur l'axe horizontal, adimensionnés par U.
    y, u, x, v = cavity_centerlines(s)
    ex["centerlines"] = (y, u / U, x, v / U)
    # Table de Ghia la plus proche (à 1 % près) : Re = 100, 400 ou 1000.
    for Re in GHIA:
        if abs(s.reynolds - Re) <= 0.01 * Re:
            ex["ghia_Re"] = Re
            ex["ghia_errors"] = ghia_errors(s, Re=Re)
            ex["ghia_vortex"] = GHIA_PRIMARY_VORTEX[Re]
    # Centre du tourbillon principal : minimum de la fonction de courant adimensionnée ψ/(U L).
    psi = an.stream_function(s.state.u, s.state.v, s.grid) / (U * L)
    i, j = np.unravel_index(int(np.argmin(psi)), psi.shape)
    ex["vortex"] = (float(s.grid.x_f[i]), float(s.grid.y_f[j]), float(psi.min()))
    # Stationnarité : variation relative de l'énergie cinétique sur les 10 derniers % du calcul.
    d = s.diagnostics.as_arrays()
    tail = d["time"] >= 0.9 * d["time"][-1]
    energy = d["kinetic_energy"][tail]
    ex["steadiness"] = float(np.ptp(energy) / max(abs(energy[-1]), 1e-30))


def _analyze_channel(result: SimulationResult) -> None:
    """Profils de vitesse, vitesse sur l'axe, gradient de pression (comparaison à Poiseuille)."""
    s, fields, ex = result.solver, result.fields, result.extra
    g, U = s.grid, s.U_ref
    H = g.Ly
    # Vitesse et pression sur l'axe y = H/2 (interpolation aux centres des cellules).
    axis = np.full(g.Nx, 0.5 * H)
    u_axis = an.interpolate(fields.u, g, g.x_c, axis) / U
    p_axis = an.interpolate(fields.p, g, g.x_c, axis)
    # Profils u(y) sur les faces les plus proches des stations x/H.
    from .common import CHANNEL_STATIONS

    stations = [st for st in CHANNEL_STATIONS if st * H <= 0.95 * g.Lx]
    profiles = [(st, s.state.u[int(round(st * H / g.dx)), 1:-1] / U) for st in stations]
    # Longueur d'établissement : première abscisse où u_axe atteint 99 % de 1,5 U.
    reached = np.flatnonzero(u_axis >= 0.99 * 1.5)
    entry = float(g.x_c[reached[0]] / H) if reached.size else float("nan")
    # Gradient de pression : pente (moindres carrés) sur la zone établie, après la longueur
    # d'établissement et avant la sortie (au moins la seconde moitié du canal).
    x_start = max(0.5 * g.Lx, (entry + 0.5) * H) if np.isfinite(entry) else 0.5 * g.Lx
    sel = (g.x_c >= x_start) & (g.x_c <= 0.95 * g.Lx)
    slope = float(np.polyfit(g.x_c[sel], p_axis[sel], 1)[0]) if sel.sum() >= 2 else float("nan")
    eta = g.y_c / H
    ex.update(
        u_axis=u_axis, p_axis=p_axis, profiles=profiles, eta=eta, dpdx=slope,
        # Solution de Poiseuille : dp/dx = -12 μ U / H².
        dpdx_analytic=-12.0 * s.rho * s.nu * U / H**2, entry_length=entry, fit_start=float(x_start / H),
        # Écart maximal du dernier profil à la parabole 6 η (1 - η).
        profile_error=float(np.abs(profiles[-1][1] - 6.0 * eta * (1.0 - eta)).max()) if profiles else float("nan"),
    )


# ===================================================================== lecture commentée
def _fmt(x: float, digits: int = 3) -> str:
    """Nombre avec virgule décimale (texte en français)."""
    return f"{x:.{digits}f}".replace(".", ",")


def interpret(result: SimulationResult) -> list[str]:
    """Commentaires automatiques : régime, comparaison aux références, fiabilité, conseils."""
    if result.case == "obstacle":
        notes = _interpret_obstacle(result)
    elif result.case == "cavity":
        notes = _interpret_cavity(result)
    else:
        notes = _interpret_channel(result)
    # Incompressibilité (toujours vérifiée par la projection).
    div = an.divergence_report(result.solver).history_max
    notes.append(f"Incompressibilité : max |∇·u| = {div:.1e} sur tout le calcul (précision machine), conséquence "
                 "directe de la projection.")
    if result.stopped:
        notes.insert(0, f"⚠️ Calcul arrêté avant la fin (t = {_fmt(result.solver.time, 2)}) : les grandeurs portent "
                        "sur la partie calculée.")
    return notes


def _interpret_obstacle(result: SimulationResult) -> list[str]:
    s, p, ex, summary = result.solver, result.params, result.extra, result.summary
    notes: list[str] = []
    Re = s.reynolds
    if summary is None:
        return ["Historique trop court pour une analyse : allongez la durée simulée."]
    beta = ex.get("blockage", 0.0)
    # Régime de l'écoulement.
    if p["shape"] == "cylinder":
        if summary.unsteady and Re < 47:
            notes.append(f"Oscillations détectées à Re = {Re:.4g} < 47 : elles devraient s'amortir (régime stationnaire). "
                         "Mettre la perturbation à 0 ou prolonger le calcul.")
        elif not summary.unsteady and Re > 50:
            notes.append(f"Écoulement encore stationnaire à Re = {Re:.4g} > 47 : le lâcher tourbillonnaire n'est pas "
                         "déclenché (prolonger le calcul ou ajouter une perturbation initiale).")
        elif summary.unsteady:
            notes.append(f"Re = {Re:.4g} : allée de Von Kármán périodique, comme attendu au-delà de Re ≈ 47.")
        else:
            notes.append(f"Re = {Re:.4g} : écoulement stationnaire à deux tourbillons attachés, comme attendu sous Re ≈ 47.")
    # Strouhal : fiabilité, loi de Williamson et effet du blocage.
    if summary.unsteady:
        n = summary.n_periods
        quality = "fiable" if n >= 10 else "acceptable" if n >= 5 else "peu fiable"
        notes.append(f"Strouhal St = {_fmt(summary.strouhal)} sur {_fmt(n, 1)} périodes ({quality} ; 10 périodes au "
                     f"moins conseillées) ; passages par zéro : {_fmt(ex.get('st_crossing', float('nan')))}.")
        if "st_williamson" in ex:
            notes.append(f"Loi de Williamson (milieu infini) : St = {_fmt(ex['st_williamson'])}. Le blocage de "
                         f"{_fmt(100 * beta, 1)} % accélère l'écoulement entre les parois ; corrigé par conservation du "
                         f"débit (vitesse U/(1 − β)), St ≈ {_fmt(ex['st_blockage'])}.")
    # Traînée et recirculation comparées à la littérature.
    refs = ex.get("references")
    if refs:
        lo, hi = refs["Cd"]
        notes.append(f"Cd = {_fmt(summary.cd_mean)} ; littérature en milieu infini : {_fmt(lo, 2)} à {_fmt(hi, 2)}"
                     + (f". L'écart vient surtout du blocage ({_fmt(100 * beta, 1)} %) : agrandir la hauteur du domaine "
                        "pour s'en approcher." if beta > 0.05 else "."))
        if "Lr" in refs and np.isfinite(result.recirculation):
            lo, hi = refs["Lr"]
            notes.append(f"Recirculation Lr/D = {_fmt(result.recirculation, 2)} ; littérature : {_fmt(lo, 2)} à "
                         f"{_fmt(hi, 2)} (Dennis & Chang 1970, Fornberg 1980).")
        if "Cl_amplitude" in refs and summary.unsteady:
            lo, hi = refs["Cl_amplitude"]
            notes.append(f"Amplitude de Cl = {_fmt(summary.cl_amplitude)} ; littérature : {_fmt(lo, 2)} à {_fmt(hi, 2)}.")
    # Corps élancé : portance, finesse et théorie des profils minces.
    if result.slender:
        notes.append(f"Cl = {_fmt(summary.cl_mean)}, Cd = {_fmt(summary.cd_mean)} : finesse L/D = "
                     f"{_fmt(ex.get('lift_to_drag', float('nan')), 2)}.")
        if "cl_thin_airfoil" in ex:
            notes.append(f"Théorie des profils minces (fluide parfait) : Cl ≈ {_fmt(ex['cl_thin_airfoil'], 2)}. À "
                         f"Re = {Re:.4g}, la viscosité (couches limites épaisses, décollement) réduit fortement la "
                         "portance : l'écart est attendu à bas Reynolds.")
    # Moyennes et résolution.
    if result.mean_window is not None:
        t0, t1 = result.mean_window
        notes.append(f"Champs moyens (Cp, sillage) calculés sur t ∈ [{_fmt(t0, 1)} ; {_fmt(t1, 1)}].")
    cells = p["cells"]
    if cells < 16:
        notes.append(f"Maillage grossier ({cells} mailles par D) : vérifiez la convergence avec une étude paramétrique "
                     "en résolution (12, 18, 27 mailles par D).")
    return notes


def _interpret_cavity(result: SimulationResult) -> list[str]:
    ex = result.extra
    notes: list[str] = []
    if "ghia_errors" in ex:
        err_u, err_v = ex["ghia_errors"]
        verdict = "très bon accord" if max(err_u, err_v) < 0.01 else "accord correct" if max(err_u, err_v) < 0.03 else \
            "écart notable : raffiner le maillage ou prolonger le calcul"
        notes.append(f"Écart maximal à Ghia et al. (1982, Re = {ex['ghia_Re']}) : {_fmt(err_u, 4)} sur u et "
                     f"{_fmt(err_v, 4)} sur v (en vitesse du couvercle) : {verdict}.")
        x, y, psi = ex["vortex"]
        gx, gy, gpsi = ex["ghia_vortex"]
        notes.append(f"Tourbillon principal en ({_fmt(x)} ; {_fmt(y)}), ψ = {_fmt(psi, 4)} ; Ghia : ({_fmt(gx, 4)} ; "
                     f"{_fmt(gy, 4)}), ψ = {_fmt(gpsi, 4)} (position à une maille près).")
    else:
        notes.append("Pas de table de Ghia à ce Reynolds (disponibles : 100, 400, 1000) : comparaison qualitative.")
    if ex["steadiness"] > 1e-3:
        notes.append(f"L'énergie cinétique varie encore de {100 * ex['steadiness']:.2g} % sur les 10 % finaux : "
                     "le régime stationnaire n'est pas atteint, prolongez le calcul.")
    else:
        notes.append("Régime stationnaire atteint (énergie cinétique stable).")
    if result.solver.reynolds >= 1000:
        notes.append("À Re ≥ 1000, des tourbillons secondaires apparaissent dans les coins inférieurs : visibles sur "
                     "les lignes de courant (onglet Champs).")
    return notes


def _interpret_channel(result: SimulationResult) -> list[str]:
    ex = result.extra
    notes: list[str] = []
    entry = ex["entry_length"]
    if np.isfinite(entry):
        notes.append(f"La vitesse sur l'axe atteint 99 % de sa valeur de Poiseuille (1,5 U) à x = {_fmt(entry, 2)} H ; "
                     "cette longueur d'établissement croît à peu près linéairement avec Re (à vérifier par une étude "
                     "paramétrique en Re).")
    else:
        notes.append("Profil non établi avant la sortie : allongez le canal ou réduisez Re.")
    error = 100.0 * (ex["dpdx"] - ex["dpdx_analytic"]) / abs(ex["dpdx_analytic"])
    sign = "+" if error >= 0 else ""
    notes.append(f"Gradient de pression dans la zone établie (x ≥ {_fmt(ex['fit_start'], 1)} H) : écart de "
                 f"{sign}{_fmt(error, 2)} % à la solution de Poiseuille −12 μ U / H².")
    notes.append(f"Écart maximal du dernier profil à la parabole : {_fmt(ex['profile_error'], 4)} U.")
    return notes

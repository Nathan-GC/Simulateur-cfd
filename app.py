"""Application web interactive du simulateur cfd2d (Streamlit).

Lancement, depuis la racine du projet ::

    .venv\\Scripts\\python -m streamlit run app.py

L'application s'ouvre dans le navigateur (http://localhost:8501) avec deux onglets :

* **📚 Théorie & Documentation** : théorie illustrée (équations, grille MAC, projection,
  stabilité, régimes de sillage), README et tutoriel (formules LaTeX rendues), exercices
  corrigés et code source commenté, dans l'ordre de lecture conseillé ;
* **🚀 Simulation Interactive** : à gauche, le cas d'étude, les scénarios guidés, l'étude
  paramétrique et tous les réglages (fluide, géométrie, maillage, conditions aux limites et
  initiales, pas de temps, schémas) ; à droite, l'aperçu du domaine puis les résultats
  (indicateurs, lecture commentée, champs interactifs, efforts, paroi et sillage, animations,
  méthode, diagnostics, exports), l'historique des calculs et les études paramétriques.

Les calculs tournent en arrière-plan : la page reste utilisable et un bouton permet de les
arrêter. L'application n'implémente aucun calcul : elle assemble une
:class:`~cfd2d.config.SimulationConfig`, exécute :class:`~cfd2d.solver.NavierStokesSolver` et
présente les résultats des modules ``analytics``, ``visualization`` et ``io`` du paquet
``cfd2d`` (dossier ``src/cfd2d``). Le code de l'interface est rangé dans le dossier ``webapp/``.
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``os.environ`` : mode synchrone (tests) ; ``shutil.which`` : présence d'ffmpeg ; ``sys.path``.
import os
import shutil
import sys

# Chemins de fichiers portables.
from pathlib import Path

# Backend matplotlib sans fenêtre (« Agg ») : à choisir avant tout autre import de matplotlib.
import matplotlib

matplotlib.use("Agg")

# Streamlit : interface web.
import streamlit as st  # noqa: E402

# Le paquet cfd2d est importé depuis src/ même s'il n'a pas été installé (pip install -e .) ;
# le dossier du projet (celui de ce fichier) contient le paquet de l'interface, webapp/.
ROOT = Path(__file__).resolve().parent
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import cfd2d  # noqa: E402

from webapp import history, results, runner, scenarios, store  # noqa: E402
from webapp.codegen import python_script  # noqa: E402
from webapp.common import PRECONDITIONERS, RUNS_DIR  # noqa: E402
from webapp.docs import documentation_tab, topic_extras  # noqa: E402
from webapp.params import (  # noqa: E402
    PEAK_SPEED,
    WIDGETS,
    apply_params,
    case_selector,
    check_params,
    estimate_cost,
    geometry_key,
    init_state,
    read_params,
    regime_caption,
    run_summary,
)
from webapp.preview import domain_preview  # noqa: E402

# Onglets du panneau de résultats (à droite).
RESULTS_TAB, HISTORY_TAB, STUDIES_TAB = "📊 Résultats", "🗂️ Historique", "📈 Études"
# Texte alternatif du schéma du domaine (lecteurs d'écran).
DOMAIN_ALT = "Schéma du domaine de calcul : obstacles et conditions aux limites (entrée, sortie, parois)"


def synchronous() -> bool:
    """Calculs dans le fil de la page (tests automatiques) plutôt qu'en arrière-plan."""
    return os.environ.get("CFD2D_SYNC_RUNS") == "1" or bool(st.session_state.get("sync_runs"))


def flash(kind: str, text: str) -> None:
    """Message à afficher une fois, en tête des résultats (kind : success, info, warning, error)."""
    st.session_state.setdefault("flash", []).append((kind, text))


# =========================================================================== rappels (callbacks)
def launch(case: str, spec: history.StudySpec | None) -> None:
    """Prépare la tâche de calcul (calcul unique ou étude) ; le fil est démarré par la page."""
    params = read_params(case)
    runs = history.study_params(case, params, spec) if spec is not None else [(params, None)]
    requests = []
    for p, study in runs:
        config, errors, _ = check_params(p)
        if config is None:
            flash("error", f"{study['label'] if study else 'Calcul'} : {' '.join(errors)}")
            continue
        # Durée prévue (non calibrée) : sert ensuite à calibrer les estimations sur la vitesse réelle.
        predicted = estimate_cost(config, PEAK_SPEED[case])["base_seconds"]
        label = runner.run_label(p) + (f" · étude {study['label']}" if study else "")
        requests.append(runner.RunRequest(params=p, config=config, label=label, predicted_time=predicted, study=study))
    if requests:
        st.session_state["job"] = runner.Job(requests)
        # Les résultats s'afficheront dans l'onglet « Résultats » (ou « Études » pour une série).
        st.session_state["right_tabs"] = RESULTS_TAB


def prolong(extra_time: float) -> None:
    """Prolonge le calcul affiché de ``extra_time`` unités de temps (moniteurs conservés)."""
    result = st.session_state.get("result")
    if result is None:
        return
    request = runner.RunRequest(params=result.params, config=None, label=result.label, base=result,
                                extra_time=float(extra_time))
    st.session_state["job"] = runner.Job([request])


def show_run(run_id: str) -> None:
    """Affiche un calcul de l'historique (relu sur disque, analyse refaite)."""
    try:
        st.session_state["result"] = store.load_run(run_id)
        st.session_state["right_tabs"] = RESULTS_TAB
    except (OSError, ValueError, KeyError) as exc:
        flash("error", f"Calcul illisible ({exc}).")


def reload_run_params(run_id: str) -> None:
    """Remplace les réglages du panneau de gauche par ceux d'un calcul de l'historique."""
    params = store.load_params(run_id)
    apply_params(params["case"], params)
    if params.get("shape") == "image":
        flash("warning", "Réglages rechargés ; l'image de l'obstacle doit être chargée à nouveau.")
    else:
        flash("success", "Réglages du calcul rechargés dans le panneau de gauche.")


def collect_finished_job() -> None:
    """Range les résultats d'une tâche terminée (appelé en début de page, avant les widgets)."""
    job = st.session_state.get("job")
    if job is None or job.active:
        return
    del st.session_state["job"]
    for message in job.errors:
        flash("error", message)
    if job.results:
        result = job.results[-1]
        st.session_state["result"] = result
        if job.total > 1:
            flash("success", f"Étude terminée : {len(job.results)} calcul(s) sur {job.total} (onglet Études).")
            st.session_state["right_tabs"] = STUDIES_TAB
        elif job.requests[0].base is not None:
            # Prolongation : la durée simulée affichée suit le calcul (pas d'avertissement « paramètres modifiés »).
            apply_params(result.case, {"t_end": result.params["t_end"]})
    if job.status == "stopped":
        flash("warning", "Calcul arrêté à votre demande : les résultats portent sur la partie calculée.")


# =============================================================================== suivi du calcul
@st.fragment(run_every=0.8)
def job_status() -> None:
    """Avancement de la tâche, rafraîchi toutes les 0,8 s ; recharge la page à la fin du calcul."""
    job = st.session_state.get("job")
    if job is None:
        return
    if not job.active:
        # Fin du calcul : nouvelle exécution complète de la page (affichage des résultats).
        st.rerun()
    st.progress(job.fraction, text=job.text)
    st.button("Arrêter le calcul", icon=":material/stop_circle:", on_click=job.cancel.set, width="stretch",
              help="Interrompt le calcul ; ce qui est déjà calculé est analysé et enregistré.")


@st.fragment(run_every=0.8)
def job_preview() -> None:
    """Indicateur de chargement et aperçu en direct du calcul (panneau de résultats)."""
    job = st.session_state.get("job")
    if job is None or not job.active:
        return
    title = f"Calcul {job.index + 1} sur {job.total} en cours…" if job.total > 1 else "Simulation en cours…"
    # st.status : indicateur de chargement (icône animée) contenant l'avancement et l'aperçu.
    with st.status(title, state="running", expanded=True):
        st.progress(job.fraction, text=job.text)
        if job.preview is not None:
            st.image(job.preview, caption=job.preview_caption, alt="Aperçu en direct du champ en cours de calcul")
        st.caption("La page reste utilisable : vous pouvez lire la théorie ou l'historique pendant le calcul.")


def start_pending_job() -> None:
    """Démarre la tâche préparée par « Lancer » (en arrière-plan, ou ici même en mode synchrone)."""
    job = st.session_state.get("job")
    if job is None or job.status != "pending":
        return
    if synchronous():
        # Mode synchrone (tests) : indicateur de chargement pendant le calcul, puis nouvelle exécution de
        # la page (les résultats sont rangés en tête de page, avant la création des widgets).
        with st.spinner("Simulation en cours…", show_time=True):
            runner.run_job(job)
        st.rerun()
    else:
        runner.start_in_background(job)


# ============================================================================ onglet Simulation
def parameter_panel() -> tuple[str, dict, object]:
    """Panneau de gauche : cas, scénarios, étude, lancement, récapitulatif, réglages, script."""
    case = case_selector()
    scenarios.scenario_panel()
    params = read_params(case)
    spec, study_errors = history.study_controls(case, params)
    config, errors, advice = check_params(params)
    job = st.session_state.get("job")
    if job is not None and job.active:
        job_status()
    else:
        label = f"Lancer l'étude ({len(spec.values)} calculs)" if spec is not None else "Lancer la simulation"
        st.button(label, type="primary", icon=":material/rocket_launch:", width="stretch", key="run",
                  disabled=config is None or bool(study_errors), on_click=launch, args=(case, spec))
    for message in errors + study_errors:
        st.error(message, icon=":material/block:")
    if config is not None:
        rows, cost = run_summary(params, config, store.calibration(params["pressure_solver"]))
        if spec is not None:
            rows.append(("Durée estimée de l'étude", f"≈ {len(spec.values)} × la durée d'un calcul"))
        with st.expander("Récapitulatif avant calcul", expanded=True, icon=":material/fact_check:"):
            st.table({"Grandeur": [r[0] for r in rows], "Valeur": [r[1] for r in rows]}, hide_index=True)
            caption = regime_caption(params, config)
            if caption:
                st.caption(f"Régime attendu : {caption}")
        for message in advice:
            st.warning(message, icon=":material/lightbulb:")
        if cost["seconds"] > 300:
            st.warning("Calcul long : réduire la résolution ou la durée pour un premier essai.", icon=":material/schedule:")
    WIDGETS[case](topic_extras)
    if config is not None:
        with st.expander("Script Python équivalent", icon=":material/code:"):
            script = python_script(params, runner.run_label(params))
            st.caption("Le même calcul hors de l'application : à copier dans `examples/` puis lancer.")
            st.code(script, language="python", line_numbers=True, height=360)
            st.download_button("Télécharger mon_calcul.py", data=script, file_name="mon_calcul.py", mime="text/x-python",
                               on_click="ignore", icon=":material/download:", key="download_script")
    return case, params, config


def results_panel(case: str, params: dict, config: object) -> None:
    """Onglet « Résultats » : messages, calcul en cours, aperçu du domaine, résultats du dernier calcul."""
    for kind, text in st.session_state.pop("flash", []):
        getattr(st, kind)(text)
    job = st.session_state.get("job")
    if job is not None and job.active:
        job_preview()
    result = st.session_state.get("result")
    # Aperçu du domaine des réglages actuels : déplié tant qu'aucun résultat n'est affiché.
    if config is not None:
        with st.expander("Aperçu du domaine (réglages actuels)", expanded=result is None, icon=":material/grid_view:"):
            domain_png, zoom_png = domain_preview(geometry_key(params))
            tall = config.domain.Ly > 0.8 * config.domain.Lx
            if zoom_png is None:
                st.image(domain_png, width=420 if tall else "stretch", alt=DOMAIN_ALT)
            else:
                c1, c2 = st.columns([2.6, 1])
                c1.image(domain_png, width="stretch", alt=DOMAIN_ALT)
                c2.image(zoom_png, width="stretch", alt="Zoom sur le maillage : cellules solides de l'obstacle et "
                         "contour exact")
    if result is None:
        if job is None:
            st.info("Réglez les paramètres du calcul (ou choisissez un scénario guidé), puis cliquez sur "
                    "« Lancer la simulation ».", icon=":material/info:")
        return
    if job is not None and job.active:
        st.caption("Résultats du calcul précédent :")
    results.show_results(result, params if result.case == case else None,
                         on_prolong=None if job is not None and job.active else prolong)


def simulation_tab() -> None:
    """Onglet 2 : réglages à gauche, résultats, historique et études à droite."""
    left, right = st.columns([1.25, 2.2], gap="large")
    with left:
        case, params, config = parameter_panel()
    with right:
        start_pending_job()
        tabs = st.tabs([RESULTS_TAB, HISTORY_TAB, STUDIES_TAB], key="right_tabs", on_change="rerun")
        if tabs[0].open:
            with tabs[0]:
                results_panel(case, params, config)
        if tabs[1].open:
            with tabs[1]:
                history.history_tab(show_run, reload_run_params)
        if tabs[2].open:
            with tabs[2]:
                history.studies_tab()


# ================================================================================ point d'entrée
def sidebar() -> None:
    """Barre latérale : version, commande de lancement, environnement, calculs enregistrés."""
    with st.sidebar:
        st.markdown("### 🌀 cfd2d")
        st.caption(f"Solveur Navier–Stokes 2D, version {cfd2d.__version__} · Streamlit {st.__version__}")
        st.markdown("**Lancer l'application**")
        st.code(".venv\\Scripts\\python -m streamlit run app.py", language="bash", wrap_lines=True)
        st.markdown("**Environnement**")
        st.markdown(
            f"- ffmpeg : {'trouvé, animations MP4' if shutil.which('ffmpeg') else 'absent, animations GIF'}\n"
            f"- pyamg : {'installé' if 'amg' in PRECONDITIONERS else 'absent (multigrille indisponible)'}\n"
            f"- calculs enregistrés : {len(store.list_runs())} (`{RUNS_DIR.relative_to(ROOT).as_posix()}`)"
        )
        st.caption("Thème clair ou sombre : menu ⋮ en haut à droite, Settings.")
        if st.session_state.get("result") is not None and st.button("Effacer le résultat affiché", icon=":material/delete:"):
            del st.session_state["result"]
            st.rerun()


def main() -> None:
    """Page de l'application : titre, barre latérale, deux onglets."""
    st.set_page_config(page_title="cfd2d · Simulateur CFD 2D", page_icon="🌀", layout="wide")
    init_state()
    # Résultats d'une tâche terminée rangés avant la création des widgets (ils peuvent modifier les réglages).
    collect_finished_job()
    sidebar()
    # Titre court (lisible sur téléphone) et sous-titre.
    st.title("🌀 cfd2d · simulateur 2D")
    st.caption("Navier–Stokes incompressible · grille MAC décalée · projection de Chorin · obstacles par masque binaire")
    # Onglets principaux sans état : en changer n'interrompt rien (le calcul tourne en arrière-plan).
    tab_doc, tab_sim = st.tabs(["📚 Théorie & Documentation", "🚀 Simulation Interactive"])
    with tab_doc:
        documentation_tab()
    with tab_sim:
        simulation_tab()


# Exécuté par « streamlit run app.py » (le script est lancé sous le nom __main__).
if __name__ == "__main__":
    main()

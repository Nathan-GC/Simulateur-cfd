"""Scénarios pédagogiques guidés : réglages par étapes, ce qu'il faut observer, questions.

Chaque étape écrit ses paramètres dans l'état de session (les widgets du panneau de gauche
les affichent aussitôt) ; certaines préparent une étude paramétrique. Les comportements
annoncés ont été vérifiés par des calculs (seuils de divergence, régimes de sillage...).
"""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``dataclass`` : étapes et scénarios.
from dataclasses import dataclass, field

# ``Any`` : valeurs de types variés.
from typing import Any

# Streamlit : widgets et état de session.
import streamlit as st

from .params import apply_params


@dataclass(frozen=True)
class Step:
    """Étape d'un scénario : réglages appliqués et observation attendue."""

    label: str
    params: dict[str, Any]
    observe: str
    # Étude paramétrique à préparer : (paramètre, valeurs), ou None pour un calcul unique.
    study: tuple[str, str] | None = None


@dataclass(frozen=True)
class Scenario:
    """Scénario guidé : objectif, réglages communs, étapes, questions (énoncé, réponse)."""

    title: str
    case: str
    goal: str
    base: dict[str, Any]
    steps: tuple[Step, ...]
    questions: tuple[tuple[str, str], ...] = field(default_factory=tuple)


# Réglages communs du cylindre « rapide » (12 mailles par D, domaine 16 D x 8 D).
_CYLINDER = {"shape": "cylinder", "cells": 12, "upstream": 4.0, "downstream": 12.0, "height": 8.0, "y_offset": 0.0,
             "inlet_profile": "uniform", "outlet": "convective", "north": "slip", "south": "slip", "scheme": "ab2",
             "advection": "quick", "pressure_solver": "direct", "dt": None, "cfl": None, "fourier": None,
             "avg_start": None, "ramp_time": 0.0, "U": 1.0, "rho": 1.0}

SCENARIOS: dict[str, Scenario] = {
    "regimes": Scenario(
        title="Régimes de sillage du cylindre",
        case="obstacle",
        goal="Observer comment le sillage d'un cylindre change de nature quand le nombre de Reynolds augmente : "
             "bulle de recirculation stationnaire, puis allée de Von Kármán au-delà de Re ≈ 47.",
        base=_CYLINDER | {"animation": True, "anim_duration": 20.0},
        steps=(
            Step("Re = 20", {"Re": 20.0, "perturbation": 0.0, "t_end": 40.0},
                 "Écoulement stationnaire : deux tourbillons attachés, Cl nul, Lr/D ≈ 0,9 (lignes de courant du champ moyen)."),
            Step("Re = 40", {"Re": 40.0, "perturbation": 0.0, "t_end": 50.0},
                 "Toujours stationnaire, mais la bulle s'allonge (Lr/D ≈ 2,2) et Cd diminue."),
            Step("Re = 60", {"Re": 60.0, "perturbation": 0.05, "t_end": 150.0},
                 "Juste au-dessus du seuil : la portance se met à osciller lentement puis atteint un cycle régulier."),
            Step("Re = 150", {"Re": 150.0, "perturbation": 0.05, "t_end": 80.0, "cells": 16},
                 "Allée de Von Kármán intense ; le Strouhal augmente (comparez à la loi de Williamson)."),
        ),
        questions=(
            ("Pourquoi Cd diminue-t-il quand Re augmente ?",
             "La part visqueuse de la traînée (frottement) décroît comme 1/√Re environ, car les couches limites "
             "s'amincissent ; la traînée de pression, liée au décollement, varie moins."),
            ("À partir de quel Reynolds la portance oscille-t-elle, et pourquoi ?",
             "Vers Re ≈ 47 : l'écoulement stationnaire devient instable (bifurcation de Hopf) ; une petite "
             "perturbation croît jusqu'à un cycle limite, le lâcher alterné de tourbillons."),
            ("Comment évolue St avec Re ?",
             "Il augmente (environ 0,14 à Re = 60, 0,18 à Re = 150 en milieu infini, loi de Williamson) ; le "
             "blocage du domaine le décale vers le haut de quelques %."),
        ),
    ),
    "diffusion": Scenario(
        title="Diffusion numérique des schémas d'advection",
        case="obstacle",
        goal="Comparer trois schémas d'advection sur une maille grossière (10 mailles par D, Re = 100) : le schéma "
             "décentré ajoute une viscosité numérique, le schéma centré oscille.",
        base=_CYLINDER | {"Re": 100.0, "cells": 10, "perturbation": 0.05, "t_end": 80.0, "animation": False},
        steps=(
            Step("QUICK", {"advection": "quick"}, "Référence : lâcher régulier, St ≈ 0,18."),
            Step("Décentré amont", {"advection": "upwind"},
                 "La viscosité numérique (≈ U Δx / 2 = 0,05, cinq fois ν) amortit fortement le lâcher, voire le supprime."),
            Step("Centré", {"advection": "central"},
                 "Re_Δ = U Δx / ν = 10 > 2 : oscillations parasites (« wiggles ») visibles sur la vorticité près de l'obstacle."),
        ),
        questions=(
            ("Quel Reynolds « effectif » simule le schéma décentré ?",
             "La viscosité totale vaut ν + U Δx / 2 = 0,01 + 0,05 : Re effectif ≈ 100 × 0,01 / 0,06 ≈ 17, sous le seuil "
             "du lâcher (47)."),
            ("Comment réduire la diffusion numérique sans changer de schéma ?",
             "En raffinant la maille : la viscosité numérique est proportionnelle à Δx."),
        ),
    ),
    "stability": Scenario(
        title="Stabilité d'un schéma explicite",
        case="obstacle",
        goal="Dépasser la limite de stabilité d'Adams–Bashforth 2 et constater la divergence, puis voir qu'un schéma "
             "plus robuste (RK3) tolère un pas plus grand. La contrainte de diffusion est relâchée (Fourier 2) pour "
             "isoler celle de l'advection.",
        base=_CYLINDER | {"Re": 100.0, "perturbation": 0.05, "t_end": 15.0, "animation": False, "fourier": 2.0},
        steps=(
            Step("AB2, CFL 0,4", {"scheme": "ab2", "cfl": 0.4}, "Valeur sûre : calcul stable."),
            Step("AB2, CFL 4", {"scheme": "ab2", "cfl": 4.0},
                 "Au-delà de la limite : les erreurs s'amplifient à chaque pas et le calcul diverge (vers t ≈ 1)."),
            Step("RK3, CFL 3", {"scheme": "rk3", "cfl": 3.0},
                 "Stable : la région de stabilité de RK3 est plus grande (mais chaque pas coûte trois évaluations)."),
        ),
        questions=(
            ("Pourquoi le Courant effectif (onglet Diagnostics) est-il inférieur à la valeur visée ?",
             "Le critère combine advection et diffusion (1/Δt = taux advectif + taux diffusif) et prend la vitesse "
             "maximale en x et en y : le pas réel est plus petit que celui du seul critère de Courant."),
            ("Que se passe-t-il si l'on impose un Δt trop grand ?",
             "Le solveur le réduit automatiquement au pas stable (avec un avertissement) : il ne diverge pas."),
        ),
    ),
    "blockage": Scenario(
        title="Effet du blocage (confinement)",
        case="obstacle",
        goal="Mesurer l'effet des parois : plus le domaine est étroit, plus l'écoulement s'accélère autour de "
             "l'obstacle, et plus Cd et St augmentent.",
        base=_CYLINDER | {"Re": 100.0, "perturbation": 0.05, "t_end": 80.0},
        steps=(
            Step("Étude H = 4, 8, 16", {}, "Préparation d'une étude paramétrique en hauteur de domaine (3 calculs) : "
                 "lancez-la, puis ouvrez l'onglet Études.", study=("height", "4, 8, 16")),
        ),
        questions=(
            ("La correction St (1 − β) rapproche-t-elle les points de la loi de Williamson ?",
             "Oui en grande partie : au droit de l'obstacle, la vitesse moyenne entre les parois vaut U / (1 − β) par "
             "conservation du débit ; c'est la correction de blocage la plus simple."),
        ),
    ),
    "convergence": Scenario(
        title="Convergence en maillage",
        case="obstacle",
        goal="Vérifier que les résultats ne dépendent plus du maillage : Cd et St pour 8, 12 et 18 mailles par D, "
             "ordre de convergence observé et extrapolation de Richardson.",
        base=_CYLINDER | {"Re": 100.0, "perturbation": 0.05, "t_end": 80.0},
        steps=(
            Step("Étude 8, 12, 18 mailles par D", {}, "Préparation d'une étude en résolution (rapport constant 1,5) : "
                 "lancez-la, puis ouvrez l'onglet Études (Cd puis St).", study=("cells", "8, 12, 18")),
        ),
        questions=(
            ("Pourquoi l'ordre observé est-il souvent proche de 1 ?",
             "Le schéma est d'ordre 2 en écoulement libre, mais la frontière en escalier introduit une erreur "
             "géométrique d'ordre Δx près de la paroi."),
        ),
    ),
    "cavity": Scenario(
        title="Cavité : de Re = 100 à Re = 1000",
        case="cavity",
        goal="Valider le solveur sur la cavité entraînée contre les tables de Ghia et al. (1982) et voir le tourbillon "
             "principal se recentrer quand Re augmente.",
        base={"U": 1.0, "rho": 1.0, "scheme": "ab2", "advection": "quick", "pressure_solver": "direct", "dt": None,
              "cfl": None, "fourier": None},
        steps=(
            Step("Re = 100", {"Re": 100.0, "N": 48, "t_end": 25.0}, "Écart à Ghia de l'ordre de 0,005 ; tourbillon vers (0,62 ; 0,74)."),
            Step("Re = 400", {"Re": 400.0, "N": 64, "t_end": 40.0}, "Le tourbillon descend vers (0,55 ; 0,61)."),
            Step("Re = 1000", {"Re": 1000.0, "N": 96, "t_end": 60.0},
                 "Tourbillon vers (0,53 ; 0,56) ; tourbillons secondaires marqués dans les coins inférieurs."),
        ),
        questions=(
            ("Pourquoi faut-il raffiner la grille quand Re augmente ?",
             "Les couches limites le long des parois s'amincissent (∝ 1/√Re) : il faut plus de mailles pour les résoudre."),
        ),
    ),
    "channel": Scenario(
        title="Établissement de l'écoulement en canal",
        case="channel",
        goal="Mesurer la longueur nécessaire pour que le profil d'entrée uniforme devienne la parabole de Poiseuille.",
        base={"U": 1.0, "rho": 1.0, "length": 10.0, "Ny": 20, "inlet_profile": "uniform", "outlet": "neumann",
              "ramp_time": 0.0, "scheme": "ab2", "advection": "quick", "pressure_solver": "direct", "dt": None, "cfl": None,
              "fourier": None, "t_end": 25.0},
        steps=(
            Step("Étude Re = 10, 50, 100", {}, "Étude paramétrique en Re (3 calculs) : la longueur d'établissement "
                 "croît avec Re (onglet Études, grandeur « longueur d'établissement »).", study=("Re", "10, 50, 100")),
        ),
        questions=(
            ("Comment la longueur d'établissement varie-t-elle avec Re ?",
             "À peu près linéairement aux Reynolds modérés : la couche limite doit diffuser jusqu'à l'axe, en un temps "
             "∝ H²/ν, pendant lequel le fluide parcourt ∝ U H²/ν = Re H."),
        ),
    ),
    "airfoil": Scenario(
        title="Profil d'aile : portance et incidence",
        case="obstacle",
        goal="Mesurer la portance d'un NACA 0012 à Re = 1000 pour plusieurs incidences et la comparer à la théorie "
             "des profils minces (fluide parfait). Calculs plus longs (48 mailles par corde).",
        base={"shape": "naca", "naca_code": "0012", "cells": 48, "upstream": 2.5, "downstream": 6.0, "height": 4.0,
              "y_offset": 0.0, "Re": 1000.0, "perturbation": 0.0, "t_end": 30.0, "inlet_profile": "uniform",
              "outlet": "convective", "north": "slip", "south": "slip", "scheme": "ab2", "advection": "quick",
              "pressure_solver": "direct", "dt": None, "cfl": None, "fourier": None, "avg_start": None, "U": 1.0,
              "rho": 1.0, "ramp_time": 0.0},
        steps=(
            Step("Étude α = 0, 4, 8°", {"incidence": 0.0}, "Étude en incidence (3 calculs) : Cl(α) dans l'onglet Études, "
                 "avec la droite des profils minces.", study=("incidence", "0, 4, 8")),
        ),
        questions=(
            ("Pourquoi Cl est-il bien inférieur à 2π α ?",
             "À Re = 1000, les couches limites sont épaisses et la viscosité réduit la circulation autour du profil : "
             "la théorie des profils minces suppose un fluide parfait (Re très grand)."),
        ),
    ),
}


def load_step(scenario: Scenario, step: Step) -> None:
    """Fonction de rappel : applique les réglages d'une étape (et prépare son étude éventuelle)."""
    apply_params(scenario.case, scenario.base | step.params)
    s = st.session_state
    if step.study is None:
        s["study_on"] = False
    else:
        param, values = step.study
        s["study_on"] = True
        s[f"study_param_{scenario.case}"] = param
        s[f"study_values_{scenario.case}_{param}"] = values


def scenario_panel() -> None:
    """Panneau des scénarios guidés (en tête du panneau de paramètres)."""
    s = st.session_state
    with st.expander("Scénarios guidés", icon=":material/school:", expanded=s.get("scenario", "none") != "none"):
        key = st.selectbox("Scénario", ["none", *SCENARIOS], key="scenario",
                           format_func=lambda k: "Choisir un scénario…" if k == "none" else SCENARIOS[k].title)
        if key == "none":
            st.caption("Chaque scénario règle les paramètres étape par étape, dit ce qu'il faut observer et pose "
                       "quelques questions.")
            return
        scenario = SCENARIOS[key]
        st.markdown(scenario.goal)
        for k, step in enumerate(scenario.steps, 1):
            with st.container(border=True):
                st.markdown(f"**Étape {k} : {step.label}**")
                st.caption(step.observe)
                st.button("Charger ces réglages", key=f"scenario_{key}_{k}", on_click=load_step, args=(scenario, step),
                          icon=":material/download:", width="stretch")
        st.markdown("**Questions**")
        for k, (question, answer) in enumerate(scenario.questions, 1):
            st.markdown(f"{k}. {question}")
            if st.toggle("Voir la réponse", key=f"scenario_answer_{key}_{k}"):
                st.caption(answer)

"""Configurations types prêtes à l'emploi (cylindre, cavité, canal, profil NACA)."""

# Annotations de type évaluées paresseusement.
from __future__ import annotations

# ``Literal`` : paramètres texte restreints à quelques valeurs.
from typing import Literal

# Conditions aux limites (constructeurs channel / lid_driven_cavity).
from .boundary import BoundaryConfig

# Blocs de configuration assemblés par les fonctions ci-dessous.
from .config import DomainConfig, FlowConfig, NumericsConfig, SimulationConfig, TimeConfig

# Formes d'obstacles utilisées par les cas types.
from .geometry import NACA4, Cylinder


def cylinder_flow(
    Re: float = 100.0,
    cells_per_diameter: int = 20,
    diameter: float = 1.0,
    length: float = 20.0,
    height: float = 10.0,
    x_center: float = 5.0,
    walls: Literal["slip", "no-slip"] = "slip",
    outlet: Literal["neumann", "convective"] = "convective",
    t_end: float = 150.0,
    scheme: Literal["euler", "ab2", "rk3"] = "ab2",
    advection: Literal["upwind", "central", "quick", "tvd"] = "quick",
    perturbation: float = 0.05,
) -> SimulationConfig:
    """Cylindre en soufflerie (allée de Von Kármán pour Re > ~47).

    ``length``, ``height`` et ``x_center`` sont exprimés en diamètres ; le cylindre est
    centré verticalement (blocage ``1/height``) et ``U_inf = 1``.
    """
    # Diamètre (unité de longueur des autres paramètres).
    D = diameter
    return SimulationConfig(
        # Nom lisible, ex. "cylinder_Re100" ({Re:g} : format compact sans zéros inutiles).
        name=f"cylinder_Re{Re:g}",
        # Domaine length x height diamètres ; round(...) donne le nombre entier de cellules
        # correspondant à la résolution demandée (cellules par diamètre).
        domain=DomainConfig(
            Lx=length * D, Ly=height * D, Nx=round(length * cells_per_diameter), Ny=round(height * cells_per_diameter)
        ),
        # U_inf = 1 et Re donné : la viscosité vaut nu = U D / Re.
        flow=FlowConfig(U_inf=1.0, Re=Re, perturbation=perturbation),
        # Entrée uniforme à l'ouest, sortie à l'est, parois haut/bas.
        boundaries=BoundaryConfig.channel(walls=walls, outlet=outlet),
        # Cylindre à x_center diamètres de l'entrée, à mi-hauteur.
        obstacles=[Cylinder(x_center * D, 0.5 * height * D, D)],
        # Durée simulée et schéma temporel.
        time=TimeConfig(t_end=t_end, scheme=scheme),
        # Schéma d'advection.
        numerics=NumericsConfig(advection=advection),
    )


def lid_driven_cavity(
    Re: float = 100.0,
    N: int = 64,
    t_end: float = 30.0,
    scheme: Literal["euler", "ab2", "rk3"] = "ab2",
    advection: Literal["upwind", "central", "quick", "tvd"] = "quick",
) -> SimulationConfig:
    """Cavité carrée unité entraînée par sa paroi supérieure (cas test de Ghia et al., 1982)."""
    return SimulationConfig(
        name=f"cavity_Re{Re:g}",
        # Carré unité discrétisé en N x N cellules.
        domain=DomainConfig(Lx=1.0, Ly=1.0, Nx=N, Ny=N),
        # Re basé sur la vitesse du couvercle et le côté de la cavité (L_ref = 1).
        flow=FlowConfig(U_inf=1.0, Re=Re, L_ref=1.0),
        # Quatre parois adhérentes, couvercle mobile à vitesse 1.
        boundaries=BoundaryConfig.lid_driven_cavity(lid_velocity=1.0),
        time=TimeConfig(t_end=t_end, scheme=scheme),
        numerics=NumericsConfig(advection=advection),
    )


def channel_flow(
    Re: float = 10.0,
    length: float = 4.0,
    height: float = 1.0,
    Nx: int = 64,
    Ny: int = 16,
    profile: Literal["uniform", "parabolic"] = "uniform",
    t_end: float = 15.0,
    scheme: Literal["euler", "ab2", "rk3"] = "ab2",
) -> SimulationConfig:
    """Canal plan à parois adhérentes (écoulement de Poiseuille, Re basé sur la hauteur)."""
    return SimulationConfig(
        name=f"channel_Re{Re:g}",
        domain=DomainConfig(Lx=length, Ly=height, Nx=Nx, Ny=Ny),
        # Re = U_moyen H / nu.
        flow=FlowConfig(U_inf=1.0, Re=Re, L_ref=height),
        # Parois adhérentes : le profil se développe vers la parabole de Poiseuille.
        boundaries=BoundaryConfig.channel(walls="no-slip", profile=profile, outlet="neumann"),
        time=TimeConfig(t_end=t_end, scheme=scheme),
    )


def naca_airfoil(
    code: str = "0012",
    alpha_deg: float = 10.0,
    Re: float = 1000.0,
    cells_per_chord: int = 64,
    length: float = 8.0,
    height: float = 4.0,
    x_le: float = 2.0,
    t_end: float = 40.0,
    scheme: Literal["euler", "ab2", "rk3"] = "ab2",
) -> SimulationConfig:
    """Profil NACA 4 chiffres en incidence (dimensions en cordes, corde unité)."""
    return SimulationConfig(
        # Ex. "naca0012_a10_Re1000".
        name=f"naca{code}_a{alpha_deg:g}_Re{Re:g}",
        domain=DomainConfig(
            Lx=length, Ly=height, Nx=round(length * cells_per_chord), Ny=round(height * cells_per_chord)
        ),
        # Re basé sur la corde (longueur de référence du premier obstacle).
        flow=FlowConfig(U_inf=1.0, Re=Re),
        # Soufflerie à parois glissantes, sortie convective (sillage instationnaire).
        boundaries=BoundaryConfig.channel(walls="slip", outlet="convective"),
        # Profil de corde 1, bord d'attaque à x_le, à mi-hauteur du domaine.
        obstacles=[NACA4(code=code, chord=1.0, x_le=x_le, y_le=0.5 * height, alpha_deg=alpha_deg)],
        time=TimeConfig(t_end=t_end, scheme=scheme),
    )

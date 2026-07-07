"""Derived mechanical and thermal descriptors from elastic constants.

Given the isotropic aggregate moduli (bulk ``K`` and shear ``G``, in GPa),
density and cell geometry, we derive the engineering properties that map onto
the design targets (light / strong / tough / heat-resistant). All formulas are
standard; references are noted inline.

Unit conventions (inputs):
    K, G            : GPa
    density         : g/cm^3
    volume          : Angstrom^3 (unit cell)
    n_atoms         : atoms per unit cell
    mean_mass       : mean atomic mass (amu)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

H_PLANCK = 6.62607015e-34  # J*s
K_B = 1.380649e-23  # J/K


def youngs_modulus(K: float, G: float) -> float:
    """Young's modulus E = 9KG/(3K+G) (GPa)."""
    denom = 3.0 * K + G
    return 9.0 * K * G / denom if denom > 0 else float("nan")


def poisson_ratio(K: float, G: float) -> float:
    """Poisson's ratio nu = (3K-2G)/(2(3K+G))."""
    denom = 2.0 * (3.0 * K + G)
    return (3.0 * K - 2.0 * G) / denom if denom > 0 else float("nan")


def pugh_ratio(K: float, G: float) -> float:
    """Pugh's ratio k = G/K. Lower (k < ~0.57, i.e. K/G > 1.75) => ductile /
    tougher; higher => brittle. (Pugh 1954.)"""
    return G / K if K > 0 else float("nan")


def vickers_hardness(K: float, G: float) -> float:
    """Vickers hardness (GPa) via Chen et al. (2011): Hv = 2*(k^2 G)^0.585 - 3,
    with k = G/K. Clamped at 0 (model can go slightly negative for soft metals)."""
    k = pugh_ratio(K, G)
    if not math.isfinite(k) or k <= 0:
        return float("nan")
    hv = 2.0 * (k * k * G) ** 0.585 - 3.0
    return max(hv, 0.0)


def fracture_toughness(K: float, G: float, volume: float, n_atoms: int) -> float:
    """Fracture toughness K_IC (MPa*m^0.5) via Niu et al. (2019):
        K_IC = V0^(1/6) * G * (K/G)^(1/2)
    where V0 is the volume per atom (m^3), G, K in Pa. Result converted to
    MPa*m^0.5."""
    if K <= 0 or G <= 0 or n_atoms <= 0:
        return float("nan")
    v0 = (volume * 1e-30) / n_atoms  # m^3 per atom
    g_pa, k_pa = G * 1e9, K * 1e9
    k_ic = (v0 ** (1.0 / 6.0)) * g_pa * (k_pa / g_pa) ** 0.5  # Pa*m^0.5
    return k_ic / 1e6  # MPa*m^0.5


def sound_velocities(K: float, G: float, density: float) -> tuple[float, float, float]:
    """(v_longitudinal, v_transverse, v_mean) in m/s from K, G (GPa), rho (g/cm^3)."""
    rho = density * 1000.0  # kg/m^3
    if rho <= 0 or G <= 0:
        return (float("nan"),) * 3
    g_pa, k_pa = G * 1e9, K * 1e9
    v_t = math.sqrt(g_pa / rho)
    v_l = math.sqrt((k_pa + 4.0 * g_pa / 3.0) / rho)
    v_m = (1.0 / 3.0 * (1.0 / v_l**3 + 2.0 / v_t**3)) ** (-1.0 / 3.0)
    return v_l, v_t, v_m


def debye_temperature(K: float, G: float, density: float, volume: float, n_atoms: int) -> float:
    """Debye temperature (K) from the mean sound velocity (Anderson 1963):
        theta_D = (h/k_B) * (3n / (4*pi*V))^(1/3) * v_m
    with V the cell volume (m^3), n atoms in the cell."""
    _, _, v_m = sound_velocities(K, G, density)
    if not math.isfinite(v_m) or volume <= 0 or n_atoms <= 0:
        return float("nan")
    V = volume * 1e-30  # m^3
    number_density = 3.0 * n_atoms / (4.0 * math.pi * V)
    return (H_PLANCK / K_B) * number_density ** (1.0 / 3.0) * v_m


def gruneisen(poisson: float) -> float:
    """Gruneisen parameter from Poisson ratio: gamma = 3(1+nu)/(2(2-3nu))."""
    denom = 2.0 * (2.0 - 3.0 * poisson)
    return 3.0 * (1.0 + poisson) / denom if denom != 0 else float("nan")


def slack_thermal_conductivity(
    K: float,
    G: float,
    density: float,
    volume: float,
    n_atoms: int,
    mean_mass: float,
    temperature: float = 300.0,
) -> float:
    """Lattice thermal conductivity (W/m/K) via the Slack model:
        kappa = A * (M_avg * theta_D^3 * delta) / (gamma^2 * n^(2/3) * T)
    Practical unit convention (Slack 1979; widely used in high-throughput work):
    M_avg in amu, delta = cube-root of the average atomic volume in Angstrom,
    theta_D and T in K, with A = 3.1e-6 giving kappa in W/m/K. This is the
    canonical high-temperature proxy for heat resistance."""
    theta_d = debye_temperature(K, G, density, volume, n_atoms)
    gamma = gruneisen(poisson_ratio(K, G))
    if not all(map(math.isfinite, (theta_d, gamma, mean_mass))) or gamma == 0 or n_atoms <= 0:
        return float("nan")
    delta = (volume / n_atoms) ** (1.0 / 3.0)  # Angstrom
    A = 3.1e-6
    return (A * mean_mass * theta_d**3 * delta) / (gamma**2 * n_atoms ** (2.0 / 3.0) * temperature)


@dataclass
class DerivedProperties:
    youngs_modulus: float
    poisson_ratio: float
    pugh_ratio: float
    vickers_hardness: float
    fracture_toughness: float
    debye_temperature: float
    sound_velocity_mean: float
    gruneisen: float
    slack_thermal_conductivity: float


def derive_all(
    K: float,
    G: float,
    density: float,
    volume: float,
    n_atoms: int,
    mean_mass: float,
) -> DerivedProperties:
    """Compute the full derived-property set from K, G and cell geometry."""
    _, _, v_m = sound_velocities(K, G, density)
    return DerivedProperties(
        youngs_modulus=youngs_modulus(K, G),
        poisson_ratio=poisson_ratio(K, G),
        pugh_ratio=pugh_ratio(K, G),
        vickers_hardness=vickers_hardness(K, G),
        fracture_toughness=fracture_toughness(K, G, volume, n_atoms),
        debye_temperature=debye_temperature(K, G, density, volume, n_atoms),
        sound_velocity_mean=v_m,
        gruneisen=gruneisen(poisson_ratio(K, G)),
        slack_thermal_conductivity=slack_thermal_conductivity(
            K, G, density, volume, n_atoms, mean_mass
        ),
    )

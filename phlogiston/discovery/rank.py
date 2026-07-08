"""Multi-objective ranking of screened candidates (pipeline §7).

Goal profile: **light + strong + tough + heat-resistant**. We treat density as a
hard-ish constraint (flight) and stability as a gate, then score/rank the
survivors on the competing mechanical + thermal objectives (all higher-better):

  * specific stiffness = (K + G)/2 / ρ   (strength-to-weight)
  * fracture toughness K_IC
  * Vickers hardness
  * Debye temperature   (thermal/stiffness proxy for melting resistance)
  * Slack thermal conductivity κ  (dissipates heat)

``multi_objective_score`` gives a scalar (min-max normalized weighted sum over
the pool) for a total order; ``pareto_front`` gives the non-dominated set for an
honest trade-off view. ``rank_candidates`` applies the gate + ceiling and
returns candidates sorted by score with the Pareto flag set.
"""

from __future__ import annotations

from collections.abc import Callable

OBJECTIVES: dict[str, Callable] = {
    "specific_stiffness": lambda p: 0.5
    * (p.get("bulk_modulus_vrh", 0.0) + p.get("shear_modulus_vrh", 0.0))
    / max(p.get("density", 1e-3), 1e-3),
    "fracture_toughness": lambda p: p.get("fracture_toughness", 0.0),
    "vickers_hardness": lambda p: p.get("vickers_hardness", 0.0),
    "debye_temperature": lambda p: p.get("debye_temperature", 0.0),
    "slack_thermal_conductivity": lambda p: p.get("slack_thermal_conductivity", 0.0),
}


def _objective_matrix(candidates) -> list[list[float]]:
    return [[fn(c.properties) for fn in OBJECTIVES.values()] for c in candidates]


def multi_objective_score(candidates, weights: dict[str, float] | None = None) -> list[float]:
    """Min-max normalize each objective over the pool, then weighted-sum. Higher
    is better. Returns one score per candidate (empty pool -> [])."""
    if not candidates:
        return []
    names = list(OBJECTIVES)
    w = {n: 1.0 for n in names}
    if weights:
        w.update(weights)
    mat = _objective_matrix(candidates)
    scores = [0.0] * len(candidates)
    for j in range(len(names)):
        col = [row[j] for row in mat]
        lo, hi = min(col), max(col)
        span = (hi - lo) or 1.0
        wj = w[names[j]]
        for i, v in enumerate(col):
            scores[i] += wj * (v - lo) / span
    total_w = sum(w[n] for n in names) or 1.0
    return [s / total_w for s in scores]


def pareto_front(candidates) -> list[int]:
    """Indices of non-dominated candidates (all objectives higher-better).

    i is dominated if some j is >= in every objective and > in at least one.
    """
    mat = _objective_matrix(candidates)
    front = []
    for i, vi in enumerate(mat):
        dominated = False
        for k, vk in enumerate(mat):
            if k == i:
                continue
            # vk dominates vi if it's >= on every objective and > on at least one
            if all(b >= a for a, b in zip(vi, vk, strict=False)) and any(
                b > a for a, b in zip(vi, vk, strict=False)
            ):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def rank_candidates(
    candidates,
    *,
    rho_max: float | None = None,
    e_hull_max: float = 0.1,
    weights: dict[str, float] | None = None,
):
    """Gate on stability (and optional density ceiling), score, and sort.

    Returns the surviving candidates sorted by descending multi-objective score,
    with ``.score`` and ``.is_pareto`` populated.
    """
    survivors = [
        c
        for c in candidates
        if c.energy_above_hull <= e_hull_max
        and (rho_max is None or c.properties.get("density", float("inf")) <= rho_max)
    ]
    if not survivors:
        return []
    scores = multi_objective_score(survivors, weights)
    front = set(pareto_front(survivors))
    for i, c in enumerate(survivors):
        c.score = scores[i]
        c.is_pareto = i in front
    survivors.sort(key=lambda c: c.score, reverse=True)
    return survivors

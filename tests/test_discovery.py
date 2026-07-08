"""Tests for the discovery half (EMA, sampler, screen, rank, novelty).
Run: python -m tests.test_discovery"""

from __future__ import annotations

import sys

import torch
from pymatgen.core import Lattice, Structure

from phlogiston.discovery.novelty import canonical_formula, dedup, novelty_filter
from phlogiston.discovery.rank import multi_objective_score, pareto_front, rank_candidates
from phlogiston.discovery.screen import PropertyScreen, ScoredCandidate
from phlogiston.models.cdvae import CDVAE
from phlogiston.models.predictor import PREDICT_KEYS, Predictor
from phlogiston.train.ema import EMA

_results: list[tuple[str, bool, str]] = []


def _check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _toy_structure(elements=("Fe", "Al"), a=3.0):
    lat = Lattice.cubic(a)
    return Structure(lat, list(elements), [[0, 0, 0], [0.5, 0.5, 0.5]])


def _candidate(props, formula="FeAl"):
    return ScoredCandidate(structure=_toy_structure(), properties=props, formula=formula)


def test_ema():
    model = torch.nn.Linear(4, 4)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # move weights
    ema.update(model)  # shadow = 0.5*old + 0.5*new
    before = [p.detach().clone() for p in model.parameters()]
    with ema.averaged(model):
        inside = [p.detach().clone() for p in model.parameters()]
    after = [p.detach().clone() for p in model.parameters()]
    swapped = any(not torch.equal(a, b) for a, b in zip(inside, before, strict=False))
    restored = all(torch.equal(a, b) for a, b in zip(after, before, strict=False))
    _check("EMA swaps then restores weights", swapped and restored)


def test_sample_returns_structure():
    m = CDVAE(latent_dim=16, mul=16, n_max=16, n_elements=100, n_levels=5, n_layers=2, correlation=1)
    s = m.sample(n_atoms=6, steps_per_level=1)
    _check("CDVAE.sample returns a Structure", isinstance(s, Structure), f"N={len(s)}")


def test_property_screen():
    model = Predictor(mul=16, n_layers=2, correlation=1)
    screen = PropertyScreen(model, device="cpu")
    structs = [_toy_structure(), _toy_structure(("Ni", "Ti"))]
    scored = screen.score(structs)
    ok = len(scored) == 2 and all(
        set(PREDICT_KEYS) <= set(c.properties) and "density" in c.properties for c in scored
    )
    _check("PropertyScreen scores all targets + density", bool(ok), f"n={len(scored)}")


def test_decoupled_screen():
    # decoupled: stability columns must come from the stability_model. Force the
    # stability model to emit a distinctive constant for energy_above_hull.
    from phlogiston.models.predictor import PREDICT_KEYS, STABILITY_KEYS

    prop = Predictor(mul=16, n_layers=2, correlation=1)
    stab = Predictor(mul=16, n_layers=2, correlation=1)
    # make stab predict a fixed, recognizable value for every target
    with torch.no_grad():
        for h in stab.heads:
            for p in h.parameters():
                p.zero_()
        stab.target_mean.fill_(-7.5)  # de-standardized output == mean when head==0
    screen = PropertyScreen(prop, stability_model=stab, device="cpu")
    scored = screen.score([_toy_structure()])
    ehull_col = PREDICT_KEYS.index("energy_above_hull")
    ok = ehull_col in [PREDICT_KEYS.index(k) for k in STABILITY_KEYS] and abs(
        scored[0].properties["energy_above_hull"] - (-7.5)
    ) < 1e-3
    _check("decoupled screen takes stability from stability_model", bool(ok),
           f"Ehull={scored[0].properties['energy_above_hull']:.2f}")


def test_ranking_and_pareto():
    # a dominates b on every objective; c trades off
    a = _candidate({"bulk_modulus_vrh": 300, "shear_modulus_vrh": 200, "density": 3.0,
                    "vickers_hardness": 30, "fracture_toughness": 5, "debye_temperature": 800,
                    "slack_thermal_conductivity": 50, "energy_above_hull": 0.0}, "A")
    b = _candidate({"bulk_modulus_vrh": 100, "shear_modulus_vrh": 80, "density": 6.0,
                    "vickers_hardness": 10, "fracture_toughness": 1, "debye_temperature": 300,
                    "slack_thermal_conductivity": 10, "energy_above_hull": 0.02}, "B")
    c = _candidate({"bulk_modulus_vrh": 120, "shear_modulus_vrh": 90, "density": 2.0,
                    "vickers_hardness": 8, "fracture_toughness": 8, "debye_temperature": 400,
                    "slack_thermal_conductivity": 5, "energy_above_hull": 0.05}, "C")
    ranked = rank_candidates([a, b, c], e_hull_max=0.1)
    front = pareto_front([a, b, c])
    _check("ranking sorts by score desc", ranked[0].score >= ranked[-1].score,
           f"top={ranked[0].formula}")
    _check("dominated candidate B not on Pareto front", 1 not in front, str(front))
    # stability gate drops unstable
    gated = rank_candidates([a, b, c], e_hull_max=0.01)
    _check("stability gate filters e_hull>tau", {x.formula for x in gated} == {"A"},
           str([x.formula for x in gated]))


def test_multi_objective_score_range():
    cands = [_candidate({"bulk_modulus_vrh": v, "shear_modulus_vrh": v, "density": 3.0,
                         "vickers_hardness": v, "fracture_toughness": v, "debye_temperature": v,
                         "slack_thermal_conductivity": v, "energy_above_hull": 0.0})
             for v in (10, 50, 100)]
    scores = multi_objective_score(cands)
    ok = len(scores) == 3 and min(scores) >= 0 and max(scores) <= 1 and scores[2] > scores[0]
    _check("multi-objective score normalized + monotone", bool(ok), str([f"{s:.2f}" for s in scores]))


def test_novelty_and_dedup():
    c1 = _candidate({}, canonical_formula(_toy_structure(("Fe", "Al"))))
    c2 = _candidate({}, canonical_formula(_toy_structure(("Ni", "Ti"))))
    ref = {c1.formula}
    novel, known = novelty_filter([c1, c2], ref)
    _check("novelty_filter splits by formula", len(novel) == 1 and len(known) == 1,
           f"novel={[x.formula for x in novel]}")
    uniq = dedup([_candidate({}), _candidate({})])  # identical structures
    _check("dedup collapses identical structures", len(uniq) == 1, f"n={len(uniq)}")


if __name__ == "__main__":
    test_ema()
    test_sample_returns_structure()
    test_property_screen()
    test_decoupled_screen()
    test_ranking_and_pareto()
    test_multi_objective_score_range()
    test_novelty_and_dedup()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)

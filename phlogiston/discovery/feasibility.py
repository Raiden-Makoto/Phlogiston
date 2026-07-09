"""Tier-0 composition feasibility / synthesizability sanity (pipeline §7.5).

Stability + good properties answer *"if this crystal existed, would it be
stable and useful?"*. Feasibility answers a different question -- *"could this
thing actually be made?"* -- and it's where composition realism (which we
otherwise ignore) comes back. This is a cheap, rule-based first pass that culls
the obviously-unmakeable candidates the generator loves to emit:
high-entropy soups (10-15 elements), radioactive/synthetic species, and
implausibly large ordered stoichiometries.

It's deliberately conservative -- it should reject the clearly-impossible, not
adjudicate the merely-unusual. The learned Tier-1 synthesizability model and
Tier-2 physics verification (ensemble uMLIP) are the sharper (and more
expensive) arbiters.

No new hard dependency: radioactivity and charge balance come from pymatgen. If
``smact`` is importable we additionally use its validity test as a soft signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Noble gases: don't form bulk crystalline compounds under normal conditions.
_NOBLE_GASES = frozenset({"He", "Ne", "Ar", "Kr", "Xe", "Rn"})

# Common anions used to decide whether a composition is "likely ionic" (and thus
# ought to be charge-balanceable) vs. a metallic/intermetallic phase (which need
# not be).
_ANIONS = frozenset({"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "P", "As"})


@dataclass
class FeasibilityReport:
    """Verdict for one candidate composition.

    ``passed`` is the hard gate (fatal violations); ``score`` in [0, 1] is a soft
    plausibility used for ranking/tie-breaking; ``reasons`` lists every fired
    rule (both hard and soft) for transparency.
    """

    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)


def _composition(structure_or_comp):
    from pymatgen.core import Composition, Structure

    if isinstance(structure_or_comp, Structure):
        return structure_or_comp.composition
    if isinstance(structure_or_comp, Composition):
        return structure_or_comp
    return Composition(str(structure_or_comp))


def _charge_balanceable(comp) -> bool:
    """True if the composition admits a charge-neutral integer oxidation-state
    assignment. All-metal (intermetallic) compositions bond metallically and are
    exempted (treated as balanceable), matching SMACT's alloy handling."""
    symbols = [str(e) for e in comp.elements]
    has_anion = any(s in _ANIONS for s in symbols)
    if not has_anion:  # metallic / intermetallic -> not an ionic-balance question
        return True
    try:
        guesses = comp.oxi_state_guesses()  # empty tuple if none balance
        return len(guesses) > 0
    except Exception:  # noqa: BLE001  degenerate composition -> treat as not balanceable
        return False


def _smact_valid(comp) -> bool | None:
    """Optional SMACT validity (charge neutral + Pauling electronegativity). Returns
    None if smact isn't installed so callers can ignore this axis."""
    try:
        from smact.screening import smact_validity
    except Exception:  # noqa: BLE001  smact optional
        return None
    try:
        symbols = tuple(str(e) for e in comp.elements)
        return bool(smact_validity(symbols, use_pauling_test=True, include_alloys=True))
    except Exception:  # noqa: BLE001
        return False


def composition_feasibility(
    structure_or_comp,
    *,
    max_elements: int = 5,
    max_reduced_atoms: int = 40,
    allow_radioactive: bool = False,
) -> FeasibilityReport:
    """Rule-based feasibility of a single composition.

    Hard gate (any -> ``passed=False``):
      * a radioactive / no-stable-isotope element (unless ``allow_radioactive``),
      * a noble gas,
      * more than ``max_elements`` distinct elements,
      * a reduced formula with more than ``max_reduced_atoms`` atoms.

    Soft ``score`` in [0, 1] averages: element-count parsimony, stoichiometric
    compactness, and charge-balanceability (+ SMACT validity when available).
    """
    from pymatgen.core import Element

    comp = _composition(structure_or_comp)
    symbols = [str(e) for e in comp.elements]
    reasons: list[str] = []
    hard_ok = True

    # --- hard checks -------------------------------------------------------
    if not allow_radioactive:
        rads = [s for s in symbols if Element(s).is_radioactive]
        if rads:
            hard_ok = False
            reasons.append(f"radioactive element(s): {', '.join(sorted(set(rads)))}")

    nobles = [s for s in symbols if s in _NOBLE_GASES]
    if nobles:
        hard_ok = False
        reasons.append(f"noble gas: {', '.join(sorted(set(nobles)))}")

    n_el = len(symbols)
    if n_el > max_elements:
        hard_ok = False
        reasons.append(f"{n_el} distinct elements > max {max_elements}")

    reduced = comp.reduced_composition
    n_reduced = int(round(reduced.num_atoms))
    if n_reduced > max_reduced_atoms:
        hard_ok = False
        reasons.append(f"reduced formula has {n_reduced} atoms > max {max_reduced_atoms}")

    # --- soft score --------------------------------------------------------
    count_score = max(0.0, 1.0 - max(0, n_el - 3) / max(1, max_elements - 3 + 1))
    compact_score = max(0.0, 1.0 - max(0, n_reduced - 5) / max(1, max_reduced_atoms - 5))
    parts = [count_score, compact_score]

    # Charge balance via oxi_state_guesses() is combinatorial in the number of
    # elements, so only run it on compositions that already clear the cheap hard
    # gates (few elements, compact) -- otherwise a 12-element soup can hang for
    # minutes before we reject it anyway.
    if hard_ok:
        balanceable = _charge_balanceable(comp)
        parts.append(1.0 if balanceable else 0.0)
        if not balanceable:
            reasons.append("no charge-neutral oxidation-state assignment")
        smact_ok = _smact_valid(comp)
        if smact_ok is not None:
            parts.append(1.0 if smact_ok else 0.0)
            if not smact_ok:
                reasons.append("fails SMACT validity")

    score = sum(parts) / len(parts)
    return FeasibilityReport(passed=hard_ok, score=round(score, 4), reasons=reasons)


def feasibility_filter(
    candidates: list,
    *,
    max_elements: int = 5,
    max_reduced_atoms: int = 40,
    allow_radioactive: bool = False,
    min_score: float = 0.0,
) -> tuple[list, list]:
    """Split candidates into (feasible, rejected) by the Tier-0 rules.

    Each candidate's ``properties['feasibility']`` is set to the soft score.
    ``candidates`` items must expose ``.structure`` or ``.formula``.
    """
    feasible, rejected = [], []
    for c in candidates:
        target = getattr(c, "structure", None) or getattr(c, "formula", "")
        try:
            rep = composition_feasibility(
                target,
                max_elements=max_elements,
                max_reduced_atoms=max_reduced_atoms,
                allow_radioactive=allow_radioactive,
            )
        except Exception:  # noqa: BLE001  unparseable -> reject
            rejected.append(c)
            continue
        if hasattr(c, "properties"):
            c.properties["feasibility"] = rep.score
        (feasible if (rep.passed and rep.score >= min_score) else rejected).append(c)
    return feasible, rejected

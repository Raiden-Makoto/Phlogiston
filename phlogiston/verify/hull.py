"""Stage 2b: self-consistent uMLIP convex hull (DESIGN.md §4).

We place each candidate on a convex hull built **entirely in the uMLIP's own
energy frame**: the competing phases of the candidate's chemical system are
pulled from Materials Project and re-relaxed with the *same* potential, so
systematic model errors cancel and the resulting ``e_above_hull_umlip`` is an
apples-to-apples, MP-comparable stability estimate independent of the
predictor's (potentially biased) score.

Why re-relax competitors instead of using MP's DFT energies directly? A uMLIP's
absolute energies are offset from DFT per-element; mixing uMLIP candidate
energies with DFT competitor energies would fold that offset straight into the
hull distance. Relaxing both sides with one potential removes it.

To stay tractable we only relax the phases that can actually touch the lower
hull -- MP entries at or near their DFT hull (``ehull_cutoff``) plus the
elemental references -- and cache every relaxed competitor energy on disk keyed
by MP entry id, so phases shared across candidates (and reruns) are relaxed once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


@dataclass
class HullResult:
    """Placement of one candidate on the self-consistent uMLIP hull."""

    e_above_hull_umlip: float  # eV/atom (may be < 0 if below the known hull)
    formation_energy_umlip: float  # eV/atom (uMLIP frame)
    n_competitors: int  # phases used to build the local hull
    chemsys: str


class CompetitorCache:
    """Disk-backed cache of uMLIP-relaxed competitor energies, keyed by MP entry
    id. Maps ``entry_id -> {"e_total", "composition", "nsites"}`` where ``e_total``
    is the total potential energy (eV) of the uMLIP-relaxed MP structure."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:  # noqa: BLE001  corrupt cache -> start fresh
                self._data = {}

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def put(self, key: str, e_total: float, composition: str, nsites: int) -> None:
        self._data[key] = {"e_total": float(e_total), "composition": composition, "nsites": int(nsites)}

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data))
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._data)


def _mp_entries_in_chemsys(elements: list[str], api_key: str | None = None):
    """MP ComputedStructureEntries for the full chemical system (all subsystems),
    each carrying a structure and its DFT ``energy_above_hull`` in ``.data``.

    Pinned to the ``GGA_GGA+U`` thermo type: ``get_entries_in_chemsys`` otherwise
    mixes functionals (GGA and r2SCAN), whose energies aren't hull-comparable, and
    ``property_data`` attaches the DFT hull distance we prune on.
    """
    from mp_api.client import MPRester

    from phlogiston.data.materials_project import get_api_key

    key = get_api_key(api_key)
    with MPRester(key) as mpr:
        entries = mpr.get_entries_in_chemsys(
            elements,
            inc_structure=True,
            additional_criteria={"thermo_types": ["GGA_GGA+U"]},
            property_data=["energy_above_hull"],
        )
    return entries


def _entry_ehull_mp(entry) -> float | None:
    """DFT energy-above-hull an MP entry reports (varies by client version)."""
    data = getattr(entry, "data", None) or {}
    for k in ("energy_above_hull", "e_above_hull"):
        if k in data and data[k] is not None:
            return float(data[k])
    return None


def _prune_near_hull(raw: list, ehull_cutoff: float) -> list:
    """Keep only phases that can define the lower hull: elemental references plus
    entries within ``ehull_cutoff`` of the DFT hull. Prefers the ``energy_above_hull``
    attached to each entry; if that's unavailable, builds a DFT PhaseDiagram from
    the raw entries' own energies and prunes on that."""
    ehulls = {id(e): _entry_ehull_mp(e) for e in raw}
    if sum(1 for v in ehulls.values() if v is not None) < max(1, len(raw) // 2):
        # Field mostly missing -> derive from a DFT phase diagram.
        try:
            from pymatgen.analysis.phase_diagram import PhaseDiagram

            pd_dft = PhaseDiagram(raw)
            ehulls = {id(e): pd_dft.get_e_above_hull(e) for e in raw}
        except Exception:  # noqa: BLE001  keep everything if the DFT hull fails
            return list(raw)
    keep = []
    for e in raw:
        eh = ehulls.get(id(e))
        if len(e.composition.elements) == 1 or eh is None or eh <= ehull_cutoff:
            keep.append(e)

    # Only the lowest-energy phase per composition can be a hull vertex; drop
    # higher polymorphs so we don't relax redundant structures.
    best: dict[str, object] = {}
    for e in keep:
        f = e.composition.reduced_formula
        if f not in best or e.energy_per_atom < best[f].energy_per_atom:  # type: ignore[attr-defined]
            best[f] = e
    return list(best.values())


def build_competitor_entries(
    elements: list[str],
    calc: "Calculator",
    cache: CompetitorCache,
    *,
    ehull_cutoff: float = 0.05,
    relax_steps: int = 200,
    relax_fmax: float = 0.05,
    api_key: str | None = None,
    verbose: bool = True,
) -> list:
    """Fetch the near-hull MP phases for ``elements`` and return uMLIP-frame
    ``PDEntry`` objects (relaxing any not already cached).

    Only phases within ``ehull_cutoff`` of the MP DFT hull (plus all elemental
    references) are relaxed -- higher phases can't define the lower hull, so
    skipping them is exact for the hull and dramatically cheaper.
    """
    from pymatgen.analysis.phase_diagram import PDEntry
    from pymatgen.core import Composition

    from phlogiston.verify.relax import relax_structure

    raw = _mp_entries_in_chemsys(elements, api_key=api_key)
    keep = _prune_near_hull(raw, ehull_cutoff)
    if verbose:
        print(f"[verify] {'-'.join(elements)}: {len(raw)} MP entries -> {len(keep)} near-hull to relax", flush=True)

    pdentries: list = []
    for e in keep:
        eid = str(getattr(e, "entry_id", None) or getattr(e, "material_id", ""))
        cached = cache.get(eid) if eid else None
        if cached is not None:
            comp = Composition(cached["composition"])
            pdentries.append(PDEntry(comp, cached["e_total"]))
            continue
        struct = getattr(e, "structure", None)
        if struct is None:
            continue
        try:
            r = relax_structure(struct, calc, steps=relax_steps, fmax=relax_fmax)
        except Exception:  # noqa: BLE001  one bad competitor shouldn't abort the hull
            continue
        comp = r.structure.composition
        pdentries.append(PDEntry(comp, r.energy))
        if eid:
            cache.put(eid, r.energy, comp.formula, len(r.structure))
    cache.flush()
    return pdentries


def refined_hull_distance(
    candidate: "Structure",
    candidate_energy_total: float,
    competitor_entries: list,
) -> HullResult:
    """Place the (relaxed) candidate on the uMLIP-frame hull of its competitors.

    ``candidate_energy_total`` is the uMLIP total energy (eV) of the relaxed
    candidate cell. The candidate is *not* added to the phase diagram, so a
    genuinely new stable phase reports a negative ``e_above_hull_umlip``.
    """
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

    chemsys = "-".join(sorted({str(el) for el in candidate.composition.elements}))
    cand_entry = PDEntry(candidate.composition, float(candidate_energy_total))

    pd = PhaseDiagram(competitor_entries)
    _, ehull = pd.get_decomp_and_e_above_hull(cand_entry, allow_negative=True)
    eform = pd.get_form_energy_per_atom(cand_entry)
    return HullResult(
        e_above_hull_umlip=float(ehull),
        formation_energy_umlip=float(eform),
        n_competitors=len(competitor_entries),
        chemsys=chemsys,
    )

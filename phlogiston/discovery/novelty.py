"""Novelty / dedup filtering (pipeline §7).

Two cheap, composable checks:
  * ``dedup`` — collapse near-identical generated structures (StructureMatcher).
  * ``novelty_filter`` — drop candidates whose reduced formula already exists in
    the training corpora (GNoME + MP). This is a *composition-level* novelty
    proxy (not full structure matching against 600k crystals, which is
    intractable here); a known formula with a genuinely new structure is
    conservatively treated as "not novel".
"""

from __future__ import annotations


def canonical_formula(structure_or_str) -> str:
    """Canonical reduced formula (pymatgen), for consistent set membership."""
    from pymatgen.core import Composition, Structure

    if isinstance(structure_or_str, Structure):
        return structure_or_str.composition.reduced_formula
    return Composition(str(structure_or_str)).reduced_formula


def load_reference_formulas(data_root: str) -> set[str]:
    """Best-effort set of reduced formulas known in GNoME + MP. Missing sources
    are skipped (returns whatever is available, possibly empty)."""
    import pandas as pd

    formulas: set[str] = set()

    # GNoME summary: normalized column "reduced_formula"
    try:
        from phlogiston.data import gnome

        df = gnome.load_summary(data_root)
        col = next((c for c in ("reduced_formula", "composition") if c in df.columns), None)
        if col is not None:
            for v in df[col].dropna().astype(str):
                try:
                    formulas.add(canonical_formula(v))
                except (ValueError, KeyError):
                    pass
    except Exception:  # noqa: BLE001  (best-effort; missing/renamed columns are fine)
        pass

    # MP metadata: "formula_pretty"
    try:
        from phlogiston.data import materials_project as mp

        meta = mp.metadata_path(data_root)
        if meta.exists():
            df = pd.read_csv(meta, usecols=["formula_pretty"])
            for v in df["formula_pretty"].dropna().astype(str):
                try:
                    formulas.add(canonical_formula(v))
                except (ValueError, KeyError):
                    pass
    except (FileNotFoundError, ValueError, ImportError):
        pass

    return formulas


def dedup(candidates: list, ltol: float = 0.3, stol: float = 0.5, angle_tol: float = 10.0) -> list:
    """Return one representative per group of matching structures.

    ``candidates`` items must expose a ``.structure`` (pymatgen Structure).
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    unique: list = []
    for c in candidates:
        if not any(matcher.fit(c.structure, u.structure) for u in unique):
            unique.append(c)
    return unique


def novelty_filter(candidates: list, reference: set[str]) -> tuple[list, list]:
    """Split candidates into (novel, known) by reduced-formula membership."""
    novel, known = [], []
    for c in candidates:
        f = getattr(c, "formula", "") or canonical_formula(c.structure)
        (known if f in reference else novel).append(c)
    return novel, known

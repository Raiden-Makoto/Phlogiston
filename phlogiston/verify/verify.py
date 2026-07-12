"""Tier-2 verification orchestrator (DESIGN.md §4, stages 2a+2b).

Runs the cheap, high-signal half of Tier 2 over a discovery registry:

  2a  relax each candidate with the primary uMLIP (canonical, lower-energy cell),
  2b  place it on a self-consistent uMLIP convex hull built from re-relaxed MP
      competitors -> ``e_above_hull_umlip`` (an outside, unbiased stability check),

then records the **predictor residual** ``e_above_hull_umlip - e_above_hull_pred``
-- the calibration signal that tells us whether the discovery predictor is
systematically optimistic or being gamed off-manifold (DESIGN.md §5).

Verified columns are appended back to ``candidates.csv``; relaxed CIFs are written
to ``relaxed/`` (originals in ``cifs/`` are left untouched for provenance). Ensemble
cross-check (2c) and phonons (2d) are deliberately out of scope for this pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from phlogiston.verify.ensemble import ensemble_cross_check
from phlogiston.verify.hull import CompetitorCache, build_competitor_entries, refined_hull_distance
from phlogiston.verify.phonons import phonon_stability
from phlogiston.verify.potential import DEFAULT_BACKEND, load_calculator
from phlogiston.verify.relax import relax_structure

# Columns verification appends to candidates.csv (fixed order).
VERIFY_COLUMNS = [
    "e_above_hull_umlip",
    "formation_energy_umlip",
    "predictor_residual",
    "relax_rmsd",
    "relax_dvol",
    "relax_de",
    "relax_converged",
    "e_above_hull_umlip_secondary",
    "ensemble_e_hull_spread",
    "ensemble_confidence",
    "dynamically_stable",
    "min_phonon_freq",
    "verify_tier",
    "relaxed_cif",
]


@dataclass
class VerifyRow:
    """Per-candidate verification outcome (mirrors the appended CSV columns)."""

    id: int
    formula: str
    e_above_hull_pred: float | None
    e_above_hull_umlip: float
    formation_energy_umlip: float
    predictor_residual: float | None
    relax_rmsd: float
    relax_dvol: float
    relax_de: float
    relax_converged: bool
    verify_tier: str
    relaxed_cif: str
    chemsys: str
    n_competitors: int
    # 2c ensemble cross-check (None if not run)
    e_above_hull_secondary: float | None = None
    ensemble_spread: float | None = None
    ensemble_confidence: str | None = None
    # 2d phonons (None if not run)
    min_phonon_freq: float | None = None
    dynamically_stable: bool | None = None


@dataclass
class VerifyReport:
    rows: list[VerifyRow] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)  # (id, reason)


def _find_cif(cif_dir: Path, cid: int) -> Path | None:
    hits = sorted(cif_dir.glob(f"{cid:05d}_*.cif"))
    return hits[0] if hits else None


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def verify_registry(
    save_dir: str,
    *,
    backend: str = DEFAULT_BACKEND,
    top_k: int | None = None,
    verify_e_hull_max: float = 0.1,
    ehull_cutoff: float = 0.05,
    relax_steps: int = 500,
    competitor_relax_steps: int = 200,
    cross_backend: str | None = "mattersim",
    ensemble_spread_max: float = 0.05,
    do_phonons: bool = True,
    phonon_e_hull_max: float = 0.05,
    phonon_supercell_min_len: float = 8.0,
    phonon_displacement: float = 0.03,
    phonon_mesh: int = 8,
    phonon_tol: float = 0.1,
    device: str | None = None,
    api_key: str | None = None,
    verbose: bool = True,
) -> VerifyReport:
    """Verify (a subset of) a discovery registry and append results in place.

    Runs 2a relax + 2b self-consistent hull on every target, then -- only on the
    candidates that pass the 2b gate -- 2c ensemble cross-check (``cross_backend``)
    and 2d phonons (on those within ``phonon_e_hull_max`` of the hull).

    Parameters
    ----------
    save_dir : registry dir holding ``candidates.csv`` and ``cifs/``.
    top_k : verify only the top-N by discovery score (None = all).
    verify_e_hull_max : gate; candidates with ``e_above_hull_umlip`` above this
        are tagged ``screened`` rather than ``verified``.
    ehull_cutoff : only MP competitors within this DFT hull distance are relaxed.
    cross_backend : secondary uMLIP for the ensemble check (None disables 2c).
    do_phonons : run 2d dynamical-stability phonons on near-hull survivors.
    """
    import pandas as pd

    def log(m):
        if verbose:
            print(m, flush=True)

    save = Path(save_dir)
    csv_path = save / "candidates.csv"
    cif_dir = save / "cifs"
    if not csv_path.exists():
        raise FileNotFoundError(f"No registry at {csv_path}")

    df = pd.read_csv(csv_path)
    if "score" in df.columns:
        df_sorted = df.sort_values("score", ascending=False)
    else:
        df_sorted = df
    targets = df_sorted.head(top_k) if top_k else df_sorted
    stages = f"2a+2b{'+2c' if cross_backend else ''}{'+2d' if do_phonons else ''}"
    log(f"[verify] loaded {len(df)} candidates; verifying {len(targets)} "
        f"[{stages}] backend={backend}" + (f" cross={cross_backend}" if cross_backend else ""))

    calc = load_calculator(backend, device=device)
    cache = CompetitorCache(save / "verify_cache" / f"umlip_{backend}.json")
    calc2 = load_calculator(cross_backend, device=device) if cross_backend else None
    cache2 = CompetitorCache(save / "verify_cache" / f"umlip_{cross_backend}.json") if cross_backend else None
    relaxed_dir = save / "relaxed"
    relaxed_dir.mkdir(parents=True, exist_ok=True)

    from pymatgen.core import Structure

    report = VerifyReport()
    for _, row in targets.iterrows():
        cid = int(row["id"])
        formula = str(row["formula"])
        cif = _find_cif(cif_dir, cid)
        if cif is None:
            report.skipped.append((cid, "no CIF"))
            log(f"[verify] {cid:05d} {formula}: SKIP (no CIF)")
            continue
        try:
            structure = Structure.from_file(str(cif))
        except Exception as exc:  # noqa: BLE001
            report.skipped.append((cid, f"CIF read: {exc}"))
            continue

        # 2a relax
        rr = relax_structure(structure, calc, steps=relax_steps)

        # 2b self-consistent uMLIP hull
        elements = sorted({str(el) for el in rr.structure.composition.elements})
        try:
            competitors = build_competitor_entries(
                elements, calc, cache,
                ehull_cutoff=ehull_cutoff, relax_steps=competitor_relax_steps,
                api_key=api_key, verbose=verbose,
            )
            hull = refined_hull_distance(rr.structure, rr.energy, competitors)
        except Exception as exc:  # noqa: BLE001  (MP/hull failure shouldn't abort the batch)
            report.skipped.append((cid, f"hull: {exc}"))
            log(f"[verify] {cid:05d} {formula}: SKIP (hull failed: {exc})")
            continue

        e_pred = _to_float(row.get("energy_above_hull"))
        residual = hull.e_above_hull_umlip - e_pred if e_pred is not None else None
        tier = "verified" if hull.e_above_hull_umlip <= verify_e_hull_max else "screened"

        relaxed_cif = relaxed_dir / cif.name
        try:
            rr.structure.to(filename=str(relaxed_cif), fmt="cif")
            rel_path = str(Path("relaxed") / cif.name)
        except Exception:  # noqa: BLE001
            rel_path = ""

        vrow = VerifyRow(
            id=cid, formula=formula, e_above_hull_pred=e_pred,
            e_above_hull_umlip=hull.e_above_hull_umlip,
            formation_energy_umlip=hull.formation_energy_umlip,
            predictor_residual=residual,
            relax_rmsd=rr.rmsd, relax_dvol=rr.dvol_frac, relax_de=rr.de,
            relax_converged=rr.converged, verify_tier=tier, relaxed_cif=rel_path,
            chemsys=hull.chemsys, n_competitors=hull.n_competitors,
        )
        log(
            f"[verify] {cid:05d} {formula:<16} e_hull_umlip={hull.e_above_hull_umlip:+.3f} "
            f"(pred {('%+.3f' % e_pred) if e_pred is not None else '  --'}, "
            f"resid {('%+.3f' % residual) if residual is not None else '--'})  "
            f"rmsd={rr.rmsd:.2f} de={rr.de:+.2f}  [{tier}]  ({hull.n_competitors} competitors)"
        )

        # 2c/2d only on candidates that pass the 2b stability gate.
        if tier == "verified":
            if calc2 is not None:
                try:
                    ens = ensemble_cross_check(
                        rr.structure, hull.e_above_hull_umlip, calc2, cache2, cross_backend,
                        spread_max=ensemble_spread_max, relax_steps=relax_steps,
                        competitor_relax_steps=competitor_relax_steps, ehull_cutoff=ehull_cutoff,
                        api_key=api_key, verbose=verbose,
                    )
                    vrow.e_above_hull_secondary = ens.e_above_hull_secondary
                    vrow.ensemble_spread = ens.spread
                    vrow.ensemble_confidence = ens.confidence
                    log(f"[verify]       2c {cross_backend}: e_hull={ens.e_above_hull_secondary:+.3f} "
                        f"spread={ens.spread:.3f} -> {ens.confidence}-confidence")
                except Exception as exc:  # noqa: BLE001
                    log(f"[verify]       2c cross-check failed: {exc}")

            if do_phonons and hull.e_above_hull_umlip <= phonon_e_hull_max:
                try:
                    ph = phonon_stability(
                        rr.structure, calc,
                        supercell_min_len=phonon_supercell_min_len,
                        displacement=phonon_displacement, mesh=phonon_mesh, tol_thz=phonon_tol,
                    )
                    vrow.min_phonon_freq = ph.min_freq_thz
                    vrow.dynamically_stable = ph.dynamically_stable
                    log(f"[verify]       2d phonons: min_freq={ph.min_freq_thz:+.2f} THz "
                        f"({'stable' if ph.dynamically_stable else 'IMAGINARY'}) "
                        f"[{'x'.join(map(str, ph.supercell))} sc, {ph.n_displacements} disp]")
                except Exception as exc:  # noqa: BLE001
                    log(f"[verify]       2d phonons failed: {exc}")

        report.rows.append(vrow)

    _write_back(df, report, csv_path)
    _write_calibration_report(report, save / "verify_report.txt", verify_e_hull_max)
    log(f"[verify] done: {len(report.rows)} verified, {len(report.skipped)} skipped "
        f"-> {csv_path}")
    return report


def _write_back(df, report: VerifyReport, csv_path: Path) -> None:
    """Append the verification columns to candidates.csv, matched by id."""
    import pandas as pd

    # object dtype so a column can hold floats, bools, and "" together (a bare
    # `df[col] = ""` infers pandas' string dtype and rejects float assignment).
    for col in VERIFY_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series([""] * len(df), index=df.index, dtype=object)
        else:
            # A prior verify pass may have written these columns all-empty,
            # which pandas re-infers as float64 on re-read; force object so a
            # later bool/str assignment does not trip pandas 2.x strict dtype.
            df[col] = df[col].astype(object)
    by_id = {r.id: r for r in report.rows}
    for i, cid in df["id"].items():
        r = by_id.get(int(cid))
        if r is None:
            continue
        df.at[i, "e_above_hull_umlip"] = round(r.e_above_hull_umlip, 4)
        df.at[i, "formation_energy_umlip"] = round(r.formation_energy_umlip, 4)
        df.at[i, "predictor_residual"] = round(r.predictor_residual, 4) if r.predictor_residual is not None else ""
        df.at[i, "relax_rmsd"] = round(r.relax_rmsd, 4)
        df.at[i, "relax_dvol"] = round(r.relax_dvol, 4)
        df.at[i, "relax_de"] = round(r.relax_de, 4)
        df.at[i, "relax_converged"] = bool(r.relax_converged)
        df.at[i, "e_above_hull_umlip_secondary"] = (
            round(r.e_above_hull_secondary, 4) if r.e_above_hull_secondary is not None else "")
        df.at[i, "ensemble_e_hull_spread"] = (
            round(r.ensemble_spread, 4) if r.ensemble_spread is not None else "")
        df.at[i, "ensemble_confidence"] = r.ensemble_confidence or ""
        df.at[i, "dynamically_stable"] = (
            bool(r.dynamically_stable) if r.dynamically_stable is not None else "")
        df.at[i, "min_phonon_freq"] = (
            round(r.min_phonon_freq, 4) if r.min_phonon_freq is not None else "")
        df.at[i, "verify_tier"] = r.verify_tier
        df.at[i, "relaxed_cif"] = r.relaxed_cif
    df.to_csv(csv_path, index=False)


def _write_calibration_report(report: VerifyReport, path: Path, gate: float) -> None:
    """Batch calibration signal: residual distribution + gate outcome (DESIGN §5)."""
    import statistics

    rows = report.rows
    lines = ["Tier-2 verification calibration report", "=" * 38, ""]
    n = len(rows)
    verified = sum(1 for r in rows if r.verify_tier == "verified")
    lines.append(f"verified {verified}/{n} at e_above_hull_umlip <= {gate:g} eV/atom "
                 f"({len(report.skipped)} skipped)")
    resid = [r.predictor_residual for r in rows if r.predictor_residual is not None]
    if resid:
        mean = statistics.fmean(resid)
        med = statistics.median(resid)
        sd = statistics.pstdev(resid) if len(resid) > 1 else 0.0
        lines += [
            "",
            "predictor residual  (e_above_hull_umlip - e_above_hull_pred), eV/atom:",
            f"  n={len(resid)}  mean={mean:+.4f}  median={med:+.4f}  std={sd:.4f}"
            f"  min={min(resid):+.4f}  max={max(resid):+.4f}",
            "",
            "  interpretation: a consistent positive mean => predictor is optimistic",
            "  (shift the discovery gate); large std => predictor gamed off-manifold",
            "  (tighten cond-trust-radius).",
        ]
    # 2c ensemble confidence
    ens = [r for r in rows if r.ensemble_confidence is not None]
    if ens:
        high = sum(1 for r in ens if r.ensemble_confidence == "high")
        spreads = [r.ensemble_spread for r in ens if r.ensemble_spread is not None]
        lines += [
            "",
            f"ensemble cross-check (2c): {high}/{len(ens)} high-confidence "
            f"(spread <= threshold); mean spread={statistics.fmean(spreads):.4f} eV/atom"
            if spreads else "",
        ]
    # 2d phonons
    ph = [r for r in rows if r.dynamically_stable is not None]
    if ph:
        stable = sum(1 for r in ph if r.dynamically_stable)
        lines += ["", f"phonons (2d): {stable}/{len(ph)} dynamically stable "
                       f"(no imaginary modes beyond tolerance)"]
    if report.skipped:
        lines += ["", "skipped:"]
        lines += [f"  {cid:05d}: {reason}" for cid, reason in report.skipped]
    path.write_text("\n".join(lines) + "\n")

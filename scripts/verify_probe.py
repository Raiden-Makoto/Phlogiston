"""Quick end-to-end probe of the Tier-2 hull path on a small chemical system.

Confirms MP access (get_entries_in_chemsys returns structures + DFT hull
distance), uMLIP relaxation of competitors, and PhaseDiagram placement, without
running the full registry. Uses a tiny chemsys + low relax steps for speed.

Usage (inside the ROCm container, with MP_API_KEY in env):
  python scripts/verify_probe.py --chemsys Ti-O --backend chgnet
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chemsys", default="Ti-O", help="e.g. Ti-O")
    ap.add_argument("--backend", default="chgnet")
    ap.add_argument("--ehull-cutoff", type=float, default=0.03)
    ap.add_argument("--relax-steps", type=int, default=60)
    args = ap.parse_args()

    from phlogiston.verify.hull import build_competitor_entries, refined_hull_distance
    from phlogiston.verify.potential import load_calculator
    from phlogiston.verify.relax import relax_structure

    elements = sorted(args.chemsys.split("-"))
    print(f"[probe] chemsys={elements} backend={args.backend}", flush=True)

    calc = load_calculator(args.backend)

    # Raw MP fetch sanity: structures + hull field present?
    from phlogiston.verify.hull import _entry_ehull_mp, _mp_entries_in_chemsys

    raw = _mp_entries_in_chemsys(elements)
    with_struct = sum(1 for e in raw if getattr(e, "structure", None) is not None)
    with_ehull = sum(1 for e in raw if _entry_ehull_mp(e) is not None)
    print(f"[probe] {len(raw)} MP entries: {with_struct} w/ structure, {with_ehull} w/ e_above_hull")
    if raw:
        e0 = raw[0]
        print(f"[probe] sample entry_id={getattr(e0, 'entry_id', '?')} "
              f"comp={e0.composition.reduced_formula} ehull_mp={_entry_ehull_mp(e0)}")

    import tempfile

    cache_dir = tempfile.mkdtemp()
    from phlogiston.verify.hull import CompetitorCache

    cache = CompetitorCache(f"{cache_dir}/cache.json")
    competitors = build_competitor_entries(
        elements, calc, cache, ehull_cutoff=args.ehull_cutoff, relax_steps=args.relax_steps
    )
    print(f"[probe] built {len(competitors)} uMLIP competitor entries; cache size={len(cache)}")

    # Place one near-hull MP structure back on the uMLIP hull -> should be ~0.
    target = None
    for e in raw:
        if len(e.composition.elements) == len(elements) and getattr(e, "structure", None):
            target = e
            break
    if target is None:
        print("[probe] no multi-element target structure found; skipping placement")
        return 0
    rr = relax_structure(target.structure, calc, steps=args.relax_steps)
    hull = refined_hull_distance(rr.structure, rr.energy, competitors)
    print(f"[probe] placed {target.composition.reduced_formula}: "
          f"e_above_hull_umlip={hull.e_above_hull_umlip:+.4f} "
          f"eform_umlip={hull.formation_energy_umlip:+.4f} "
          f"({hull.n_competitors} competitors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

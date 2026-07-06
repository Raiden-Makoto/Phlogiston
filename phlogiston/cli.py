"""Phlogiston command-line interface.

Phase 1 exposes dataset acquisition:

    phlogiston fetch-gnome              # download default GNoME files
    phlogiston fetch-gnome --all        # download everything (incl. structure zips)
    phlogiston gnome-info               # download+load summary, print stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phlogiston.data import gnome
from phlogiston.data import materials_project as mp


def _cmd_fetch_gnome(args: argparse.Namespace) -> int:
    keys = list(gnome.GNOME_FILES) if args.all else args.keys
    paths = gnome.download(args.data_root, keys=keys, force=args.force)
    print("\n[gnome] downloaded:")
    for key, path in paths.items():
        size_mb = path.stat().st_size / 1e6 if path.exists() else 0.0
        print(f"  {key:32s} -> {path}  ({size_mb:.1f} MB)")
    return 0


def _cmd_gnome_info(args: argparse.Namespace) -> int:
    df = gnome.load_summary(args.data_root, functional=args.functional)
    stable = gnome.filter_stable(df, max_decomposition_energy=args.max_decomp)
    print(f"[gnome] summary ({args.functional}): {len(df):,} rows, {df.shape[1]} columns")
    print(f"[gnome] columns: {list(df.columns)}")
    print(
        f"[gnome] stable (decomp <= {args.max_decomp} eV/atom): {len(stable):,} "
        f"({100 * len(stable) / max(len(df), 1):.1f}%)"
    )
    if "elements" in df.columns:
        print(f"[gnome] example rows:\n{df.head(3).to_string()}")
    return 0


def _cmd_fetch_mp(args: argparse.Namespace) -> int:
    exclude = list(args.exclude_elements or [])
    if args.exclude_radioactive:
        exclude = sorted(set(exclude) | set(mp.RADIOACTIVE_ELEMENTS))
    df = mp.fetch_structures(
        args.data_root,
        elements=args.elements,
        exclude_elements=exclude or None,
        num_elements=tuple(args.num_elements) if args.num_elements else None,
        num_sites_max=args.num_sites_max,
        is_stable=(True if args.stable_only else None),
        max_energy_above_hull=args.max_energy_above_hull,
        limit=args.limit,
        save_cif=not args.no_cif,
        force=args.force,
    )
    print(f"[mp] fetched {len(df):,} materials")
    if len(df):
        print(df.head(3).to_string())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phlogiston", description=__doc__)
    p.add_argument(
        "--data-root", default="data",
        help="Root directory for datasets (default: ./data)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch-gnome", help="Download GNoME dataset files")
    f.add_argument(
        "--keys", nargs="+", default=list(gnome.DEFAULT_KEYS),
        choices=list(gnome.GNOME_FILES),
        help=f"Which files to fetch (default: {list(gnome.DEFAULT_KEYS)})",
    )
    f.add_argument("--all", action="store_true", help="Fetch every file (large!)")
    f.add_argument("--force", action="store_true", help="Re-download even if present")
    f.set_defaults(func=_cmd_fetch_gnome)

    g = sub.add_parser("gnome-info", help="Load the summary and print statistics")
    g.add_argument("--functional", default="pbe", choices=["pbe", "r2scan"])
    g.add_argument(
        "--max-decomp", type=float, default=0.0,
        help="Decomposition-energy threshold (eV/atom) for 'stable' count",
    )
    g.set_defaults(func=_cmd_gnome_info)

    m = sub.add_parser("fetch-mp", help="Download Materials Project structures + labels")
    m.add_argument("--elements", nargs="+", default=None,
                   help="Restrict to materials containing these elements")
    m.add_argument("--num-elements", nargs=2, type=int, default=None,
                   metavar=("MIN", "MAX"), help="Min/max number of distinct elements")
    m.add_argument("--num-sites-max", type=int, default=None,
                   help="Cap sites per unit cell (keeps graphs tractable)")
    m.add_argument("--stable-only", action="store_true",
                   help="Only fetch thermodynamically stable materials (e_above_hull == 0)")
    m.add_argument("--max-energy-above-hull", type=float, default=None,
                   help="Keep materials with e_above_hull <= this (eV/atom); "
                        "e.g. 0.05 for stable + near-stable")
    m.add_argument("--exclude-elements", nargs="+", default=None,
                   help="Drop materials containing any of these elements")
    m.add_argument("--exclude-radioactive", action="store_true",
                   help="Drop materials containing radioactive elements (Tc, Pm, Z>=84)")
    m.add_argument("--limit", type=int, default=None,
                   help="Max number of materials to fetch (default: all matching)")
    m.add_argument("--no-cif", action="store_true", help="Skip writing CIF files")
    m.add_argument("--force", action="store_true", help="Overwrite existing CIFs")
    m.set_defaults(func=_cmd_fetch_mp)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

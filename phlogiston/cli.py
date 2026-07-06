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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

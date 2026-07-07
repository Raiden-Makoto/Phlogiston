"""Phlogiston command-line interface.

Phase 1 exposes dataset acquisition:

    phlogiston fetch-gnome              # download default GNoME files
    phlogiston fetch-gnome --all        # download everything (incl. structure zips)
    phlogiston gnome-info               # download+load summary, print stats
"""

from __future__ import annotations

import argparse

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


def _cmd_fetch_mp_elasticity(args: argparse.Namespace) -> int:
    df = mp.fetch_elasticity(
        args.data_root,
        limit=args.limit,
        save_cif=not args.no_structures,
        force=args.force,
    )
    print(f"[mp] elasticity: {len(df):,} labeled materials")
    if len(df):
        cols = [
            "material_id",
            "formula_pretty",
            "bulk_modulus_vrh",
            "shear_modulus_vrh",
            "vickers_hardness",
            "fracture_toughness",
            "debye_temperature",
            "slack_thermal_conductivity",
        ]
        print(df[cols].head(5).to_string())
    return 0


def _count_data_lines(path) -> int:
    with open(path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)  # minus header


def _cmd_featurize(args: argparse.Namespace) -> int:
    from phlogiston.data import precompute

    stats = precompute.featurize_all(
        args.data_root,
        sources=tuple(args.sources),
        cutoff=args.cutoff,
        workers=args.workers,
        shard_size=args.shard_size,
        limit=args.limit,
    )
    print(f"[featurize] {stats}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from phlogiston.train import train

    train(
        args.data_root,
        stage=args.stage,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        mul=args.mul,
        n_layers=args.n_layers,
        correlation=args.correlation,
        max_shards=args.max_shards,
        out_dir=args.out_dir,
        init_ckpt=args.init_ckpt,
        resume=args.resume,
    )
    return 0


def _cmd_datasets_summary(args: argparse.Namespace) -> int:
    import zipfile

    import pandas as pd

    root = args.data_root
    print(f"Phlogiston dataset summary  (root: {root})\n")

    # --- GNoME ---
    gnome_summary = gnome.local_path(root, "summary_pbe")
    gnome_n = _count_data_lines(gnome_summary) if gnome_summary.exists() else 0
    gnome_zip = gnome.local_path(root, "structures_by_id")
    gnome_cifs = 0
    if gnome_zip.exists():
        with zipfile.ZipFile(gnome_zip) as z:
            gnome_cifs = sum(1 for n in z.namelist() if n.lower().endswith(".cif"))

    # --- Materials Project ---
    meta_ids: set[str] = set()
    meta_path = mp.metadata_path(root)
    if meta_path.exists():
        meta_ids = set(pd.read_csv(meta_path, usecols=["material_id"])["material_id"])

    elas_ids: set[str] = set()
    elas_path = mp.elasticity_path(root)
    if elas_path.exists():
        elas_ids = set(pd.read_csv(elas_path, usecols=["material_id"])["material_id"])

    cifs = mp.cif_dir(root)
    n_cifs = len(list(cifs.glob("*.cif"))) if cifs.exists() else 0

    both = len(meta_ids & elas_ids)
    elas_only = len(elas_ids - meta_ids)

    def row(name, n, struct, stab, mech):
        print(f"  {name:<28} {n:>10,}   {struct:^9} {stab:^9} {mech:^12}")

    print("  SOURCE                            count   structure stability  mech/thermal")
    print("  " + "-" * 74)
    row("GNoME (summary)", gnome_n, "zip" if gnome_cifs else "-", "yes", "-")
    row("MP near-stable", len(meta_ids), "yes", "yes", f"{both:,} of them")
    row("MP elasticity", len(elas_ids), "yes", "yes", "yes")
    print()
    print(f"  GNoME structure CIFs (by_id.zip): {gnome_cifs:,}")
    print(f"  MP structure CIFs on disk:        {n_cifs:,}")
    print()
    print("  Label overlap (MP):")
    print(f"    stability + mech/thermal (both): {both:,}")
    print(f"    mech/thermal only (metastable):  {elas_only:,}")
    print(f"    total mech/thermal-labeled:      {len(elas_ids):,}")
    print()
    print("  -> No single material has every label. Shared encoder trains on all")
    print("     structures (stability/density); property heads train on the mech")
    print("     subset via masked multi-task losses.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phlogiston", description=__doc__)
    p.add_argument(
        "--data-root",
        default="data",
        help="Root directory for datasets (default: ./data)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch-gnome", help="Download GNoME dataset files")
    f.add_argument(
        "--keys",
        nargs="+",
        default=list(gnome.DEFAULT_KEYS),
        choices=list(gnome.GNOME_FILES),
        help=f"Which files to fetch (default: {list(gnome.DEFAULT_KEYS)})",
    )
    f.add_argument("--all", action="store_true", help="Fetch every file (large!)")
    f.add_argument("--force", action="store_true", help="Re-download even if present")
    f.set_defaults(func=_cmd_fetch_gnome)

    g = sub.add_parser("gnome-info", help="Load the summary and print statistics")
    g.add_argument("--functional", default="pbe", choices=["pbe", "r2scan"])
    g.add_argument(
        "--max-decomp",
        type=float,
        default=0.0,
        help="Decomposition-energy threshold (eV/atom) for 'stable' count",
    )
    g.set_defaults(func=_cmd_gnome_info)

    m = sub.add_parser("fetch-mp", help="Download Materials Project structures + labels")
    m.add_argument(
        "--elements",
        nargs="+",
        default=None,
        help="Restrict to materials containing these elements",
    )
    m.add_argument(
        "--num-elements",
        nargs=2,
        type=int,
        default=None,
        metavar=("MIN", "MAX"),
        help="Min/max number of distinct elements",
    )
    m.add_argument(
        "--num-sites-max",
        type=int,
        default=None,
        help="Cap sites per unit cell (keeps graphs tractable)",
    )
    m.add_argument(
        "--stable-only",
        action="store_true",
        help="Only fetch thermodynamically stable materials (e_above_hull == 0)",
    )
    m.add_argument(
        "--max-energy-above-hull",
        type=float,
        default=None,
        help="Keep materials with e_above_hull <= this (eV/atom); "
        "e.g. 0.05 for stable + near-stable",
    )
    m.add_argument(
        "--exclude-elements",
        nargs="+",
        default=None,
        help="Drop materials containing any of these elements",
    )
    m.add_argument(
        "--exclude-radioactive",
        action="store_true",
        help="Drop materials containing radioactive elements (Tc, Pm, Z>=84)",
    )
    m.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of materials to fetch (default: all matching)",
    )
    m.add_argument("--no-cif", action="store_true", help="Skip writing CIF files")
    m.add_argument("--force", action="store_true", help="Overwrite existing CIFs")
    m.set_defaults(func=_cmd_fetch_mp)

    e = sub.add_parser(
        "fetch-mp-elasticity", help="Fetch MP elastic constants + derive mechanical/thermal labels"
    )
    e.add_argument(
        "--limit", type=int, default=None, help="Max number of elasticity records to fetch"
    )
    e.add_argument(
        "--no-structures",
        action="store_true",
        help="Don't download structures for elasticity materials",
    )
    e.add_argument("--force", action="store_true", help="Overwrite existing CIFs")
    e.set_defaults(func=_cmd_fetch_mp_elasticity)

    fz = sub.add_parser(
        "featurize", help="Precompute crystal graphs for the whole corpus (CPU, sharded)"
    )
    fz.add_argument("--sources", nargs="+", default=["mp", "gnome"], choices=["mp", "gnome"])
    fz.add_argument("--cutoff", type=float, default=6.0)
    fz.add_argument(
        "--workers",
        type=int,
        default=8,
        help="CPU worker processes (bounded to be shared-box friendly)",
    )
    fz.add_argument("--shard-size", type=int, default=4096)
    fz.add_argument("--limit", type=int, default=None)
    fz.set_defaults(func=_cmd_featurize)

    sub.add_parser(
        "datasets-summary", help="Print label coverage across GNoME + MP datasets"
    ).set_defaults(func=_cmd_datasets_summary)

    tr = sub.add_parser("train", help="Train the predictor (schedule B)")
    tr.add_argument("--stage", type=int, default=1, choices=[1, 2])
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument(
        "--encoder-lr", type=float, default=1e-4, help="Encoder LR in stage 2 (fine-tune)"
    )
    tr.add_argument("--mul", type=int, default=128)
    tr.add_argument("--n-layers", type=int, default=2)
    tr.add_argument("--correlation", type=int, default=3)
    tr.add_argument(
        "--max-shards", type=int, default=None, help="Load only N shards (for quick runs)"
    )
    tr.add_argument("--out-dir", default="runs")
    tr.add_argument(
        "--init-ckpt", default=None, help="Warm-start weights only (e.g. stage-1 -> stage-2)"
    )
    tr.add_argument(
        "--resume",
        default=None,
        help="Resume from a checkpoint (restores optimizer/scheduler/epoch/best)",
    )
    tr.set_defaults(func=_cmd_train)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

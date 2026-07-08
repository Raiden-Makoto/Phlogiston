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
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        compile=args.compile,
        select_by=args.select_by,
        grad_clip=args.grad_clip,
        restrict_labeled=args.restrict_labeled,
    )
    return 0


def _cmd_train_cdvae(args: argparse.Namespace) -> int:
    from phlogiston.train import train_cdvae

    train_cdvae(
        args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        mul=args.mul,
        n_layers=args.n_layers,
        correlation=args.correlation,
        n_max=args.n_max,
        beta=args.beta,
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        stable_max=args.stable_max,
        max_shards=args.max_shards,
        out_dir=args.out_dir,
        resume=args.resume,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
    )
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    from phlogiston.discovery import discover, format_report

    ranked = discover(
        args.generator,
        args.predictor,
        args.data_root,
        n_samples=args.n_samples,
        steps_per_level=args.steps_per_level,
        e_hull_max=args.e_hull_max,
        rho_max=args.rho_max,
        do_dedup=not args.no_dedup,
        check_novelty=not args.no_novelty,
    )
    print("\n" + format_report(ranked, top_k=args.top_k))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from phlogiston.train import evaluate_checkpoint

    evaluate_checkpoint(
        args.ckpt,
        args.data_root,
        split=args.split,
        stage=args.stage,
        batch_size=args.batch_size,
        max_shards=args.max_shards,
        num_workers=args.num_workers,
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
    tr.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Large default: MI350X 288GB has headroom, keeps GPUs fed",
    )
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
    tr.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Linear LR warmup epochs before cosine decay",
    )
    tr.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stop after N epochs without val improvement (0 disables)",
    )
    tr.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (parallel collation; keeps GPUs fed)",
    )
    tr.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (fuses e3nn kernels; experimental on ROCm)",
    )
    tr.add_argument(
        "--select-by",
        default=None,
        choices=["loss", "auc", "r2"],
        help="Best-checkpoint metric (default: stage 1 -> auc, stage 2 -> r2)",
    )
    tr.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
        help="Max grad norm (0 disables); guards against divergence",
    )
    tr.add_argument(
        "--restrict-labeled",
        action="store_true",
        help="Train only on structures with a present label for the stage's "
        "targets (fast, dense signal for stage-2 property fine-tuning)",
    )
    tr.set_defaults(func=_cmd_train)

    ev = sub.add_parser("evaluate", help="Score a checkpoint (MAE + R2 + stability AUC/AP)")
    ev.add_argument("--ckpt", required=True, help="Path to a saved checkpoint (.pt)")
    ev.add_argument("--split", default="val", choices=["train", "val", "test"])
    ev.add_argument("--stage", type=int, default=2, choices=[1, 2])
    ev.add_argument("--batch-size", type=int, default=512)
    ev.add_argument("--max-shards", type=int, default=None)
    ev.add_argument("--num-workers", type=int, default=4)
    ev.set_defaults(func=_cmd_evaluate)

    tc = sub.add_parser("train-cdvae", help="Train the CDVAE generator (EMA + composite loss)")
    tc.add_argument("--epochs", type=int, default=30)
    tc.add_argument("--batch-size", type=int, default=256)
    tc.add_argument("--lr", type=float, default=1e-3)
    tc.add_argument("--latent-dim", type=int, default=256)
    tc.add_argument("--mul", type=int, default=128)
    tc.add_argument("--n-layers", type=int, default=3)
    tc.add_argument("--correlation", type=int, default=2)
    tc.add_argument("--n-max", type=int, default=64, help="Max atoms per generated cell")
    tc.add_argument("--beta", type=float, default=0.01, help="KL weight (VAE regularization)")
    tc.add_argument("--ema-decay", type=float, default=0.999)
    tc.add_argument("--grad-clip", type=float, default=5.0)
    tc.add_argument(
        "--stable-max",
        type=float,
        default=None,
        help="Train only on structures with e_above_hull <= this (eV/atom)",
    )
    tc.add_argument("--max-shards", type=int, default=None)
    tc.add_argument("--out-dir", default="runs")
    tc.add_argument("--resume", default=None)
    tc.add_argument("--warmup-epochs", type=int, default=2)
    tc.add_argument("--patience", type=int, default=8)
    tc.add_argument("--num-workers", type=int, default=4)
    tc.set_defaults(func=_cmd_train_cdvae)

    dc = sub.add_parser("discover", help="Generate -> screen -> rank novel stable candidates")
    dc.add_argument("--generator", required=True, help="CDVAE checkpoint (.pt)")
    dc.add_argument("--predictor", required=True, help="Predictor checkpoint (.pt)")
    dc.add_argument("--n-samples", type=int, default=128)
    dc.add_argument("--steps-per-level", type=int, default=4)
    dc.add_argument("--e-hull-max", type=float, default=0.1, help="Stability gate (eV/atom)")
    dc.add_argument("--rho-max", type=float, default=None, help="Density ceiling (g/cm^3)")
    dc.add_argument("--top-k", type=int, default=10)
    dc.add_argument("--no-dedup", action="store_true")
    dc.add_argument("--no-novelty", action="store_true")
    dc.set_defaults(func=_cmd_discover)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

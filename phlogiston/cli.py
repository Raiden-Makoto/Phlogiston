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


def _cmd_fetch_mp_synth(args: argparse.Namespace) -> int:
    df = mp.fetch_synthesizability(args.data_root, chunk=args.chunk)
    if len(df):
        n_obs = int((~df["theoretical"] | df["has_icsd"]).sum())
        print(f"[mp] {n_obs:,}/{len(df):,} experimentally observed (positives)")
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
        feedback_root=args.feedback_root,
        feedback_weight=args.feedback_weight,
    )
    return 0


def _cmd_train_synth(args: argparse.Namespace) -> int:
    from phlogiston.train import train_synth

    train_synth(
        args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        mul=args.mul,
        n_layers=args.n_layers,
        correlation=args.correlation,
        max_shards=args.max_shards,
        include_gnome=not args.no_gnome,
        out_dir=args.out_dir,
        init_ckpt=args.init_ckpt,
        resume=args.resume,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
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
        init_ckpt=args.init_ckpt,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
    )
    return 0


def _cmd_fit_latent_head(args: argparse.Namespace) -> int:
    import torch

    from phlogiston.discovery.loop import load_generator
    from phlogiston.models.cdvae import fit_latent_property_head

    gen = load_generator(args.generator)
    head = fit_latent_property_head(
        gen,
        args.data_root,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        max_shards=args.max_shards,
        num_workers=args.num_workers,
        feedback_root=args.feedback_root,
        feedback_weight=args.feedback_weight,
    )
    torch.save({"model": head.state_dict(), "hidden": args.hidden, "latent_dim": gen.latent_dim},
               args.out)
    print(f"[fit-latent-head] saved -> {args.out}")
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    from phlogiston.discovery import discover, format_report

    stats: dict = {}
    ranked = discover(
        args.generator,
        args.predictor,
        args.data_root,
        stability_ckpt=args.stability_ckpt,
        stability_bias=args.stability_bias,
        synth_ckpt=args.synth_ckpt,
        synth_min=args.synth_min,
        latent_head_ckpt=args.latent_head,
        cond_steps=args.cond_steps,
        cond_trust_radius=args.cond_trust_radius,
        n_samples=args.n_samples,
        steps_per_level=args.steps_per_level,
        gen_batch_size=args.gen_batch_size,
        e_hull_max=args.e_hull_max,
        rho_max=args.rho_max,
        do_dedup=not args.no_dedup,
        check_novelty=not args.no_novelty,
        check_feasibility=not args.no_feasibility,
        save_dir=args.save_dir,
        max_elements=args.max_elements,
        max_reduced_atoms=args.max_reduced_atoms,
        allow_radioactive=args.allow_radioactive,
        umlip_gate=args.umlip_gate,
        umlip_backend=args.umlip_backend,
        umlip_e_hull_max=args.umlip_e_hull_max,
        umlip_relax_steps=args.umlip_relax_steps,
        umlip_with_hull=not args.umlip_no_hull,
        umlip_max_candidates=args.umlip_max_candidates,
        umlip_max_rmsd=args.umlip_max_rmsd,
        umlip_ehull_cutoff=args.umlip_ehull_cutoff,
        umlip_cross_backend=args.umlip_cross_backend,
        umlip_ensemble_spread_max=args.umlip_ensemble_spread_max,
        umlip_phonons=args.umlip_phonons,
        umlip_require_phonon_stable=not args.umlip_keep_phonon_unstable,
        umlip_phonon_e_hull_max=args.umlip_phonon_e_hull_max,
        stats_out=stats,
    )
    print("\n" + format_report(ranked, top_k=args.top_k, stats=stats))
    return 0


def _cmd_show_candidates(args: argparse.Namespace) -> int:
    import csv
    from pathlib import Path

    from phlogiston.discovery import format_report

    path = Path(args.save_dir) / "candidates.csv"
    if not path.exists():
        print(f"No registry at {path}; run `discover --save-dir {args.save_dir}` first.")
        return 1
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    print(format_report(rows, top_k=args.top_k))
    print(f"\n{len(rows)} candidates in registry: {path}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from phlogiston.verify import verify_registry

    report = verify_registry(
        args.save_dir,
        backend=args.backend,
        top_k=args.top_k,
        verify_e_hull_max=args.verify_e_hull_max,
        ehull_cutoff=args.ehull_cutoff,
        relax_steps=args.relax_steps,
        competitor_relax_steps=args.competitor_relax_steps,
        cross_backend=None if args.no_ensemble else args.cross_backend,
        ensemble_spread_max=args.ensemble_spread_max,
        do_phonons=not args.no_phonons,
        phonon_e_hull_max=args.phonon_e_hull_max,
        phonon_supercell_min_len=args.phonon_supercell_min_len,
        phonon_displacement=args.phonon_displacement,
        phonon_mesh=args.phonon_mesh,
        phonon_tol=args.phonon_tol,
        device=args.device,
    )
    if not report.rows:
        print("[verify] no candidates verified.")
        return 1
    ranked = sorted(report.rows, key=lambda r: r.e_above_hull_umlip)
    print(f"\nVerified {len(report.rows)} candidates (sorted by uMLIP hull distance):\n")
    header = (f"{'id':>5}  {'formula':<16}{'Ehull_uMLIP':>12}{'Ehull_pred':>11}"
              f"{'resid':>8}{'spread':>8}{'conf':>6}{'phonon':>8}{'dyn':>5}  tier")
    print(header)
    print("-" * len(header))
    for r in ranked:
        ep = f"{r.e_above_hull_pred:+.3f}" if r.e_above_hull_pred is not None else "--"
        rs = f"{r.predictor_residual:+.3f}" if r.predictor_residual is not None else "--"
        sp = f"{r.ensemble_spread:.3f}" if r.ensemble_spread is not None else "--"
        cf = r.ensemble_confidence or "--"
        pf = f"{r.min_phonon_freq:+.2f}" if r.min_phonon_freq is not None else "--"
        dyn = ("yes" if r.dynamically_stable else "no") if r.dynamically_stable is not None else "--"
        print(f"{r.id:>5}  {r.formula:<16}{r.e_above_hull_umlip:>+12.3f}{ep:>11}"
              f"{rs:>8}{sp:>8}{cf:>6}{pf:>8}{dyn:>5}  {r.verify_tier}")
    return 0


def _cmd_harvest_verified(args: argparse.Namespace) -> int:
    from phlogiston.train import harvest_verified

    harvest_verified(args.registry, args.out, cutoff=args.cutoff, dedup=not args.no_dedup)
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

    es = sub.add_parser(
        "fetch-mp-synth",
        help="Fetch MP experimental-provenance flags (theoretical/ICSD) for Tier-1 synthesizability",
    )
    es.add_argument("--chunk", type=int, default=1000, help="Material IDs per API call")
    es.set_defaults(func=_cmd_fetch_mp_synth)

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
    tr.add_argument(
        "--feedback-root", default=None,
        help="Active-learning: mix Tier-2-verified feedback shards (from "
        "`harvest-verified`) into the train split only. Single-process.",
    )
    tr.add_argument(
        "--feedback-weight", type=int, default=1,
        help="Replicate the feedback set this many times to up-weight the "
        "scarce hard negatives (e.g. 20-50 for ~100 records vs ~150k corpus)",
    )
    tr.set_defaults(func=_cmd_train)

    ts = sub.add_parser("train-synth", help="Train the Tier-1 synthesizability classifier")
    ts.add_argument("--epochs", type=int, default=8)
    ts.add_argument("--batch-size", type=int, default=512)
    ts.add_argument("--lr", type=float, default=1e-3)
    ts.add_argument(
        "--encoder-lr", type=float, default=None,
        help="Lower encoder LR (fine-tune) when warm-starting; default: single LR",
    )
    ts.add_argument("--mul", type=int, default=128)
    ts.add_argument("--n-layers", type=int, default=2)
    ts.add_argument("--correlation", type=int, default=3)
    ts.add_argument("--max-shards", type=int, default=None)
    ts.add_argument("--no-gnome", action="store_true", help="Exclude the GNoME negative pool")
    ts.add_argument("--out-dir", default="runs")
    ts.add_argument(
        "--init-ckpt", default=None,
        help="Warm-start the encoder from a Predictor/stability checkpoint",
    )
    ts.add_argument("--resume", default=None, help="Resume (restores optimizer/scheduler/epoch)")
    ts.add_argument("--warmup-epochs", type=int, default=1)
    ts.add_argument("--patience", type=int, default=6)
    ts.add_argument("--num-workers", type=int, default=4)
    ts.add_argument("--grad-clip", type=float, default=5.0)
    ts.set_defaults(func=_cmd_train_synth)

    ev = sub.add_parser("evaluate", help="Score a checkpoint (MAE + R2 + stability AUC/AP)")
    ev.add_argument("--ckpt", required=True, help="Path to a saved checkpoint (.pt)")
    ev.add_argument("--split", default="val", choices=["train", "val", "test"])
    ev.add_argument("--stage", type=int, default=2, choices=[1, 2])
    ev.add_argument("--batch-size", type=int, default=512)
    ev.add_argument("--max-shards", type=int, default=None)
    ev.add_argument("--num-workers", type=int, default=4)
    ev.set_defaults(func=_cmd_evaluate)

    hv = sub.add_parser(
        "harvest-verified",
        help="Active-learning: build feedback shards from Tier-2 verified registries",
    )
    hv.add_argument(
        "--registry", action="append", required=True,
        help="Registry dir with a verified candidates.csv + relaxed/ (repeatable)",
    )
    hv.add_argument(
        "--out", default="data/runs/feedback",
        help="Feedback root: writes processed/shards + manifest.jsonl here",
    )
    hv.add_argument("--cutoff", type=float, default=6.0, help="Graph neighbor cutoff (A)")
    hv.add_argument("--no-dedup", action="store_true", help="Skip StructureMatcher dedup")
    hv.set_defaults(func=_cmd_harvest_verified)

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
    tc.add_argument("--resume", default=None, help="Full resume (weights+opt+sched+epoch)")
    tc.add_argument(
        "--init-ckpt",
        default=None,
        help="Warm-start weights+EMA only, fresh schedule (continue a capped run)",
    )
    tc.add_argument("--warmup-epochs", type=int, default=2)
    tc.add_argument("--patience", type=int, default=8)
    tc.add_argument("--num-workers", type=int, default=4)
    tc.set_defaults(func=_cmd_train_cdvae)

    dc = sub.add_parser("discover", help="Generate -> screen -> rank novel stable candidates")
    dc.add_argument("--generator", required=True, help="CDVAE checkpoint (.pt)")
    dc.add_argument(
        "--predictor", required=True, help="Property Predictor checkpoint (.pt)"
    )
    dc.add_argument(
        "--stability-ckpt",
        default=None,
        help="Separate stability specialist for the gate (recommended: Stage-1 best)",
    )
    dc.add_argument(
        "--stability-bias",
        type=float,
        default=0.0,
        help="Calibration offset (eV/atom) added to predicted energy_above_hull before "
        "the gate. Set to the measured Tier-2 residual mean (~+0.25) to correct the "
        "predictor's optimism so survivors actually land near the uMLIP hull.",
    )
    dc.add_argument("--n-samples", type=int, default=128)
    dc.add_argument("--steps-per-level", type=int, default=4)
    dc.add_argument(
        "--gen-batch-size", type=int, default=None,
        help="Decode generation in chunks of this many structures to avoid GPU "
        "OOM at large --n-samples (the e3nn tensor product scales with total "
        "atoms in the batch). ~384 is safe on MI300; None = single batch.",
    )
    dc.add_argument("--e-hull-max", type=float, default=0.1, help="Stability gate (eV/atom)")
    dc.add_argument("--rho-max", type=float, default=None, help="Density ceiling (g/cm^3)")
    dc.add_argument("--top-k", type=int, default=10)
    dc.add_argument("--no-dedup", action="store_true")
    dc.add_argument("--no-novelty", action="store_true")
    dc.add_argument(
        "--synth-ckpt", default=None,
        help="Tier-1 synthesizability model (.pt); scores + optionally gates candidates",
    )
    dc.add_argument(
        "--synth-min", type=float, default=0.3,
        help="Tier-1 gate: drop candidates below this synthesizability (loose by "
        "design to allow near-future synthesis advances; 0 = score only)",
    )
    dc.add_argument("--no-feasibility", action="store_true", help="Skip Tier-0 composition feasibility gate")
    dc.add_argument(
        "--save-dir", default=None,
        help="Persist survivors here: CIFs + an accumulating, deduped candidates.csv",
    )
    dc.add_argument("--max-elements", type=int, default=5, help="Tier-0: max distinct elements")
    dc.add_argument("--max-reduced-atoms", type=int, default=40, help="Tier-0: max atoms in reduced formula")
    dc.add_argument("--allow-radioactive", action="store_true", help="Tier-0: permit radioactive elements")
    dc.add_argument(
        "--umlip-gate", action="store_true",
        help="Tier-1.5: relax survivors with a uMLIP and gate on the self-consistent "
        "uMLIP hull distance. Closes the predictor's off-manifold blind spot so the "
        "saved candidates are already physically verified (relax+hull).",
    )
    dc.add_argument("--umlip-backend", default="chgnet", help="Tier-1.5 uMLIP backend (chgnet | mattersim)")
    dc.add_argument("--umlip-e-hull-max", type=float, default=0.1,
                    help="Tier-1.5 gate: keep iff e_above_hull_umlip <= this (eV/atom)")
    dc.add_argument("--umlip-relax-steps", type=int, default=300, help="Tier-1.5: max relax steps per candidate")
    dc.add_argument("--umlip-no-hull", action="store_true",
                    help="Tier-1.5: fast relax+drift-only pass (no Materials Project hull round-trip)")
    dc.add_argument("--umlip-max-candidates", type=int, default=None,
                    help="Tier-1.5: relax at most the top-N survivors by predicted hull distance "
                    "(bounds cost; default: all)")
    dc.add_argument("--umlip-max-rmsd", type=float, default=None,
                    help="Tier-1.5 drift prefilter: drop candidates whose relaxation RMSD exceeds "
                    "this (Angstrom); large drift => off-manifold generator guess")
    dc.add_argument("--umlip-ehull-cutoff", type=float, default=0.05,
                    help="Tier-1.5: only relax MP competitors within this DFT hull distance (eV/atom)")
    dc.add_argument("--umlip-cross-backend", default=None,
                    help="Tier-1.5 (2c): second, independent uMLIP (e.g. mattersim) that re-relaxes "
                    "hull-passers and flags member disagreement (off-distribution). None = skip 2c.")
    dc.add_argument("--umlip-ensemble-spread-max", type=float, default=0.05,
                    help="Tier-1.5 (2c): high-confidence iff |e_hull spread| <= this (eV/atom)")
    dc.add_argument("--umlip-phonons", action="store_true",
                    help="Tier-1.5 (2d): run finite-displacement phonons on near-hull survivors and "
                    "drop candidates with confirmed imaginary modes (dynamically unstable)")
    dc.add_argument("--umlip-keep-phonon-unstable", action="store_true",
                    help="Tier-1.5 (2d): annotate phonon stability but do NOT drop unstable candidates")
    dc.add_argument("--umlip-phonon-e-hull-max", type=float, default=0.05,
                    help="Tier-1.5 (2d): run phonons only on candidates within this hull distance (eV/atom)")
    dc.add_argument(
        "--latent-head",
        default=None,
        help="Fitted latent property head (.pt) -> property-conditioned generation",
    )
    dc.add_argument("--cond-steps", type=int, default=100, help="Latent-optimization steps")
    dc.add_argument(
        "--cond-trust-radius",
        type=float,
        default=4.0,
        help="Trust-region radius for latent optimization (keeps z on-manifold; "
        "with d=256 the anchor norm is ~16, so 4 is a quarter-norm move)",
    )
    dc.set_defaults(func=_cmd_discover)

    scmd = sub.add_parser("show-candidates", help="Pretty-print the saved candidate registry")
    scmd.add_argument("--save-dir", required=True, help="Directory holding candidates.csv")
    scmd.add_argument("--top-k", type=int, default=20, help="How many candidates to show")
    scmd.set_defaults(func=_cmd_show_candidates)

    vc = sub.add_parser("verify", help="Tier-2: uMLIP relax + self-consistent hull over a registry")
    vc.add_argument("--save-dir", required=True, help="Registry dir (candidates.csv + cifs/)")
    vc.add_argument("--backend", default="chgnet", help="Primary uMLIP backend (chgnet | mattersim)")
    vc.add_argument("--top-k", type=int, default=None, help="Verify only the top-N by score (default: all)")
    vc.add_argument("--verify-e-hull-max", type=float, default=0.1,
                    help="Tag as 'verified' iff e_above_hull_umlip <= this (eV/atom)")
    vc.add_argument("--ehull-cutoff", type=float, default=0.05,
                    help="Only relax MP competitors within this DFT hull distance (eV/atom)")
    vc.add_argument("--relax-steps", type=int, default=500, help="Max relax steps per candidate")
    vc.add_argument("--competitor-relax-steps", type=int, default=200, help="Max relax steps per competitor")
    vc.add_argument("--cross-backend", default="mattersim",
                    help="2c ensemble cross-check uMLIP (independent member)")
    vc.add_argument("--no-ensemble", action="store_true", help="Skip 2c ensemble cross-check")
    vc.add_argument("--ensemble-spread-max", type=float, default=0.05,
                    help="2c: high-confidence iff |e_hull spread| <= this (eV/atom)")
    vc.add_argument("--no-phonons", action="store_true", help="Skip 2d phonon stability")
    vc.add_argument("--phonon-e-hull-max", type=float, default=0.05,
                    help="2d: run phonons only on candidates within this hull distance (eV/atom)")
    vc.add_argument("--phonon-supercell-min-len", type=float, default=8.0,
                    help="2d: target min supercell axis length (Angstrom)")
    vc.add_argument("--phonon-displacement", type=float, default=0.03,
                    help="2d: finite displacement (Angstrom)")
    vc.add_argument("--phonon-mesh", type=int, default=8, help="2d: q-point mesh density (NxNxN)")
    vc.add_argument("--phonon-tol", type=float, default=0.1,
                    help="2d: imaginary-frequency tolerance (THz); stable iff min_freq >= -tol")
    vc.add_argument("--device", default=None, help="torch device (default: auto)")
    vc.set_defaults(func=_cmd_verify)

    fh = sub.add_parser("fit-latent-head", help="Fit f_p(z) on a CDVAE for conditioning")
    fh.add_argument("--generator", required=True, help="CDVAE checkpoint (.pt)")
    fh.add_argument("--out", required=True, help="Output path for the fitted head (.pt)")
    fh.add_argument("--hidden", type=int, default=256)
    fh.add_argument("--epochs", type=int, default=100)
    fh.add_argument("--lr", type=float, default=1e-3)
    fh.add_argument("--max-shards", type=int, default=None)
    fh.add_argument("--num-workers", type=int, default=4)
    fh.add_argument(
        "--feedback-root", default=None,
        help="Active-learning: mix Tier-2-verified feedback shards into the fit",
    )
    fh.add_argument("--feedback-weight", type=int, default=1,
                    help="Replicate feedback this many times (up-weight hard negatives)")
    fh.set_defaults(func=_cmd_fit_latent_head)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

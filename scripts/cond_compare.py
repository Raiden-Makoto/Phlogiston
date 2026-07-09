"""Head-to-head diagnostic: does property conditioning actually shift the
generated distribution toward the target profile?

Generates two matched batches from the SAME trained CDVAE decoder:
  (A) unconditional  -- z ~ N(0, I)
  (B) conditioned    -- z optimized toward the profile (on-manifold, projected)

Both are decoded with the identical batched sampler and scored by the SAME
decoupled predictor screen (properties from the Stage-2 model, stability from
the Stage-1 specialist). We report validity yield and the median of each target
for both batches, plus the conditioned-minus-unconditional delta. Conditioning
"works" if the maximize-targets shift up and energy_above_hull shifts down.

Usage (inside the ROCm container):
    python -m scripts.cond_compare --n 128 --steps 150 --steps-per-level 8
"""

from __future__ import annotations

import argparse
import statistics as st

import torch

from phlogiston.discovery.loop import load_generator, load_latent_head
from phlogiston.discovery.screen import PropertyScreen, load_predictor
from phlogiston.models.cdvae import DEFAULT_PROFILE, generate_conditioned
from phlogiston.models.cdvae.sampler import batched_sample
from phlogiston.models.predictor import PREDICT_KEYS

# targets we want to move + desired direction (+1 maximize, -1 minimize)
REPORT = {
    "energy_above_hull": -1,
    "bulk_modulus_vrh": +1,
    "shear_modulus_vrh": +1,
    "vickers_hardness": +1,
    "fracture_toughness": +1,
    "debye_temperature": +1,
    "slack_thermal_conductivity": +1,
}


def _median(scored, key):
    vals = [c.properties[key] for c in scored if key in c.properties]
    return st.median(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="data/runs/cdvae_long/cdvae_best.pt")
    ap.add_argument("--head", default="data/runs/latent_head_long.pt")
    ap.add_argument("--predictor", default="data/runs/property/predictor_stage2_best.pt")
    ap.add_argument("--stability", default="data/runs/predictor_stage1_best.pt")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=150, help="latent-optimization steps")
    ap.add_argument("--steps-per-level", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--reward-cap", type=float, default=2.0)
    ap.add_argument("--trust", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"[cmp] device={device}  n={args.n}  cond-steps={args.steps}", flush=True)

    gen = load_generator(args.generator, device)
    head = load_latent_head(args.head, gen.latent_dim, device)
    predictor = load_predictor(args.predictor, device)
    stability = load_predictor(args.stability, device)
    screen = PropertyScreen(predictor, stability_model=stability, device=device)

    # (A) unconditional: matched-size batch from N(0, I)
    print("[cmp] generating unconditional batch ...", flush=True)
    z0 = torch.randn(args.n, gen.latent_dim, device=device)
    uncond = batched_sample(gen, z0, steps_per_level=args.steps_per_level)
    uscored = screen.score(uncond)

    umed = {k: _median(uscored, k) for k in REPORT}
    print(f"\n[cmp] unconditional valid/scored = {len(uscored)}/{args.n}")

    # (B) conditioned: sweep the trust radius to see where the INDEPENDENT
    # predictor (not the head) confirms a real shift toward the profile.
    for tr in args.trust:
        print(f"\n[cmp] === conditioned, trust_radius={tr} ===", flush=True)
        cond = generate_conditioned(
            gen, head, args.n, profile=DEFAULT_PROFILE, steps=args.steps, lr=args.lr,
            trust_radius=tr, reward_cap=args.reward_cap, steps_per_level=args.steps_per_level, device=device,
        )
        cscored = screen.score(cond)
        print(f"[cmp] conditioned valid/scored = {len(cscored)}/{args.n}")
        print(f"{'target':<28}{'uncond med':>12}{'cond med':>12}{'delta':>12}  dir  hit")
        print("-" * 80)
        hits = 0
        for key, direction in REPORT.items():
            um, cm = umed[key], _median(cscored, key)
            delta = cm - um
            good = (delta * direction) > 0
            hits += int(good)
            arrow = "max" if direction > 0 else "min"
            print(f"{key:<28}{um:>12.3f}{cm:>12.3f}{delta:>+12.3f}  {arrow:>3}  {'YES' if good else 'no'}")
        print("-" * 80)
        print(f"[cmp] trust_radius={tr}: conditioning moved {hits}/{len(REPORT)} targets the right way")


if __name__ == "__main__":
    main()

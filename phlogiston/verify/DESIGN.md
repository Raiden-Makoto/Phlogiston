# Verify (Tier 2) — DESIGN

Independent **physics verification** of shortlisted candidates. This is the first
point in the pipeline where a model *outside the generation loop* — a pretrained
universal ML interatomic potential (uMLIP) — touches the candidate, so it both
**gates** on real physics and **measures the bias** of our in-loop predictor.
Consumes the durable registry from `discovery` (`candidates.csv` + CIFs) and
writes verified verdicts back. This document is the plan; it is intentionally
code-free. See `docs/pipeline.md` §7 and `discovery/` (Tier 0/1).

---

## 0. Why this layer exists

**The bias problem (grading our own homework).** The discovery stability score is
a *property regressor* that is also inside the generation loop: latent
optimization gradient-ascends the predictor's own `energy_above_hull` estimate,
so surviving candidates cluster exactly where that predictor is *optimistic*. A
model used to both steer and judge subscribes to its own errors. We need
independent evidence.

**Why the predictor cannot do this job (it is not redundant with uMLIP).**

| | Stability score (predictor) | uMLIP (this layer) |
|---|---|---|
| Kind of model | scalar-energy regressor | full potential energy surface (E + forces) |
| Relaxes geometry? | no — scores the *as-generated* (noisy) cell | yes — finds the real local minimum |
| Gives forces / phonons? | no | yes → dynamical stability |
| In the generation loop? | yes (circular) | no (independent) |
| Trained on | equilibrium structures, masked scalar labels | DFT relaxation trajectories (`MPtrj`) |

Three things this buys that the predictor structurally can't:
1. **Relaxation** — the CDVAE emits approximate geometry; "the material" is the
   relaxed structure at the bottom of its well, which can differ substantially.
   Scoring the unrelaxed cell answers a question about a structure that doesn't
   physically exist.
2. **Dynamical stability** — "no imaginary phonon modes" needs second derivatives
   of energy w.r.t. displacement, i.e. a consistent force field. A scalar
   regressor gives one point and cannot distinguish a minimum from a saddle.
3. **Independence + calibration** — an out-of-loop verdict, *and* the residual
   `E_uMLIP − E_predictor` measures the predictor's optimism bias (see §5).

## 1. Why uMLIP is the primary tool (not a DFT stopgap)

For screening fictional candidates at scale, a foundation uMLIP is *strictly the
better choice* here — not a cheap approximation we tolerate until DFT is
available:

- **Hull compatibility for free.** `energy_above_hull` is only meaningful in a
  consistent reference frame. Foundation potentials (MACE-MP-0, CHGNet) are
  trained on **`MPtrj`** — Materials Project relaxation trajectories — so their
  energies live in MP's frame and are directly hull-comparable *without* DFT.
- **Runs on the hardware we have.** uMLIPs are PyTorch models → run on the gbt
  AMD MI350X (ROCm). DFT (VASP/QE) is CPU-MPI or **CUDA/NVIDIA** GPU; ROCm DFT is
  essentially unsupported, so DFT would need a separate CPU HPC/cloud cluster.
- **Cost.** uMLIP relaxation is seconds, phonons minutes — vs DFT hours per
  relaxation and 50–200 runs per phonon calc. This turns "verify thousands of
  fictional structures" (infeasible) into a routine batch on the existing box.
- **Consistency = robustness.** One fixed model applies identical physics to
  every candidate, so its errors are *systematic and cancel* in hull/residual
  comparisons. A DFT campaign is a fleet of independent runs, each needing
  per-system convergence babysitting (k-points, smearing, magnetism, mixing) —
  a single mis-converged run silently poisons a verdict. At screening scale the
  uMLIP is often the *more* trustworthy signal, not just the faster one.
- **Honest caveat.** uMLIPs are least reliable **off-distribution** — exactly
  where our more exotic fictional compositions live. So the uMLIP verdict is
  trustworthy for "relaxes to a sane, dynamically-stable, near-hull structure,"
  but an exotic *winner* still deserves one DFT confirmation. That is **Tier 3**:
  a manual, on-demand, top-1-or-2 DFT check on HPC/cloud — not a pipeline stage
  (see §9).

## 2. Model choice

Default **MACE-MP-0** (best relaxation + phonon fidelity; heavier). Kept behind a
thin adapter so the backend is swappable:

| Backend | Notes |
|---|---|
| **MACE-MP-0** (default) | best accuracy, strong phonons; larger/slower |
| CHGNet | lighter, includes magnetic moments |
| ORB / MatterSim | fastest; for cheap mass pre-screen if throughput bites |

All are ASE-`Calculator`-compatible, so relaxation/phonons use the same driver
regardless of backend.

## 3. Decisions locked in

- **Relaxation replaces the structure.** Relaxation moves the cell downhill to
  the nearest local minimum, so the relaxed structure is lower-energy and the
  physically real one → it **replaces** the generated structure as the canonical
  candidate. The original generated CIF is **kept for provenance**, and we record
  the **relaxation drift** (RMSD of positions, |Δvolume|, energy drop): large
  drift is itself a signal that the generator's geometry was poor / off-manifold.
- **Phonons only on near-hull survivors.** Phonons are the expensive-ish step;
  running them on structures far above the hull is wasted compute. They run only
  on candidates below a (tight) `phonon_e_hull_max` threshold after §4 relaxation.

## 4. Pipeline (staged, cheap → expensive)

```
candidates.csv + CIFs  (from discovery)
  └─ 2a  RELAX  (uMLIP, gbt GPU)
        • ASE relaxation (cell + positions) to local minimum
        • replace structure with relaxed; keep original CIF; record drift
        • total energy E_uMLIP
  └─ 2b  REFINED HULL
        • formation energy from uMLIP; build local convex hull from the
          candidate's chemical system (MP entries), place candidate on it
        • → e_above_hull_umlip   (MP-frame, hull-comparable)
        • residual = e_above_hull_umlip − e_above_hull_predictor   (bias meter)
        • GATE: drop candidates with e_above_hull_umlip > verify_e_hull_max
  └─ 2c  DYNAMICAL STABILITY  (only if e_above_hull_umlip ≤ phonon_e_hull_max)
        • finite-displacement phonons (phonopy) with the same uMLIP
        • min phonon frequency; flag imaginary modes (allow small Γ tolerance)
        • GATE: dynamically_stable = (no significant imaginary modes)
  └─ 2d  (optional, later) ELASTIC TENSOR
        • strain–stress with the uMLIP → K, G; re-check vs predicted moduli
  └─ WRITE BACK to candidates.csv + a batch calibration report
```

**Local hull construction (2b detail).** For each candidate, pull the competing
phases of its chemical system (from MP metadata / `mp-api`) and either (a) place
the candidate's uMLIP formation energy on the MP DFT hull, or (b) — preferred for
self-consistency — relax the competing phases with the *same* uMLIP so systematic
model errors cancel, then build the hull with `pymatgen`'s `PhaseDiagram`. Choice
is an open decision (§10).

## 5. The calibration signal (bias meter)

Beyond a per-candidate verdict, the batch of residuals
`e_above_hull_umlip − e_above_hull_predictor` characterizes the predictor:

- **Consistent offset** → the predictor is systematically optimistic by a fixed
  amount; we can shift the discovery gate to compensate.
- **Large, structure-dependent scatter** → the predictor is being *gamed*
  off-manifold by the latent optimizer → signal to tighten the conditioning
  trust region / lower `cond-trust-radius` in discovery.

Emitted as a batch report (distribution stats + parity plot) alongside the
verified registry.

## 6. Registry integration (I/O)

Verification **appends columns** to the existing `candidates.csv` (no new store):

| New column | Meaning |
|---|---|
| `e_above_hull_umlip` | refined hull distance in MP frame |
| `formation_energy_umlip` | uMLIP formation energy per atom |
| `predictor_residual` | `e_above_hull_umlip − e_above_hull_predictor` |
| `relax_rmsd`, `relax_dvol`, `relax_de` | relaxation drift diagnostics |
| `dynamically_stable` | phonon verdict (True/False/`--` if not run) |
| `min_phonon_freq` | most-negative phonon frequency (THz) |
| `verify_tier` | `screened` (discovery) vs `verified` (passed Tier 2) |
| `relaxed_cif` | path to the relaxed CIF (canonical structure) |

- Relaxed CIFs written to `DIR/relaxed/`; original generated CIFs untouched in
  `DIR/cifs/` (provenance).
- `show-candidates` gains a `--verified` view and shows the new columns / tier.

## 7. Reuse & new dependencies

- **Reused**: `pymatgen` (Structure/CIF I/O, `PhaseDiagram`, hull), the existing
  registry read/write in `discovery/loop.py`, MP access in
  `data/materials_project.py`.
- **New**: `mace-torch` (or `chgnet`), `ase` (relaxation driver / FrechetCellFilter),
  `phonopy` (finite-displacement phonons). All CPU/ROCm-friendly; no CUDA-only
  ops. Add to `requirements.txt` (and note in README install).

## 8. Module layout (planned)

```
phlogiston/verify/
  DESIGN.md          # this file
  potential.py       # uMLIP backend adapter (MACE default) → ASE Calculator
  relax.py           # ASE relaxation; drift metrics; replace-with-relaxed
  hull.py            # local PhaseDiagram + e_above_hull_umlip + residual
  phonons.py         # phonopy finite-displacement; imaginary-mode gate
  elastic.py         # (optional, 2d) strain–stress moduli
  verify.py          # orchestrator: registry → stages → write-back + report
```
CLI: a `verify` subcommand (`--save-dir`, thresholds, `--backend`, `--no-phonons`,
`--elastic`) mirroring `discover`'s ergonomics.

## 9. Tier 3 (DFT) — deliberately out of pipeline

A single DFT confirmation for an exotic finalist, run manually on HPC/cloud
(atomate2 + MP-compatible input set + custodian for error recovery). Not built
now; the verified registry (relaxed CIFs + uMLIP verdicts) is exactly the
hand-off artifact it would consume.

## 10. Build plan (incremental, validated)

1. `potential.py` backend adapter + a sanity relax of a known solid (e.g. Si,
   NaCl) → energies/lattice match references.
2. `relax.py` on a handful of registry CIFs; verify drift metrics + replacement.
3. `hull.py`: reproduce a known MP `e_above_hull` for a real material within
   tolerance (validates the MP-frame comparison).
4. `phonons.py`: confirm a stable solid has no imaginary modes and a known
   unstable one does.
5. `verify.py` end-to-end on the current 10-candidate registry → write-back +
   calibration report; eyeball the residuals.

## 11. Open decisions

- uMLIP backend (MACE-MP-0 vs CHGNet) and version pinning.
- Local hull: MP-DFT competitors vs uMLIP-relaxed competitors (self-consistent).
- Thresholds: `verify_e_hull_max` (stability gate) and `phonon_e_hull_max`
  (phonon trigger) — likely tighter than discovery's 0.1 eV/atom.
- Phonon rigor: Γ-point only (fast) vs small mesh; imaginary-mode tolerance.
- Whether to feed the calibration offset back into discovery automatically or
  keep it advisory (report only).
- Include Tier-2d elastic re-verification in v1 or defer.
```

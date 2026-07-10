# Verify (Tier 2) — DESIGN

Independent **physics verification** of shortlisted candidates. This is the first
point in the pipeline where a model *outside the generation loop* — a pretrained
universal ML interatomic potential (uMLIP) — touches the candidate, so it both
**gates** on real physics and **measures the bias** of our in-loop predictor.
Consumes the durable registry from `discovery` (`candidates.csv` + CIFs) and
writes verified verdicts back. This document is the plan; it is intentionally
code-free. See `docs/pipeline.md` §7.5 and `discovery/` (Tier 0/1).

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
| Trained on | equilibrium structures, masked scalar labels | Materials Project relaxation trajectories (`MPtrj`) |

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

## 1. Why an ensemble of uMLIPs

Foundation universal ML interatomic potentials are the right verification tool
for this pipeline, and we use **an ensemble of independent ones** rather than a
single model. Why uMLIP at all, then why an ensemble:

- **Hull compatibility for free.** `energy_above_hull` is only meaningful in a
  consistent reference frame. Foundation potentials (MACE-MP-0, CHGNet, ORB) are
  trained on **`MPtrj`** — Materials Project relaxation trajectories — so their
  energies live in MP's frame and are directly hull-comparable.
- **Runs on the hardware we have.** uMLIPs are PyTorch models → they run on the
  gbt AMD MI350X (ROCm) GPUs, the same box as the rest of the pipeline.
- **Cost.** A relaxation is seconds and phonons minutes, so verifying the whole
  registry (and re-scoring with several potentials) is a routine batch, not a
  campaign.
- **Consistency.** A fixed model applies identical physics to every candidate,
  so its errors are *systematic* and largely cancel in hull/residual comparisons.

**Why an ensemble — the off-distribution safeguard.** Any single uMLIP is least
reliable **off-distribution**, which is exactly where our more exotic fictional
compositions live. Instead of trusting one model there, we run several
*independently-trained* potentials (MACE, CHGNet, ORB) and treat their
**disagreement** as the confidence signal: when they concur on the relaxed
structure and energy, the verdict is trustworthy; when they diverge, the
candidate is flagged **low-confidence / off-distribution** for manual scrutiny.
The ensemble spread is our built-in "don't trust me here" detector — no external
ground-truth step required (see §9).

## 2. The ensemble

| Member | Status | Role |
|---|---|---|
| **CHGNet** (primary) | **wired + tested on ROCm** | relaxation, hull, phonons, elastic |
| **MatterSim** | **wired + tested on ROCm** | independent cross-check (disagreement) |
| ~~MACE-MP-0~~ | excluded | hard-pins `e3nn==0.4.4` vs our `e3nn>=0.5` |
| ~~ORB~~ | excluded | 0.7.0 dropped its ASE calc + needs CUDA-only `warp` |

The primary model drives relaxation and phonons; the other **re-scores the
relaxed structure** (and optionally re-relaxes) to produce the disagreement
signal. Backends are constructed through one adapter (`potential.py`,
`load_calculator(backend, device)`) so every member runs through the same ASE
driver, and membership is configurable.

> **Reality notes (from wiring it up).** Two obvious members didn't fit this
> ROCm + from-scratch stack. **MACE-MP-0** hard-pins `e3nn==0.4.4`, conflicting
> with the `e3nn>=0.5` our models are built/verified on (same interpreter).
> **ORB** 0.7.0 dropped its ASE `Calculator`, changed its loader API, and depends
> on `warp-lang` (no ROCm GPU path). **CHGNet** (primary) and **MatterSim** both
> ship ASE calculators, install conflict-free, and relax Si/NaCl/Cu to correct
> energies on the MI350X GPUs — and their energies differ enough to give a real
> disagreement signal. CHGNet bundles its weights (offline); MatterSim downloads
> them once on first use (then cached).

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
  └─ 2a  RELAX  (primary uMLIP, gbt GPU)
        • ASE relaxation (cell + positions) to local minimum
        • replace structure with relaxed; keep original CIF; record drift
        • total energy E_uMLIP
  └─ 2b  REFINED HULL
        • formation energy from uMLIP; build local convex hull from the
          candidate's chemical system (MP entries), place candidate on it
        • → e_above_hull_umlip   (MP-frame, hull-comparable)
        • residual = e_above_hull_umlip − e_above_hull_predictor   (bias meter)
        • GATE: drop candidates with e_above_hull_umlip > verify_e_hull_max
  └─ 2c  ENSEMBLE CROSS-CHECK  (CHGNet, ORB re-score the relaxed cell)
        • per-member energy / e_above_hull on the relaxed structure
        • disagreement = spread across members (energy + structural)
        • → ensemble_confidence flag (high / low = off-distribution)
  └─ 2d  DYNAMICAL STABILITY  (only if e_above_hull_umlip ≤ phonon_e_hull_max)
        • finite-displacement phonons (phonopy) with the primary uMLIP
        • min phonon frequency; flag imaginary modes (allow small Γ tolerance)
        • GATE: dynamically_stable = (no significant imaginary modes)
  └─ 2e  (optional, later) ELASTIC TENSOR
        • strain–stress with the uMLIP → K, G; re-check vs predicted moduli
  └─ WRITE BACK to candidates.csv + a batch calibration report
```

**Local hull construction (2b detail).** For each candidate, pull the competing
phases of its chemical system (from MP metadata / `mp-api`) and either (a) place
the candidate's uMLIP formation energy on the MP hull, or (b) — preferred for
self-consistency — relax the competing phases with the *same* uMLIP so systematic
model errors cancel, then build the hull with `pymatgen`'s `PhaseDiagram`. Choice
is an open decision (§11).

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
| `ensemble_e_hull_spread` | disagreement across ensemble members (eV/atom) |
| `ensemble_confidence` | `high` vs `low` (low = off-distribution, scrutinize) |
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
- **New (installed + tested on ROCm)**: `chgnet` (primary uMLIP), `mattersim`
  (cross-check uMLIP), `ase` (relaxation driver / `FrechetCellFilter`), `phonopy`
  (finite-displacement phonons). `mace-torch` and `orb-models` are deliberately
  **not** used (e3nn pin / dropped-ASE-calc + CUDA-only warp, respectively). All
  are in `requirements.txt`; the ROCm image builds and the full relaxation path
  runs on the MI350X GPUs. NB: `mattersim` pulls a heavy dependency tail
  (`atomate2`, `wandb`, `azure-*`, ...) that our code never imports — candidate
  for slimming later.

## 8. Module layout (planned)

```
phlogiston/verify/
  DESIGN.md          # this file
  potential.py       # uMLIP backend adapters (CHGNet + MatterSim) → ASE Calculator  [DONE]
  ensemble.py        # multi-potential re-score + disagreement/confidence
  relax.py           # ASE relaxation; drift metrics; replace-with-relaxed
  hull.py            # local PhaseDiagram + e_above_hull_umlip + residual
  phonons.py         # phonopy finite-displacement; imaginary-mode gate
  elastic.py         # (optional, 2e) strain–stress moduli
  verify.py          # orchestrator: registry → stages → write-back + report
```
CLI: a `verify` subcommand (`--save-dir`, thresholds, `--members`, `--no-phonons`,
`--elastic`) mirroring `discover`'s ergonomics.

## 9. Off-distribution safeguard — ensemble disagreement

The ensemble is how we stay honest on exotic compositions without any external
ground-truth step. After relaxation, every member re-evaluates the relaxed cell;
we summarize their **spread** in energy / `e_above_hull` (and, optionally,
whether independent re-relaxations converge to the same geometry):

- **Tight agreement** → the region is well-covered by the potentials' training
  data; the verdict (near-hull + dynamically stable) is trustworthy.
- **Wide disagreement** → the candidate sits off-distribution for at least one
  model; it's flagged `ensemble_confidence = low` and surfaced for manual review
  rather than silently trusted or dropped.

This turns the single biggest weakness of foundation potentials — unreliability
off-distribution — into an explicit, per-candidate confidence estimate, and it's
the documented substitute for a separate physics-confirmation tier.

## 10. Build plan (incremental, validated)

1. ~~`potential.py` backend adapter + a sanity relax of a known solid (e.g. Si,
   NaCl) → energies/lattice match references.~~ **DONE** — CHGNet + MatterSim on
   ROCm GPU; `scripts/verify_potential.py` relaxes Si/NaCl/Cu to correct energies.
2. ~~`relax.py` on a handful of registry CIFs; verify drift metrics +
   replacement.~~ **DONE** — `scripts/verify_relax.py`. NB: the current registry
   candidates relax with *large* drift (rmsd ~0.2-1.6 A, |dV| up to 0.7, energy
   drops of 1-11 eV/atom) and mostly fail to converge in 500 steps — direct
   evidence the generator's raw cells are far off-manifold (the drift red-flag
   working as intended). May warrant a higher default `steps` in verify.py.
3. `hull.py`: reproduce a known MP `e_above_hull` for a real material within
   tolerance (validates the MP-frame comparison).
4. `phonons.py`: confirm a stable solid has no imaginary modes and a known
   unstable one does.
5. `verify.py` end-to-end on the current 10-candidate registry → write-back +
   calibration report; eyeball the residuals.

## 11. Open decisions

- Ensemble = CHGNet (primary) + MatterSim (cross-check), both wired. Optional
  third member later (would need to be ASE-ready + ROCm/stack-compatible).
- Slim MatterSim's heavy dependency tail (atomate2/wandb/azure) if image size
  matters.
- Disagreement metric + `ensemble_confidence` threshold; energy-spread only vs
  also structural (re-relax) agreement.
- Local hull: MP competitors vs uMLIP-relaxed competitors (self-consistent).
- Thresholds: `verify_e_hull_max` (stability gate) and `phonon_e_hull_max`
  (phonon trigger) — likely tighter than discovery's 0.1 eV/atom.
- Phonon rigor: Γ-point only (fast) vs small mesh; imaginary-mode tolerance.
- Whether to feed the calibration offset back into discovery automatically or
  keep it advisory (report only).
- Include Tier-2e elastic re-verification in v1 or defer.

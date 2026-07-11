# Active-Learning Flywheel — DESIGN

Close the discovery loop by feeding **Tier-2 verdicts back into the models that
steer and gate generation**. Tier-2 (uMLIP relax + self-consistent hull) is the
first out-of-loop signal that tells us where the in-loop predictor is wrong; this
document turns that signal into training data so the *next* batch is better.
Intentionally code-free — this is the plan. See `phlogiston/verify/DESIGN.md`
(Tier 2) and `docs/pipeline.md` §7.

---

## 0. Why this layer exists

The recalibration experiment (2026-07-11) made the problem concrete. With the
Tier-1 synthesizability gate on, the discovery predictor's `energy_above_hull`
is a **consistent optimist on the generator manifold**: the uMLIP hull put the
survivors ~+0.25 eV/atom higher than predicted (n=9, mean +0.247, std 0.12).
Applying that measured offset as a static gate bias (`--stability-bias 0.25`)
and re-running produced **0 survivors at e_hull ≤ 0.1** — i.e. the candidates
were never actually near the hull; the predictor's ~0 estimates were optimism.

A constant bias is therefore only a band-aid: it makes the gate *honest*, but it
makes yield collapse because the underlying models still (a) mis-rank
generator-manifold structures and (b) steer the latent optimizer straight into
the optimistic region. The fix is to **retrain on the failures** so the models
stop being wrong where the generator actually samples.

## 1. What the failures give us (the label)

Every verified candidate carries, after Tier-2:

- the **relaxed structure** (`relaxed/<id>.cif`) — the real local minimum, not
  the as-generated cell,
- `e_above_hull_umlip`, `formation_energy_umlip` — an **outside** stability label,
- `predictor_residual = e_above_hull_umlip − e_above_hull_pred` — the error we
  want to drive to zero,
- (when run) ensemble spread and phonon stability.

These are **hard negatives**: structures the in-loop models scored as near-hull
that an independent potential says are not. That is exactly the signal an
active-learning loop wants — informative, on-manifold, and expensive to get.

## 2. The uMLIP-vs-DFT reference subtlety (must not ignore)

The stability specialist (Stage-1) and the latent head were trained on **DFT**
hull distances (Materials Project + GNoME). `e_above_hull_umlip` is computed on a
**self-consistent uMLIP hull** (candidate and competitors both relaxed with the
same potential). These are two different reference surfaces; they differ by a
per-chemistry offset (typically 0.05–0.15 eV/atom).

Consequences for how we use the label:

- **Do NOT** naively concatenate uMLIP `energy_above_hull` values into the DFT
  label column as if they were the same quantity — that injects the reference
  offset as label noise comparable to our signal.
- The **robust, reference-agnostic** part of the signal is the *sign and rank*:
  "these generator structures are less stable than the model thinks / than real
  MP polymorphs." Stage-1 is used as a **gate** and is selected on **AUC**
  (stable-vs-unstable discrimination), so teaching it "these are on the unstable
  side" is both well-posed and exactly what the gate needs.

Chosen v1 label policy:

1. **Stage-1 stability fine-tune** — inject the harvested structures with their
   `energy_above_hull` set to the **uMLIP hull distance directly** (mask on).
   Rationale: Stage-1 is selected on stability **AUC**, so what matters is that
   the labels *order* structures correctly; the failures sit far above the gate
   threshold (0.1–11 eV/atom) where the small uMLIP↔DFT offset (~0.1) is not the
   dominant signal, and the handful of genuine near-hull survivors keep their
   true low value so they remain positives. We deliberately avoid clamping to a
   fixed "unstable" value, which would corrupt those positives. (A future v2
   could learn an explicit DFT↔uMLIP delta head given DFT labels for a
   calibration subset.)
2. **Latent head refit** — the head only needs to know "this latent region
   decodes to something the outside judge dislikes," so the same uMLIP label
   steers conditioning away from it.

## 3. Which models get retrained (and which don't)

| Model | Retrain? | Why |
|---|---|---|
| **Stage-1 stability specialist** | **Yes** — warm-start fine-tune | It is the gate. Fixing its on-manifold optimism is the highest-leverage change. |
| **Latent property head** `f_p(z)` | **Yes** — refit including failures | It is what the latent optimizer games; teaching it the bad regions improves *conditioning* (the "better generation" lever). |
| Stage-2 property predictor | Not yet | The mechanical/thermal targets were never uMLIP-verified, so we have no outside label for them. |
| CDVAE generator/decoder | Not yet | Retraining the decoder on *failures* would teach it to reproduce them. The right generator-side move is reward-weighted fine-tuning on the *survivors*, which needs a larger verified positive set first (see §5). |

## 4. The loop

```
discover ──▶ Tier-2 verify ──▶ harvest (failures + survivors, uMLIP labels)
   ▲                                        │
   │                                        ▼
   └──── refit latent head  ◀── fine-tune Stage-1 stability  ◀── feedback shards
```

Concretely, one turn of the flywheel:

1. **harvest-verified** — read one or more registries' `candidates.csv` +
   `relaxed/` CIFs, keep rows with a numeric `e_above_hull_umlip`, featurize the
   *relaxed* structure, and write feedback records in the shard format
   (`{graph, y, mask}`) to a separate feedback root (kept out of the main corpus
   so it can be up-weighted and never leaks into val/test).
2. **Stage-1 fine-tune** — warm-start from the current `predictor_stage1_best.pt`,
   train on a **replay mix** (main DFT corpus + feedback replicated `w×` to
   up-weight the scarce hard negatives), low LR, early-stop on stability AUC, and
   hold out the feedback from validation so the metric stays honest.
3. **Latent-head refit** — re-run `fit-latent-head` on the CDVAE with the feedback
   records included, so `f_p(z)` learns the bad latent regions.
4. **Re-discover → re-verify** the same config and measure: (a) predictor residual
   mean/std shrink, (b) survivor yield at the honest gate rises, (c) fraction
   dynamically stable rises.

## 5. Data-sufficiency caveat

The current harvested set is small: ~51 verified structures across all runs (40
+ 2 + 9), almost all *negatives*. Two implications:

- Fine-tuning on ~51 hard negatives will nudge, not transform, the models — the
  effect on residual std should be measurable but modest.
- The generator-side improvement (reward-weighted decoder fine-tune on
  *survivors*) is blocked until we have enough verified **positives**. The
  cheapest way to grow both sets is to verify a **larger unbiased batch** (e.g.
  all Tier-0/Tier-1 survivors, a few hundred structures) rather than only the
  top-k. That harvest run is the recommended precursor to a serious flywheel.

## 6. Success criteria

- Predictor residual on a fresh verified batch: **mean → 0, std < 0.10**.
- Honest-gate yield (survivors at e_hull_umlip ≤ 0.1 without a static bias):
  **rises** relative to the current ~2/9.
- No regression in Stage-1 held-out DFT stability AUC (guard against the
  feedback distorting the base task).

## 7. Non-goals

- No DFT (uMLIP is the reference of record, per project scope).
- No decoder retrain in v1 (needs a positive set; §5).
- No online/continual learning — the loop is run in discrete, logged turns so
  each model version is reproducible and comparable.

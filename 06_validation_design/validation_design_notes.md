# The two validation experiments

## What is already done, and what needs your GPU

Both experiments split into a part that needs no model and a part that needs the trained
checkpoints. **The no-model parts are finished and verified here.** The model parts need one
Colab run against your Drive checkpoints, which is what `CMIN_validation.ipynb` does.

| | done here | needs the Colab run |
|---|---|---|
| **Exp 1** | the target: PC1 factor, loadings fitted on train, projected to test | the province latent from each model, and the ridge probe |
| **Exp 2** | the event log, the peak rule, the null, the specificity control, a pilot | ten-seed attribution and graph statistics |

Nothing in the notebook trains, and nothing on Drive is moved, overwritten or deleted. The
cells that train (P6, H3) and the cell that quarantines stale files (H3) are not copied into
it. Results are written to a new folder, `protocol/validation/`.

## Three findings that changed the design

**1. The global mixture cannot support an attribution claim, at all.**
I piloted the whole Experiment-2 pipeline on the June single-seed daily attribution
(`CMIM/CMIN 1/global_context_timeseries.csv`) before writing the notebook. The naive test
looked like a win: weather scored F1 = 0.522 against a circular-shift null at p = 0.028.
It is an artefact. The four global weights correlate at |r| > 0.999 and 99.96% of their
joint variation sits on one principal component, because the softmax is saturated at
price = 0.989. Weather, supply and economy peak on *identical days*. The series is one
scalar wearing four labels.

So the notebook adds a **domain-specificity control**: for events of domain *e*, compare
*e*'s own recall against the mean recall of the other three domains on the same events. It
drops that weather result to p = 0.089, correctly. Experiment 2 runs on the **node-level**
attention, which is not degenerate (in Ramadan its weather weight falls while economy
rises, which one scalar cannot do).

**2. The event log is thinner than the plan assumed.** Counts over the full 1,826 days,
with the test year in brackets:

| domain | event type | n | independent? |
|---|---|---|---|
| weather | extreme rain, train p95 | 100 (27) | no, model input |
| economy | Ramadan | 160 (40) | no, model input |
| economy | Idul Fitri | 5 (1) | no, model input |
| economy | fuel-price change | 6 (**0**) | no, model input |
| supply | regulatory events | 23 (**2**) | **yes**, never an input |

Two consequences. Fuel prices are **constant through the whole test year**, so that arm
cannot run on the test period at all. And the *BPS harvest calendar does not exist in this
data*: `Panen tahun lalu` changes only on 1 January each year, so it is an annual constant,
not an event series. The alignment test is therefore run over the full sample, and the only
genuinely independent arm carries 23 events. That arm is reported as underpowered, not as
null. Everything else is a faithfulness check: does the model point at the domain whose own
input moved?

**3. The tests have the power to detect a real effect.** I ran the notebook's own analysis
cells against synthetic model outputs where the answer is known. A planted weather signal is
recovered at specificity 0.467, median p = 0.002, significant in all six synthetic seeds, and it holds
across all twelve combinations of the two pre-specified thresholds. A planted factor latent
is recovered at r = 0.99, R² = 0.97 at h = 14. A collapsed attribution is correctly refused,
and an unplanted domain correctly returns null (p = 0.54). So a negative result from the
real run will mean the effect is absent, not that the test could not see it.

## What the outcomes mean

**Experiment 1.** The interesting result is the *ordering*, not the R². If uniform pooling
tracks the factor as well as the learned graph, that supports the paper as written: the
graph is a common-factor estimator, not a map of pairwise trade. The only damaging outcome
is no relationship in any arm, which would withdraw §7.3.

**Experiment 2.** Three outcomes, all publishable:
1. specificity positive and stable → the attribution is validated, and §7.6 stops being a
   limitation;
2. regimes stable across seeds but specificity null → report the regime tables, drop the
   domain-attribution claim, keep the synchronisation index;
3. neither survives → the single-seed regime figure comes out and the instrument claim
   narrows to the two results already at ten seeds, adjacency reproducibility (r = 0.987)
   and regime response under the shifted-context control (p = 0.045 vs 0.461).

## Files

```
CMIN_validation.ipynb   the run. Open in Colab, run top to bottom, paste V6's output back.
build_targets.py        builds factor_series.csv and event_log.csv from the source files
factor_series.csv       PC1 factor at h in {1,3,7,14,30}, train-fitted loadings
factor_meta.json        per-horizon PC1 shares, train vs test under frozen loadings
event_log.csv           294 dated events, four types, with an independence flag
align.py                the alignment test as a standalone module
pilot.py                the pilot on the June series
pilot_alignment.csv     the pilot's raw result (the one that looks like a win)
pilot_specificity.csv   the control that rejects it
dryrun.py               executes the notebook's V2-V6 against synthetic outputs
```

## Running it

Upload `CMIN_validation.ipynb` to Colab, point it at the same Drive folder, switch on a GPU
and run every cell. **V0 prints a checkpoint inventory — read it before going further.** It
needs ten seeds of `L1_encoder_only@R`, `H_uniform_price@R`, `L3_learned_static@R`,
`L6_ctx_conditioned@R` and `L6_ctx_conditioned@S`. If a rung is short, every table reports
its own n rather than silently pooling. V1 is the only slow cell. Paste V6's summary block
back into the chat and I will write the results into the paper.

---

# Update after the first run (9 September)

The run came back with **one seed per rung**, not ten: Drive held a single
`w_*__s*.pt` per rung and no `H_uniform_price@R` at all. What that produced:

**Experiment 1 -- positive, and usable.** On the untouched test year, ridge fitted on
training days and loadings frozen on training days, the province latent recovers the
common factor at r = 0.69 and R2 = 0.47-0.53 at h = 7, 14, 30 (t = 17 on 337 test days).
The no-graph arm does it as well as CMIN-S and better than full CMIN, so message passing
adds nothing to factor recovery -- which is the paper's own claim, now measured. At one
seed the relationship is claimable; the ordering between arms is not.

**Experiment 2 -- negative, and robust.** The attribution is nearly uniform: spread across
the four domains is 0.020 against 0.187 in the June run, and the largest regime shift is
0.011 against June's 0.105. Nothing clears the specificity control; weather scores -0.309,
meaning its peaks avoid extreme-rain days, and that holds in all twelve pre-specified
threshold settings (-0.24 to -0.39, p 0.93 to 1.00). This is not the collapse artefact:
degrees of freedom on the standard head is 0.45, so the four weights do carry separate
signals. The null is genuine.

**A reporting bug this exposed, now fixed.** At one seed `ci95` is NaN, every comparison
against it is False, and V6 printed "0 of 27 ... the regime story does not survive ten
seeds" without having tested anything. `sign_stability` likewise returned 1.0 because a
single value agrees with its own mean. Both now report the seed count instead of a verdict.
The conclusion happens to be right for an independent reason -- the effect sizes above --
but a NaN must not write a sentence.

**Do not compare my density figure to June's.** I defined density as the share of adjacency
entries above 1/R; the June figure used a definition I do not have. What is valid within
this run is that density is identical across all four regimes to four decimal places.

## Next step: `CMIN_retrain.ipynb`

Run it before re-running the validation notebook. It trains the missing seeds of
`L6_ctx_conditioned@S` and stops. It writes only checkpoint files that do not already
exist, so seed 0 is never overwritten and an interrupted run resumes. It never calls
`record()`, so the `.pkl` prediction files behind every MAE in the paper cannot move.
R0 prints the exact run list before anything starts; R1 reports the wall time of the first
run and extrapolates, so you can stop early. Two optional flags in R0 (`ALSO_R`,
`ALSO_UNIFORM`) fill the anchored-head twin and the missing uniform-adjacency arm; both
default to off.

Then re-run `CMIN_validation.ipynb` unchanged and paste V6 back.

---

# Result, ten seeds (9 September)

`CMIN_retrain.ipynb` filled the nine missing seeds in 9.8 minutes; all ten L6@S checkpoints
verified. The validation notebook was then re-run.

**Experiment 1 -- positive.** Out-of-sample factor recovery at r = 0.69, R2 = 0.47-0.53
(h = 7/14/30, 337 test origins). The no-graph rung matches CMIN-S and beats full CMIN.
Still one checkpoint per arm, so the relationship is claimed and the ordering is not.

**Experiment 2 -- negative, at ten seeds.** No attribution or density regime effect is
stable in more than 8 seeds of 10 (sign test p = 0.109). Attribution spread across seeds
(weather 0.393 +/- 0.077) is as large as any regime effect in it (largest 0.087). Event
alignment: every specificity negative, no domain clears p < 0.05 in any seed, and the
pattern holds across all twelve threshold settings. Not a power failure -- the same
pipeline recovers a planted signal at p = 0.002.

Five graph parameters do shift reliably across all ten seeds, by at most 1.6% of their own
level: anchor alpha in harvest, fusion gate in peak rain and harvest, row entropy in
harvest and Ramadan. Reliable and negligible.

**A statistical error found and fixed.** V3 computed the confidence interval on the
*level* while the effect is a *paired* difference between two regimes for the same seed.
Testing a paired effect against the level's interval is the unpaired-test mistake and is
far too conservative; V6 then printed a verdict on that basis. Both now use a paired
interval and a two-sided sign test on the ten per-seed differences, and the summary shows
the effect as a percentage of its own level so that "reliable" and "large" cannot be
confused. Verified on synthetic inputs: a tiny effect with 10/10 sign agreement is now
detected (the old test missed it) and a large effect with 8/10 agreement is still rejected.

**One number still to verify: V7.** The paper's abstract used to lead with "adjacency
reproducibility, edge r = 0.987 over 45 seed pairs". 45 pairs means ten seeds, but only one
L6@S checkpoint existed when that ran, and the routine silently skips a seed whose
checkpoint will not load -- so the figure cannot have come from L6@S as implied. It most
likely came from the hierarchy rung H_real@S. The new **V7** cell recomputes it from the ten
L6@S checkpoints that now exist. The paper currently states the reproducibility claim
without the number; run V7 and I will put the right one in.

---

# The final run for Q1 (13 September): `CMIN_final.ipynb`

One notebook closes every experimental gap still open. It trains about 137 runs in three
tiers (roughly 2.5 hours on a T4), is resumable, and never deletes, moves or overwrites
anything. F12 zips every result into `protocol/final/cmin_final_results.zip`.

| gap | cell |
|---|---|
| r = 0.987 attributed to the wrong model, and possibly inflated by the shared physical anchor | F7: edge correlation on the full adjacency AND its learned part, against ten untrained models |
| factor probe at one seed per arm; uniform arm missing | T1 training, F4 |
| shifted-context placebo and regime response at one seed | T1 (L7@S), F8 |
| market-level co-movement and dispersion tests at one seed | F9, with a max-over-lags null |
| temperature (Eq. 3) never tested alone; four conditioning points only tested together | T2: each point removed alone at identical capacity, F11 |
| head-count and width sensitivity at three seeds | T3, F11 |
| no check that checkpoints reproduce the published tables | F3 |

Two guards run before training: every one-point variant must have exactly L6's parameter
count, and each removed point must be measurably constant across windows while it varied
in L6.

**A bug caught before it shipped.** The first draft looped `for sd in seeds` at notebook top
level. `sd` is the protocol's per-market price scale used by `RP()`, so every MAE after that
loop would have been silently wrong. All loop variables are renamed, and a guard now asserts
`mu` and `sd` are unchanged wherever a price is computed. `dryrun_final.py` runs F0, F2a and
F3-F12 against synthetic extraction files and the guard holds through all of them.

**References verified.** 21 entries were checked against Crossref and the rest by hand.
One did not exist: `arXiv:2108.05927`, cited as a hierarchical graph network paper, is a
hate-speech workshop overview. It is replaced by Guo et al., AAAI 2021, verified. 32
entries now carry verified DOIs.

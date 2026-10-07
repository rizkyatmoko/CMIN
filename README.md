# CMIN — reproduction and evidence package

Supporting material for the manuscript *"A Context-Conditioned Market Integration Network for
East Java Chilli Prices: Distinguishing Learned Graph Structure from a Provincial Common
Factor"*, submitted to *Computers and Electronics in Agriculture*.

Authors: Rizky Alfanio Atmoko, Slamin (Universitas Jember) and Yuqing Lin (University of
Newcastle).

This folder holds the data, the saved outputs, the analysis scripts and the notebooks behind
every table and figure in the paper, together with scripts that rebuild the model-free results
from the raw panel. 143 files, 153 MB, plus an index of 3.05 GB of per-seed model outputs that
are too large to include and are listed in `11_raw_runs_index/`.

Last updated 7 October 2026.

## How to use this package

`MANIFEST.csv` is the entry point. It has one row per numbered table and figure in the paper,
plus the paper's headline claims, and for each one gives the file here that supports it, the
script that produced that file, and what to look for inside it.

`CHECKSUMS.sha256` records a SHA-256 for every file, so any copy can be checked against the
version used for the paper.

## Where each result comes from

| Result in the paper | Files in this package |
|---|---|
| Forecast accuracy (Table 10) | `03_protocol_runs_544/mae_by_model_horizon.csv` — each MAE in the table is a ten-seed mean here (for example, persistence at h=1 is 2035.86, printed as 2,036). `metrics_long.csv` holds all 544 runs, one row per run, seed and horizon. |
| Model comparisons and verdicts | The three `contrast_*.csv` files in `03_protocol_runs_544/` carry, for every reported comparison, the two-level paired bootstrap interval, the equivalence outcome at the 2% margin and the Benjamini–Hochberg-corrected *p*-value. |
| Independent re-scoring | `05_dgcrn_baseline/reference_reproduction.csv`: 110 of the published model-seeds were re-scored offline from their stored predictions and match `metrics_long.csv` to 0.000 IDR. The 60 published capacity-family verdicts reproduce exactly in `dgcrn_contrasts_capacity70.csv`. |
| Adjacency collapse (Section 7) | `04_ten_seed_final_run/adjacency_means.npz` holds the learned adjacency per seed, `conditioning_weights_original.csv` the weight magnitudes from the original checkpoints, and `final_analysis.json` the summary statistics. The learned component is exactly uniform in 9 of 10 seeds on the persistence-anchored head. |
| Factor probe (Experiment 1) | `04_ten_seed_final_run/f4_factor_probe*.csv` for the results; `06_validation_design/factor_series.csv` for the target, whose loadings are fitted on training data only. |
| Attribution tests (Experiment 2) | `04_ten_seed_final_run/f5_*` for the regime effects and `f6_*` for event alignment, the circular-shift null, the domain-specificity control and the synthetic positive control that recovers a planted signal at median *p* = 0.002. |
| Road distances | `02_data_diagnostics/verify_distances.py` and `distance_check.csv`: the distance matrix cross-checked against OpenStreetMap routing between the 38 regency and city seats, r = 0.92 over 703 pairs, 81% of each city's eight nearest neighbours shared. `decay_robustness.json` repeats the distance-decay result on the routed distances. |
| Panel size | 120 distinct (city, market) pairs across 113 distinct market names; seven names recur in more than one regency. Both counts are printed by `02_data_diagnostics/recompute_section3.py`. |
| Per-seed model outputs | `11_raw_runs_index/` indexes 1,307 files across four exports, with sizes and checksums. Every run the paper scores has a stored prediction file: 544 of 544 protocol runs and 24 of 24 DGCRN runs, as established by `coverage_check.py`. |
| Training runs as executed | `07_notebooks/executed/` holds the twelve notebooks as returned from Colab, with training logs, parameter audits and printed tables. The copies one level up are the same code without outputs. |

## Rebuilding results from the raw data

Two scripts reconstruct the model-free results with no model and no trained checkpoint, and
print each recomputed value beside the published one:

```
cd 02_data_diagnostics
python recompute_section3.py          # Tables 3, 4, 5 and 6, chilli panel
python verify_commodity_panels.py     # Tables 3-5 for the five comparison commodities
```

```
cd 03_protocol_runs_544
python recompute_context_cost.py      # Table 11, from the per-seed metrics
```

Saved outputs of all three runs are stored alongside the scripts as `*_output.txt`.

## Layout

```
01_manuscript/          the submitted PDF and its single-file LaTeX source
02_data_diagnostics/    Section 3: model-free diagnostics, the distance cross-check, and
                        scripts that rebuild them from the raw panel
03_protocol_runs_544/   the 544-run protocol: per-run metrics, every contrast, and the
                        split-boundary sensitivity analysis
04_ten_seed_final_run/  the ten-seed final run: factor probe, regime effects, event
                        alignment, adjacency statistics, mechanism contrasts
05_dgcrn_baseline/      the DGCRN runs, the 70 paired comparisons, and the offline
                        reproduction of the published results
06_validation_design/   the design of the two validation experiments, the event log, the
                        factor target, and the pilot analyses that shaped them
07_notebooks/           the Colab notebooks that produced the runs, and their generators;
                        executed/ holds the copies returned with their run logs
08_supervisor_report/   a 12-page internal report on the two experiments, with its build script
09_raw_inputs/          the chilli panel, the five comparison commodity panels, the distance
                        matrix, weather, policy and CPI inputs, and the trained checkpoints
10_figures/             the nine figures as they appear in the paper, and make_figures.py
11_raw_runs_index/      an index, with checksums, of the 3.05 GB of per-seed predictions and
                        trained weights held outside this package
```

## Coverage of the tables

Every numbered table was checked by extracting its printed values from the LaTeX source and
locating them in the files listed above.

* Tables 3, 4, 5, 6, 10, 11, 12, 13, 14, 17, 18, 19, 21, B.1, C.1 and D.1: every printed value
  located in the cited file.
* Tables 8, 9, 16 and 20: the measured values are located; the remaining entries are quantities
  the paper derives from them, such as percentage changes, or protocol constants such as the
  675-day circular shift, which is set in the notebook.
* Tables 1 and 2 summarise previously published work and have no underlying data.
* Tables 7 and 15 are definitions and verdict summaries; their content is the model code and the
  contrast files respectively.

Two tables had no surviving output file from the original session. Both are rebuilt by the
scripts above and both reproduce exactly: Table 6 (daily, weekly and change-only rows) and
Table 11 (every percentage and every *p*-value).

## Reproduction notes

Three caveats affect how these files should be read.

1. **PC1 in Table 4 recomputes 0.002–0.008 lower.** The code that produced the published column
   is not among the saved files, so its exact pooling cannot be confirmed. Per-market
   normalisation, the mean of market logs, and dropping the first lookback window were all
   tried and move further from the published values, so the construction in
   `recompute_section3.py` is the closest available. Every other column of Tables 3, 5 and 6
   reproduces exactly, and the ordering and magnitude of the PC1 entries are unaffected.

2. **The inter-city road-distance matrix was compiled with an AI-assisted search of online map
   routes**, not taken from an official source; Section 3.1 of the paper states this. The
   cross-check in `02_data_diagnostics/` is what supports its use: r = 0.92 against
   OpenStreetMap routing over 703 city pairs, 81% of each city's eight nearest neighbours
   shared, and the distance-decay result unchanged when the routed distances are substituted.

3. **The split boundaries are defined on forecast origins, and target windows are thirty days
   long,** so 29 of the 143 validation windows contain calendar dates that also appear in the
   test targets — 435 of 4,290 validation target-days, or 10.1%. No test origin was used for
   parameter fitting or checkpoint selection, and the same split applies to every
   configuration, but validation and test are not fully separated in calendar time.
   `03_protocol_runs_544/leakage_robustness.py` re-scores all stored predictions with those
   dates excluded; `leakage_robustness_FINDINGS.md` reports the outcome and explains why that
   exclusion measures sensitivity to the March 2025 price spike rather than the effect of the
   overlap itself.

## Data sources

Retail prices are published by the East Java Provincial Office of Industry and Trade through
SISKAPERBAPO, weather variables by the NASA POWER project, and production statistics by BPS
Jawa Timur. Full citations are in the manuscript.

## Not included

The 3.05 GB of per-seed predictions and trained weights are indexed with checksums in
`11_raw_runs_index/` and are available from the corresponding author on request;
`09_raw_inputs/cmin_original_checkpoints.zip` holds the subset from which the weight-magnitude
results were computed. Everything else the paper relies on is in this folder.

Scripts that read the per-seed outputs refer to local paths on the machine where the analyses
were run. Those paths are recorded in `11_raw_runs_index/WHERE_THE_RAW_RUNS_LIVE.md` and need
to be changed to wherever the exports are placed.

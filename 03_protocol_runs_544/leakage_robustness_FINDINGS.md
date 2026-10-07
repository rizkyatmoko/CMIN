# The validation/test overlap, and what the re-scoring test can and cannot show

## The overlap, measured

Splits are defined on forecast origins. Test origins are those whose thirty-day target window
begins on or after 28 February 2025. Validation origins are the 143 immediately before them —
but their windows also run thirty days, so the last of them reach past the boundary:

* **29 of the 143** validation windows cross the boundary (origins 1430–1458).
* **435 of the 4,290** validation target-days — **10.1%** — fall inside the test period.
* They touch **29 of the 366** test days, 28 February to 28 March 2025.

Training never saw a test day. Checkpoint selection, which used a horizon-normalised validation
MAE, did see those 29 days as part of a score averaged over 143 windows.

## The test that was run

`leakage_robustness.py` re-scores every stored prediction on the 337 published test origins and
again on the 308 whose windows start after the last overlapping day, then re-runs the paper's own
inference on both. 105 comparisons across four families.

**42 of the 105 verdicts change**, and percentage differences move by up to 17 points.

## Why that result does not answer the leakage question

The 29 dropped origins are not an ordinary slice of the test year. Their windows cover
28 February to 26 April 2025, which contains Ramadan (1–30 March 2025) and the chilli price
spike around it:

| | mean price in window | CMIN window MAE at h=30 |
|---|---|---|
| first 29 origins | 81,050 IDR/kg | 50,353 |
| remaining 308 | 40,857 IDR/kg | 15,279 |
| ratio | 2.0× | **3.3×** |

Dropping them removes the hardest and most expensive regime of the test year. The verdict
changes therefore measure **sensitivity to that regime**, not the effect of the overlap. Every
comparison shifts in the same direction — models look better once the spike is gone — which is
what removing a hard period does, whatever the selection history.

**The leakage question cannot be settled this way.** The contamination is in checkpoint
selection, so answering it properly means re-selecting checkpoints from a validation set that
stops before the test boundary. Per-epoch validation predictions were not saved, so that means
retraining every arm.

## What the test does establish, and it is worth reporting

The comparisons are strongly period-dependent. With the Ramadan window in the test set, the
capacity family gives 44 inconclusive, 15 different, 1 equivalent; without it, 18 inconclusive,
40 different, 2 equivalent. A paper that reports "no resolved difference" should say that the
verdict depends on whether a single extreme regime is included, because a reviewer who
re-evaluates on a different year will find exactly this.

## Bottom line for the manuscript

1. Keep the disclosure of the overlap, with the measured numbers.
2. Do **not** claim the published verdicts are robust to it; this test does not show that.
3. Consider adding the regime-sensitivity result as a limitation. It is an honest finding, it is
   reproducible from the stored predictions, and it reinforces the paper's own position that
   differences at these sample sizes are often unresolved.

Files: `leakage_robustness.py`, `leakage_robustness.csv` (all 105 comparisons, both test sets).

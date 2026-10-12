# Robust Demand Prediction for Cutting-Stock Optimization

An end-to-end pipeline that connects the robust, regularized estimation work
in my thesis and the companion
[`slts-kle-hybrid`](../slts-kle-hybrid) project directly to the proposed
endeavor: reducing material waste in cutting-stock operations by improving
the demand predictions that cutting plans are built from.

## Pipeline

1. **Simulated production-order data** (`src/demand_model.py`): correlated,
   noisy process/sensor features (standardized machine and batch readings,
   generated from a low-rank factor model to induce realistic
   multicollinearity) plus a contamination fraction of anomalous orders
   (misreads, rework jobs), driving a dense true relationship to required
   piece length. This is a clearly-labeled simulation, not a claim about any
   specific company's real production data.
2. **Robust demand prediction** (`src/demand_model.py`): the SLTS +
   Kibria-Lukman hybrid estimator from the companion project is fit on the
   training orders and used to predict required piece lengths for new
   orders, compared against a naive OLS fit (no robustness, no
   regularization) and a plain Ridge fit (regularization, no outlier
   screening).
3. **Cutting-stock optimization** (`src/cutting_stock.py`): a
   First-Fit-Decreasing (FFD) solver, a standard heuristic for the 1D
   cutting-stock / bin-packing problem, turns a list of predicted piece
   lengths into an actual cutting plan against fixed-length stock rolls.
4. **Realized-waste evaluation** (`src/run_pipeline.py`): plans are built
   from each method's *predicted* lengths, then scored against the *true*
   lengths the customer actually needed — an overcut piece is scrap, an
   undercut piece requires a full extra stock roll to redo. This measures
   the quantity the proposed endeavor is actually about: material waste
   driven by prediction error, not just abstract coefficient accuracy.

## Results

Full output: `results/waste_comparison.csv`, `results/waste_comparison.png`.
8 repeats per contamination level, 90 simulated orders per run (63
train / 27 test split).

**Prediction accuracy (RMSE on held-out orders):**

| Contamination | Robust (SLTS+KLE) | Naive OLS | Plain Ridge |
|---:|---:|---:|---:|
| 0.00 | 3.2 | 2.1 | 1.8 |
| 0.05 | 12.0 | 19.2 | 14.0 |
| 0.10 | 26.8 | 39.5 | 28.8 |
| 0.15 | 38.3 | 46.7 | 39.7 |
| 0.20 | 41.3 | 46.7 | 38.5 |

With no contamination, the naive and plain-ridge models are slightly more
accurate, which is the expected cost of robustness when it isn't needed. Once
any meaningful contamination is present, the robust pipeline is consistently
more accurate than the naive OLS pipeline, and competitive with (sometimes
better than, sometimes slightly behind) plain Ridge, which is itself
regularized but has no protection against the contaminated orders in this
simulation.

**Realized material waste** (length units; see `results/waste_raw.csv` for
the per-run data) is noisier across only 8 repeats and does not show a clean
ordering at every contamination level — this is reported honestly rather
than smoothed over. The clearer and more load-bearing result is the
prediction-accuracy table above: lower upstream prediction error is the
mechanism by which better statistical estimation is supposed to reduce
downstream material waste, and that mechanism holds here once contamination
is present.

## Honest limitations

- The FFD cutting-stock solver is a standard heuristic, not an optimal
  column-generation solver; it is adequate for measuring the *relative*
  effect of prediction quality on waste, which is this project's purpose,
  but would not be the production solver for a real deployment.
- All data here is simulated. The simulation is built to resemble the
  statistical structure of real production telemetry (correlated process
  variables, a dense effect on outcome, a contamination fraction), not to
  reproduce any specific real dataset.
- Realized-waste results are noisy at 8 repeats per setting; the
  prediction-RMSE results, which are the mechanism the waste reduction
  depends on, are the more stable finding.

## Reproducing

```bash
pip install numpy scipy scikit-learn pandas matplotlib
cd src
python3 run_pipeline.py
```

Requires the companion `slts-kle-hybrid` project to be checked out as a
sibling directory (`demand_model.py` imports its estimator code directly).

## Files

```
src/demand_model.py      Simulated production-order data + robust/baseline model fitting
src/cutting_stock.py     First-Fit-Decreasing cutting-stock solver
src/run_pipeline.py      End-to-end pipeline: predict lengths -> cut plan -> realized waste
results/                 CSV results and plots from the runs reported above
```

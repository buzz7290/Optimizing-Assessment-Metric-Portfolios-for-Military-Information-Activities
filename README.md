# OIE Metric-Portfolio Reproducibility Code

These scripts consolidate the latest numerical logic used for the manuscript.
Some late-stage figure edits were originally made with short one-off plotting
calls, so this folder places the current logic into clean, reusable scripts.

## Files

### `metric_portfolio_core.py`
Single source of truth for the numerical example:
- 15 candidate metrics
- source classes
- causal-node coverage
- metric-type constraints
- source-diversity requirements
- sparse pairwise redundancy matrix
- base weights
- objective and feasibility functions
- exhaustive enumeration
- fast re-optimization over the feasible set

### `enumerate_portfolios.py`
Reproduces the exhaustive search used in the numerical application.
It reports the number of feasible portfolios, the optimum, runner-up, metric
costs, and top-ranked portfolios. It also writes CSV files containing the
complete feasible set and the top 20 portfolios.

Expected headline results:
- 32,768 possible portfolios
- 10,290 feasible portfolios
- base optimum `{1,4,6,9,12,13}`
- objective `F = 5.260`
- runner-up `{1,4,6,9,10,13}` with `F = 5.280`

### `analyze_weight_sensitivity.py`
Reproduces the individual objective-weight sensitivity calculations and writes
contiguous optimal-portfolio intervals to `weight_sensitivity_segments.csv`.

### `generate_manuscript_figures.py`
Generates the figures currently used for the numerical application and
sensitivity section:
- coverage map
- individual cost decomposition
- five individual weight-sensitivity figures
- joint collection-cost/latency stability map

## Requirements

Python 3 with:

```bash
pip install numpy matplotlib
```

## Typical workflow

```bash
python enumerate_portfolios.py
python analyze_weight_sensitivity.py
python generate_manuscript_figures.py
```

All scripts should be run from this directory so that imports resolve cleanly.

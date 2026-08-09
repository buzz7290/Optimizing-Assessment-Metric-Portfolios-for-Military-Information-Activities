# OIE Metric-Portfolio Reproducibility Code

This repository contains the Python code used to reproduce the numerical
application, sensitivity analyses, robustness check, and manuscript figures for
the article *Optimizing Assessment-Metric Portfolios for Military Information
Activities: A Quadratic Set-Covering Approach*.

The scripts use a common numerical instance defined in
`metric_portfolio_core.py` so that portfolio enumeration, sensitivity analysis,
and figure generation remain consistent across the reproducibility package.

## Files

### `metric_portfolio_core.py`

Single source of truth for the numerical application. It defines:

- 15 candidate assessment metrics
- source classes
- causal-node coverage
- metric-type constraints
- source-diversity requirements
- candidate planning attributes
- sparse pairwise redundancy matrix
- base-case objective weights
- objective and feasibility functions
- exhaustive enumeration
- fast re-optimization over the feasible set

### `enumerate_portfolios.py`

Reproduces the exhaustive search reported in the numerical application. It
reports the number of feasible portfolios, the optimum and runner-up portfolios,
objective values, and top-ranked feasible portfolios. It also writes CSV files
containing the complete feasible set and the top 20 portfolios.

Expected headline results:

- 32,768 possible portfolios
- 10,290 feasible portfolios
- base optimum `{1,4,6,9,12,13}`
- objective value `F = 5.260`
- runner-up `{1,4,6,9,10,13}` with `F = 5.280`

### `analyze_weight_sensitivity.py`

Reproduces the individual objective-weight sensitivity analysis. Each objective
weight is varied while the remaining weights are held at their base-case values.
The script identifies contiguous intervals over which each portfolio is optimal
and writes the results to `weight_sensitivity_segments.csv`.

### `input_score_robustness.py`

Reproduces the manuscript's input-score robustness check. The analysis
perturbs each candidate attribute score and each nonzero pairwise redundancy
score by one adjacent level on the planning scale, re-solves the optimization
problem after each perturbation, and records whether the base-case portfolio
remains optimal.

Expected headline result:

- 172 adjacent-level perturbations evaluated
- base-case portfolio remains optimal in 140 cases
- robustness rate: 81.4%

This script reports the robustness results in text; it does not generate a
figure.

### `generate_manuscript_figures.py`

Generates the figures used in the numerical application and sensitivity
analysis:

- candidate-to-node coverage map
- individual generalized-cost decomposition
- five individual weight-sensitivity figures
- joint collection-cost/latency stability map

## Requirements

Python 3 with:

```bash
pip install numpy matplotlib
```

No commercial optimization solver is required for the numerical instance
because the scripts use exhaustive enumeration over the 15 candidate metrics.

## Typical workflow

Run the scripts from the repository directory so that local imports resolve
correctly:

```bash
python enumerate_portfolios.py
python analyze_weight_sensitivity.py
python input_score_robustness.py
python generate_manuscript_figures.py
```

The first three scripts reproduce the principal numerical and robustness results
reported in the manuscript. The final script regenerates the manuscript
figures.

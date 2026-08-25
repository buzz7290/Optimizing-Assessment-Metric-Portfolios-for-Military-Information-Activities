# Reproducibility Code for the AMSP Numerical Application

This repository contains the Python code used to reproduce the numerical application of the **Assessment Metric Selection Problem (AMSP)** for Ukraine's *I Want to Live* campaign.

The script enumerates all possible portfolios of 15 candidate assessment metrics, applies the model's evidentiary requirements, evaluates feasible portfolios under the quadratic objective, performs sensitivity and robustness analyses, and generates the figures used in the study.

## Files

```text
.
├── reproduce_iwanttolive_case.py
├── README.md
└── figures/                         # created automatically when the script runs
```

The main script is:

```text
reproduce_iwanttolive_case.py
```

No optimization solver is required. Because the numerical application contains only 15 candidate metrics, the script uses exact enumeration of all \(2^{15}=32{,}768\) possible portfolios.

## Requirements

The script uses Python 3 and the following external packages:

- `numpy`
- `matplotlib`

The remaining imports (`itertools`, `collections`, and `pathlib`) are part of the Python standard library.

A simple installation is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

On Windows, activate the virtual environment with:

```bash
.venv\Scripts\activate
```

## Running the Analysis

From the repository directory, run:

```bash
python reproduce_iwanttolive_case.py
```

The script prints the principal numerical results to the terminal and creates a `figures/` directory containing the generated PDF figures.

## Expected Base-Case Results

The current version of the script produces:

```text
Feasible full-model portfolios: 816
Base optimum: (2, 4, 7, 8, 11, 13, 15)
Objective value: 5.780
Coverage-only baseline: (1, 4, 8, 12, 15)
Adjacent-level perturbations: 130
Base optimum retained: 113 of 130 (86.9%)
```

The five lowest-cost feasible portfolios are also printed, together with their individual-cost and pairwise-overlap components.

The one-at-a-time weight analysis produces the following base-portfolio stability intervals:

| Weight | Base value | Stability interval |
|---|---:|---:|
| Collection burden \(w_c\) | 0.50 | \(0 \leq w_c \leq 0.675\) |
| Unreliability \(w_u\) | 0.80 | \(0.20 \leq w_u \leq 1.60\) |
| Latency \(w_\ell\) | 0.60 | \(0.25 \leq w_\ell \leq 1.20\) |
| Manipulation vulnerability \(w_v\) | 0.80 | \(0.50 \leq w_v \leq 1.60\) |
| Evidentiary overlap \(w_R\) | 0.50 | \(0.267 \leq w_R \leq 1.00\) |

Small floating-point differences in the printed breakpoint values may occur depending on the Python environment.

## What the Script Implements

The numerical application includes:

- 15 candidate metrics;
- seven assessment nodes:
  - dissemination,
  - exposure,
  - consideration,
  - private contact,
  - coordination,
  - completed surrender,
  - countermeasures;
- node-specific minimum metric-count requirements;
- node-specific source-diversity requirements;
- two required metric types for the completed-surrender node:
  - surrender-outcome measures,
  - campaign-influence measures;
- normalized candidate attributes for:
  - collection burden,
  - reliability,
  - latency,
  - manipulation vulnerability;
- pairwise evidentiary-duplication scores \(R_{ik}\);
- the base-case objective weights;
- a coverage-only benchmark;
- exact one-at-a-time weight stability calculations;
- adjacent-level perturbations of candidate attribute and overlap scores; and
- joint sensitivity analysis for collection burden and manipulation vulnerability.

The objective evaluated for a selected portfolio \(S\) is

\[
F(S)
=
w_c \sum_{i\in S} c_i^{\mathrm{col}}
+
w_u \sum_{i\in S} (1-r_i)
+
w_\ell \sum_{i\in S} \ell_i
+
w_v \sum_{i\in S} v_i
+
w_R \sum_{\substack{i<k\\i,k\in S}} R_{ik}.
\]

When objective values tie, the implementation prefers the portfolio with fewer selected metrics and then uses lexicographic ordering as the final deterministic tie-break.

## Generated Figures

Running the script creates the following files in `figures/`:

```text
fig_coverage_map_iwanttolive.pdf
fig_cost_decomposition_iwanttolive.pdf
fig_key_weight_breakpoints_iwanttolive.pdf
fig_collection_manipulation_stability_iwanttolive.pdf
fig_weight_sweep_wc_iwanttolive.pdf
fig_weight_sweep_wu_iwanttolive.pdf
fig_weight_sweep_wl_iwanttolive.pdf
fig_weight_sweep_wv_iwanttolive.pdf
fig_weight_sweep_wR_iwanttolive.pdf
```

The first four correspond to the principal visualizations used in the current numerical analysis. The five `fig_weight_sweep_*` files provide additional one-at-a-time sensitivity plots.

## Interpretation of the Case Inputs

The *I Want to Live* case is **real-world grounded but analytically parameterized**.

The campaign structure, communication channels, process stages, and candidate observables are based on publicly documented features of the campaign. However, the normalized attribute scores, objective weights, evidentiary requirements, candidate metric construction, and pairwise overlap values are analytical inputs developed for the study.

They should therefore **not** be interpreted as reconstructed estimates or judgments made by Ukrainian planners. Their purpose is to demonstrate how the AMSP framework can be instantiated for a documented operational problem while keeping the study's analytical assumptions explicit.

## Reproducibility Scope

This script reproduces the numerical analysis for the current *I Want to Live* application. Earlier development versions of the paper used separate helper scripts and a superseded numerical example. Those files are not required to reproduce the current results.

## Suggested Citation

If this repository accompanies a published or working-paper version of the study, cite the paper itself as the primary methodological source and reference this repository as the associated reproducibility code.


"""Weight-sensitivity calculations used in Section 6.

Each weight is varied independently from 0 to 2x its base-case value while
all other weights remain fixed. The script reports contiguous intervals over
which each portfolio is optimal and writes them to CSV.
"""
from pathlib import Path
import csv
import numpy as np

from metric_portfolio_core import W, BASE_S, best_with_weights, portfolio_set_string

OUT = Path(__file__).resolve().parent

WEIGHTS = [
    ('wc', r'w_c', 0.50),
    ('wu', r'w_u', 0.80),
    ('wl', r'w_l', 0.60),
    ('wv', r'w_v', 0.80),
    ('wR', r'w_R', 0.50),
]


def sweep_segments(key, base_value, n=4001):
    xs = np.linspace(0.0, 2.0*base_value, n)
    sols = []
    vals = []
    for x in xs:
        w = W.copy()
        w[key] = float(x)
        F, _, S = best_with_weights(w)
        sols.append(S)
        vals.append(F)

    segments = []
    start = 0
    for j in range(1, len(xs)+1):
        if j == len(xs) or sols[j] != sols[start]:
            segments.append((xs[start], xs[j-1], sols[start]))
            start = j
    return xs, np.asarray(vals), segments


def main():
    rows = []
    for key, symbol, base in WEIGHTS:
        _, _, segs = sweep_segments(key, base)
        print(f'\n{symbol} (base {base:.2f})')
        for lo, hi, S in segs:
            marker = '  <-- base portfolio' if S == BASE_S and lo <= base <= hi else ''
            print(f'  {lo:.4f} to {hi:.4f}: {portfolio_set_string(S)}{marker}')
            rows.append([key, symbol, base, lo, hi, portfolio_set_string(S), S == BASE_S])

    with (OUT/'weight_sensitivity_segments.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['weight_key','symbol','base_value','segment_start','segment_end','optimal_portfolio','is_base_portfolio'])
        writer.writerows(rows)

    print('\nWrote weight_sensitivity_segments.csv')


if __name__ == '__main__':
    main()

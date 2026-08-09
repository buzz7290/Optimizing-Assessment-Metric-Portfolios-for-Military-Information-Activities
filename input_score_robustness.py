"""Adjacent-level input-score robustness check for the OIE metric-portfolio model.

Place this file beside metric_portfolio_core.py and run:
    python input_score_robustness.py

It reproduces the local score-perturbation results reported in the manuscript.
"""
from collections import Counter
from copy import deepcopy

from metric_portfolio_core import M, R, W, BASE_RESULTS, BASE_S, objective, portfolio_set_string

SCALE = [0.1, 0.3, 0.5, 0.7, 0.9]
R_SCALE = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


def adjacent(scale, value, direction):
    idx = min(range(len(scale)), key=lambda k: abs(scale[k] - value))
    new_idx = idx + direction
    if 0 <= new_idx < len(scale):
        return scale[new_idx]
    return None


def best_over_base_feasible_set(Mx=M, Rx=R):
    candidates = []
    for _, _, S in BASE_RESULTS:
        candidates.append((objective(S, W, Mx, Rx), len(S), S))
    candidates.sort(key=lambda z: (round(z[0], 12), z[1], z[2]))
    return candidates[0]


def run():
    cases = []

    for i, metric in M.items():
        for attr in ['c', 'r', 'l', 'v']:
            for direction in (-1, 1):
                new_value = adjacent(SCALE, metric[attr], direction)
                if new_value is None:
                    continue
                Mx = deepcopy(M)
                Mx[i][attr] = new_value
                _, _, S = best_over_base_feasible_set(Mx=Mx, Rx=R)
                cases.append(('attribute', f'{i}:{attr}:{direction}', S))

    for pair, value in R.items():
        for direction in (-1, 1):
            new_value = adjacent(R_SCALE, value, direction)
            if new_value is None:
                continue
            Rx = dict(R)
            Rx[pair] = new_value
            _, _, S = best_over_base_feasible_set(Mx=M, Rx=Rx)
            cases.append(('redundancy', f'R{pair}:{direction}', S))

    attr_cases = [c for c in cases if c[0] == 'attribute']
    red_cases = [c for c in cases if c[0] == 'redundancy']

    def retained(group):
        return sum(S == BASE_S for _, _, S in group)

    print('Base portfolio:', portfolio_set_string(BASE_S))
    print(f'All perturbations: {retained(cases)}/{len(cases)} = {retained(cases)/len(cases):.3%}')
    print(f'Attribute perturbations: {retained(attr_cases)}/{len(attr_cases)} = {retained(attr_cases)/len(attr_cases):.3%}')
    print(f'Redundancy perturbations: {retained(red_cases)}/{len(red_cases)} = {retained(red_cases)/len(red_cases):.3%}')

    print('\nAlternative portfolios when the base solution changes:')
    changed = Counter(S for _, _, S in cases if S != BASE_S)
    for S, count in changed.most_common():
        print(f'  {portfolio_set_string(S)}: {count}')


if __name__ == '__main__':
    run()

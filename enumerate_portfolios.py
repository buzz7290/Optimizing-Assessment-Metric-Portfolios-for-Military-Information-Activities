"""Exhaustive enumeration used for the numerical application.

Run:
    python enumerate_portfolios.py

Outputs:
    all_feasible_portfolios.csv
    top_20_portfolios.csv
"""
from pathlib import Path
import csv

from metric_portfolio_core import (
    M, R, W, BASE_RESULTS, BASE_S, BASE_F, features, metric_cost,
    portfolio_set_string,
)

OUT = Path(__file__).resolve().parent


def write_results_csv(path: Path, rows):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'objective_F', 'cardinality', 'portfolio',
            'sum_collection', 'sum_unreliability', 'sum_latency',
            'sum_manipulation_vulnerability', 'pairwise_redundancy_sum',
            'individual_cost_component', 'redundancy_penalty'
        ])
        for rank, (F, card, S) in enumerate(rows, start=1):
            feat = features(S)
            individual = (W['wc']*feat['sc'] + W['wu']*feat['su'] +
                          W['wl']*feat['sl'] + W['wv']*feat['sv'])
            red_penalty = W['wR']*feat['rr']
            writer.writerow([
                rank, f'{F:.12f}', card, portfolio_set_string(S),
                feat['sc'], feat['su'], feat['sl'], feat['sv'], feat['rr'],
                f'{individual:.12f}', f'{red_penalty:.12f}'
            ])


def main():
    print(f'Total candidate portfolios: {2**len(M):,}')
    print(f'Feasible portfolios: {len(BASE_RESULTS):,}')
    print(f'Base optimum: {portfolio_set_string(BASE_S)}')
    print(f'Objective F = {BASE_F:.3f}')

    feat = features(BASE_S)
    individual = (W['wc']*feat['sc'] + W['wu']*feat['su'] +
                  W['wl']*feat['sl'] + W['wv']*feat['sv'])
    print(f'  individual-cost component = {individual:.3f}')
    print(f'  redundancy sum             = {feat["rr"]:.3f}')
    print(f'  redundancy penalty         = {W["wR"]*feat["rr"]:.3f}')

    if len(BASE_RESULTS) > 1:
        F2, _, S2 = BASE_RESULTS[1]
        print(f'Runner-up: {portfolio_set_string(S2)}, F = {F2:.3f}, gap = {F2-BASE_F:.3f}')

    print('\nIndividual generalized metric costs:')
    for i in M:
        print(f'  {i:2d}: C_i = {metric_cost(i):.3f}  {M[i]["short"]}')

    print('\nTop 10 feasible portfolios:')
    for rank, (F, card, S) in enumerate(BASE_RESULTS[:10], start=1):
        print(f'  {rank:2d}. F={F:.3f}  n={card}  {portfolio_set_string(S)}')

    write_results_csv(OUT/'all_feasible_portfolios.csv', BASE_RESULTS)
    write_results_csv(OUT/'top_20_portfolios.csv', BASE_RESULTS[:20])
    print('\nWrote CSV outputs to:', OUT)


if __name__ == '__main__':
    main()

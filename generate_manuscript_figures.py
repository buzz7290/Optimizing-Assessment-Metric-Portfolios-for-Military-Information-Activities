"""Generate the current numerical and sensitivity figures for the manuscript.

Run:
    python generate_manuscript_figures.py

Outputs are written to ./figures/:
    fig_coverage_map.pdf
    fig_cost_decomposition.pdf
    fig_weight_sweep_wc.pdf
    fig_weight_sweep_wu.pdf
    fig_weight_sweep_wl.pdf
    fig_weight_sweep_wv.pdf
    fig_weight_sweep_wR.pdf
    fig_collection_latency_stability.pdf
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from metric_portfolio_core import (
    M, W, NODE_ORDER, NODE_LABELS, BASE_S, BASE_PORTFOLIOS,
    metric_cost, objective, best_with_weights, best_index_for_weights,
    portfolio_set_string,
)

OUT = Path(__file__).resolve().parent
FIG = OUT/'figures'
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Match the current sensitivity-figure styling.
BASE_COLOR = '#1f77b4'  # blue
ALT_COLORS = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
LINESTYLES = ['--', '-.', ':', (0,(5,2)), (0,(3,1,1,1))]
ENVELOPE_COLOR = '0.75'


def fig_coverage_map():
    fig, ax = plt.subplots(figsize=(7.0,5.8))
    ax.set_xlim(-0.5,4.5)
    ax.set_ylim(len(M)-0.5,-0.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(NODE_LABELS)

    ylabels = [
        f'{i}{"*" if i in BASE_S else ""}  [{M[i]["source"]}]  {M[i]["short"]}'
        for i in M
    ]
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(ylabels, fontsize=7.4)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)

    for y, i in enumerate(M):
        for x, node in enumerate(NODE_ORDER):
            ax.add_patch(plt.Rectangle((x-.48,y-.46),.96,.92,
                                       facecolor='white',edgecolor='0.82',lw=.55))
            if node in M[i]['nodes']:
                ax.add_patch(plt.Rectangle((x-.40,y-.36),.80,.72,
                                           facecolor='0.20',edgecolor='0.20',lw=.4))
                txt = ''
                if node == 'op':
                    roles = M[i].get('types',{}).get('op',set())
                    if 'incident' in roles: txt = 'I'
                    if 'route' in roles: txt = 'R'
                if txt:
                    ax.text(x,y,txt,color='white',ha='center',va='center',
                            fontsize=7.2,fontweight='bold')

    ax.set_xlabel('Required causal node')
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG/'fig_coverage_map.pdf', bbox_inches='tight')
    plt.close(fig)


def fig_cost_decomposition():
    ids = list(M)
    x = np.arange(len(ids))
    comps = [
        (np.array([W['wc']*M[i]['c'] for i in ids]), 'Collection', '0.80','///'),
        (np.array([W['wu']*(1-M[i]['r']) for i in ids]), 'Unreliability', '0.62','\\\\\\'),
        (np.array([W['wl']*M[i]['l'] for i in ids]), 'Latency', '0.44','xx'),
        (np.array([W['wv']*M[i]['v'] for i in ids]), 'Manipulation vulnerability', '0.25','..'),
    ]

    fig, ax = plt.subplots(figsize=(7.0,3.8))
    bottom = np.zeros(len(ids))
    for arr, label, shade, hatch in comps:
        ax.bar(x, arr, bottom=bottom, label=label, color=shade,
               edgecolor='black', linewidth=.35, width=.68, hatch=hatch)
        bottom += arr

    for k, i in enumerate(ids):
        ax.text(k, bottom[k]+.025, f'{bottom[k]:.2f}', ha='center', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}{"*" if i in BASE_S else ""}' for i in ids])
    ax.set_xlabel('Candidate metric  (* selected in base-case optimum)')
    ax.set_ylabel(r'Individual generalized cost $C_i$')
    ax.set_ylim(0, max(bottom)*1.22)
    ax.legend(ncol=2, frameon=False, fontsize=7.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIG/'fig_cost_decomposition.pdf', bbox_inches='tight')
    plt.close(fig)


def optimal_portfolios_in_sweep(key, xmax, n=1201):
    xs = np.linspace(0.0, xmax, n)
    best_vals = []
    optimal_seq = []
    for x in xs:
        w = W.copy()
        w[key] = float(x)
        F, _, S = best_with_weights(w)
        best_vals.append(F)
        optimal_seq.append(S)

    sols = []
    for S in optimal_seq:
        if S not in sols:
            sols.append(S)
    if BASE_S in sols:
        sols = [BASE_S] + [S for S in sols if S != BASE_S]
    return xs, np.asarray(best_vals), sols


def set_legend_label(S, letter):
    base = ' (base case)' if S == BASE_S else ''
    return f'{letter}: ' + r'$\{' + ','.join(map(str,S)) + r'\}$' + base


def fig_individual_weight_sweep(key, title, base_value, filename):
    xs, envelope, sols = optimal_portfolios_in_sweep(key, 2.0*base_value)
    fig, ax = plt.subplots(figsize=(7.1,4.1))

    for idx, S in enumerate(sols):
        ys = []
        for x in xs:
            w = W.copy()
            w[key] = float(x)
            ys.append(objective(S, w))

        letter = chr(65+idx)
        if S == BASE_S:
            ax.plot(xs, ys, color=BASE_COLOR, linewidth=2.0,
                    label=set_legend_label(S, letter))
        else:
            ax.plot(xs, ys,
                    color=ALT_COLORS[(idx-1) % len(ALT_COLORS)],
                    linestyle=LINESTYLES[(idx-1) % len(LINESTYLES)],
                    linewidth=2.0,
                    label=set_legend_label(S, letter))

    # Draw the exact optimum last so its lower envelope is visible.
    ax.plot(xs, envelope, color=ENVELOPE_COLOR, linewidth=4.0,
            label='global optimum envelope', zorder=0)
    ax.axvline(base_value, color='0.35', linestyle=':', linewidth=1.5)

    tag = {'wc':r'$w_c$','wu':r'$w_u$','wl':r'$w_\ell$','wv':r'$w_v$','wR':r'$w_R$'}[key]
    ymin, ymax = ax.get_ylim()
    ax.text(base_value, ymin + 0.08*(ymax-ymin),
            f'base {tag} = {base_value:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_title(title, fontsize=15)
    ax.set_xlabel(title, fontsize=13)
    ax.set_ylabel(r'Total portfolio cost $F$', fontsize=13)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=False, fontsize=9.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIG/filename, bbox_inches='tight')
    plt.close(fig)


def fig_collection_latency_stability():
    # Same ranges used in the manuscript: +/-50% around the base values.
    wc_vals = np.linspace(.25,.75,301)
    wl_vals = np.linspace(.30,.90,361)

    solutions = [BASE_S]
    Z = np.zeros((len(wl_vals),len(wc_vals)), dtype=int)

    for yi, wl in enumerate(wl_vals):
        for xi, wc in enumerate(wc_vals):
            w = W.copy()
            w['wc'] = float(wc)
            w['wl'] = float(wl)
            idx, _ = best_index_for_weights(w)
            S = BASE_PORTFOLIOS[idx]
            if S not in solutions:
                solutions.append(S)
            Z[yi,xi] = solutions.index(S)

    # Explicit current palette for the six regions visible in the current map.
    palette = ['#c6d9e8','#8fb9d6','#efb287','#df8da4','#b7c98d','#9e94c7']
    if len(solutions) > len(palette):
        # Fallback if future instance changes create additional regions.
        palette += [str(v) for v in np.linspace(.75,.35,len(solutions)-len(palette))]
    cmap = ListedColormap(palette[:len(solutions)])

    fig, ax = plt.subplots(figsize=(8.8,4.9))
    ax.pcolormesh(wc_vals, wl_vals, Z, cmap=cmap, shading='nearest',
                  vmin=-.5, vmax=len(solutions)-.5, alpha=.72)

    # White decision boundaries for readability.
    levels = np.arange(.5, len(solutions)-.5, 1.0)
    if len(levels):
        ax.contour(wc_vals, wl_vals, Z, levels=levels, colors='white', linewidths=1.0)

    letters = {S: chr(65+i) for i,S in enumerate(solutions)}

    # Choose robust in-region label locations by finding a point far from other regions.
    # For the current six-region example these hand-tuned locations give a cleaner figure.
    preferred = {
        'A':(.41,.68), 'B':(.60,.45), 'C':(.715,.34),
        'D':(.715,.74), 'E':(.695,.845), 'F':(.742,.885),
    }
    for S, idx in [(S,i) for i,S in enumerate(solutions)]:
        letter = letters[S]
        x, y = preferred.get(letter, (None,None))
        if x is None:
            ys, xs = np.where(Z==idx)
            if not len(xs):
                continue
            x = wc_vals[int(np.median(xs))]
            y = wl_vals[int(np.median(ys))]
        ax.text(x, y, letter, ha='center', va='center', fontsize=17,
                fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='0.55',
                          boxstyle='round,pad=.18', linewidth=.7))

    # Base case and analytically interpretable A/B boundary.
    ax.plot(W['wc'], W['wl'], 'ko', ms=5)
    ax.annotate('base case', xy=(W['wc'],W['wl']), xytext=(.535,.635),
                fontsize=9, arrowprops=dict(arrowstyle='-', lw=.8))
    line = np.linspace(.30,.75,200)
    ax.plot(line, line, linestyle=':', linewidth=1.1, color='black')
    ax.text(.685,.705,r'$w_c=w_\ell$',fontsize=9,rotation=38,
            ha='center',va='bottom')

    ax.set_xlabel(r'Collection-cost weight $w_c$', fontsize=15)
    ax.set_ylabel(r'Latency weight $w_\ell$', fontsize=15)
    ax.set_xlim(.25,.75)
    ax.set_ylim(.30,.90)
    ax.tick_params(labelsize=11)

    handles = []
    for i,S in enumerate(solutions):
        suffix = ' (base case)' if S == BASE_S else ''
        handles.append(Patch(facecolor=palette[i], edgecolor='0.5', alpha=.72,
                             label=f'{letters[S]}: {portfolio_set_string(S)}{suffix}'))
    leg = ax.legend(handles=handles, title='Optimal portfolio', frameon=False,
                    fontsize=10.5, title_fontsize=13, loc='upper left',
                    bbox_to_anchor=(1.02,1.0), borderaxespad=0.)
    leg._legend_box.align = 'left'

    fig.tight_layout(rect=(0,0,.80,1))
    fig.savefig(FIG/'fig_collection_latency_stability.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    fig_coverage_map()
    fig_cost_decomposition()

    fig_individual_weight_sweep('wc', r'Collection-cost weight $w_c$', .50,
                                'fig_weight_sweep_wc.pdf')
    fig_individual_weight_sweep('wu', r'Unreliability weight $w_u$', .80,
                                'fig_weight_sweep_wu.pdf')
    fig_individual_weight_sweep('wl', r'Latency weight $w_\ell$', .60,
                                'fig_weight_sweep_wl.pdf')
    fig_individual_weight_sweep('wv', r'Manipulation-vulnerability weight $w_v$', .80,
                                'fig_weight_sweep_wv.pdf')
    fig_individual_weight_sweep('wR', r'Redundancy weight $w_R$', .50,
                                'fig_weight_sweep_wR.pdf')

    fig_collection_latency_stability()
    print('Generated manuscript figures in:', FIG)


if __name__ == '__main__':
    main()

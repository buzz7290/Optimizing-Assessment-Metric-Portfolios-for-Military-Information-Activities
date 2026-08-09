"""Core numerical model for the OIE metric-portfolio paper.

Contains the numerical instance, feasibility rules, objective function,
exhaustive enumeration, and fast re-optimization under changed weights.

The notation follows the manuscript:
    wc  = collection-cost weight
    wu  = unreliability weight
    wl  = latency weight
    wv  = manipulation-vulnerability weight
    wR  = pairwise-redundancy weight
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple, FrozenSet
import numpy as np

Portfolio = Tuple[int, ...]

# -----------------------------------------------------------------------------
# Numerical instance
# -----------------------------------------------------------------------------
M = {
    1: dict(short='Dissemination completion rate',
            name='Dissemination completion rate from unit transmission logs',
            nodes={'ex'}, source='A', c=.1, r=.9, l=.3, v=.1),
    2: dict(short='Platform-confirmed delivery rate',
            name='Platform-confirmed delivery rate',
            nodes={'ex'}, source='B', c=.3, r=.7, l=.1, v=.3),
    3: dict(short='Correct-comprehension survey',
            name='Proportion of sampled respondents correctly explaining the intended message',
            nodes={'co'}, source='C', c=.5, r=.7, l=.5, v=.5),
    4: dict(short='Comprehension interviews',
            name='Proportion of key-informant interviews demonstrating correct comprehension',
            nodes={'co'}, source='D', c=.7, r=.9, l=.7, v=.3),
    5: dict(short='Coded digital comprehension',
            name='Proportion of coded digital responses demonstrating correct interpretation',
            nodes={'co'}, source='B', c=.3, r=.5, l=.3, v=.7),
    6: dict(short='Patrol-observed interference',
            name='Patrol-observed civilian-interference incidents per assessment period',
            nodes={'be','op'}, source='E', c=.5, r=.9, l=.3, v=.3,
            types={'op':{'incident'}}),
    7: dict(short='Convoy AAR interference',
            name='Convoy after-action reports of civilian-interference incidents per assessment period',
            nodes={'be','op'}, source='E', c=.3, r=.7, l=.5, v=.3,
            types={'op':{'incident'}}),
    8: dict(short='Police-confirmed interference',
            name='Police-confirmed civilian-interference incidents per assessment period',
            nodes={'be','op'}, source='F', c=.5, r=.7, l=.5, v=.5,
            types={'op':{'incident'}}),
    9: dict(short='Geospatial obstructions',
            name='Geospatially confirmed road obstructions per assessment period',
            nodes={'be','op'}, source='G', c=.7, r=.9, l=.5, v=.1,
            types={'op':{'incident'}}),
    10: dict(short='Route-closure hours',
             name='Route-closure hours attributable to civilian interference',
             nodes={'op'}, source='H', c=.3, r=.9, l=.3, v=.1,
             types={'op':{'route'}}),
    11: dict(short='Convoy-delay minutes',
             name='Median convoy-delay minutes attributable to civilian interference',
             nodes={'op'}, source='E', c=.5, r=.7, l=.3, v=.3,
             types={'op':{'route'}}),
    12: dict(short='Unobstructed movement rate',
             name='Proportion of scheduled movements completed without civilian interference',
             nodes={'op'}, source='H', c=.5, r=.9, l=.1, v=.1,
             types={'op':{'route'}}),
    13: dict(short='Confirmed hostile recruitment',
             name='Confirmed hostile recruitment events linked to the activity',
             nodes={'ad'}, source='G', c=.7, r=.7, l=.7, v=.3),
    14: dict(short='Community retaliation reports',
             name='Community reports of retaliatory intimidation or recruitment linked to the activity',
             nodes={'ad'}, source='D', c=.5, r=.5, l=.5, v=.7),
    15: dict(short='Online recruitment activity',
             name='Hostile online recruitment activity index associated with the campaign',
             nodes={'ad'}, source='B', c=.3, r=.5, l=.3, v=.9),
}

SOURCES = {
    'A':'own-force technical reporting',
    'B':'platform analytics',
    'C':'audience survey',
    'D':'qualitative audience research',
    'E':'friendly-force observation/reporting',
    'F':'partner reporting',
    'G':'intelligence/technical reporting',
    'H':'movement-control data',
}

# Sparse pairwise redundancy matrix. Unlisted pairs have R_ik = 0.
R = {
    (1,2):.3,
    (3,4):.5,(3,5):.5,(4,5):.3,
    (6,7):.7,(6,8):.3,(6,9):.3,(7,8):.3,(7,9):.3,(8,9):.3,
    (10,11):.3,(10,12):.7,(11,12):.3,
    (6,10):.3,(6,11):.5,(6,12):.3,
    (7,10):.1,(7,11):.7,(7,12):.1,
    (8,10):.3,(8,11):.3,(8,12):.3,
    (9,10):.3,(9,11):.3,(9,12):.3,
    (13,14):.3,(13,15):.3,(14,15):.3,
    (2,5):.1,(2,15):.1,(5,15):.1,(4,14):.1,(9,13):.1,
}

# Minimum metric counts by required causal node.
REQ = {'ex':1,'co':1,'be':2,'op':2,'ad':1}
# Minimum number of distinct source classes by required causal node.
DREQ = {'ex':1,'co':1,'be':2,'op':2,'ad':1}
# Prescribed metric types for the operational-effect node.
TYPES = {'op':{'route','incident'}}
MANDATORY = set()

# Base-case objective weights.
W = dict(wc=.50, wu=.80, wl=.60, wv=.80, wR=.50)

NODE_ORDER = ['ex','co','be','op','ad']
NODE_LABELS = ['Execution','Comprehension','Behavior','Operational\neffect','Adverse\neffect']


def metric_cost(i: int, w: Dict[str, float] = W, Mx=M) -> float:
    """Individual generalized cost C_i."""
    m = Mx[i]
    return (w['wc']*m['c'] + w['wu']*(1-m['r']) +
            w['wl']*m['l'] + w['wv']*m['v'])


def feasible(
    S: Iterable[int],
    Mx=M,
    removed_sources: FrozenSet[str] = frozenset(),
) -> bool:
    """Return True iff portfolio S satisfies all hard constraints."""
    S = set(S)

    if not MANDATORY.issubset(S):
        return False
    if any(Mx[i]['source'] in removed_sources for i in S):
        return False

    # Coverage count + source diversity.
    for node, k in REQ.items():
        covered = [i for i in S if node in Mx[i]['nodes']]
        if len(covered) < k:
            return False
        if len({Mx[i]['source'] for i in covered}) < DREQ[node]:
            return False

    # Required metric types.
    for node, roles in TYPES.items():
        for role in roles:
            if not any(role in Mx[i].get('types',{}).get(node,set()) for i in S):
                return False

    return True


def features(S: Iterable[int], Mx=M, Rx=R) -> Dict[str, float]:
    """Return additive portfolio features used by the objective."""
    S = tuple(sorted(S))
    return dict(
        sc=sum(Mx[i]['c'] for i in S),
        su=sum(1-Mx[i]['r'] for i in S),
        sl=sum(Mx[i]['l'] for i in S),
        sv=sum(Mx[i]['v'] for i in S),
        rr=sum(Rx.get((min(i,j),max(i,j)),0.0)
               for a,i in enumerate(S) for j in S[a+1:]),
    )


def objective(S: Iterable[int], w: Dict[str, float] = W, Mx=M, Rx=R) -> float:
    """Quadratic portfolio objective written using precomputed pairwise sum."""
    f = features(S, Mx, Rx)
    return (w['wc']*f['sc'] + w['wu']*f['su'] + w['wl']*f['sl'] +
            w['wv']*f['sv'] + w['wR']*f['rr'])


def enumerate_portfolios(
    Mx=M,
    Rx=R,
    w: Dict[str, float] = W,
    removed_sources: FrozenSet[str] = frozenset(),
    excluded: FrozenSet[int] = frozenset(),
):
    """Exhaustively enumerate and rank all feasible portfolios."""
    ids = [i for i in sorted(Mx)
           if i not in excluded and Mx[i]['source'] not in removed_sources]
    out = []
    for mask in range(1 << len(ids)):
        S = tuple(ids[b] for b in range(len(ids)) if (mask >> b) & 1)
        if feasible(S, Mx, removed_sources):
            out.append((objective(S, w, Mx, Rx), len(S), S))

    # Primary objective F. Cardinality and tuple order only break exact ties.
    out.sort(key=lambda z: (round(z[0],12), z[1], z[2]))
    return out


# Enumerate once at import. The feasible set is unchanged by weight sensitivity.
BASE_RESULTS = enumerate_portfolios()
BASE_F, _, BASE_S = BASE_RESULTS[0]
BASE_PORTFOLIOS = [S for _,_,S in BASE_RESULTS]
BASE_CARD = np.array([len(S) for S in BASE_PORTFOLIOS], dtype=int)
BASE_FEATURES = np.array([
    [features(S)['sc'], features(S)['su'], features(S)['sl'],
     features(S)['sv'], features(S)['rr']]
    for S in BASE_PORTFOLIOS
], dtype=float)


def best_index_for_weights(w: Dict[str, float]):
    """Fast exact re-optimization over the already-enumerated feasible set."""
    coeff = np.array([w['wc'], w['wu'], w['wl'], w['wv'], w['wR']], dtype=float)
    vals = BASE_FEATURES @ coeff
    minv = vals.min()
    idxs = np.flatnonzero(np.isclose(vals, minv, rtol=0, atol=1e-12))

    if len(idxs) > 1:
        mincard = BASE_CARD[idxs].min()
        idxs = idxs[BASE_CARD[idxs] == mincard]

    return int(idxs[0]), float(minv)


def best_with_weights(w: Dict[str, float]):
    idx, value = best_index_for_weights(w)
    S = BASE_PORTFOLIOS[idx]
    return value, len(S), S


def portfolio_set_string(S: Iterable[int]) -> str:
    return '{' + ','.join(str(i) for i in S) + '}'

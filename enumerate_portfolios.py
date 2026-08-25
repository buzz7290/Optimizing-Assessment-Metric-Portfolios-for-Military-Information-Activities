import itertools
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Illustrative parameterization for the real-world-grounded I Want to Live case.
# Campaign structure and candidate observables are grounded in public sources;
# normalized scores, weights, requirements, and overlap judgments are illustrative.
# ---------------------------------------------------------------------
M = list(range(1,16))
nodes = [
    'Dissemination', 'Exposure', 'Consideration', 'Private contact',
    'Coordination', 'Completed surrender', 'Countermeasures'
]
node_keys = ['dissem','exposure','consideration','contact','coord','completion','counter']

candidate_names = {
1:'Campaign messages/ads released',
2:'Telegram post reach/engagement',
3:'Paid digital-ad impressions',
4:'Unique website visits from Russia',
5:'Surrender-guidance/FAQ engagement',
6:'Hotline/chatbot inquiries',
7:'Unique contacts after screening',
8:'Formal surrender applications',
9:'Verified active-duty applicants',
10:'Coordinated surrender operations opened',
11:'Completed project-attributed surrenders',
12:'Application-to-surrender conversion rate',
13:'Post-surrender influence interviews',
14:'Independent records linking surrender to project',
15:'Channel disruption/hostile-probe incidents',
}

source = {1:'A',2:'B',3:'B',4:'C',5:'C',6:'D',7:'D',8:'E',9:'E',10:'F',11:'F',12:'F',13:'G',14:'H',15:'I'}

# c_col, reliability, latency, manipulation vulnerability
attrs = {
1:(.1,.9,.1,.1),
2:(.3,.9,.1,.3),
3:(.1,.9,.1,.5),
4:(.1,.9,.1,.5),
5:(.3,.7,.1,.5),
6:(.3,.7,.1,.5),
7:(.5,.9,.3,.3),
8:(.3,.9,.3,.3),
9:(.5,.9,.5,.1),
10:(.7,.9,.5,.1),
11:(.5,.9,.3,.1),
12:(.1,.9,.5,.1),
13:(.7,.7,.7,.5),
14:(.7,.9,.7,.1),
15:(.3,.7,.3,.3),
}

coverage = {
1:{'dissem'},
2:{'dissem','exposure'},
3:{'dissem','exposure'},
4:{'exposure','consideration'},
5:{'consideration'},
6:{'contact'},
7:{'contact','coord'},
8:{'contact','coord'},
9:{'coord'},
10:{'coord'},
11:{'completion'},
12:{'coord','completion'},
13:{'consideration','completion'},
14:{'completion'},
15:{'counter'},
}

k = {'dissem':1,'exposure':2,'consideration':2,'contact':2,'coord':2,'completion':2,'counter':1}
d = {'dissem':1,'exposure':2,'consideration':2,'contact':2,'coord':2,'completion':2,'counter':1}

# metric types only for completed-surrender node
metric_types = defaultdict(set)
for i in [11,12]:
    metric_types[i].add(('completion','surrender_outcome'))
for i in [13,14]:
    metric_types[i].add(('completion','campaign_influence'))
required_types = {'completion': {'surrender_outcome','campaign_influence'}}

R = {
(2,3):.5, (2,4):.1, (3,4):.1,
(4,5):.7, (4,6):.1, (5,6):.1,
(6,7):.7, (6,8):.3, (7,8):.5,
(8,9):.7, (9,10):.5, (10,11):.5,
(11,13):.3, (11,14):.3, (13,14):.1,
(8,12):.5, (12,13):.1, (11,12):.9, (12,14):.1,
}

base_w = {'wc':.50,'wu':.80,'wl':.60,'wv':.80,'wR':.50}


def feasible(S):
    S=set(S)
    for n in node_keys:
        cov=[i for i in S if n in coverage[i]]
        if len(cov) < k[n]:
            return False
        if len({source[i] for i in cov}) < d[n]:
            return False
    for n, reqs in required_types.items():
        for h in reqs:
            if not any((n,h) in metric_types[i] for i in S):
                return False
    return True


def coeffs(S):
    S=set(S)
    return {
        'wc':sum(attrs[i][0] for i in S),
        'wu':sum(1-attrs[i][1] for i in S),
        'wl':sum(attrs[i][2] for i in S),
        'wv':sum(attrs[i][3] for i in S),
        'wR':sum(v for (i,j),v in R.items() if i in S and j in S),
    }


def objective(S, w=base_w):
    c=coeffs(S)
    return sum(c[p]*w[p] for p in w)

feasible_sets=[]
for bits in itertools.product([0,1], repeat=len(M)):
    S=tuple(i for i,b in zip(M,bits) if b)
    if feasible(S):
        feasible_sets.append(S)

co = {S:coeffs(S) for S in feasible_sets}

def best_for_weights(w):
    return min((sum(c[p]*w[p] for p in w), len(S), S) for S,c in co.items())

best = best_for_weights(base_w)
base_obj, _, baseS = best

# Controlled comparison 1: relax the evidentiary requirements while retaining
# the same full objective, including the pairwise-overlap penalty.
def relaxed_feasible(S):
    S=set(S)
    return all(any(n in coverage[i] for i in S) for n in node_keys)

relaxed_sets=[]
for bits in itertools.product([0,1], repeat=len(M)):
    S=tuple(i for i,b in zip(M,bits) if b)
    if relaxed_feasible(S):
        relaxed_sets.append(S)
relaxed_baseline=min((objective(S),len(S),S) for S in relaxed_sets)

# Controlled comparison 2: retain the full evidentiary requirements but
# remove the pairwise-overlap penalty by setting wR=0.
no_overlap_w=dict(base_w)
no_overlap_w['wR']=0.0
no_overlap_best=best_for_weights(no_overlap_w)

# Print numerical summary for manuscript cross-checking.
print('Feasible full-model portfolios:', len(feasible_sets))
print('Base optimum:', baseS, 'F=', round(base_obj,3), 'coeffs=', co[baseS])
print('Relaxed-requirements optimum:', relaxed_baseline)
print('No-overlap-penalty optimum:', no_overlap_best)
print('No-overlap portfolio under base objective:',
      round(objective(no_overlap_best[2], base_w),3))
print('Top five full-model portfolios:')
for val,_,S in sorted((objective(S),len(S),S) for S in feasible_sets)[:5]:
    c=co[S]
    indiv=sum(c[p]*base_w[p] for p in ['wc','wu','wl','wv'])
    pen=c['wR']*base_w['wR']
    print(S, round(indiv,3), round(pen,3), round(val,3), 'Rsum', c['wR'])

# Exact base-stability interval in each one-dimensional tested sweep [0, 2*base].
def stability_interval(param):
    lo, hi = 0.0, 2*base_w[param]
    lowS=highS=None
    cb=co[baseS]
    for S,c in co.items():
        if S==baseS: continue
        da=cb[param]-c[param]
        db=sum((cb[p]-c[p])*base_w[p] for p in base_w if p!=param)
        if abs(da)<1e-12:
            continue
        bp=-db/da
        if da>0 and bp<hi:
            hi=bp; highS=S
        elif da<0 and bp>lo:
            lo=bp; lowS=S
    return max(0.0,lo), min(2*base_w[param],hi), lowS, highS

print('Stability intervals:')
for p in base_w:
    print(p, stability_interval(p))

# Adjacent-level input perturbation robustness.
attr_scale=[.1,.3,.5,.7,.9]
overlap_scale=[0,.1,.3,.5,.7,.9]

def solve_modified(attrsX, RX):
    def cmod(S):
        S=set(S)
        return {
            'wc':sum(attrsX[i][0] for i in S),
            'wu':sum(1-attrsX[i][1] for i in S),
            'wl':sum(attrsX[i][2] for i in S),
            'wv':sum(attrsX[i][3] for i in S),
            'wR':sum(v for (i,j),v in RX.items() if i in S and j in S),
        }
    return min((sum(cmod(S)[p]*base_w[p] for p in base_w),len(S),S) for S in feasible_sets)[2]

attr_cases=[]
for i in M:
    for q in range(4):
        v=attrs[i][q]
        idx=attr_scale.index(v)
        nbr=[]
        if idx>0: nbr.append(attr_scale[idx-1])
        if idx<len(attr_scale)-1: nbr.append(attr_scale[idx+1])
        for nv in nbr:
            ax=dict(attrs)
            t=list(ax[i]); t[q]=nv; ax[i]=tuple(t)
            attr_cases.append(solve_modified(ax,R))

R_cases=[]
for pair,v in R.items():
    idx=overlap_scale.index(v)
    nbr=[]
    if idx>0: nbr.append(overlap_scale[idx-1])
    if idx<len(overlap_scale)-1: nbr.append(overlap_scale[idx+1])
    for nv in nbr:
        rx=dict(R)
        if nv==0: rx.pop(pair,None)
        else: rx[pair]=nv
        R_cases.append(solve_modified(attrs,rx))

print('Input perturbations:',len(attr_cases),len(R_cases), 'total',len(attr_cases)+len(R_cases))
print('Base retained:',sum(s==baseS for s in attr_cases),sum(s==baseS for s in R_cases),
      'total',sum(s==baseS for s in attr_cases+R_cases))
print('Alternative counts:',Counter(s for s in attr_cases+R_cases if s!=baseS))

# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
selected=set(baseS)

# Coverage map (larger labels, wrapped metric names, and shape-coded metric types)
wrap_names = {
1:'Campaign messages or\nadvertisements released',
2:'Telegram post reach\nand engagement',
3:'Paid digital-ad\nimpressions',
4:'Unique website visits\nfrom Russia',
5:'Surrender-guidance or\nFAQ engagement',
6:'Hotline or chatbot\ninquiries',
7:'Unique contacts after\nscreening',
8:'Formal surrender\napplications',
9:'Verified active-duty\napplicants',
10:'Coordinated surrender\noperations opened',
11:'Completed project-attributed\nsurrenders',
12:'Application-to-surrender\nconversion rate',
13:'Post-surrender interviews on\ncampaign influence',
14:'Independent records linking\nsurrender to project',
15:'Channel disruption or\nhostile-probe incidents',
}
from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(11.7,10.9))
ax.set_xlim(0,len(nodes)); ax.set_ylim(0,len(M))
for r,i in enumerate(M):
    y=len(M)-1-r
    for c,n in enumerate(node_keys):
        ax.add_patch(Rectangle((c,y),1,1,facecolor='white',edgecolor='0.82',linewidth=.8))
        if n in coverage[i]:
            ax.add_patch(Rectangle((c+.07,y+.09),.86,.82,facecolor='0.22',edgecolor='0.22'))
            if n=='completion':
                if ('completion','surrender_outcome') in metric_types[i]:
                    ax.plot(c+.5,y+.5,marker='o',markersize=7.2,markerfacecolor='white',markeredgecolor='white',linestyle='None')
                if ('completion','campaign_influence') in metric_types[i]:
                    ax.plot(c+.5,y+.5,marker='^',markersize=7.8,markerfacecolor='white',markeredgecolor='white',linestyle='None')
ax.set_xticks(np.arange(len(nodes))+.5)
ax.set_xticklabels(['Dissemination','Exposure','Consideration','Private\ncontact','Coordination','Completed\nsurrender','Counter-\nmeasure'],fontsize=10.8)
ax.set_yticks(np.arange(len(M))+.5)
ylabels=[]
for i in reversed(M):
    star='*' if i in selected else ''
    ylabels.append(f'{i}{star} [{source[i]}]  {wrap_names[i]}')
ax.set_yticklabels(ylabels,fontsize=11.5,linespacing=1.05)
ax.tick_params(length=0,pad=6)
for spine in ax.spines.values(): spine.set_visible(False)
legend_handles=[
    Line2D([0],[0],marker='o',linestyle='None',markersize=7.2,markerfacecolor='0.2',markeredgecolor='0.2',label='Surrender-outcome metric type'),
    Line2D([0],[0],marker='^',linestyle='None',markersize=7.6,markerfacecolor='0.2',markeredgecolor='0.2',label='Campaign-influence metric type'),
]
ax.legend(handles=legend_handles,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,frameon=False,fontsize=11.2,handletextpad=.5,columnspacing=1.8)
fig.text(.5,.016,'Filled cell = metric can contribute evidence   |   * = selected in base-case optimum   |   [A]–[I] = source class',ha='center',fontsize=10.6)
fig.subplots_adjust(left=.365,right=.995,top=.985,bottom=.18)
fig.savefig(FIG/'fig_coverage_map_iwanttolive.pdf',bbox_inches='tight')
plt.close(fig)

# Cost decomposition
wc,wu,wl,wv=[base_w[x] for x in ['wc','wu','wl','wv']]
parts=np.array([[wc*attrs[i][0], wu*(1-attrs[i][1]), wl*attrs[i][2], wv*attrs[i][3]] for i in M])
fig,ax=plt.subplots(figsize=(10.5,5.8))
x=np.arange(1,16)
bottom=np.zeros(15)
hatches=['////','\\\\\\\\','xx','..']
labels=['Collection','Unreliability','Latency','Manipulation vulnerability']
for j in range(4):
    ax.bar(x,parts[:,j],bottom=bottom,edgecolor='black',linewidth=.6,hatch=hatches[j],label=labels[j],facecolor='white')
    bottom+=parts[:,j]
for idx,total in enumerate(bottom):
    ax.text(idx+1,total+.025,f'{total:.2f}',ha='center',va='bottom',fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([f'{i}*' if i in selected else str(i) for i in M],fontsize=8)
ax.set_xlabel('Candidate metric (* selected in base-case optimum)')
ax.set_ylabel(r'Individual generalized cost $C_i$')
ax.legend(ncol=2,fontsize=8,frameon=False)
ax.set_ylim(0,max(bottom)+.25)
fig.tight_layout()
fig.savefig(FIG/'fig_cost_decomposition_iwanttolive.pdf',bbox_inches='tight')
plt.close(fig)

# Decision-relevant breakpoint figure used as Figure 4 in the manuscript.
# The annotations are positioned away from axes, tick labels, and plotted lines.
fig, axs = plt.subplots(2, 1, figsize=(8.4, 6.9))

xx = np.linspace(0, 1.0, 300)
yy = .27 - .4 * xx
ax = axs[0]
ax.plot(xx, yy, linewidth=2.2)
ax.axhline(0, linewidth=1)
ax.axvline(.50, linestyle=':', linewidth=1.4)
ax.axvline(.675, linestyle='--', linewidth=1.2)
ax.scatter([.50], [.07], s=38, zorder=4)
ax.set_xlim(0, 1.0)
ax.set_ylim(-.15, .31)
ax.set_ylabel(r'Alternative $-$ base objective', fontsize=12)
ax.set_xlabel(r'Collection-burden weight $w_c$', fontsize=12)
ax.set_title('Completed surrender (11) vs. conversion rate (12)', fontsize=13, pad=8)
ax.text(.03, .282, 'Base portfolio preferred', fontsize=10.5, va='top')
ax.text(.515, .092, 'base 0.50', fontsize=10.5)
ax.text(.69, .020, 'breakpoint 0.675', fontsize=10.5)
ax.text(.76, -.095, 'Metric 12 preferred', fontsize=10.5, ha='center')
ax.tick_params(labelsize=10.5)

xx = np.linspace(0, 1.6, 320)
yy = .2 * (xx - .5)
ax = axs[1]
ax.plot(xx, yy, linewidth=2.2)
ax.axhline(0, linewidth=1)
ax.axvline(.80, linestyle=':', linewidth=1.4)
ax.axvline(.50, linestyle='--', linewidth=1.2)
ax.scatter([.80], [.06], s=38, zorder=4)
ax.set_xlim(0, 1.6)
ax.set_ylim(-.12, .235)
ax.set_ylabel(r'Alternative $-$ base objective', fontsize=12)
ax.set_xlabel(r'Manipulation-vulnerability weight $w_v$', fontsize=12)
ax.set_title('Telegram reach (2) vs. paid advertisement impressions (3)', fontsize=13, pad=8)
ax.text(1.05, .195, 'Base portfolio preferred', fontsize=10.5)
ax.text(.82, .073, 'base 0.80', fontsize=10.5)
ax.text(.515, .016, 'breakpoint 0.50', fontsize=10.5)
ax.text(.20, -.087, 'Metric 3 preferred', fontsize=10.5, ha='center')
ax.tick_params(labelsize=10.5)

fig.tight_layout(h_pad=1.8)
fig.savefig(FIG/'fig_key_weight_breakpoints_iwanttolive.pdf', bbox_inches='tight')
plt.close(fig)

# Weight sweep plots
pretty={'wc':r'Collection-burden weight $w_c$','wu':r'Unreliability weight $w_u$','wl':r'Latency weight $w_\ell$','wv':r'Manipulation-vulnerability weight $w_v$','wR':r'Evidentiary-overlap weight $w_R$'}
short={
(2,4,7,8,11,13,15):'{2,4,7,8,11,13,15}',
(3,4,7,8,11,13,15):'{3,4,7,8,11,13,15}',
(2,4,7,8,12,13,15):'{2,4,7,8,12,13,15}',
(2,4,6,8,12,13,15):'{2,4,6,8,12,13,15}',
(3,4,6,8,12,13,15):'{3,4,6,8,12,13,15}',
}
for p,b in base_w.items():
    xx=np.linspace(0,2*b,401)
    opts=[]
    for v in xx:
        w=dict(base_w);w[p]=float(v)
        opts.append(best_for_weights(w)[2])
    unique=[]
    for s in opts:
        if s not in unique: unique.append(s)
    fig,ax=plt.subplots(figsize=(6.2,4.3))
    # plot only portfolios optimal somewhere on sweep
    for s in unique:
        yy=[]
        c=co[s]
        for v in xx:
            w=dict(base_w);w[p]=float(v)
            yy.append(sum(c[q]*w[q] for q in w))
        ax.plot(xx,yy,label=short.get(s,str(s)))
    envelope=[]
    for v in xx:
        w=dict(base_w);w[p]=float(v)
        envelope.append(best_for_weights(w)[0])
    ax.plot(xx,envelope,linewidth=3,label='Optimum envelope')
    ax.axvline(base_w[p],linestyle=':',linewidth=1)
    lo,hi,lowS,highS=stability_interval(p)
    if lo>0 and lo<2*b: ax.axvline(lo,linestyle='--',linewidth=1)
    if hi<2*b and hi>0: ax.axvline(hi,linestyle='--',linewidth=1)
    ax.set_xlabel(pretty[p]);ax.set_ylabel('Total portfolio cost')
    ax.legend(fontsize=6.5,frameon=False)
    fig.tight_layout()
    fig.savefig(FIG/f'fig_weight_sweep_{p}_iwanttolive.pdf',bbox_inches='tight')
    plt.close(fig)

# Joint stability over wc and wv.
# Only the region containing portfolio changes is plotted (wv >= 0.35), avoiding
# unused white space below the decision regions.
wc_grid = np.linspace(.25, .85, 241)
wv_grid = np.linspace(.35, 1.20, 241)

# Sort by the same secondary criteria used in best_for_weights so np.argmin
# reproduces the objective -> portfolio size -> lexicographic tie-breaking rule.
ordered_sets = sorted(feasible_sets, key=lambda S: (len(S), S))
coef = {p: np.array([co[S][p] for S in ordered_sets], dtype=float) for p in base_w}
fixed = (base_w['wu'] * coef['wu'] +
         base_w['wl'] * coef['wl'] +
         base_w['wR'] * coef['wR'])

Z_idx = np.empty((len(wv_grid), len(wc_grid)), dtype=int)
for iy, vv in enumerate(wv_grid):
    # rows = wc values, columns = feasible portfolios
    obj = (fixed + vv * coef['wv'])[None, :] + wc_grid[:, None] * coef['wc'][None, :]
    Z_idx[iy, :] = np.argmin(obj, axis=1)

# Convert portfolio indices to compact region labels in first-appearance order.
regions = []
region_lookup = {}
Z = np.empty_like(Z_idx)
for iy in range(Z_idx.shape[0]):
    for ix in range(Z_idx.shape[1]):
        S = ordered_sets[Z_idx[iy, ix]]
        if S not in region_lookup:
            region_lookup[S] = len(regions)
            regions.append(S)
        Z[iy, ix] = region_lookup[S]

fig, ax = plt.subplots(figsize=(8.0, 6.2))
ax.imshow(
    Z,
    origin='lower',
    aspect='auto',
    extent=[wc_grid[0], wc_grid[-1], wv_grid[0], wv_grid[-1]],
    interpolation='nearest'
)
ax.plot([wc_grid[0], wc_grid[-1]], [wc_grid[0], wc_grid[-1]], linestyle=':', linewidth=1.4)
ax.scatter([base_w['wc']], [base_w['wv']], s=35, marker='o')
ax.set_xlabel(r'Collection-burden weight $w_c$')
ax.set_ylabel(r'Manipulation-vulnerability weight $w_v$')
ax.set_ylim(wv_grid[0], wv_grid[-1])

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for idx, S in enumerate(regions):
    ys, xs = np.where(Z == idx)
    if len(xs) == 0:
        continue
    cx = wc_grid[int(np.median(xs))]
    cy = wv_grid[int(np.median(ys))]
    ax.text(cx, cy, letters[idx], ha='center', va='center', fontsize=10, fontweight='bold')

legend_text = '\n'.join(f'{letters[i]} = {short.get(S, str(S))}' for i, S in enumerate(regions))
ax.text(1.02, .98, legend_text, transform=ax.transAxes, va='top', fontsize=7)
fig.tight_layout()
fig.savefig(FIG/'fig_collection_manipulation_stability_iwanttolive.pdf', bbox_inches='tight')
plt.close(fig)


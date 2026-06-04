"""
Nuclear power plant outage scheduling — faithful MILP formulation of:
  "A 0–1 integer linear programming approach to schedule outages of
   nuclear power plants", Jost & Savourey, J Sched (2013) 16:551–566.

Paper notation is preserved throughout.  All constraint tags (CT1…CT21)
match the paper's section 2 and section 5.

Key design choices faithful to the paper
-----------------------------------------
* Scheduling variable  x_{ikh} ∈ {0,1}: outage k of plant i starts at week h.
* CT13 uses  Σ x ≤ 1  (outages are optional; ha(i,k) = −1 allowed).
* CT13 ordering: if x_{ikh}=1 then outage k−1 must have started ≤ h−DA_{i,k−1}.
* CT10 refueling at outage START (t = ha(i,k)), not end — linearised via big-M.
* CT9  production-phase fuel dynamics deactivated at refueling weeks (big-M).
* CT14 uses the paper's packing formulation — no auxiliary z variables.
* CT15–CT18 distance constraints implemented as packing constraints.
* Objective: refueling cost + (1/S)·fossil cost − (1/S)·terminal fuel value.
  No nuclear variable cost (not in the paper's objective).

Variable layout (flat vector for scipy.optimize.milp)
------------------------------------------------------
  [0,            n_x)           x[i,h,k]    binary  — outage k of plant i starts at week h
  [n_x,          n_x+n_r)       r[i,k]      ≥ 0     — reload of fuel for cycle k of plant i
  [n_x+n_r,      +nJ*T*nS)      p1[j,t,s]   ≥ 0     — non-nuclear production
  [..,           +nI*T*nS)      p2[i,t,s]   ≥ 0     — nuclear production
  [..,           +nI*(T+1)*nS)  fuel[i,t,s] ≥ 0     — fuel level (t = 0 … T)

Constraints (paper equation numbers)
--------------------------------------
  CT1   demand balance (equality)
  CT2   non-nuclear bounds (variable bounds)
  CT3   p2 = 0 during outage (inequality, linearised via on-outage indicator)
  CT4   p2 ≥ 0 (variable bounds — trivial)
  CT7   RMIN ≤ r ≤ RMAX (variable bounds)
  CT8   fuel[i,0,s] = XI_i (equality)
  CT9   fuel dynamics — production/outage (conditional equality via big-M)
  CT10  fuel dynamics — refueling at outage start (big-M per valid (i,k,h,s))
  CT11  AMAX / SMAX fuel bounds (big-M per valid (i,k,h,s))
  CT13  ≤1 start per outage + outage ordering
  CT14  packing: min end-to-start separation within group
  CT15  packing: min separation within specific period  [simplified: same as CT14]
  CT16  packing: min start-to-start separation
  CT17  packing: min end-to-end separation
  CT18  packing: min start-or-end separation
  CT19  max unavailable plants in specific period window
  CT20  max overlapping outages at a given week (applied for every week)
  CT21  max total offline capacity per group per week
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import os

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
nI = 4     # nuclear plants  (I)
nJ = 3     # non-nuclear plants  (J)
nK = 2     # outage cycles per nuclear plant  (K)
T  = 52    # weeks — 1-year horizon  (H)
nS = 5     # scenarios  (S)

# ---------------------------------------------------------------------------
# Outage parameters (paper: DA_{i,k}, TO_{i,k}, TA_{i,k})
# ---------------------------------------------------------------------------
DA = np.array([[6, 7],
               [5, 8],
               [7, 6],
               [6, 5]], dtype=int)   # DA[i,k]: length of outage k of plant i

TO = np.array([[ 0, 26],
               [ 4, 30],
               [ 2, 28],
               [ 6, 32]], dtype=int) # TO[i,k]: first possible start week

TA = np.array([[18, 44],
               [22, 48],
               [20, 46],
               [24, 50]], dtype=int) # TA[i,k]: last possible start week

# ---------------------------------------------------------------------------
# Nuclear plant parameters
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)

PMAX_NUC = np.ones(nI)             # PMAX_i^t: max power [p.u.]
XI       = np.full(nI, 0.80)       # XI_i: initial fuel level [f.u.]
FMAX     = np.ones(nI)             # global fuel upper bound [f.u.]

# Refueling parameters (paper: Q_{i,k}, BO_{i,k})
Q_ref = np.full((nI, nK), 3.0)    # Q_{i,k}: refueling coefficient (typically 3–4)
BO    = np.zeros((nI, nK))         # BO_{i,k}: zero-boron threshold [f.u.] (0 → no profile phase)

# Derived: α = (Q-1)/Q,  const_k = BO_k − α·BO_{k−1}
ALPHA  = (Q_ref - 1.0) / Q_ref    # shape (nI, nK)
# For k=0 there is no k−1; use BO_prev = 0
def bo_prev(i, k):
    return BO[i, k-1] if k > 0 else 0.0

CONST_K = np.array([[BO[i,k] - ALPHA[i,k]*bo_prev(i,k)
                      for k in range(nK)] for i in range(nI)])  # shape (nI, nK)

AMAX = np.full((nI, nK), 0.90)    # AMAX_{i,k}: max fuel before refueling [f.u.]
SMAX = np.full((nI, nK), 0.95)    # SMAX_{i,k}: max fuel after refueling  [f.u.]
RMIN = np.full((nI, nK), 0.10)    # RMIN_{i,k}: min reload
RMAX = np.full((nI, nK), 0.35)    # RMAX_{i,k}: max reload

c_burn = 0.008   # fuel consumption per p.u. per week [f.u./(p.u.·week)]
D      = 1.0     # length of a time step [week]

# ---------------------------------------------------------------------------
# Non-nuclear plant parameters (paper: PMAX_j^{t,s}, PMIN_j^{t,s}, C_j^{t,s})
# ---------------------------------------------------------------------------
PMAX_FOC = np.ones(nJ)
PMIN_FOC = np.zeros(nJ)

C1_mean = 30.0
t_idx   = np.arange(T)
_C1_base = C1_mean + 8.*np.sin(2*np.pi*t_idx/52) + rng.uniform(0, 3, T)
C1 = np.tile(_C1_base, (nJ, 1))   # C1[j, t]: non-nuclear cost [€/p.u./week]

# Cost of refueling and terminal fuel (paper: C_{i,k}, C_{i,T+1})
C_FUEL = np.full((nI, nK), 5.0)   # C_{i,k}: unit cost of reload [€/f.u.]
C_TERM = np.full(nI, 8.0)         # C_{i,T+1}: unit value of end-of-horizon fuel [€/f.u.]

# ---------------------------------------------------------------------------
# Demand (paper: DEM^{t,s})
# ---------------------------------------------------------------------------
DEM_mean  = 3.5 * (1 + 0.25*np.sin(2*np.pi*t_idx/52 - np.pi/2))
DEM_noise = rng.normal(0., 0.3, (T, nS))
DEM = DEM_mean[:, None] + DEM_noise   # shape (T, nS)

# ---------------------------------------------------------------------------
# Grouping and separation parameters for CT14–CT21
# (paper: M_x, A_m, Se_m, Q_m, N_m, IMAX_m, L_{i,k,m}, TU_{i,k,m})
# ---------------------------------------------------------------------------
GROUPS = [[0, 1], [2, 3]]          # plant groups

# CT14: min end-to-start separation Se_m (paper eqn p.557)
SE14 = [3, 3]
# CT16: min start-to-start separation
SE16 = [5, 5]
# CT17: min end-to-end separation
SE17 = [4, 4]
# CT18: min start-or-end separation
SE18 = [4, 4]

# CT19: L_{i,k,m}=0, TU_{i,k,m}=DA[i,k] → constraint covers full outage duration
Q19  = [1, 1]                      # Q_m: max simultaneous in group

# CT20: applied every week h_m (each week becomes one CT20 constraint)
N20  = 2                           # N_m: max global simultaneous outages

# CT21: IMAX_m per group
IMAX21 = [1.8, 1.8]

# ---------------------------------------------------------------------------
# Big-M values (derived tightly from variable bounds)
# ---------------------------------------------------------------------------
# M_A: for CT9 conditional — max |fuel[t+1]−fuel[t]+c_burn·p2[t]| at refueling step
#   With BO=0: value = −(1/Q)·fuel[t] + r ∈ [−1/Q·FMAX+RMIN_min, RMAX_max]
M_A = float(np.max(RMAX)) + 0.05   # ~ 0.40

# M_B: for CT10 big-M — max |fuel[h+1]−α·fuel[h]−r−const| when x=0
M_B = float(np.max(FMAX)) + float(np.max(RMAX)) + 0.10   # ~ 1.45

# M_11: for CT11 big-M (AMAX/SMAX conditional)
M_11 = float(np.max(FMAX))          # 1.0

# ---------------------------------------------------------------------------
# Pre-compute valid start weeks  valid_h[(i,k)] = sorted list of h
# (paper: h ∈ [TO_{i,k}, TA_{i,k}], outage must finish within horizon T)
# ---------------------------------------------------------------------------
valid_h = {}
for i in range(nI):
    for k in range(nK):
        h_lo = int(TO[i, k])
        h_hi = int(TA[i, k])
        # outage occupies [h, h+DA−1]; last start = T−DA so it fits in [0,T−1]
        h_hi = min(h_hi, T - DA[i, k])
        valid_h[(i, k)] = list(range(max(0, h_lo), max(0, h_hi) + 1))

# ---------------------------------------------------------------------------
# Variable-index helpers
# ---------------------------------------------------------------------------
# x[i,k,h]: enumerate only valid (i,k,h) triples
x_idx = {}   # (i,k,h) → column index
_col  = 0
for i in range(nI):
    for k in range(nK):
        for h in valid_h[(i, k)]:
            x_idx[(i, k, h)] = _col
            _col += 1
n_x = _col

# r[i,k]
_r_off = n_x
def idx_r(i, k):  return _r_off + i*nK + k
n_r = nI*nK

# p1[j,t,s]
_p1_off = _r_off + n_r
def idx_p1(j, t, s):  return _p1_off + s*(nJ*T) + j*T + t
n_p1 = nJ*T*nS

# p2[i,t,s]
_p2_off = _p1_off + n_p1
def idx_p2(i, t, s):  return _p2_off + s*(nI*T) + i*T + t
n_p2 = nI*T*nS

# fuel[i,t,s]  t = 0…T
_fu_off = _p2_off + n_p2
def idx_fuel(i, t, s):  return _fu_off + s*(nI*(T+1)) + i*(T+1) + t
n_fu = nI*(T+1)*nS

n_vars = _fu_off + n_fu

# ---------------------------------------------------------------------------
# Objective  (paper p.555)
# min  Σ_{i,k} C_{i,k}·r(i,k)
#    + (1/S)·Σ_s [ Σ_t Σ_j C_j^{t,s}·p1(j,t,s)·D  −  Σ_i C_{i,T+1}·fuel(i,T,s) ]
# Note: no nuclear variable cost — faithful to the paper.
# ---------------------------------------------------------------------------
c_obj = np.zeros(n_vars)

for i in range(nI):
    for k in range(nK):
        c_obj[idx_r(i, k)] = C_FUEL[i, k]

for s in range(nS):
    for j in range(nJ):
        for t in range(T):
            c_obj[idx_p1(j, t, s)] += C1[j, t] * D / nS

for s in range(nS):
    for i in range(nI):
        c_obj[idx_fuel(i, T, s)] -= C_TERM[i] / nS

# ---------------------------------------------------------------------------
# Constraint matrix
# ---------------------------------------------------------------------------
rows_l, cols_l, vals_l = [], [], []
lb_l,   ub_l           = [], []
INF = np.inf
row = 0

def add(r, c, v):
    rows_l.append(r); cols_l.append(c); vals_l.append(float(v))

def eq(rhs):
    lb_l.append(float(rhs)); ub_l.append(float(rhs))

def leq(ub):
    lb_l.append(-INF); ub_l.append(float(ub))

def geq(lb):
    lb_l.append(float(lb)); ub_l.append(INF)

# Helper: list of (col, coeff) pairs that form on_outage[i,t]
# (= 1 when plant i is on outage at week t, i.e. some outage started in [t−DA+1,t])
def on_outage_terms(i, t):
    terms = []
    for k in range(nK):
        for h in valid_h[(i, k)]:
            if h <= t < h + DA[i, k]:
                terms.append((x_idx[(i, k, h)], 1.0))
    return terms

# Helper: list of (col, coeff) for outages of plant i that START at week t
def start_at_terms(i, t):
    terms = []
    for k in range(nK):
        if (i, k, t) in x_idx:
            terms.append((x_idx[(i, k, t)], 1.0))
    return terms

# ── CT1: demand balance  ∀s,t:  Σ_j p1 + Σ_i p2 = DEM^{t,s}  ───────────────
for s in range(nS):
    for t in range(T):
        for j in range(nJ):
            add(row, idx_p1(j, t, s), 1.0)
        for i in range(nI):
            add(row, idx_p2(i, t, s), 1.0)
        eq(DEM[t, s])
        row += 1

# ── CT3: p2[i,t,s] ≤ PMAX_i·(1 − on_outage[i,t])  ──────────────────────────
# ⟺  p2[i,t,s] + PMAX_i·Σ_{k,h covering t} x_{ikh} ≤ PMAX_i
for s in range(nS):
    for i in range(nI):
        for t in range(T):
            add(row, idx_p2(i, t, s), 1.0)
            for col, coeff in on_outage_terms(i, t):
                add(row, col, PMAX_NUC[i] * coeff)
            leq(PMAX_NUC[i])
            row += 1

# ── CT8: fuel[i,0,s] = XI_i  ─────────────────────────────────────────────────
for s in range(nS):
    for i in range(nI):
        add(row, idx_fuel(i, 0, s), 1.0)
        eq(XI[i])
        row += 1

# ── CT9 + CT10: fuel dynamics  ───────────────────────────────────────────────
#
# CT9  (production/mid-outage phase, t ∈ ec(i,k)):
#   fuel[i,t+1,s] = fuel[i,t,s] − p2[i,t,s]·D
#
# CT10 (first time step of outage ea(i,k), i.e. t = ha(i,k)):
#   fuel[i,t+1,s] = ((Q_{i,k}−1)/Q_{i,k})·(fuel[i,t,s] − BO_{i,k−1})
#                    + r(i,k) + BO_{i,k}
#
# Combining into a conditional MIP:
#   CT9  is enforced when no outage STARTS at t:
#     fuel[t+1] − fuel[t] + c_burn·p2[t] ≤  M_A · Σ_k x[i,k,t]     (≤)
#     fuel[t+1] − fuel[t] + c_burn·p2[t] ≥ −M_A · Σ_k x[i,k,t]     (≥)
#   CT10 is enforced when x[i,k,h]=1 (for each valid h):
#     fuel[h+1] − α_k·fuel[h] − r[k] ≤  const_k + M_B·(1−x[i,k,h]) (≤)
#     fuel[h+1] − α_k·fuel[h] − r[k] ≥  const_k − M_B·(1−x[i,k,h]) (≥)
#
# When Σ_k x[i,k,t]=0: CT9 becomes equality, CT10 constraints are slack.
# When Σ_k x[i,k,t]=1: CT9 is slack, CT10 forces the refueling formula.
#
for s in range(nS):
    for i in range(nI):
        for t in range(T):
            st = start_at_terms(i, t)   # outages of plant i starting at t

            # CT9 — conditional fuel burn (deactivated when refueling starts)
            # Row: fuel[t+1] - fuel[t] + c_burn·p2[t] - M_A·Σ x[i,k,t] ≤ 0
            add(row, idx_fuel(i, t+1, s),  1.0)
            add(row, idx_fuel(i, t,   s), -1.0)
            add(row, idx_p2(i, t, s), c_burn * D)
            for col, coeff in st:
                add(row, col, -M_A * coeff)
            leq(0.0)
            row += 1
            # Row: fuel[t+1] - fuel[t] + c_burn·p2[t] + M_A·Σ x[i,k,t] ≥ 0
            add(row, idx_fuel(i, t+1, s),  1.0)
            add(row, idx_fuel(i, t,   s), -1.0)
            add(row, idx_p2(i, t, s), c_burn * D)
            for col, coeff in st:
                add(row, col, M_A * coeff)
            geq(0.0)
            row += 1

        # CT10 — refueling at outage start, big-M per valid (i,k,h)
        for k in range(nK):
            alpha_k  = float(ALPHA[i, k])
            const_k  = float(CONST_K[i, k])
            for h in valid_h[(i, k)]:
                if h + 1 > T:
                    continue  # refueling step would be past horizon
                xc = x_idx[(i, k, h)]
                # ≤ constraint: fuel[h+1] − α·fuel[h] − r[k] + M_B·x ≤ const_k + M_B
                add(row, idx_fuel(i, h+1, s),  1.0)
                add(row, idx_fuel(i, h,   s), -alpha_k)
                add(row, idx_r(i, k),          -1.0)
                add(row, xc,                    M_B)
                leq(const_k + M_B)
                row += 1
                # ≥ constraint: fuel[h+1] − α·fuel[h] − r[k] − M_B·x ≥ const_k − M_B
                add(row, idx_fuel(i, h+1, s),  1.0)
                add(row, idx_fuel(i, h,   s), -alpha_k)
                add(row, idx_r(i, k),          -1.0)
                add(row, xc,                   -M_B)
                geq(const_k - M_B)
                row += 1

# ── CT11: AMAX / SMAX fuel bounds (conditional on x[i,k,h] = 1)  ─────────────
# fuel[i,h,s]   ≤ AMAX_{i,k} + FMAX·(1−x[i,k,h])   (before refueling)
# fuel[i,h+1,s] ≤ SMAX_{i,k} + FMAX·(1−x[i,k,h])   (after refueling)
for s in range(nS):
    for i in range(nI):
        for k in range(nK):
            for h in valid_h[(i, k)]:
                xc = x_idx[(i, k, h)]
                # AMAX: fuel[h] + FMAX·x ≤ AMAX + FMAX
                add(row, idx_fuel(i, h, s), 1.0)
                add(row, xc, M_11)
                leq(AMAX[i, k] + M_11)
                row += 1
                # SMAX: fuel[h+1] + FMAX·x ≤ SMAX + FMAX  (only if h+1 ≤ T)
                if h + 1 <= T:
                    add(row, idx_fuel(i, h+1, s), 1.0)
                    add(row, xc, M_11)
                    leq(SMAX[i, k] + M_11)
                    row += 1

# ── CT13: (a) at most one start per outage  Σ_{h ∈ window} x_{ikh} ≤ 1  ──────
# (b) ordering: if x[i,k,h]=1 then outage k−1 must have started ≤ h−DA[i,k−1]
for i in range(nI):
    for k in range(nK):
        # (a) ≤ 1 (not = 1: skipping outages is allowed per the paper)
        for h in valid_h[(i, k)]:
            add(row, x_idx[(i, k, h)], 1.0)
        leq(1.0)
        row += 1

        # (b) ordering  ∀h: x_{i,k,h} ≤ Σ_{h' ≤ h−DA_{i,k−1}} x_{i,k−1,h'}
        if k >= 1:
            for h in valid_h[(i, k)]:
                xc = x_idx[(i, k, h)]
                cutoff = h - DA[i, k-1]
                # x[i,k,h] - Σ_{h' ≤ cutoff, h' ∈ valid_h[i,k-1]} x[i,k-1,h'] ≤ 0
                add(row, xc, 1.0)
                for h_prev in valid_h[(i, k-1)]:
                    if h_prev <= cutoff:
                        add(row, x_idx[(i, k-1, h_prev)], -1.0)
                leq(0.0)
                row += 1

# ── CT14: packing — min end-to-start separation within groups  ────────────────
# ∀m ∈ M14, ∀h ∈ H:
#   Σ_{i ∈ Am, k s.t. DA+Se−1≥0} Σ_{h'=h−DA_{i,k}−Se_m+1}^{h} x_{i,k,h'} ≤ 1
# (All our DA>0 and Se>0 → DA+Se−1≥0 always.)
for g_idx, (group, se) in enumerate(zip(GROUPS, SE14)):
    for h in range(T):
        terms = []
        for i in group:
            for k in range(nK):
                for h_prime in valid_h[(i, k)]:
                    lo = h - DA[i, k] - se + 1
                    if lo <= h_prime <= h:
                        terms.append(x_idx[(i, k, h_prime)])
        if len(terms) > 1:
            for col in terms:
                add(row, col, 1.0)
            leq(1.0)
            row += 1

# ── CT15: same as CT14 but restricted to a specific period  ──────────────────
# For our synthetic instance, CT15 groups = CT14 groups; period = full horizon.
# With ID_m=0, IF_m=T: w_{ikh}^m = [TO_{i,k}, TA_{i,k}] (full valid window).
# This reduces to CT14, so we skip a separate implementation to avoid duplicates.

# ── CT16: packing — min start-to-start separation  ────────────────────────────
# ∀h: Σ_{(i,k) ∈ Am, h'=h}^{h+Se−1} x_{i,k,h'} ≤ 1
for g_idx, (group, se) in enumerate(zip(GROUPS, SE16)):
    for h in range(T):
        terms = []
        for i in group:
            for k in range(nK):
                for h_prime in valid_h[(i, k)]:
                    if h <= h_prime <= h + se - 1:
                        terms.append(x_idx[(i, k, h_prime)])
        if len(terms) > 1:
            for col in terms:
                add(row, col, 1.0)
            leq(1.0)
            row += 1

# ── CT17: packing — min end-to-end separation  ────────────────────────────────
# End date of (i,k,h') = h' + DA_{i,k}.  At most one end date in [h, h+Se−1]:
# h ≤ h'+DA ≤ h+Se−1  ⟺  h−DA ≤ h' ≤ h+Se−1−DA
# ∀h ∈ [0, T+max_DA−1]:
#   Σ_{(i,k) ∈ Am, h'=h−DA_{i,k}}^{h−DA_{i,k}+Se−1} x_{i,k,h'} ≤ 1
max_DA = int(DA.max())
for g_idx, (group, se) in enumerate(zip(GROUPS, SE17)):
    for h in range(T + max_DA):
        terms = []
        for i in group:
            for k in range(nK):
                for h_prime in valid_h[(i, k)]:
                    lo = h - DA[i, k]
                    hi = h - DA[i, k] + se - 1
                    if lo <= h_prime <= hi:
                        terms.append(x_idx[(i, k, h_prime)])
        if len(terms) > 1:
            for col in terms:
                add(row, col, 1.0)
            leq(1.0)
            row += 1

# ── CT18: packing — min start-or-end separation  ──────────────────────────────
# Outage (i,k,h') couples or decouples in [h, h+Se−1]
# iff  ha ∈ [h−DA_{i,k}, h+Se−1]  (outage overlaps the window either at start or end)
# ∀h ∈ [0, T+max_DA−1]:
#   Σ_{(i,k) ∈ Am, h'=h−DA_{i,k}}^{h+Se−1} x_{i,k,h'} ≤ 1
for g_idx, (group, se) in enumerate(zip(GROUPS, SE18)):
    for h in range(T + max_DA):
        terms = []
        for i in group:
            for k in range(nK):
                for h_prime in valid_h[(i, k)]:
                    lo = h - DA[i, k]
                    hi = h + se - 1
                    if lo <= h_prime <= hi:
                        terms.append(x_idx[(i, k, h_prime)])
        if len(terms) > 1:
            for col in terms:
                add(row, col, 1.0)
            leq(1.0)
            row += 1

# ── CT19: max unavailable plants in specific period window  ────────────────────
# L_{i,k,m} = 0, TU_{i,k,m} = DA[i,k] → constraint window = full outage duration
# ∀h: Σ_{(i,k) ∈ Am, h'=h−DA_{i,k}+1}^{h} x_{i,k,h'} ≤ Q_m
for g_idx, (group, q) in enumerate(zip(GROUPS, Q19)):
    for h in range(T):
        terms = []
        for i in group:
            for col, coeff in on_outage_terms(i, h):
                terms.append(col)
        if terms:
            for col in terms:
                add(row, col, 1.0)
            leq(q)
            row += 1

# ── CT20: max outages overlapping each week h_m  ──────────────────────────────
# Applied for every week (each week defines one CT20 constraint globally):
# ∀h_m: Σ_{(i,k), h'=h_m−DA_{i,k}+1}^{h_m} x_{i,k,h'} ≤ N20
for h_m in range(T):
    terms = []
    for i in range(nI):
        for col, coeff in on_outage_terms(i, h_m):
            terms.append(col)
    if terms:
        for col in terms:
            add(row, col, 1.0)
        leq(N20)
        row += 1

# ── CT21: max offline capacity per group per week  ────────────────────────────
# ∀g, ∀h ∈ IT_m (all weeks): Σ_{i ∈ Cm, k, h'} PMAX_i · x_{i,k,h'} ≤ IMAX_m
for g_idx, (group, imax) in enumerate(zip(GROUPS, IMAX21)):
    for h in range(T):
        has_term = False
        for i in group:
            for col, coeff in on_outage_terms(i, h):
                add(row, col, PMAX_NUC[i] * coeff)
                has_term = True
        if has_term:
            leq(imax)
            row += 1

n_rows = row
A = csr_matrix((vals_l, (rows_l, cols_l)), shape=(n_rows, n_vars))
lb_arr = np.array(lb_l)
ub_arr = np.array(ub_l)

# ---------------------------------------------------------------------------
# Variable bounds
# ---------------------------------------------------------------------------
lb_vars = np.zeros(n_vars)
ub_vars = np.full(n_vars, INF)

for col in x_idx.values():            # x ∈ [0,1]
    ub_vars[col] = 1.0

for i in range(nI):                   # r ∈ [RMIN, RMAX]
    for k in range(nK):
        lb_vars[idx_r(i, k)] = RMIN[i, k]
        ub_vars[idx_r(i, k)] = RMAX[i, k]

for s in range(nS):
    for j in range(nJ):               # p1 ∈ [PMIN, PMAX]
        for t in range(T):
            lb_vars[idx_p1(j, t, s)] = PMIN_FOC[j]
            ub_vars[idx_p1(j, t, s)] = PMAX_FOC[j]
    for i in range(nI):               # p2 ∈ [0, PMAX_NUC]
        for t in range(T):
            ub_vars[idx_p2(i, t, s)] = PMAX_NUC[i]
        for t in range(T+1):          # fuel ∈ [0, FMAX]
            ub_vars[idx_fuel(i, t, s)] = FMAX[i]

bounds = Bounds(lb_vars, ub_vars)

# ---------------------------------------------------------------------------
# Integrality: only x variables are binary
# ---------------------------------------------------------------------------
integrality = np.zeros(n_vars)
for col in x_idx.values():
    integrality[col] = 1

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
print(f"Problem: n_vars={n_vars}  n_rows={n_rows}")
print(f"  x binary (scheduling): {n_x}  |  r continuous: {n_r}")
print("Solving...")

res = milp(c_obj,
           constraints=LinearConstraint(A, lb_arr, ub_arr),
           integrality=integrality,
           bounds=bounds)

if res.status == 0:
    print(f"Optimal — cost = {res.fun:,.2f} €")
    v = res.x
else:
    print(f"Solver status {res.status}: {res.message}")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Extract solution arrays
# ---------------------------------------------------------------------------
outage_start = {}   # (i,k) → chosen start week
for i in range(nI):
    for k in range(nK):
        for h in valid_h[(i, k)]:
            if v[x_idx[(i, k, h)]] > 0.5:
                outage_start[(i, k)] = h

r_sol    = np.array([[v[idx_r(i, k)] for k in range(nK)] for i in range(nI)])
p1_sol   = np.zeros((nJ, T, nS))
p2_sol   = np.zeros((nI, T, nS))
fuel_sol = np.zeros((nI, T+1, nS))

for s in range(nS):
    for j in range(nJ):
        for t in range(T):
            p1_sol[j, t, s]    = v[idx_p1(j, t, s)]
    for i in range(nI):
        for t in range(T):
            p2_sol[i, t, s]    = v[idx_p2(i, t, s)]
        for t in range(T+1):
            fuel_sol[i, t, s]  = v[idx_fuel(i, t, s)]

# ---------------------------------------------------------------------------
# Export solution
# ---------------------------------------------------------------------------
out_dir  = os.path.dirname(os.path.abspath(__file__))
sol_path = os.path.join(out_dir, "solution_paper.txt")

header = (["s", "t"]
          + [f"p1_{j}"    for j in range(nJ)]
          + [f"p2_{i}"    for i in range(nI)]
          + [f"fuel_{i}"  for i in range(nI)]
          + [f"r_{i}_{k}" for i in range(nI) for k in range(nK)])

with open(sol_path, "w") as f:
    f.write("\t".join(header) + "\n")
    for s in range(nS):
        for t in range(T):
            row_vals = ([s, t]
                        + [p1_sol[j, t, s]   for j in range(nJ)]
                        + [p2_sol[i, t, s]   for i in range(nI)]
                        + [fuel_sol[i, t, s] for i in range(nI)]
                        + [r_sol[i, k]       for i in range(nI) for k in range(nK)])
            f.write("\t".join(f"{x:.6g}" for x in row_vals) + "\n")
print(f"Solution written to {sol_path}")

# ---------------------------------------------------------------------------
# Visualisation  (6 plots, matching milp_paper.py style)
# ---------------------------------------------------------------------------
weeks    = np.arange(T)
weeks_fu = np.arange(T+1)
alpha    = max(0.2, 1.0/nS)
lw       = 1.8
C_NUC_PLT = ["tab:green", "tab:olive", "tab:cyan",   "tab:teal"]
C_FOC_PLT = ["tab:orange","tab:red",   "tab:brown"]
C_OUT_PLT = ["tab:blue",  "tab:purple","tab:green",  "tab:red"]

def make_fig(n_rows, title, filename, plot_fn):
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3*n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]
    plot_fn(axes)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()
    print(f"  {filename}")

# ── 1. Gantt chart — outage schedule  ─────────────────────────────────────────
def plot_gantt(axes):
    ax = axes[0]
    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, nI - 0.5)
    ax.set_yticks(range(nI))
    ax.set_yticklabels([f"Plant {i}" for i in range(nI)])
    ax.set_xlabel("Week")
    ax.set_title("Outage schedule (Gantt) — ha(i,k) from ILP")
    ax.grid(True, axis="x", alpha=0.3)
    for i in range(nI):
        for k in range(nK):
            h = outage_start.get((i, k))
            if h is not None:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (h, i - 0.38), DA[i, k], 0.76,
                    boxstyle="round,pad=0.04",
                    facecolor=C_OUT_PLT[k % len(C_OUT_PLT)],
                    edgecolor="white", linewidth=0.8, alpha=0.85))
                ax.text(h + DA[i, k]/2, i, f"k={k}\n{DA[i,k]}w",
                        ha="center", va="center", fontsize=7, color="white")
    ax.legend(handles=[mpatches.Patch(color=C_OUT_PLT[k], label=f"Cycle k={k}")
                        for k in range(nK)],
              loc="upper right", fontsize=8)

make_fig(1, "Nuclear Outage Schedule", "gantt_schedule.png", plot_gantt)

# ── 2. Fuel levels  ────────────────────────────────────────────────────────────
def plot_fuel(axes):
    for i, ax in enumerate(axes):
        for s in range(nS):
            ax.plot(weeks_fu, fuel_sol[i, :, s],
                    color=C_NUC_PLT[i], linewidth=lw, alpha=alpha)
        ax.axhline(FMAX[i],       color="grey",   linewidth=1.2, linestyle=":", label="F_max")
        ax.axhline(float(AMAX[i,0]), color="tab:orange", linewidth=1.0,
                   linestyle="--", label=f"AMAX={AMAX[i,0]:.2f}")
        ax.axhline(float(SMAX[i,0]), color="tab:blue",   linewidth=1.0,
                   linestyle="--", label=f"SMAX={SMAX[i,0]:.2f}")
        for k in range(nK):
            h = outage_start.get((i, k))
            if h is not None:
                ax.axvspan(h, h + DA[i, k], alpha=0.12, color="tab:orange",
                           label="outage" if k == 0 else "")
        ax.set_ylabel("Fuel [f.u.]"); ax.set_title(f"Nuclear plant {i}")
        ax.set_xlabel("Week"); ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

make_fig(nI, f"Fuel levels — {nS} scenarios", "fuel_levels.png", plot_fuel)

# ── 3. Nuclear generation  ─────────────────────────────────────────────────────
def plot_nuc_gen(axes):
    for i, ax in enumerate(axes):
        for s in range(nS):
            ax.plot(weeks, p2_sol[i, :, s],
                    color=C_NUC_PLT[i], linewidth=lw, alpha=alpha)
        for k in range(nK):
            h = outage_start.get((i, k))
            if h is not None:
                ax.axvspan(h, h + DA[i, k], alpha=0.12, color="tab:orange")
        ax.set_ylim(-0.05, PMAX_NUC[i]*1.15)
        ax.set_ylabel("Gen. [p.u.]"); ax.set_title(f"Nuclear plant {i}")
        ax.set_xlabel("Week"); ax.grid(True, alpha=0.3)

make_fig(nI, f"Nuclear generation — {nS} scenarios",
         "nuclear_generation.png", plot_nuc_gen)

# ── 4. Non-nuclear generation  ────────────────────────────────────────────────
def plot_foc_gen(axes):
    for j, ax in enumerate(axes):
        for s in range(nS):
            ax.plot(weeks, p1_sol[j, :, s],
                    color=C_FOC_PLT[j], linewidth=lw, alpha=alpha)
        ax.set_ylim(-0.05, PMAX_FOC[j]*1.15)
        ax.set_ylabel("Gen. [p.u.]"); ax.set_title(f"Non-nuclear plant {j}")
        ax.set_xlabel("Week"); ax.grid(True, alpha=0.3)

make_fig(nJ, f"Non-nuclear generation — {nS} scenarios",
         "fossil_generation.png", plot_foc_gen)

# ── 5. Refueling amounts  ──────────────────────────────────────────────────────
def plot_refuel(axes):
    ax = axes[0]
    xpos = np.arange(nI * nK)
    colors = [C_NUC_PLT[i] for i in range(nI) for _ in range(nK)]
    bars = ax.bar(xpos, [r_sol[i, k] for i in range(nI) for k in range(nK)],
                  color=colors, alpha=0.8, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f f.u.", fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"P{i} k{k}" for i in range(nI) for k in range(nK)], fontsize=9)
    ax.axhline(float(RMIN[0,0]), color="grey", linestyle="--", linewidth=1.2, label=f"RMIN={RMIN[0,0]:.2f}")
    ax.axhline(float(RMAX[0,0]), color="grey", linestyle=":",  linewidth=1.2, label=f"RMAX={RMAX[0,0]:.2f}")
    ax.set_ylabel("Reload r(i,k) [f.u.]")
    ax.set_title("Refueling amounts per plant / cycle  (r(i,k) — paper CT7)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

make_fig(1, "Refueling amounts", "refueling_amounts.png", plot_refuel)

# ── 6. Cost breakdown  ────────────────────────────────────────────────────────
def plot_cost(axes):
    ax = axes[0]
    refuel_cost = float(sum(C_FUEL[i,k]*r_sol[i,k]
                            for i in range(nI) for k in range(nK)))
    gen_costs   = [float(sum(C1[j,t]*p1_sol[j,t,s]*D
                             for j in range(nJ) for t in range(T)))
                   for s in range(nS)]
    term_vals   = [float(sum(C_TERM[i]*fuel_sol[i,T,s]
                             for i in range(nI)))
                   for s in range(nS)]
    cats   = ["Refueling\ncost", "Avg fossil\ngen. cost", "Avg terminal\nfuel credit"]
    values = [refuel_cost, np.mean(gen_costs), -np.mean(term_vals)]
    colors = ["tab:orange", "tab:blue", "tab:green"]
    b = ax.bar(cats, values, color=colors, alpha=0.8, edgecolor="white")
    ax.bar_label(b, fmt="%.0f €", fontsize=9)
    ax.set_ylabel("Cost [€]")
    ax.set_title("Objective cost breakdown")
    ax.grid(True, axis="y", alpha=0.3)

make_fig(1, "Cost breakdown", "cost_breakdown.png", plot_cost)

print("All plots saved.")

"""
Nuclear plant maintenance scheduling — deterministic MILP solved independently
for each load scenario.

Variable layout (flat vector of length 2*nN*T + (nF+2*nN)*T):

  [0,  nN*T)                      : d[i,t]       binary  — maintenance
  [nN*T, 2*nN*T)                  : r[i,t]       ≥ 0     — refuelling
  [2*nN*T, 2*nN*T + nF*T)         : pF[j,t]      ≥ 0     — fossil generation
  [2*nN*T + nF*T, 2*nN*T+(nF+nN)*T): pN[i,t]    ≥ 0     — nuclear generation
  [(2*nN+nF+nN)*T, …)             : x[i,t] t=1..T ≥ 0   — fuel stock
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import csv

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
nF = 4      # number of fossil plants
nN = 4      # number of nuclear plants
T  = 100    # number of time periods (t = 0 … T-1), each 1 hour

# Input parameters
nW          = 10    # number of independent load scenarios
P_fos_max   = 1.    # maximum power of each fossil plant  [p.u.]
P_nuc_max   = 1.    # maximum power of each nuclear plant [p.u.]
L_bar_mean  = 3.    # average load [p.u.]
L_std       = 0.5   # standard deviation of load noise [p.u.]
C_fos_mean  = 30.   # mean marginal cost of fossil generation [€/p.u./h]
C_nuc_mean  = 10.   # marginal cost of nuclear generation [€/p.u./h]
c_fuel_rate = 0.01  # fuel consumption rate per nuclear plant [fuel-units/(p.u.·h)]
X_max_nuc   = 1.0   # maximum fuel stock per nuclear plant [fuel-units]
X0_nuc      = 0.5   # initial fuel stock per nuclear plant [fuel-units]
ramp_fos    = 0.6   # max ramp rate for fossil plants  [p.u./period]
ramp_nuc    = 0.2   # max ramp rate for nuclear plants [p.u./period]

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
t_idx = np.arange(T)

# Fossil marginal cost [€/p.u./h], shape (nF, T)
_C_profile = C_fos_mean + 10. * np.sin(2 * np.pi * t_idx / 24) + rng.uniform(0, 2, T)
C = np.tile(_C_profile, (nF, 1))

# Period duration [h]
D = np.ones(T)

# Mean load profile, shape (T,)
L_bar = L_bar_mean * (1 + 0.20 * np.sin(2 * np.pi * t_idx / 24 - np.pi / 2)
                        + 0.05 * np.sin(2 * np.pi * t_idx / (24 * 7)))

# Load noise: shape (T, nW) — each column is one independent scenario
dL = rng.normal(0., L_std, (T, nW))

P_fos  = np.full(nF, P_fos_max)
P_nuc  = np.full(nN, P_nuc_max)
c_fuel = np.full(nN, c_fuel_rate)
X_max  = np.full(nN, X_max_nuc)
X0     = np.full(nN, X0_nuc)

# ---------------------------------------------------------------------------
# Variable-index helpers  (single scenario — no w index)
# ---------------------------------------------------------------------------
_off   = 2 * nN * T
n_vars = _off + (nF + 2 * nN) * T

def idx_d(i, t):   return i * T + t
def idx_r(i, t):   return nN * T + i * T + t
def idx_pF(j, t):  return _off + j * T + t
def idx_pN(i, t):  return _off + nF * T + i * T + t
def idx_x(i, t):   return _off + (nF + nN) * T + i * T + (t - 1)   # t = 1..T

# ---------------------------------------------------------------------------
# Objective  min Σ_{j,t} C[j,t]·D[t]·pF[j,t] + Σ_{i,t} C_nuc·D[t]·pN[i,t]
# ---------------------------------------------------------------------------
c_obj = np.zeros(n_vars)
for j in range(nF):
    for t in range(T):
        c_obj[idx_pF(j, t)] = C[j, t] * D[t]
for i in range(nN):
    for t in range(T):
        c_obj[idx_pN(i, t)] = C_nuc_mean * D[t]

# ---------------------------------------------------------------------------
# Constraint matrix  (built once — power balance RHS updated per scenario)
# ---------------------------------------------------------------------------
rows_list, cols_list, vals_list = [], [], []
lb_list, ub_list = [], []
INF = np.inf
row = 0

def add(r, c, v):
    rows_list.append(r); cols_list.append(c); vals_list.append(v)

def eq(rhs):
    lb_list.append(rhs); ub_list.append(rhs)

def leq(ub):
    lb_list.append(-INF); ub_list.append(ub)

# ── 1. Power balance (T equality rows) — RHS updated per scenario ────────────
PB_ROWS = slice(0, T)   # remember which rows hold the power balance
for t in range(T):
    for j in range(nF):
        add(row, idx_pF(j, t), 1.0)
    for i in range(nN):
        add(row, idx_pN(i, t), 1.0)
    eq(0.0)              # placeholder — overwritten in the solve loop
    row += 1

# ── 2. Nuclear upper bound (nN·T inequality rows) ────────────────────────────
for i in range(nN):
    for t in range(T):
        add(row, idx_pN(i, t), 1.0)
        add(row, idx_d(i, t),  P_nuc[i])
        leq(P_nuc[i])
        row += 1

# ── 3. Fuel dynamics (nN·T equality rows) ────────────────────────────────────
for i in range(nN):
    for t in range(T):
        add(row, idx_x(i, t + 1),          1.0)
        if t > 0:
            add(row, idx_x(i, t),          -1.0)
        add(row, idx_pN(i, t), c_fuel[i] * D[t])
        add(row, idx_r(i, t),              -1.0)
        eq(X0[i] if t == 0 else 0.0)
        row += 1

# ── 4. Refuelling capacity (nN·T inequality rows) ────────────────────────────
for i in range(nN):
    for t in range(T):
        add(row, idx_r(i, t),  1.0)
        add(row, idx_d(i, t), -X_max[i])
        leq(0.0)
        row += 1

# ── 5. Fossil ramp rate (2·nF·(T-1) inequality rows) ─────────────────────────
for j in range(nF):
    for t in range(T - 1):
        add(row, idx_pF(j, t + 1),  1.0); add(row, idx_pF(j, t), -1.0)
        leq(ramp_fos); row += 1
        add(row, idx_pF(j, t),      1.0); add(row, idx_pF(j, t + 1), -1.0)
        leq(ramp_fos); row += 1

# ── 6. Nuclear ramp rate (2·nN·(T-1) inequality rows) ────────────────────────
for i in range(nN):
    for t in range(T - 1):
        add(row, idx_pN(i, t + 1),  1.0); add(row, idx_pN(i, t), -1.0)
        leq(ramp_nuc); row += 1
        add(row, idx_pN(i, t),      1.0); add(row, idx_pN(i, t + 1), -1.0)
        leq(ramp_nuc); row += 1

n_rows = row
A      = csr_matrix((vals_list, (rows_list, cols_list)), shape=(n_rows, n_vars))
lb_arr = np.array(lb_list)
ub_arr = np.array(ub_list)

# ---------------------------------------------------------------------------
# Variable bounds
# ---------------------------------------------------------------------------
lb_vars = np.zeros(n_vars)
ub_vars = np.full(n_vars, INF)

for i in range(nN):
    for t in range(T):
        ub_vars[idx_d(i, t)] = 1.0
        ub_vars[idx_r(i, t)] = X_max[i]
for j in range(nF):
    for t in range(T):
        ub_vars[idx_pF(j, t)] = P_fos[j]
for i in range(nN):
    for t in range(T):
        ub_vars[idx_pN(i, t)] = P_nuc[i]
    for t in range(1, T + 1):
        ub_vars[idx_x(i, t)] = X_max[i]

bounds = Bounds(lb_vars, ub_vars)

# ---------------------------------------------------------------------------
# Integrality
# ---------------------------------------------------------------------------
integrality = np.zeros(n_vars)
for i in range(nN):
    for t in range(T):
        integrality[idx_d(i, t)] = 1

# ---------------------------------------------------------------------------
# Solve — one independent optimisation per scenario
# ---------------------------------------------------------------------------
print(f"Optimizing {nW} scenarios independently "
      f"(n_vars={n_vars}, n_rows={n_rows})...")

solutions = {}
for w in range(nW):
    # Update power balance RHS for this scenario's load
    lb_arr[PB_ROWS] = L_bar + dL[:, w]
    ub_arr[PB_ROWS] = L_bar + dL[:, w]

    constraint = LinearConstraint(A, lb_arr, ub_arr)
    print(f"  Scenario {w + 1}/{nW}...", end=" ", flush=True)
    res = milp(c_obj, constraints=constraint,
               integrality=integrality, bounds=bounds)
    if res.status == 0:
        solutions[w] = res.x
        print(f"cost = {res.fun:,.2f} €")
    else:
        solutions[w] = None
        print(f"FAILED ({res.message})")

# ---------------------------------------------------------------------------
# Export solution to file
# ---------------------------------------------------------------------------
header = (["w", "t"]
          + [f"d_{i}"  for i in range(nN)]
          + [f"r_{i}"  for i in range(nN)]
          + [f"pF_{j}" for j in range(nF)]
          + [f"pN_{i}" for i in range(nN)]
          + [f"x_{i}"  for i in range(nN)])

with open("solution.txt", "w") as f:
    f.write("\t".join(header) + "\n")
    for w, v in solutions.items():
        if v is None:
            continue
        for t in range(T):
            row_vals = (
                [w, t]
                + [v[idx_d(i, t)]      for i in range(nN)]
                + [v[idx_r(i, t)]      for i in range(nN)]
                + [v[idx_pF(j, t)]     for j in range(nF)]
                + [v[idx_pN(i, t)]     for i in range(nN)]
                + [v[idx_x(i, t + 1)]  for i in range(nN)]
            )
            f.write("\t".join(f"{val:.6g}" for val in row_vals) + "\n")
print("Solution written to solution.txt")

# ---------------------------------------------------------------------------
# Load file and plot
# ---------------------------------------------------------------------------
data = {}
with open("solution.txt") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        w_r = int(row["w"])
        if w_r not in data:
            data[w_r] = {k: [] for k in reader.fieldnames if k != "w"}
        for k in data[w_r]:
            data[w_r][k].append(float(row[k]))

time   = np.arange(T)
time_x = np.arange(1, T + 1)
alpha  = max(0.2, 1.0 / nW)
lw     = 1.8

def make_fig(n_rows, title, filename, plot_fn):
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    plot_fn(axes)
    axes[-1].set_xlabel("Time period")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# ── 1. Nuclear generation ────────────────────────────────────────────────────
def plot_nuclear_gen(axes):
    for i, ax in enumerate(axes):
        for w in data:
            ax.plot(time, data[w][f"pN_{i}"], color="tab:green",
                    linewidth=lw, alpha=alpha)
        ax.set_ylabel("Gen. [p.u.]"); ax.set_title(f"Nuclear plant {i}")
        ax.grid(True, alpha=0.3)

make_fig(nN, f"Nuclear generation — {nW} scenarios",
         "nuclear_generation.png", plot_nuclear_gen)

# ── 2. Fossil generation ─────────────────────────────────────────────────────
def plot_fossil_gen(axes):
    for j, ax in enumerate(axes):
        for w in data:
            ax.plot(time, data[w][f"pF_{j}"], color="tab:orange",
                    linewidth=lw, alpha=alpha)
        ax.set_ylabel("Gen. [p.u.]"); ax.set_title(f"Fossil plant {j}")
        ax.grid(True, alpha=0.3)

make_fig(nF, f"Fossil generation — {nW} scenarios",
         "fossil_generation.png", plot_fossil_gen)

# ── 3. Maintenance decisions ─────────────────────────────────────────────────
def plot_maintenance(axes):
    for i, ax in enumerate(axes):
        for w in data:
            ax.step(time, data[w][f"d_{i}"], where="post",
                    color="tab:green", linewidth=lw, alpha=alpha)
        ax.set_ylim(-0.05, 1.15); ax.set_yticks([0, 1])
        ax.set_ylabel("d [0/1]"); ax.set_title(f"Nuclear plant {i}")
        ax.grid(True, alpha=0.3)

make_fig(nN, "Maintenance decisions  (1 = under maintenance)",
         "maintenance.png", plot_maintenance)

# ── 4. Fuel stock ────────────────────────────────────────────────────────────
def plot_fuel(axes):
    for i, ax in enumerate(axes):
        for w in data:
            ax.plot(time_x, data[w][f"x_{i}"], color="tab:blue",
                    linewidth=lw, alpha=alpha)
        ax.axhline(X_max_nuc, color="grey", linewidth=lw,
                   linestyle=":", label="X_max")
        ax.set_ylabel("Fuel [f.u.]"); ax.set_title(f"Nuclear plant {i}")
        ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.3)

make_fig(nN, f"Fuel stock — {nW} scenarios",
         "fuel_stock.png", plot_fuel)

# ── 5. Refuelling ────────────────────────────────────────────────────────────
def plot_refuel(axes):
    for i, ax in enumerate(axes):
        for w in data:
            ax.step(time, data[w][f"r_{i}"], where="post",
                    color="tab:purple", linewidth=lw, alpha=alpha)
        ax.set_ylabel("Refuel [f.u.]"); ax.set_title(f"Nuclear plant {i}")
        ax.grid(True, alpha=0.3)

make_fig(nN, "Refuelling amounts",
         "refuelling.png", plot_refuel)

# ── 6. Load realisations ─────────────────────────────────────────────────────
def plot_load(axes):
    ax = axes[0]
    for w in range(nW):
        ax.plot(time, L_bar + dL[:, w], color="tab:red",
                linewidth=lw, alpha=alpha)
    ax.set_ylabel("Load [p.u.]")
    ax.set_title(f"Load realisations — {nW} scenarios")
    ax.grid(True, alpha=0.3)

make_fig(1, f"Load realisations — {nW} scenarios",
         "load_scenarios.png", plot_load)

print("Plots saved.")

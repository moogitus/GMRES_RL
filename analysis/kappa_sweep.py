"""
κ-sweep on 1D convection-diffusion matrices (§4.3, Figure 1). Compares:
  - fixed-restart GMRES(20) (deterministic),
  - randGMRES with uniform m ∈ {5, 10, 15, 20} per cycle,
  - the DQN-controlled GMRES(m) restart controller.

For each of the 20 matrices in matrices/kappa_sweep_20_matrices/, generates
5 standard-normal unit-norm RHS vectors and runs each stochastic method
with 2 seeds per RHS. Records total Arnoldi iterations and wall-clock
time per run. GMRES(20) is run once per RHS (deterministic).

Writes results/kappa_sweep/kappa_sweep.json (raw runs + κ_2(A) per matrix)
and results/kappa_sweep/kappa_sweep_arnoldi_vs_kappa.png (Arnoldi vs κ
on log-log axes with std bands).

Built on stable-baselines3 (https://github.com/DLR-RM/stable-baselines3).

Examples:
    python analysis/kappa_sweep.py
    python analysis/kappa_sweep.py --limit 3       # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse
from stable_baselines3 import DQN
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env import GMRESEnv  # noqa: E402
from utils import RelativeResidualCallback as _StopOnDone  # noqa: E402

DEFAULT_MATRICES_DIR = REPO_ROOT / "matrices" / "kappa_sweep_20_matrices"
DEFAULT_OUT_JSON = REPO_ROOT / "results" / "kappa_sweep" / "kappa_sweep.json"
DEFAULT_OUT_PLOT = REPO_ROOT / "results" / "kappa_sweep" / "kappa_sweep_arnoldi_vs_kappa.png"

# randGMRES restart values (§4.2), uniform per cycle
RAND_RESTARTS = (5, 10, 15, 20)

# DQN hyperparameters from the Bayesian sweep (Table 5)
DQN_HP = dict(
    m_max=20,
    history_length=5,
    tolerance=1e-6,
    max_cycles=5_000,
    gamma=0.925,
    lambda_work=0.01,
    convergence_bonus=9.0,
    learning_rate=3e-3,
    buffer_size=10_000,
    learning_starts=25,
    batch_size=32,
    target_update_interval=100,
    exploration_fraction=0.10,
    exploration_final_eps=0.01,
)


def load_matrix(path: Path):
    A = mmread(str(path))
    if not issparse(A):
        A = csr_matrix(A)
    return csr_matrix(A.astype(np.float64))


# 2-norm condition number κ_2(A) via dense SVD (n ≤ 3500 in this sweep)
def compute_condition_number(A) -> float:
    A_dense = A.toarray() if issparse(A) else np.asarray(A, dtype=np.float64)
    s = np.linalg.svd(A_dense, compute_uv=False)
    smin = float(s[-1])
    if smin <= 0.0:
        return float("inf")
    return float(s[0] / smin)


# standard-normal RHS, rescaled to unit Euclidean norm (§4.3)
def random_rhs(n: int, rng: np.random.Generator) -> np.ndarray:
    b = rng.standard_normal(n)
    return b / max(float(np.linalg.norm(b)), 1e-12)


def discover_matrices(matrices_dir: Path):
    paths = sorted(matrices_dir.glob("*.mtx"))
    if not paths:
        raise FileNotFoundError(f"no .mtx files found in {matrices_dir}")
    return paths



def run_dqn(A, b, seed: int) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = GMRESEnv(
        A=A, b=b,
        m_max=DQN_HP["m_max"],
        tolerance=DQN_HP["tolerance"],
        max_cycles=DQN_HP["max_cycles"],
        history_length=DQN_HP["history_length"],
        lambda_work=DQN_HP["lambda_work"],
        gamma_shape=DQN_HP["gamma"],
        convergence_bonus=DQN_HP["convergence_bonus"],
    )
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=DQN_HP["learning_rate"],
        buffer_size=DQN_HP["buffer_size"],
        learning_starts=DQN_HP["learning_starts"],
        batch_size=DQN_HP["batch_size"],
        gamma=DQN_HP["gamma"],
        train_freq=1,
        gradient_steps=1,
        target_update_interval=DQN_HP["target_update_interval"],
        exploration_fraction=DQN_HP["exploration_fraction"],
        exploration_initial_eps=1.0,
        exploration_final_eps=DQN_HP["exploration_final_eps"],
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
        seed=seed,
        device="cpu",
    )
    cb = _StopOnDone()
    t0 = time.perf_counter()
    model.learn(total_timesteps=DQN_HP["max_cycles"], callback=cb, progress_bar=False)
    elapsed = time.perf_counter() - t0
    rel = np.asarray(cb.relative_residuals, dtype=np.float64)
    ms = np.asarray(cb.ms, dtype=np.int64)
    converged = bool(np.any(rel < DQN_HP["tolerance"]))
    if converged:
        idx = int(np.argmax(rel < DQN_HP["tolerance"])) + 1
        total_arnoldi = int(ms[:idx].sum())
    else:
        total_arnoldi = int(ms.sum()) if ms.size else 0
    return {
        "converged": converged,
        "total_arnoldi": int(total_arnoldi),
        "elapsed_seconds": float(elapsed),
        "final_relative_residual": float(rel[-1]) if rel.size else float("inf"),
    }


def run_fixed_gmres(A, b, m: int = 20, seed: int = 0) -> dict:
    """Step the GMRES env with the constant action `m` until convergence/cap."""
    env = GMRESEnv(
        A=A, b=b,
        m_max=DQN_HP["m_max"],
        tolerance=DQN_HP["tolerance"],
        max_cycles=DQN_HP["max_cycles"],
        history_length=DQN_HP["history_length"],
        lambda_work=0.0,
        gamma_shape=DQN_HP["gamma"],
        convergence_bonus=0.0,
    )
    env.reset(seed=seed)
    action = m - 1  # action index in {0,...,m_max-1} maps to m in {1,...,m_max}
    total_arnoldi = 0
    converged = False
    final_relres = float("inf")
    t0 = time.perf_counter()
    for _ in range(DQN_HP["max_cycles"]):
        _, _, terminated, truncated, info = env.step(action)
        total_arnoldi += int(info["current_m"])
        final_relres = float(info["relative_residual_norm"])
        if terminated:
            converged = True
            break
        if truncated:
            break
    elapsed = time.perf_counter() - t0
    return {
        "converged": bool(converged),
        "total_arnoldi": int(total_arnoldi),
        "elapsed_seconds": float(elapsed),
        "final_relative_residual": float(final_relres),
    }


def run_random_gmres(A, b, restarts=RAND_RESTARTS, seed: int = 0) -> dict:
    """Step the GMRES env with a uniformly random m in `restarts` per cycle.

    Mirrors run_dqn / run_fixed_gmres; the seed controls both the env reset
    and the per-cycle restart-value draws so reruns at the same seed are
    reproducible.
    """
    rng = np.random.default_rng(seed)
    env = GMRESEnv(
        A=A, b=b,
        m_max=DQN_HP["m_max"],
        tolerance=DQN_HP["tolerance"],
        max_cycles=DQN_HP["max_cycles"],
        history_length=DQN_HP["history_length"],
        lambda_work=0.0,
        gamma_shape=DQN_HP["gamma"],
        convergence_bonus=0.0,
    )
    env.reset(seed=seed)
    total_arnoldi = 0
    converged = False
    final_relres = float("inf")
    t0 = time.perf_counter()
    for _ in range(DQN_HP["max_cycles"]):
        m = int(rng.choice(restarts))
        _, _, terminated, truncated, info = env.step(m - 1)
        total_arnoldi += int(info["current_m"])
        final_relres = float(info["relative_residual_norm"])
        if terminated:
            converged = True
            break
        if truncated:
            break
    elapsed = time.perf_counter() - t0
    return {
        "converged": bool(converged),
        "total_arnoldi": int(total_arnoldi),
        "elapsed_seconds": float(elapsed),
        "final_relative_residual": float(final_relres),
    }


# sweep driver
def _save_json(results, args, out_json: Path):
    payload = {
        "meta": {
            "num_matrices": len(results),
            "num_rhs_per_matrix": args.num_rhs,
            "num_seeds_per_run": args.num_seeds,
            "rhs_master_seed": args.rhs_seed,
            "rhs_distribution": "standard_normal_unit_norm",
            "tolerance_relative": DQN_HP["tolerance"],
            "max_cycles": DQN_HP["max_cycles"],
            "m_max": DQN_HP["m_max"],
            "dqn_hyperparams": DQN_HP,
            "rand_gmres_restarts": list(RAND_RESTARTS),
            "notes": [
                "GMRES(20) is deterministic; only one run per RHS is recorded. "
                "DQN and randGMRES each get num_seeds independent runs per RHS, "
                "with seeds matched cell-by-cell so each (matrix, RHS, seed) "
                "triple is comparable across the two stochastic methods.",
                "randGMRES draws m uniformly from rand_gmres_restarts at every "
                "GMRES restart cycle.",
            ],
        },
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices-dir", type=str, default=str(DEFAULT_MATRICES_DIR))
    parser.add_argument("--out-json", type=str, default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-plot", type=str, default=str(DEFAULT_OUT_PLOT))
    parser.add_argument("--num-rhs", type=int, default=5)
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--rhs-seed", type=int, default=42,
                        help="Master seed for RHS generation; reproducible across runs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N matrices (debug / smoke test).")
    parser.add_argument("--skip-plot", action="store_true",
                        help="Run the sweep but skip plot generation.")
    args = parser.parse_args()

    matrices_dir = Path(args.matrices_dir).resolve()
    out_json = Path(args.out_json)
    out_plot = Path(args.out_plot)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    paths = discover_matrices(matrices_dir)
    if args.limit:
        paths = paths[:args.limit]

    # GMRES(20) is deterministic (1 run/RHS); DQN and randGMRES each get
    # num_seeds runs/RHS.
    total_runs = len(paths) * args.num_rhs * (2 * args.num_seeds + 1)
    print(f"Loading {len(paths)} matrices from {matrices_dir}")
    print(f"  RHS per matrix              : {args.num_rhs}")
    print(f"  DQN seeds per RHS           : {args.num_seeds}")
    print(f"  randGMRES seeds per RHS     : {args.num_seeds}")
    print(f"  GMRES(20) runs per RHS      : 1  (deterministic)")
    print(f"  randGMRES restart values    : {RAND_RESTARTS}")
    print(f"  Total runs                  : {total_runs}")

    rhs_rng = np.random.default_rng(args.rhs_seed)
    results = []

    pbar = tqdm(
        total=total_runs,
        desc="runs",
        dynamic_ncols=True,
        smoothing=0.05,  # smoother ETA given highly variable per-run wall-clock
    )

    for mi, path in enumerate(paths, 1):
        name = path.stem
        t_load = time.perf_counter()
        A = load_matrix(path)
        n = A.shape[0]
        kappa = compute_condition_number(A)
        load_secs = time.perf_counter() - t_load
        tqdm.write(f"\n[{mi}/{len(paths)}] {name}  "
                   f"n={n}  nnz={A.nnz}  κ₂={kappa:.3e}  "
                   f"(load+SVD: {load_secs:.1f}s)")
        pbar.set_postfix(matrix=name, kappa=f"{kappa:.2e}", refresh=True)

        rhs_vectors = [random_rhs(n, rhs_rng) for _ in range(args.num_rhs)]

        per_rhs = []
        for ri, b in enumerate(rhs_vectors):
            # GMRES(20) is deterministic — one run per RHS suffices.
            gmres_run = run_fixed_gmres(A, b, m=20, seed=0)
            pbar.update(1)
            tqdm.write(
                f"    rhs {ri}        GMRES(20)  arnoldi={gmres_run['total_arnoldi']:>7d} "
                f"t={gmres_run['elapsed_seconds']:6.1f}s "
                f"conv={int(gmres_run['converged'])}"
            )

            seeds_dqn = []
            seeds_rand = []
            for s in range(args.num_seeds):
                seed = s + 10 * ri  # decoupled across (rhs, algorithm-seed)

                dqn_run = run_dqn(A, b, seed=seed)
                pbar.update(1)
                seeds_dqn.append(dqn_run)
                tqdm.write(
                    f"    rhs {ri} seed {s}  DQN        arnoldi={dqn_run['total_arnoldi']:>7d} "
                    f"t={dqn_run['elapsed_seconds']:6.1f}s "
                    f"conv={int(dqn_run['converged'])}"
                )

                rand_run = run_random_gmres(A, b, seed=seed)
                pbar.update(1)
                seeds_rand.append(rand_run)
                tqdm.write(
                    f"    rhs {ri} seed {s}  randGMRES  arnoldi={rand_run['total_arnoldi']:>7d} "
                    f"t={rand_run['elapsed_seconds']:6.1f}s "
                    f"conv={int(rand_run['converged'])}"
                )

            per_rhs.append({
                "rhs_index": ri,
                "dqn_seeds": seeds_dqn,
                "rand_gmres_seeds": seeds_rand,
                "gmres20_seeds": [gmres_run],  # 1-element list for schema uniformity
            })

        results.append({
            "name": name,
            "path": str(path),
            "n": int(n),
            "nnz": int(A.nnz),
            "condition_number": float(kappa),
            "per_rhs": per_rhs,
        })
        # Snapshot per matrix so a crash never costs more than one matrix.
        _save_json(results, args, out_json)

    pbar.close()
    _save_json(results, args, out_json)
    print(f"\nWrote {len(results)} matrices to {out_json}")

    if not args.skip_plot:
        make_plot(results, out_plot)
        print(f"Wrote plot to {out_plot}")


# plotting (Figure 1)
def _aggregate(matrix_entry, alg_key, value_key="total_arnoldi"):
    """Flatten the per-(rhs, seed) values for one algorithm into a 1D array."""
    return np.asarray(
        [r[value_key] for rhs in matrix_entry["per_rhs"] for r in rhs[alg_key]],
        dtype=np.float64,
    )


def _all_converged(matrix_entry, alg_key) -> bool:
    return all(r["converged"] for rhs in matrix_entry["per_rhs"] for r in rhs[alg_key])


def make_plot(results, out_path: Path):
    """Paper-ready: condition number vs mean total Arnoldi for the three methods."""
    rows = sorted(results, key=lambda r: r["condition_number"])

    kappas = np.array([r["condition_number"] for r in rows])

    # Per-method (mean, std, all-converged) per matrix.
    methods = [
        ("gmres20_seeds",   "GMRES(20)",     "#d62728", "s"),
        ("rand_gmres_seeds", "randGMRES",    "#2ca02c", "^"),
        ("dqn_seeds",       "DQN (ours)",    "#1f77b4", "o"),
    ]
    series = {}
    for alg_key, _, _, _ in methods:
        means, stds, fully = [], [], []
        for r in rows:
            arr = _aggregate(r, alg_key, "total_arnoldi")
            means.append(arr.mean())
            stds.append(arr.std())
            fully.append(_all_converged(r, alg_key))
        series[alg_key] = (
            np.array(means),
            np.array(stds),
            np.array(fully, dtype=bool),
        )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Plot in order: GMRES(20) (worst, drawn first), randGMRES (middle),
    # DQN (best, drawn on top). zorder is incremented across methods so the
    # later lines overlay earlier ones at intersection points.
    for z, (alg_key, label, color, marker) in enumerate(methods):
        means, stds, fully = series[alg_key]
        ax.fill_between(kappas, means - stds, means + stds,
                        color=color, alpha=0.16, linewidth=0)
        ax.plot(kappas, means, linestyle="-", linewidth=1.8, color=color,
                label=label, zorder=3 + 2 * z)
        # Markers: filled for full-convergence matrices, hollow otherwise.
        ax.plot(kappas[fully], means[fully],
                linestyle="None", marker=marker, markersize=6, color=color,
                zorder=4 + 2 * z)
        ax.plot(kappas[~fully], means[~fully],
                linestyle="None", marker=marker, markersize=6,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.5,
                zorder=4 + 2 * z)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Condition number $\kappa_2(A)$")
    ax.set_ylabel("Total Arnoldi iterations to convergence")
    ax.set_title(
        r"Total Arnoldi iterations vs. $\kappa_2(A)$ "
        r"on convection-diffusion matrices",
        pad=14,
    )
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

    leg = ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    leg.get_frame().set_linewidth(0.6)
    ax.text(
        0.99, 0.02,
        "Hollow markers: not all runs reached tolerance.\n"
        "Shaded band: $\\pm 1$ std (DQN, randGMRES: 5 RHS $\\times$ 2 seeds = 10\n"
        "runs; GMRES(20): 5 RHS, deterministic).",
        ha="right", va="bottom", transform=ax.transAxes,
        fontsize=8.5, color="0.25",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.7", linewidth=0.5),
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

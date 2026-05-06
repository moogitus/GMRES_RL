"""
eval_reward_sweep.py

Two-phase hyperparameter sweep comparing reward functions for AK-SLRL.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hyperparameter search spaces (Bayesian TPE, Optuna):

  "original"  (authors' reward: R = c/‖r‖ + ‖r_prev‖ - ‖r‖)
      cte               ∈ [0.05, 200]    log-uniform
      convergence_bonus ∈ [0,    30]     uniform

  "shaped"    (PBRS with Φ = -log‖r‖, unbounded)
      lambda_work       ∈ [1e-4, 0.15]  log-uniform
      gamma_shape       ∈ [0.70, 0.999] uniform
      convergence_bonus ∈ [0,    30]    uniform

  "pbrs"      (τ-clamped PBRS, Φ_τ = -log(max(‖r‖,τ)/τ) ≤ 0)
      lambda_work       ∈ [1e-4, 0.15]  log-uniform
      gamma_shape       ∈ [0.70, 0.999] uniform
      convergence_bonus ∈ [0,    30]    uniform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 1 — Bayesian sweep:
  Objective: mean speedup vs GMRES(20) on 3 proxy configs × 1 seed each.
  Algorithm: Optuna TPESampler (default 40 trials, ~5 min total per reward type).

Phase 2 — Full 5-point κ eval:
  3 RHS vectors × 3 seeds = 9 SLRL runs per (reward_type, κ_level).
  Gives enough samples for std-dev comparison and basic significance testing.
  Results saved to logs/reward_sweep_results.json.

Run:
    python eval/eval_reward_sweep.py                  # full run
    python eval/eval_reward_sweep.py --n-trials 20   # faster sweep
    python eval/eval_reward_sweep.py --skip-sweep    # use hardcoded defaults
"""

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse, json
from pathlib import Path

import numpy as np
import torch
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scipy.io import mmread
from scipy.sparse import csr_matrix
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from src.env import AKSLRLEnv

# ─────────────────────────────────────────── configs ──

MATRIX_DIR = Path(__file__).parent.parent / "data" / "matrices" / "convdiff_matrices"

# Small/medium proxy configs used during the Bayesian sweep (fast: ~5 s each).
# Matrices loaded from convdiff_matrices/; max_cycles controls episode length.
SWEEP_CONFIGS = [
    {"n": 500, "eps": 0.05, "max_cycles":  600},   # κ ~50k   (easy)
    {"n": 500, "eps": 0.10, "max_cycles": 1007},   # κ ~130k  (medium)
    {"n": 900, "eps": 0.20, "max_cycles": 7050},   # κ ~280k  (hard)
]

# Standard 5 difficulty levels for the final eval (matches baseline plot).
EVAL_CONFIGS = [
    {"label": "Easy",      "n": 1000, "eps": 0.05, "max_cycles":  3000,
     "mtx": "convdiff_n1000_eps05_easy.mtx"},
    {"label": "Medium",    "n": 1000, "eps": 0.10, "max_cycles":  5000,
     "mtx": "convdiff_n1000_eps10_medium.mtx"},
    {"label": "Hard",      "n": 1000, "eps": 0.15, "max_cycles": 10000,
     "mtx": "convdiff_n1000_eps15_hard.mtx"},
    {"label": "Very Hard", "n": 1500, "eps": 0.15, "max_cycles": 25000,
     "mtx": "convdiff_n1500_eps15_very_hard.mtx"},
    {"label": "Extreme",   "n": 2000, "eps": 0.15, "max_cycles": 75000,
     "mtx": "convdiff_n2000_eps15_extreme.mtx"},
]


def load_convdiff(mtx_filename):
    path = MATRIX_DIR / mtx_filename
    return csr_matrix(mmread(str(path)).astype(np.float64))

REWARD_TYPES = ["original", "shaped", "pbrs"]

# Sensible defaults used when --skip-sweep is passed.
DEFAULT_PARAMS = {
    "original": {"cte": 1.0,    "convergence_bonus": 10.0},
    "shaped":   {"lambda_work": 0.001, "gamma_shape": 0.97, "convergence_bonus": 10.0},
    "pbrs":     {"lambda_work": 0.001, "gamma_shape": 0.97, "convergence_bonus": 10.0},
}

# Search space description (printed at startup for transparency).
SEARCH_SPACE = {
    "original": {
        "cte":               ("log-uniform", 0.05,  200.0),
        "convergence_bonus": ("uniform",     0.0,   30.0),
    },
    "shaped": {
        "lambda_work":       ("log-uniform", 1e-4,  0.15),
        "gamma_shape":       ("uniform",     0.70,  0.999),
        "convergence_bonus": ("uniform",     0.0,   30.0),
    },
    "pbrs": {
        "lambda_work":       ("log-uniform", 1e-4,  0.15),
        "gamma_shape":       ("uniform",     0.70,  0.999),
        "convergence_bonus": ("uniform",     0.0,   30.0),
    },
}


def print_search_space():
    print("\n" + "━" * 64)
    print("Hyperparameter search spaces:")
    for rt, space in SEARCH_SPACE.items():
        print(f"  [{rt}]")
        for param, (dist, lo, hi) in space.items():
            print(f"      {param:<22} ∈ [{lo}, {hi}]  ({dist})")
    print("━" * 64)


# ─────────────────────────────────────────── helpers ──

def _summarise(residuals, ms, tolerance, max_cycles, m_max):
    res = np.array(residuals)
    msa = np.array(ms)
    if np.any(res < tolerance):
        idx = int(np.argmax(res < tolerance)) + 1
        return {
            "converged":     True,
            "total_arnoldi": int(msa[:idx].sum()),
            "mean_m":        float(msa[:idx].mean()),
        }
    return {
        "converged":     False,
        "total_arnoldi": max_cycles * m_max,
        "mean_m":        float("nan"),
    }


def run_fixed_m(A, b, m, tolerance, max_cycles):
    env = AKSLRLEnv(A=A, b=b, m_max=m, tolerance=tolerance, max_cycles=max_cycles)
    env.reset()
    action = np.array([1.0], dtype=np.float32)
    residuals, ms = [], []
    for _ in range(max_cycles):
        _, _, term, trunc, info = env.step(action)
        residuals.append(float(info["residual_norm"]))
        ms.append(int(info["current_m"]))
        if term or trunc:
            break
    return _summarise(residuals, ms, tolerance, max_cycles, m)


class _StopOnDone(BaseCallback):
    def __init__(self):
        super().__init__()
        self.residuals, self.ms = [], []
        self._done = False

    def _on_step(self):
        for info, done in zip(self.locals.get("infos", []),
                               self.locals.get("dones", [False])):
            if "residual_norm" in info:
                self.residuals.append(float(info["residual_norm"]))
            if "current_m" in info:
                self.ms.append(int(info["current_m"]))
            if done:
                self._done = True
        return not self._done


def run_slrl(A, b, m_max, tolerance, max_cycles, reward_type, params, seed, gamma_sac):
    """Run one scratch SAC episode with given reward type and hyperparams."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_kwargs = dict(
        A=A, b=b, m_max=m_max, tolerance=tolerance, max_cycles=max_cycles,
        reward_type=reward_type,
        convergence_bonus=params.get("convergence_bonus", 10.0),
    )
    if reward_type == "original":
        env_kwargs["cte"] = params.get("cte", 1.0)
    else:
        env_kwargs["gamma_shape"] = params.get("gamma_shape", 0.97)
        env_kwargs["lambda_work"]  = params.get("lambda_work", 0.001)

    env = AKSLRLEnv(**env_kwargs)
    model = SAC(
        policy="MlpPolicy", env=env,
        learning_rate=3e-4,
        buffer_size=min(max_cycles * 5, 50_000),
        learning_starts=5, batch_size=256, tau=0.005,
        gamma=gamma_sac, train_freq=1, gradient_steps=1,
        ent_coef="auto", verbose=0, seed=seed, device="cpu",
    )
    logger = _StopOnDone()
    model.learn(total_timesteps=max_cycles, callback=logger, progress_bar=False)
    return _summarise(logger.residuals, logger.ms, tolerance, max_cycles, m_max)


def kappa_approx(A, n):
    lo = A.diagonal(-1)[0]; hi = A.diagonal(1)[0]; d = A.diagonal()[0]
    s = np.sqrt(lo * hi)
    lmin = d + 2*s*np.cos(n*np.pi/(n+1))
    lmax = d + 2*s*np.cos(1*np.pi/(n+1))
    return abs(lmax/lmin)


# ─────────────────────────────────── Bayesian sweep ──

def make_objective(reward_type, m_max, tolerance, gamma_sac, holdout_seed):
    """Return an Optuna objective for the given reward_type."""
    # Pre-build proxy problems (fixed across all trials for fair comparison).
    # Proxy configs (n=500/900) are not in the saved eval matrices so we
    # still generate them on the fly — they are small and fast.
    from src.matrices import make_convdiff_1d_sparse
    problems = []
    for cfg in SWEEP_CONFIGS:
        n, eps, mc = cfg["n"], cfg["eps"], cfg["max_cycles"]
        A = make_convdiff_1d_sparse(n, eps)
        rng = np.random.default_rng(holdout_seed + n + int(eps * 10000))
        b = rng.standard_normal(n).astype(np.float64)
        g20 = run_fixed_m(A, b, m=m_max, tolerance=tolerance, max_cycles=mc)
        problems.append({"A": A, "b": b, "n": n, "max_cycles": mc,
                         "g20_arnoldi": g20["total_arnoldi"]})

    def objective(trial):
        params = {}
        if reward_type == "original":
            params["cte"] = trial.suggest_float(
                "cte", *SEARCH_SPACE["original"]["cte"][1:], log=True)
            params["convergence_bonus"] = trial.suggest_float(
                "convergence_bonus", *SEARCH_SPACE["original"]["convergence_bonus"][1:])
        else:
            params["lambda_work"] = trial.suggest_float(
                "lambda_work", *SEARCH_SPACE[reward_type]["lambda_work"][1:], log=True)
            params["gamma_shape"] = trial.suggest_float(
                "gamma_shape", *SEARCH_SPACE[reward_type]["gamma_shape"][1:])
            params["convergence_bonus"] = trial.suggest_float(
                "convergence_bonus", *SEARCH_SPACE[reward_type]["convergence_bonus"][1:])

        speedups = []
        for p in problems:
            r = run_slrl(
                p["A"], p["b"], m_max=m_max, tolerance=tolerance,
                max_cycles=p["max_cycles"], reward_type=reward_type,
                params=params, seed=0, gamma_sac=gamma_sac,
            )
            if p["g20_arnoldi"] > 0 and r["total_arnoldi"] > 0:
                speedups.append(p["g20_arnoldi"] / r["total_arnoldi"])
            else:
                speedups.append(0.0)

        return -float(np.mean(speedups))   # Optuna minimises

    return objective


def run_sweep(reward_type, n_trials, m_max, tolerance, gamma_sac, holdout_seed):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    obj = make_objective(reward_type, m_max, tolerance, gamma_sac, holdout_seed)
    print(f"\n  [{reward_type}] running {n_trials} Optuna trials...")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  [{reward_type}] best: {best}  (proxy speedup ×{-study.best_value:.3f})")
    return best


# ────────────────────────────────── full 5-point eval ──

def run_full_eval(reward_type, params, m_max, tolerance, gamma_sac,
                  num_rhs, num_seeds, holdout_seed):
    """
    Run SLRL + GMRES(20) on all 5 EVAL_CONFIGS.
    Returns num_rhs GMRES runs and num_rhs × num_seeds SLRL runs per config.
    """
    results = []
    for cfg in EVAL_CONFIGS:
        n, eps, mc = cfg["n"], cfg["eps"], cfg["max_cycles"]
        A = load_convdiff(cfg["mtx"])
        kap = kappa_approx(A, n)

        rng = np.random.default_rng(holdout_seed + n + int(eps * 10000))
        problems = [rng.standard_normal(n).astype(np.float64) for _ in range(num_rhs)]

        g20_arnoldi, slrl_arnoldi = [], []

        for b in problems:
            r = run_fixed_m(A, b, m=m_max, tolerance=tolerance, max_cycles=mc)
            g20_arnoldi.append(r["total_arnoldi"])

        for b in problems:
            for seed in range(num_seeds):
                r = run_slrl(A, b, m_max=m_max, tolerance=tolerance, max_cycles=mc,
                             reward_type=reward_type, params=params,
                             seed=seed, gamma_sac=gamma_sac)
                slrl_arnoldi.append(r["total_arnoldi"])

        g20_mean = float(np.mean(g20_arnoldi))
        sl_mean  = float(np.mean(slrl_arnoldi))
        sl_std   = float(np.std(slrl_arnoldi))
        speedup  = g20_mean / sl_mean if sl_mean > 0 else float("nan")

        n_runs = len(slrl_arnoldi)
        print(f"    {cfg['label']:<10}  κ={kap:.0f}  g20={g20_mean:.0f}  "
              f"slrl={sl_mean:.0f}±{sl_std:.0f}  ×{speedup:.2f}"
              f"  (n={n_runs} runs)")

        results.append({
            "label": cfg["label"], "n": n, "eps": eps, "kappa": kap,
            "g20":  g20_arnoldi,
            "slrl": slrl_arnoldi,
        })
    return results


# ──────────────────────────────────────────────────────── main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-max",        type=int,   default=20)
    parser.add_argument("--tolerance",    type=float, default=1e-6)
    parser.add_argument("--gamma-sac",    type=float, default=0.97)
    parser.add_argument("--n-trials",     type=int,   default=40,
                        help="Optuna trials per reward type (default 40)")
    parser.add_argument("--num-rhs",      type=int,   default=3,
                        help="RHS vectors per config in final eval (default 3)")
    parser.add_argument("--num-seeds",    type=int,   default=3,
                        help="SAC seeds per (rhs, config) in final eval (default 3)")
    parser.add_argument("--holdout-seed", type=int,   default=7777)
    parser.add_argument("--skip-sweep",   action="store_true",
                        help="Skip Bayesian sweep; use default hyperparams")
    parser.add_argument("--out",          type=str,
                        default="logs/reward_sweep_results.json")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print_search_space()
    print(f"\nFinal eval: {args.num_rhs} RHS × {args.num_seeds} seeds"
          f" = {args.num_rhs * args.num_seeds} SLRL runs per (reward_type, κ-level)")

    # ── Phase 1: Bayesian sweep ───────────────────────────────────────────
    best_params: dict = {}
    if args.skip_sweep:
        print("\nSkipping sweep — using default hyperparameters.")
        best_params = DEFAULT_PARAMS.copy()
    else:
        print("\n" + "=" * 64)
        print("Phase 1: Bayesian hyperparameter sweep (Optuna TPE)")
        print("=" * 64)
        for rt in REWARD_TYPES:
            best_params[rt] = run_sweep(
                reward_type=rt,
                n_trials=args.n_trials,
                m_max=args.m_max,
                tolerance=args.tolerance,
                gamma_sac=args.gamma_sac,
                holdout_seed=args.holdout_seed,
            )

    print("\nHyperparameters entering final eval:")
    for rt, p in best_params.items():
        print(f"  {rt}: {p}")

    # ── Phase 2: full 5-point eval ────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Phase 2: Full 5-point κ eval")
    print("=" * 64)

    all_results: dict = {}
    for rt in REWARD_TYPES:
        print(f"\n[{rt}]  params={best_params[rt]}")
        all_results[rt] = run_full_eval(
            reward_type=rt,
            params=best_params[rt],
            m_max=args.m_max,
            tolerance=args.tolerance,
            gamma_sac=args.gamma_sac,
            num_rhs=args.num_rhs,
            num_seeds=args.num_seeds,
            holdout_seed=args.holdout_seed,
        )

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "best_params": best_params,
        "results":     all_results,
        "meta": {
            "num_rhs":   args.num_rhs,
            "num_seeds": args.num_seeds,
            "m_max":     args.m_max,
            "tolerance": args.tolerance,
        },
    }
    Path(args.out).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {args.out}")

    # ── Summary table with stddev ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Level':<12} {'kappa':>8}  " +
          "  ".join(f"{'mean±std':>16}" for _ in REWARD_TYPES))
    print(f"{'':>22}  " + "  ".join(f"[{rt}]" + " " * (16 - len(rt) - 2)
                                    for rt in REWARD_TYPES))
    print("-" * 80)
    for i, cfg in enumerate(EVAL_CONFIGS):
        label = cfg["label"]
        kap = all_results[REWARD_TYPES[0]][i]["kappa"]
        row = f"  {label:<10} {kap:>8.0f}  "
        for rt in REWARD_TYPES:
            sl = np.array(all_results[rt][i]["slrl"])
            row += f"  {sl.mean():>8.0f}±{sl.std():>5.0f}"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    main()

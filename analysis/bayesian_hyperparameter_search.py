"""
analysis/bayesian_hyperparameter_search.py

Bayesian hyperparameter search (Optuna TPE) for the GMRES_RL DQN agent's
PBRS reward (env.py). Sweeps the three reward-shaping coefficients over a
SINGLE study aggregated across the full sweep_matrices/ benchmark, optimizing
mean total Arnoldi steps to convergence.

Search space:
    lambda_work        log-uniform [1e-4, 1.0]      per-cycle work penalty
    gamma_shape        uniform     [0.80, 0.999]    PBRS discount + DQN gamma
    convergence_bonus  uniform     [0.0, 100.0]     terminal bonus B

Right-hand side: b = A @ ones for every matrix.

Objective: mean total_arnoldi over the full matrix set (one trial = train+solve
on every matrix; non-convergent solves penalized by max_cycles*m_max).

Run (default sweeps every .tar.gz / .mtx / .mat in matrices/sweep_matrices/):
    python analysis/bayesian_hyperparameter_search.py --n-trials 60
    python analysis/bayesian_hyperparameter_search.py --n-trials 100 --enable-pruner

Requires: optuna (`pip install optuna`).
"""

import argparse
import gzip
import io
import json
import sys
import tarfile
import time
import threading
from pathlib import Path

import numpy as np
import optuna
import torch
from scipy.io import loadmat, mmread
from scipy.sparse import csr_matrix, issparse
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env import GMRESEnv  # noqa: E402

DEFAULT_MATRICES_DIR = REPO_ROOT / "matrices" / "sweep_matrices"


# --------------------------------------------------------------------------- #
# Matrix loading (handles .tar.gz, .mtx[.gz], .mat). b = A @ ones always.
# --------------------------------------------------------------------------- #

def _consistent_rhs(A) -> np.ndarray:
    x_true = np.ones(A.shape[1], dtype=np.float64)
    return np.asarray(A @ x_true, dtype=np.float64).reshape(-1)


def _largest_mtx_member(tar: tarfile.TarFile):
    members = [
        m for m in tar.getmembers()
        if m.name.endswith(".mtx") and not m.name.endswith("_b.mtx")
    ]
    if not members:
        raise ValueError("no .mtx file in archive")
    return max(members, key=lambda m: m.size)


def _load_archive(path: Path):
    with tarfile.open(path) as tar:
        member = _largest_mtx_member(tar)
        with tar.extractfile(member) as fh:
            return mmread(io.BytesIO(fh.read()))


def _load_mtx(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as fh:
        return mmread(io.BytesIO(fh.read()))


def _load_mat(path: Path):
    data = loadmat(path)
    problem = data["Problem"]
    A = problem["A"][0, 0]
    return A


def load_matrix(path: Path):
    """Load A from .tar.gz, .mtx[.gz], or .mat. Returns (name, A_csr, b)."""
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".mtx.gz", ".mtx", ".mat"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    if path.name.endswith(".tar.gz") or path.name.endswith(".tgz"):
        A = _load_archive(path)
    elif path.name.endswith(".mat"):
        A = _load_mat(path)
    else:
        A = _load_mtx(path)

    if not issparse(A):
        A = csr_matrix(A)
    A = csr_matrix(A.astype(np.float64))
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{path.name}: not square ({A.shape})")
    b = _consistent_rhs(A)
    return name, A, b


def discover_matrices(matrices_dir: Path) -> list[Path]:
    """Find all matrix-bearing files at the top level of matrices_dir."""
    if not matrices_dir.exists():
        raise FileNotFoundError(f"matrices dir not found: {matrices_dir}")
    paths = []
    for pat in ("*.tar.gz", "*.tgz", "*.mtx", "*.mtx.gz", "*.mat"):
        paths.extend(sorted(matrices_dir.glob(pat)))
    if not paths:
        raise FileNotFoundError(f"no matrices found in {matrices_dir}")
    return paths


# --------------------------------------------------------------------------- #
# Single-solve evaluation
# --------------------------------------------------------------------------- #


class _ProgressSlots:
    """Simple slot allocator so concurrently running trials use stable tqdm rows."""

    def __init__(self, n_slots: int, start_position: int = 1):
        self._available = list(range(start_position, start_position + max(1, int(n_slots))))
        self._cond = threading.Condition()

    def acquire(self) -> int:
        with self._cond:
            while not self._available:
                self._cond.wait()
            return self._available.pop(0)

    def release(self, position: int) -> None:
        with self._cond:
            self._available.append(position)
            self._available.sort()
            self._cond.notify()

class _StopOnDone(BaseCallback):
    def __init__(self):
        super().__init__()
        self.relative_residuals = []
        self.ms = []
        self._done = False

    def _on_step(self):
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", [False]),
        ):
            if "relative_residual_norm" in info:
                self.relative_residuals.append(float(info["relative_residual_norm"]))
            if "current_m" in info:
                self.ms.append(int(info["current_m"]))
            if done:
                self._done = True
        return not self._done


def evaluate_solve(A, b, hp: dict, args, seed: int) -> dict:
    """Run one DQN training/solve under the given hyperparameters."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = GMRESEnv(
        A=A,
        b=b,
        m_max=args.m_max,
        tolerance=args.tolerance,
        max_cycles=args.max_cycles,
        history_length=args.history_length,
        lambda_work=hp["lambda_work"],
        gamma_shape=hp["gamma_shape"],
        convergence_bonus=hp["convergence_bonus"],
    )
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=hp["gamma_shape"],          # tied to the PBRS discount
        train_freq=1,
        gradient_steps=1,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
        seed=seed,
        device=args.device,
    )

    logger = _StopOnDone()
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.max_cycles, callback=logger, progress_bar=False)
    elapsed = time.perf_counter() - t0

    rel = np.asarray(logger.relative_residuals, dtype=np.float64)
    ms = np.asarray(logger.ms, dtype=np.int64)
    converged = bool(np.any(rel < args.tolerance))
    if converged:
        idx = int(np.argmax(rel < args.tolerance)) + 1
        total_arnoldi = int(ms[:idx].sum())
        cycles = idx
    else:
        total_arnoldi = int(ms.sum()) if ms.size else 0
        cycles = args.max_cycles
    return {
        "converged": converged,
        "total_arnoldi": total_arnoldi,
        "cycles": cycles,
        "elapsed_seconds": float(elapsed),
        "final_relative_residual": float(rel[-1]) if rel.size else float("inf"),
        "mean_m": float(ms.mean()) if ms.size else float("nan"),
    }


def run_score(run: dict, args) -> float:
    """Per-(matrix, seed) score: total_arnoldi if converged, else penalized."""
    if run["converged"]:
        return float(run["total_arnoldi"])
    return float(args.max_cycles * args.m_max + run["total_arnoldi"])


# --------------------------------------------------------------------------- #
# Optuna objective: aggregate over the whole matrix set
# --------------------------------------------------------------------------- #

def make_objective(matrices, args, progress_slots: _ProgressSlots | None = None):
    """One trial = train+solve on every matrix; objective = mean run score."""
    def objective(trial: optuna.Trial) -> float:
        hp = {
            "lambda_work":       trial.suggest_float("lambda_work", 1e-4, 1.0, log=True),
            "gamma_shape":       trial.suggest_float("gamma_shape", 0.80, 0.999),
            "convergence_bonus": trial.suggest_float("convergence_bonus", 0.0, 100.0),
        }

        per_matrix: dict[str, dict] = {}
        running_scores: list[float] = []
        report_step = 0

        position = progress_slots.acquire() if progress_slots is not None else 1
        pbar = tqdm(
            total=len(matrices) * args.n_seeds,
            desc=f"trial {trial.number:>3d}",
            leave=False,
            position=position,
            dynamic_ncols=True,
        )
        try:
            for matrix_idx, (name, A, b) in enumerate(matrices, start=1):
                seed_runs = []
                seed_scores = []
                for s in range(args.n_seeds):
                    seed = args.seed + s
                    run = evaluate_solve(A, b, hp, args, seed)
                    seed_runs.append(run)
                    score = run_score(run, args)
                    seed_scores.append(score)
                    running_scores.append(score)
                    pbar.update(1)
                    pbar.set_postfix({
                        "matrix": f"{matrix_idx}/{len(matrices)}:{name[:14]}",
                        "seed": f"{s + 1}/{args.n_seeds}",
                        "running_mean": f"{np.mean(running_scores):.0f}",
                        "conv": f"{sum(1 for r in seed_runs if r['converged'])}/{len(seed_runs)}",
                    })
                    trial.report(float(np.mean(running_scores)), report_step)
                    report_step += 1
                    if trial.should_prune():
                        trial.set_user_attr("per_matrix", per_matrix)
                        raise optuna.TrialPruned()
                per_matrix[name] = {
                    "converged_rate": float(np.mean([r["converged"] for r in seed_runs])),
                    "total_arnoldi_mean": float(np.mean([r["total_arnoldi"] for r in seed_runs])),
                    "total_arnoldi_std":  float(np.std([r["total_arnoldi"] for r in seed_runs])),
                    "score_mean": float(np.mean(seed_scores)),
                    "elapsed_seconds_mean": float(np.mean([r["elapsed_seconds"] for r in seed_runs])),
                    "per_seed": seed_runs,
                }

            trial.set_user_attr("per_matrix", per_matrix)
            trial.set_user_attr("converged_matrix_count",
                                int(sum(1 for v in per_matrix.values() if v["converged_rate"] >= 0.5)))
            trial.set_user_attr("total_arnoldi_sum",
                                float(sum(v["total_arnoldi_mean"] for v in per_matrix.values())))
            return float(np.mean(running_scores))
        finally:
            pbar.close()
            if progress_slots is not None:
                progress_slots.release(position)

    return objective


# --------------------------------------------------------------------------- #
# JSON export
# --------------------------------------------------------------------------- #

def _trial_to_dict(t: optuna.trial.FrozenTrial) -> dict:
    return {
        "number": t.number,
        "state": t.state.name,
        "value": t.value,
        "params": t.params,
        "user_attrs": {
            "converged_matrix_count": t.user_attrs.get("converged_matrix_count"),
            "total_arnoldi_sum": t.user_attrs.get("total_arnoldi_sum"),
            "per_matrix": t.user_attrs.get("per_matrix", {}),
        },
        "intermediate_values": dict(t.intermediate_values),
        "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
        "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
    }


def export_results(study: optuna.Study, matrix_names: list[str], args, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = study.best_trial if completed else None
    payload = {
        "meta": {
            "study_name": study.study_name,
            "matrices": matrix_names,
            "n_matrices": len(matrix_names),
            "n_trials_total": len(study.trials),
            "n_trials_completed": len(completed),
            "rhs": "b = A @ ones",
            "search_space": {
                "lambda_work":       {"low": 1e-4, "high": 1.0,   "log": True},
                "gamma_shape":       {"low": 0.80, "high": 0.999, "log": False},
                "convergence_bonus": {"low": 0.0,  "high": 100.0, "log": False},
            },
            "fixed": {
                "m_max": args.m_max,
                "tolerance": args.tolerance,
                "max_cycles": args.max_cycles,
                "history_length": args.history_length,
                "seed": args.seed,
                "n_seeds": args.n_seeds,
                "device": args.device,
                "learning_rate": args.learning_rate,
                "buffer_size": args.buffer_size,
                "learning_starts": args.learning_starts,
                "batch_size": args.batch_size,
                "target_update_interval": args.target_update_interval,
                "exploration_fraction": args.exploration_fraction,
                "exploration_final_eps": args.exploration_final_eps,
            },
            "objective": "mean total_arnoldi over (matrices x seeds); non-convergent penalized by max_cycles*m_max",
            "sampler": "TPE (multivariate)",
        },
        "best_trial": (
            None if best is None else {
                "number": best.number,
                "value": best.value,
                "params": best.params,
                "user_attrs": {
                    "converged_matrix_count": best.user_attrs.get("converged_matrix_count"),
                    "total_arnoldi_sum": best.user_attrs.get("total_arnoldi_sum"),
                    "per_matrix": best.user_attrs.get("per_matrix", {}),
                },
            }
        ),
        "trials": [_trial_to_dict(t) for t in study.trials],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices-dir", type=str, default=str(DEFAULT_MATRICES_DIR))
    parser.add_argument("--matrix-paths", nargs="+", default=None,
                        help="Optional explicit list of matrix files (overrides directory glob).")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel Optuna workers within this process. "
                             "Use modest values on CPU-bound workloads.")
    parser.add_argument("--show-live-progress", action="store_true",
                        help="Show one live tqdm row per active trial plus an overall bar.")
    parser.add_argument("--n-startup-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Seeds per (trial, matrix). 1 keeps the search cheap.")
    parser.add_argument("--m-max", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-cycles", type=int, default=5000)
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    # Fixed (not searched) DQN/training hyperparameters.
    # learning-rate: 1e-3 is a balanced default for SB3 DQN with a small MLP and
    # short-horizon single-life RL (above the 1e-4 stock default to fit the
    # ~hundreds-to-thousands-of-cycles budget; below 3e-3 to avoid instability).
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    parser.add_argument("--exploration-final-eps", type=float, default=0.01)
    parser.add_argument("--study-name", type=str, default="gmres_dqn_pbrs_sweep")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optional Optuna storage URL for resumable studies "
                             "(e.g. sqlite:///analysis/results/pbrs_sweep.db).")
    parser.add_argument("--enable-pruner", action="store_true",
                        help="Enable MedianPruner across the matrix loop within a trial.")
    parser.add_argument("--out", type=str,
                        default="analysis/results/pbrs_sweep.json")
    parser.add_argument("--export-every", type=int, default=5,
                        help="Snapshot the JSON every N completed trials.")
    args = parser.parse_args()

    matrices_dir = Path(args.matrices_dir).expanduser().resolve()
    if args.matrix_paths is not None:
        paths = [Path(p).expanduser().resolve() for p in args.matrix_paths]
    else:
        paths = discover_matrices(matrices_dir)

    print(f"Loading {len(paths)} matrices from {matrices_dir}")
    matrices = []
    for p in paths:
        name, A, b = load_matrix(p)
        print(f"  {name:24s}  n={A.shape[0]:7d}  nnz={A.nnz:9d}  ||b||={float(np.linalg.norm(b)):.3e}")
        matrices.append((name, A, b))
    matrix_names = [m[0] for m in matrices]

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=args.n_startup_trials,
        seed=args.seed,
        multivariate=True,
        group=True,
    )
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=1)
        if args.enable_pruner else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    out_path = Path(args.out)

    overall_pbar = None
    if args.show_live_progress:
        overall_pbar = tqdm(
            total=args.n_trials,
            desc="study",
            position=0,
            dynamic_ncols=True,
        )

    def _snapshot_callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial):
        if overall_pbar is not None:
            overall_pbar.update(1)
        completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        if completed and completed % args.export_every == 0:
            export_results(study, matrix_names, args, out_path)

    progress_slots = _ProgressSlots(args.n_jobs) if args.show_live_progress else None

    try:
        study.optimize(
            make_objective(matrices, args, progress_slots=progress_slots),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            callbacks=[_snapshot_callback],
            show_progress_bar=not args.show_live_progress,
            gc_after_trial=True,
        )
    finally:
        if overall_pbar is not None:
            overall_pbar.close()

    export_results(study, matrix_names, args, out_path)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = study.best_trial
        print("\nBest trial:")
        print(f"  number: {best.number}")
        print(f"  value (mean total_arnoldi): {best.value:.2f}")
        print(f"  converged matrices: {best.user_attrs.get('converged_matrix_count')}/{len(matrix_names)}")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
    else:
        print("\nNo completed trials.")
    print(f"\nWrote {len(study.trials)} trials to {out_path}")


if __name__ == "__main__":
    main()

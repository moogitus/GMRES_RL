"""
analysis/bayesian_hyperparameter_search.py

Bayesian hyperparameter search (Optuna TPE) for the GMRES_RL DQN agent on the
cavity05-08 DRIVCAV matrices. Runs TWO INDEPENDENT studies per matrix in a
single invocation:

    1. objective = "arnoldi" : minimize mean total_arnoldi-to-convergence
    2. objective = "time"    : minimize mean wall-clock seconds-to-convergence

Both studies share the same search space; reporting both lets us compare which
hyperparameters minimize numerical work vs Python/torch overhead.

Search space (wide, identical across matrices and objectives):
    learning_rate           log-uniform [1e-5, 1e-2]
    lambda_work (reward)    log-uniform [1e-4, 1.0]
    gamma_shape (reward)    uniform     [0.80, 0.999]
    convergence_bonus B     uniform     [0.0, 100.0]
    history_length k        int         [1, 20]

Right-hand side: b = A @ ones (consistent RHS) for every matrix; we ignore the
Problem.b shipped with the .mat file.

Non-convergent runs are penalized so the optimizer always prefers a convergent
trial over a non-convergent one (penalty is objective-specific).

Run (one terminal command does all 4 matrices x 2 objectives):
    python analysis/bayesian_hyperparameter_search.py --n-trials 60
    python analysis/bayesian_hyperparameter_search.py \\
        --matrices cavity05 cavity07 --n-trials 100 --enable-pruner
    python analysis/bayesian_hyperparameter_search.py --objectives arnoldi   # only one

Requires: optuna (`pip install optuna`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from scipy.io import loadmat
from scipy.sparse import csr_matrix, issparse
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env import GMRESEnv  # noqa: E402

DEFAULT_MATRICES = ["cavity05", "cavity06", "cavity07", "cavity08"]
OBJECTIVES = ("arnoldi", "time")


# --------------------------------------------------------------------------- #
# Matrix loading
# --------------------------------------------------------------------------- #

def load_matrix(name: str, matrices_dir: Path):
    """Load A from a SuiteSparse .mat file; build b = A @ ones."""
    path = matrices_dir / f"{name}.mat"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    data = loadmat(path)
    problem = data["Problem"]
    A = problem["A"][0, 0]
    if not issparse(A):
        A = csr_matrix(A)
    A = csr_matrix(A.astype(np.float64))
    x_true = np.ones(A.shape[1], dtype=np.float64)
    b = np.asarray(A @ x_true, dtype=np.float64).reshape(-1)
    return A, b


# --------------------------------------------------------------------------- #
# Single-solve evaluation
# --------------------------------------------------------------------------- #

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
        history_length=hp["history_length"],
        lambda_work=hp["lambda_work"],
        gamma_shape=hp["gamma_shape"],
        convergence_bonus=hp["convergence_bonus"],
    )
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=hp["learning_rate"],
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=hp["gamma_shape"],
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


def trial_score(run: dict, objective: str, args) -> float:
    """Per-run scalar (lower is better), dispatched on the active objective.

    arnoldi: total_arnoldi (convergent) or max_cycles*m_max + total_arnoldi
             (non-convergent, strictly worse than any convergent outcome).
    time   : elapsed_seconds (convergent) or args.time_penalty + elapsed_seconds
             (non-convergent, strictly worse than any convergent outcome).
    """
    if objective == "arnoldi":
        if run["converged"]:
            return float(run["total_arnoldi"])
        return float(args.max_cycles * args.m_max + run["total_arnoldi"])
    if objective == "time":
        if run["converged"]:
            return float(run["elapsed_seconds"])
        return float(args.time_penalty + run["elapsed_seconds"])
    raise ValueError(f"unknown objective: {objective}")


# --------------------------------------------------------------------------- #
# Optuna objective (one matrix)
# --------------------------------------------------------------------------- #

def make_objective(matrix_name: str, objective_name: str, A, b, args):
    def objective(trial: optuna.Trial) -> float:
        hp = {
            "learning_rate":     trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "lambda_work":       trial.suggest_float("lambda_work", 1e-4, 1.0, log=True),
            "gamma_shape":       trial.suggest_float("gamma_shape", 0.80, 0.999),
            "convergence_bonus": trial.suggest_float("convergence_bonus", 0.0, 100.0),
            "history_length":    trial.suggest_int("history_length", 1, 20),
        }

        seed_runs = []
        scores = []
        for s in range(args.n_seeds):
            seed = args.seed + s
            run = evaluate_solve(A, b, hp, args, seed)
            seed_runs.append(run)
            scores.append(trial_score(run, objective_name, args))
            trial.report(float(np.mean(scores)), s)
            if trial.should_prune():
                trial.set_user_attr("per_seed", seed_runs)
                trial.set_user_attr("matrix", matrix_name)
                trial.set_user_attr("objective", objective_name)
                raise optuna.TrialPruned()

        trial.set_user_attr("per_seed", seed_runs)
        trial.set_user_attr("matrix", matrix_name)
        trial.set_user_attr("objective", objective_name)
        trial.set_user_attr("converged_rate", float(np.mean([r["converged"] for r in seed_runs])))
        trial.set_user_attr("total_arnoldi_mean", float(np.mean([r["total_arnoldi"] for r in seed_runs])))
        trial.set_user_attr("total_arnoldi_std", float(np.std([r["total_arnoldi"] for r in seed_runs])))
        trial.set_user_attr("elapsed_seconds_mean", float(np.mean([r["elapsed_seconds"] for r in seed_runs])))
        trial.set_user_attr("elapsed_seconds_std", float(np.std([r["elapsed_seconds"] for r in seed_runs])))
        return float(np.mean(scores))

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
            "matrix": t.user_attrs.get("matrix"),
            "objective": t.user_attrs.get("objective"),
            "converged_rate": t.user_attrs.get("converged_rate"),
            "total_arnoldi_mean": t.user_attrs.get("total_arnoldi_mean"),
            "total_arnoldi_std": t.user_attrs.get("total_arnoldi_std"),
            "elapsed_seconds_mean": t.user_attrs.get("elapsed_seconds_mean"),
            "elapsed_seconds_std": t.user_attrs.get("elapsed_seconds_std"),
            "per_seed": t.user_attrs.get("per_seed", []),
        },
        "intermediate_values": dict(t.intermediate_values),
        "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
        "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
    }


def _study_summary(study: optuna.Study) -> dict:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = study.best_trial if completed else None
    return {
        "study_name": study.study_name,
        "n_trials_total": len(study.trials),
        "n_trials_completed": len(completed),
        "best_trial": (
            None if best is None else {
                "number": best.number,
                "value": best.value,
                "params": best.params,
                "user_attrs": {
                    "converged_rate": best.user_attrs.get("converged_rate"),
                    "total_arnoldi_mean": best.user_attrs.get("total_arnoldi_mean"),
                    "total_arnoldi_std": best.user_attrs.get("total_arnoldi_std"),
                    "elapsed_seconds_mean": best.user_attrs.get("elapsed_seconds_mean"),
                    "elapsed_seconds_std": best.user_attrs.get("elapsed_seconds_std"),
                },
            }
        ),
        "trials": [_trial_to_dict(t) for t in study.trials],
    }


def export_results(studies: dict[str, dict[str, optuna.Study]], args, out_path: Path):
    """Export results nested as studies[objective][matrix]."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _best_block(s: optuna.Study) -> dict | None:
        completed = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            return None
        b = s.best_trial
        return {
            "value": s.best_value,
            "params": s.best_params,
            "user_attrs": {
                "converged_rate": b.user_attrs.get("converged_rate"),
                "total_arnoldi_mean": b.user_attrs.get("total_arnoldi_mean"),
                "total_arnoldi_std": b.user_attrs.get("total_arnoldi_std"),
                "elapsed_seconds_mean": b.user_attrs.get("elapsed_seconds_mean"),
                "elapsed_seconds_std": b.user_attrs.get("elapsed_seconds_std"),
            },
        }

    payload = {
        "meta": {
            "matrices": sorted({m for obj_studies in studies.values() for m in obj_studies}),
            "objectives": list(studies.keys()),
            "rhs": "b = A @ ones",
            "search_space": {
                "learning_rate":     {"low": 1e-5, "high": 1e-2, "log": True},
                "lambda_work":       {"low": 1e-4, "high": 1.0,   "log": True},
                "gamma_shape":       {"low": 0.80, "high": 0.999, "log": False},
                "convergence_bonus": {"low": 0.0,  "high": 100.0, "log": False},
                "history_length":    {"low": 1,    "high": 20,    "type": "int"},
            },
            "fixed": {
                "m_max": args.m_max,
                "tolerance": args.tolerance,
                "max_cycles": args.max_cycles,
                "seed": args.seed,
                "n_seeds": args.n_seeds,
                "device": args.device,
                "buffer_size": args.buffer_size,
                "learning_starts": args.learning_starts,
                "batch_size": args.batch_size,
                "target_update_interval": args.target_update_interval,
                "exploration_fraction": args.exploration_fraction,
                "exploration_final_eps": args.exploration_final_eps,
                "time_penalty": args.time_penalty,
            },
            "objective_definitions": {
                "arnoldi": "mean total_arnoldi over seeds; non-convergent penalized by max_cycles*m_max",
                "time":    "mean elapsed_seconds over seeds; non-convergent penalized by time_penalty",
            },
            "sampler": "TPE (multivariate)",
        },
        "best_per_matrix": {
            objective: {name: _best_block(s) for name, s in obj_studies.items()}
            for objective, obj_studies in studies.items()
        },
        "studies": {
            objective: {name: _study_summary(s) for name, s in obj_studies.items()}
            for objective, obj_studies in studies.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices-dir", type=str, default="matrices")
    parser.add_argument("--matrices", nargs="+", default=DEFAULT_MATRICES,
                        help="Subset of matrix names (one independent study per name).")
    parser.add_argument("--n-trials", type=int, default=60,
                        help="Trials per matrix (each matrix gets its own Optuna study).")
    parser.add_argument("--n-startup-trials", type=int, default=15)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Seeds per trial; raise for noise reduction at proportional cost.")
    parser.add_argument("--m-max", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-cycles", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    parser.add_argument("--exploration-final-eps", type=float, default=0.01)
    parser.add_argument("--objectives", nargs="+", default=list(OBJECTIVES),
                        choices=list(OBJECTIVES),
                        help="Which objectives to run; default runs both Arnoldi and wall-clock.")
    parser.add_argument("--time-penalty", type=float, default=1.0e6,
                        help="Seconds added to non-convergent runs under the 'time' objective.")
    parser.add_argument("--study-prefix", type=str, default="gmres_dqn")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optional Optuna storage URL (e.g. sqlite:///analysis/results/cavity_search.db) for resumable studies.")
    parser.add_argument("--enable-pruner", action="store_true",
                        help="Enable Optuna MedianPruner across the seed loop within a trial.")
    parser.add_argument("--out", type=str,
                        default="analysis/results/cavity_bayes_search.json")
    parser.add_argument("--export-every", type=int, default=10,
                        help="Re-export the JSON snapshot every N completed trials so partial progress is never lost.")
    args = parser.parse_args()

    matrices_dir = Path(args.matrices_dir).resolve()
    print(f"Loading {len(args.matrices)} matrices from {matrices_dir}")
    matrices = []
    for name in args.matrices:
        A, b = load_matrix(name, matrices_dir)
        print(f"  {name}: n={A.shape[0]}, nnz={A.nnz}, ||b||={float(np.linalg.norm(b)):.3e}")
        matrices.append((name, A, b))

    out_path = Path(args.out)
    studies: dict[str, dict[str, optuna.Study]] = {obj: {} for obj in args.objectives}

    for objective_name in args.objectives:
        for name, A, b in matrices:
            print(f"\n{'='*72}\nStudy: matrix={name}  objective={objective_name}\n{'='*72}")
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
                study_name=f"{args.study_prefix}_{name}_{objective_name}",
                storage=args.storage,
                sampler=sampler,
                pruner=pruner,
                direction="minimize",
                load_if_exists=True,
            )
            studies[objective_name][name] = study

            def _snapshot_callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial):
                completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
                if completed and completed % args.export_every == 0:
                    export_results(studies, args, out_path)

            study.optimize(
                make_objective(name, objective_name, A, b, args),
                n_trials=args.n_trials,
                callbacks=[_snapshot_callback],
                show_progress_bar=True,
                gc_after_trial=True,
            )

            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed:
                best = study.best_trial
                print(f"\n  best ({objective_name}) for {name}: value={best.value:.4f}")
                print(f"    arnoldi_mean={best.user_attrs.get('total_arnoldi_mean')}  "
                      f"time_mean={best.user_attrs.get('elapsed_seconds_mean'):.3f}s  "
                      f"converged_rate={best.user_attrs.get('converged_rate')}")
                for k, v in best.params.items():
                    print(f"    {k}: {v}")

            export_results(studies, args, out_path)

    export_results(studies, args, out_path)
    print(f"\nWrote per-(objective, matrix) studies to {out_path}")
    print("\nSummary (best per matrix per objective):")
    for objective_name, obj_studies in studies.items():
        print(f"  [{objective_name}]")
        for name, s in obj_studies.items():
            completed = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed:
                print(f"    {name}: no completed trials")
                continue
            b = s.best_trial
            print(f"    {name}: value={s.best_value:.4f}  "
                  f"arnoldi={b.user_attrs.get('total_arnoldi_mean')}  "
                  f"time={b.user_attrs.get('elapsed_seconds_mean'):.3f}s")


if __name__ == "__main__":
    main()

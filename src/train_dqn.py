"""
Single-life DQN training driver for the GMRES(m) restart controller (§3, §4).
A fresh DQN agent is trained online during one solve per matrix; no
pre-training, no replay across matrices.

Defaults reproduce the hyperparameter configuration in §7.2 (Table 5):
γ = 0.925, λ_work = 0.01, B = 9, lr = 3e-3, replay buffer 10⁴, batch 32,
hard target updates every 100 grad steps, ε-greedy 1.0 → 0.01 over the
first 10% of the cycle budget, two-layer ReLU Q-network [128, 128].

RHS is the consistent one b = A·1 (§4.2). Inputs are SuiteSparse archives
or Matrix Market files (.tar.gz / .mtx[.gz]).

Built on stable-baselines3 (https://github.com/DLR-RM/stable-baselines3).

Examples:
    python train_dqn.py
    python train_dqn.py --matrices-dir matrices/full_benchmark
    python train_dqn.py --matrices-dir matrices/full_benchmark --matrix-names 1138_bus ct20stif
"""

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import mmread
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from env import GMRESEnv
from utils import (
    _consistent_rhs,
    _find_local_archive,
    _first_existing,
    _largest_matrix_member,
    _load_mtx_file,
    _recursive_candidates,
    _validate_problem,
    summarise_runs,
)


def _strip_matrix_suffix(name: str) -> str:
    for suffix in (".tar.gz", ".tgz", ".mtx.gz", ".mtx"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def discover_matrix_names(matrices_dir: Path) -> list[str]:
    matrices_dir = matrices_dir.expanduser().resolve()
    if not matrices_dir.exists():
        raise FileNotFoundError(
            f"Matrix directory not found: {matrices_dir}. "
            "Expected a local folder containing matrix archives or Matrix Market files."
        )

    names = set()
    for pattern in ("*.tar.gz", "*.tgz", "*.mtx", "*.mtx.gz"):
        for path in matrices_dir.rglob(pattern):
            name = _strip_matrix_suffix(path.name)
            if name.endswith("_rhs1") or name.endswith("_b"):
                continue
            names.add(name)

    if not names:
        raise FileNotFoundError(
            f"No matrix archives or Matrix Market files found under {matrices_dir}."
        )
    return sorted(names)


def _find_local_problem_files(name: str, matrices_dir: Path):
    matrix_patterns = [f"{name}.mtx", f"{name}.mtx.gz"]

    matrix_path = _first_existing([
        matrices_dir / f"{name}.mtx.gz",
        matrices_dir / f"{name}.mtx",
        matrices_dir / name / f"{name}.mtx.gz",
        matrices_dir / name / f"{name}.mtx",
    ])

    if matrix_path is None:
        matrix_matches = _recursive_candidates(matrices_dir, matrix_patterns)
        matrix_path = matrix_matches[0] if matrix_matches else None

    return matrix_path


def load_problem(config: dict, matrices_dir: Path):
    matrices_dir = matrices_dir.expanduser().resolve()
    if not matrices_dir.exists():
        raise FileNotFoundError(
            f"Matrix directory not found: {matrices_dir}. "
            "Expected a local folder containing matrix archives or Matrix Market files."
        )

    tar_path = _find_local_archive(config["name"], matrices_dir)
    if tar_path is not None:
        print(f"  Using local archive {tar_path}")
        with tarfile.open(tar_path) as tar:
            matrix_member = _largest_matrix_member(tar)
            with tar.extractfile(matrix_member) as handle:
                A = mmread(io.BytesIO(handle.read()))
            b = _consistent_rhs(A)
        return _validate_problem(config["name"], A, b)

    matrix_path = _find_local_problem_files(config["name"], matrices_dir)
    if matrix_path is None:
        raise FileNotFoundError(
            f"Could not find local matrix files for {config['name']} in {matrices_dir}. "
            "Expected either a .tar.gz archive or a matrix .mtx/.mtx.gz file."
        )

    print(f"  Using local matrix {matrix_path}")
    A = _load_mtx_file(matrix_path)
    b = _consistent_rhs(A)
    return _validate_problem(config["name"], A, b)


# sb3 callback that records absolute and relative residual norms and
# chosen m at every env step, halting the rollout when the env signals done
class _StopOnDone(BaseCallback):
    def __init__(self):
        super().__init__()
        self.residuals = []
        self.relative_residuals = []
        self.ms = []
        self._done = False

    def _on_step(self):
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", [False]),
        ):
            if "residual_norm" in info:
                self.residuals.append(float(info["residual_norm"]))
            if "relative_residual_norm" in info:
                self.relative_residuals.append(float(info["relative_residual_norm"]))
            if "current_m" in info:
                self.ms.append(int(info["current_m"]))
            if done:
                self._done = True
        return not self._done


def run_dqn(A, b, args, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = GMRESEnv(
        A=A,
        b=b,
        m_max=args.m_max,
        tolerance=args.tolerance,
        max_cycles=args.max_cycles,
        history_length=args.history_length,
        lambda_work=args.lambda_work,
        gamma_shape=args.gamma,
        convergence_bonus=args.convergence_bonus,
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
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

    residuals = np.array(logger.residuals, dtype=np.float64)
    relative_residuals = np.array(logger.relative_residuals, dtype=np.float64)
    ms = np.array(logger.ms, dtype=np.int64)
    if np.any(relative_residuals < args.tolerance):
        idx = int(np.argmax(relative_residuals < args.tolerance)) + 1
        used_ms = ms[:idx]
        return {
            "converged": True,
            "cycles_to_tol": idx,
            "total_arnoldi": int(used_ms.sum()),
            "mean_m": float(used_ms.mean()),
            "elapsed_seconds": float(elapsed),
            "final_residual_norm": float(residuals[idx - 1]),
            "final_relative_residual_norm": float(relative_residuals[idx - 1]),
        }

    return {
        "converged": False,
        "cycles_to_tol": args.max_cycles,
        "total_arnoldi": int(ms.sum()) if len(ms) else 0,
        "mean_m": float(ms.mean()) if len(ms) else float("nan"),
        "elapsed_seconds": float(elapsed),
        "final_residual_norm": float(residuals[-1]) if len(residuals) else float("inf"),
        "final_relative_residual_norm": float(relative_residuals[-1]) if len(relative_residuals) else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices-dir", type=str, default="matrices/full_benchmark")
    parser.add_argument("--matrix-names", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--m-max", type=int, default=20)
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-cycles", type=int, default=10000)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.925)
    parser.add_argument("--lambda-work", type=float, default=0.01)
    parser.add_argument("--convergence-bonus", type=float, default=9)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    parser.add_argument("--exploration-final-eps", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="results/train_dqn_collection.json")
    args = parser.parse_args()

    matrices_root = Path(args.matrices_dir)
    if args.matrix_names is not None:
        configs = [{"name": name} for name in args.matrix_names]
    else:
        configs = [{"name": name} for name in discover_matrix_names(matrices_root)]
    if args.limit is not None:
        configs = configs[:args.limit]
    if not configs:
        raise ValueError("No matrices selected.")

    results = {}

    print(f"Selected {len(configs)} matrices.")
    print(
        f"matrices_dir={matrices_root}, history_length={args.history_length}, m_max={args.m_max}, "
        f"tolerance={args.tolerance}, max_cycles={args.max_cycles}"
    )

    for idx, config in enumerate(configs, start=1):
        print(f"\n{'='*72}")
        print(f"[{idx}/{len(configs)}] {config['name']}")
        print(f"{'='*72}")
        A, b = load_problem(config, matrices_root)
        print(f"n={A.shape[0]}, nnz={A.nnz}")

        runs = []
        for offset in range(args.num_seeds):
            seed = args.base_seed + offset
            run = run_dqn(A, b, args, seed)
            runs.append(run)
            print(
                f"  seed={seed:02d}: arnoldi={run['total_arnoldi']:8d}  "
                f"time={run['elapsed_seconds']:8.2f}s  mean_m={run['mean_m']:5.2f}  "
                f"final_relres={run['final_relative_residual_norm']:.3e}  conv={run['converged']}"
            )

        results[config["name"]] = {
            "source": "local",
            "runs": runs,
            "summary": summarise_runs(runs),
        }

    payload = {
        "meta": {
            "selected_count": len(configs),
            "matrices_dir": str(matrices_root.resolve()),
            "history_length": args.history_length,
            "m_max": args.m_max,
            "tolerance": args.tolerance,
            "max_cycles": args.max_cycles,
            "num_seeds": args.num_seeds,
            "base_seed": args.base_seed,
            "device": args.device,
            "dqn_hyperparams": {
                "learning_rate": args.learning_rate,
                "buffer_size": args.buffer_size,
                "learning_starts": args.learning_starts,
                "batch_size": args.batch_size,
                "gamma": args.gamma,
                "target_update_interval": args.target_update_interval,
                "exploration_fraction": args.exploration_fraction,
                "exploration_final_eps": args.exploration_final_eps,
                "net_arch": [128, 128],
            },
        },
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()

"""
train_dqn.py

Simple DQN training/evaluation script for the canonical-RHS matrix collection
from the paper, excluding garon2 and xenon2.

Assumptions:
  - use the matrix's packaged RHS from a local matrices folder
  - use the plain residual-norm tolerance from the environment
  - use the standalone GMRESEnv in env.py

Run:
    python train_dqn.py
    python train_dqn.py --matrix-names e20r0000
    python train_dqn.py --matrices-dir matrices
"""

import argparse
import gzip
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import mmread
from scipy.sparse import csr_matrix
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from env import GMRESEnv


PAPER_COLLECTION = [
    {"name": "watt_1"},
    {"name": "steam2"},
    {"name": "fs_183_4"},
    {"name": "fs_183_6"},
    {"name": "cage6"},
    {"name": "steam3"},
    {"name": "pivtol"},
    {"name": "cage5"},
    {"name": "fs_183_3"},
    {"name": "pores_1"},
    {"name": "rajat11"},
    {"name": "bfwa62"},
    {"name": "circuit_2"},
    {"name": "orsreg_1"},
    {"name": "orsirr_1"},
    {"name": "sherman4"},
    {"name": "wang2"},
    {"name": "pde2961"},
    {"name": "bwm200"},
    {"name": "cfd1"},
    {"name": "lns_131"},
    {"name": "tub100"},
    {"name": "gre_115"},
    {"name": "gre_185"},
    {"name": "lop163"},
    {"name": "odepa400"},
    {"name": "olm100"},
    {"name": "rdb200"},
    {"name": "saylr1"},
    {"name": "young3c"},
    {"name": "1138_bus"},
    {"name": "finance256"},
    {"name": "ct20stif"},
    {"name": "olesnik0"},
    {"name": "ex19"},
    {"name": "crankseg_1"},
]


def _largest_matrix_member(tar: tarfile.TarFile):
    members = [
        member for member in tar.getmembers()
        if member.name.endswith(".mtx") and not member.name.endswith("_b.mtx")
    ]
    if not members:
        raise ValueError("No matrix .mtx file found.")
    return max(members, key=lambda member: member.size)


def _rhs_member(tar: tarfile.TarFile):
    members = [member for member in tar.getmembers() if member.name.endswith("_b.mtx")]
    return max(members, key=lambda member: member.size) if members else None


def _flatten_rhs(data) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _load_mtx_file(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as handle:
        return mmread(io.BytesIO(handle.read()))


def _validate_problem(name: str, A, b):
    A = csr_matrix(A.astype(np.float64))
    b = _flatten_rhs(b)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{name}: matrix is not square")
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"{name}: RHS dimension mismatch")
    return A, b


def _consistent_rhs(A) -> np.ndarray:
    x_true = np.ones(A.shape[1], dtype=np.float64)
    return np.asarray(A @ x_true, dtype=np.float64).reshape(-1)


def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _recursive_candidates(root: Path, patterns):
    candidates = []
    seen = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(path)
    return candidates


def _find_local_archive(name: str, matrices_dir: Path):
    direct = _first_existing([
        matrices_dir / f"{name}.tar.gz",
        matrices_dir / f"{name}.tgz",
        matrices_dir / name / f"{name}.tar.gz",
        matrices_dir / name / f"{name}.tgz",
    ])
    if direct is not None:
        return direct

    matches = _recursive_candidates(matrices_dir, [f"{name}.tar.gz", f"{name}.tgz"])
    return matches[0] if matches else None


def _find_local_problem_files(name: str, matrices_dir: Path):
    matrix_patterns = [f"{name}.mtx", f"{name}.mtx.gz"]
    rhs_patterns = [
        f"{name}_rhs1.mtx",
        f"{name}_rhs1.mtx.gz",
        f"{name}_b.mtx",
        f"{name}_b.mtx.gz",
    ]

    matrix_path = _first_existing([
        matrices_dir / f"{name}.mtx.gz",
        matrices_dir / f"{name}.mtx",
        matrices_dir / name / f"{name}.mtx.gz",
        matrices_dir / name / f"{name}.mtx",
    ])
    rhs_path = _first_existing([
        matrices_dir / f"{name}_rhs1.mtx.gz",
        matrices_dir / f"{name}_rhs1.mtx",
        matrices_dir / f"{name}_b.mtx.gz",
        matrices_dir / f"{name}_b.mtx",
        matrices_dir / name / f"{name}_rhs1.mtx.gz",
        matrices_dir / name / f"{name}_rhs1.mtx",
        matrices_dir / name / f"{name}_b.mtx.gz",
        matrices_dir / name / f"{name}_b.mtx",
    ])

    if matrix_path is None:
        matrix_matches = _recursive_candidates(matrices_dir, matrix_patterns)
        matrix_path = matrix_matches[0] if matrix_matches else None
    if rhs_path is None:
        rhs_matches = _recursive_candidates(matrices_dir, rhs_patterns)
        rhs_path = rhs_matches[0] if rhs_matches else None

    return matrix_path, rhs_path


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
            rhs_member = _rhs_member(tar)
            with tar.extractfile(matrix_member) as handle:
                A = mmread(io.BytesIO(handle.read()))
            if rhs_member is not None:
                with tar.extractfile(rhs_member) as handle:
                    b = mmread(io.BytesIO(handle.read()))
            else:
                print(f"  No canonical RHS found for {config['name']}; generating consistent RHS")
                b = _consistent_rhs(A)
        return _validate_problem(config["name"], A, b)

    matrix_path, rhs_path = _find_local_problem_files(config["name"], matrices_dir)
    if matrix_path is None:
        raise FileNotFoundError(
            f"Could not find local matrix files for {config['name']} in {matrices_dir}. "
            "Expected either a .tar.gz archive or a matrix .mtx/.mtx.gz file."
        )

    print(f"  Using local matrix {matrix_path}")
    A = _load_mtx_file(matrix_path)
    if rhs_path is not None:
        print(f"  Using local RHS {rhs_path}")
        b = _load_mtx_file(rhs_path)
    else:
        print(f"  No canonical RHS found for {config['name']}; generating consistent RHS")
        b = _consistent_rhs(A)
    return _validate_problem(config["name"], A, b)


class _StopOnDone(BaseCallback):
    def __init__(self):
        super().__init__()
        self.residuals = []
        self.ms = []
        self._done = False

    def _on_step(self):
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", [False]),
        ):
            if "residual_norm" in info:
                self.residuals.append(float(info["residual_norm"]))
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
    ms = np.array(logger.ms, dtype=np.int64)
    if np.any(residuals < args.tolerance):
        idx = int(np.argmax(residuals < args.tolerance)) + 1
        used_ms = ms[:idx]
        return {
            "converged": True,
            "cycles_to_tol": idx,
            "total_arnoldi": int(used_ms.sum()),
            "mean_m": float(used_ms.mean()),
            "elapsed_seconds": float(elapsed),
            "final_residual_norm": float(residuals[idx - 1]),
        }

    return {
        "converged": False,
        "cycles_to_tol": args.max_cycles,
        "total_arnoldi": int(ms.sum()) if len(ms) else 0,
        "mean_m": float(ms.mean()) if len(ms) else float("nan"),
        "elapsed_seconds": float(elapsed),
        "final_residual_norm": float(residuals[-1]) if len(residuals) else float("inf"),
    }


def summarise_runs(runs):
    def _avg(key):
        return float(np.mean([run[key] for run in runs]))

    def _std(key):
        return float(np.std([run[key] for run in runs]))

    return {
        "convergence_rate": _avg("converged"),
        "arnoldi_mean": _avg("total_arnoldi"),
        "arnoldi_std": _std("total_arnoldi"),
        "time_mean": _avg("elapsed_seconds"),
        "time_std": _std("elapsed_seconds"),
        "cycles_mean": _avg("cycles_to_tol"),
        "final_residual_norm_mean": _avg("final_residual_norm"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices-dir", type=str, default="matrices")
    parser.add_argument("--matrix-names", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--m-max", type=int, default=20)
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.9695)
    parser.add_argument("--lambda-work", type=float, default=0.01239)
    parser.add_argument("--convergence-bonus", type=float, default=22.09)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    parser.add_argument("--exploration-final-eps", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="logs/train_dqn_collection.json")
    args = parser.parse_args()

    configs = PAPER_COLLECTION
    if args.matrix_names is not None:
        wanted = set(args.matrix_names)
        configs = [cfg for cfg in configs if cfg["name"] in wanted]
    if args.limit is not None:
        configs = configs[:args.limit]
    if not configs:
        raise ValueError("No matrices selected.")

    matrices_root = Path(args.matrices_dir)
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
                f"final_res={run['final_residual_norm']:.3e}  conv={run['converged']}"
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

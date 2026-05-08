"""
Single-life SAC training driver for the AK-SLRL continuous-action env
(akslrl_env.AKSLRLEnv). Used as the SAC baseline in §4.1 (Table 2) and
the SAC reward ablation in §7.4 (Table 6). A fresh SAC agent is trained
online during one solve per matrix.

Defaults follow the AK-SLRL paper as restated in §7.2 (Table 5):
γ_SAC = 0.97, replay buffer min(n/2, 20000), automatic entropy tuning.
Stable-baselines3 defaults are used for parameters the paper does not
specify (lr 3e-4, batch 256, τ_polyak 0.005, target update every step,
MLP [256, 256]).

Mirrors train_dqn.py but uses SAC over a continuous action and the
AK-SLRL inverse-residual reward; also supports SuiteSparse .mat files.

Built on stable-baselines3 (https://github.com/DLR-RM/stable-baselines3).

Examples:
    python train_sac.py
    python train_sac.py --matrix-names cavity05
    python train_sac.py --matrices-dir matrices
"""

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, mmread
from scipy.sparse import csr_matrix, issparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from akslrl_env import AKSLRLEnv
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
    {"name": "cavity05"},
    {"name": "cavity06"},
    {"name": "cavity07"},
    {"name": "cavity08"},
]


def _rhs_member(tar: tarfile.TarFile):
    members = [member for member in tar.getmembers() if member.name.endswith("_b.mtx")]
    return max(members, key=lambda member: member.size) if members else None


def _load_mat_file(path: Path):
    """Load a SuiteSparse .mat file. Returns (A, optional_b)."""
    data = loadmat(path)
    problem = data["Problem"]
    fields = problem.dtype.names or ()
    A = problem["A"][0, 0]
    if not issparse(A):
        A = csr_matrix(A)
    b = None
    if "b" in fields:
        b_field = problem["b"][0, 0]
        if b_field.size > 0:
            b = np.asarray(b_field, dtype=np.float64).reshape(-1)
    return A, b


def _find_local_mat(name: str, matrices_dir: Path):
    direct = _first_existing([
        matrices_dir / f"{name}.mat",
        matrices_dir / name / f"{name}.mat",
    ])
    if direct is not None:
        return direct
    matches = _recursive_candidates(matrices_dir, [f"{name}.mat"])
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


def load_problem(config: dict, matrices_dir: Path, force_consistent_rhs: bool = False):
    matrices_dir = matrices_dir.expanduser().resolve()
    if not matrices_dir.exists():
        raise FileNotFoundError(
            f"Matrix directory not found: {matrices_dir}. "
            "Expected a local folder containing matrix archives or Matrix Market files."
        )

    name = config["name"]

    mat_path = _find_local_mat(name, matrices_dir)
    if mat_path is not None:
        print(f"  Using local .mat {mat_path}")
        A, b = _load_mat_file(mat_path)
        if force_consistent_rhs or b is None:
            if not force_consistent_rhs:
                print(f"  No RHS in .mat for {name}; generating consistent RHS")
            else:
                print(f"  Forcing consistent RHS (b = A @ 1) for {name}")
            b = _consistent_rhs(A)
        return _validate_problem(name, A, b)

    tar_path = _find_local_archive(name, matrices_dir)
    if tar_path is not None:
        print(f"  Using local archive {tar_path}")
        with tarfile.open(tar_path) as tar:
            matrix_member = _largest_matrix_member(tar)
            rhs_member = _rhs_member(tar)
            with tar.extractfile(matrix_member) as handle:
                A = mmread(io.BytesIO(handle.read()))
            if force_consistent_rhs or rhs_member is None:
                if not force_consistent_rhs:
                    print(f"  No canonical RHS found for {name}; generating consistent RHS")
                else:
                    print(f"  Forcing consistent RHS (b = A @ 1) for {name}")
                b = _consistent_rhs(A)
            else:
                with tar.extractfile(rhs_member) as handle:
                    b = mmread(io.BytesIO(handle.read()))
        return _validate_problem(name, A, b)

    matrix_path, rhs_path = _find_local_problem_files(name, matrices_dir)
    if matrix_path is None:
        raise FileNotFoundError(
            f"Could not find local matrix files for {name} in {matrices_dir}. "
            "Expected a .mat, .tar.gz archive, or .mtx/.mtx.gz file."
        )

    print(f"  Using local matrix {matrix_path}")
    A = _load_mtx_file(matrix_path)
    if force_consistent_rhs or rhs_path is None:
        if not force_consistent_rhs:
            print(f"  No canonical RHS found for {name}; generating consistent RHS")
        else:
            print(f"  Forcing consistent RHS (b = A @ 1) for {name}")
        b = _consistent_rhs(A)
    else:
        print(f"  Using local RHS {rhs_path}")
        b = _load_mtx_file(rhs_path)
    return _validate_problem(name, A, b)


# sb3 callback that records absolute residual norms (and relative norms
# once b_norm is set) and chosen m at every env step. AKSLRLEnv emits
# absolute ||r||, so divide by ||b|| here to align with the relative-
# residual tolerance used elsewhere
class _StopOnDone(BaseCallback):
    def __init__(self):
        super().__init__()
        self.residuals = []
        self.relative_residuals = []
        self.ms = []
        self._done = False
        self._b_norm = None

    def set_b_norm(self, b_norm: float):
        self._b_norm = max(float(b_norm), 1e-12)

    def _on_step(self):
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", [False]),
        ):
            if "residual_norm" in info:
                r = float(info["residual_norm"])
                self.residuals.append(r)
                if self._b_norm is not None:
                    self.relative_residuals.append(r / self._b_norm)
            if "current_m" in info:
                self.ms.append(int(info["current_m"]))
            if done:
                self._done = True
        return not self._done


def run_sac(A, b, args, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = AKSLRLEnv(
        A=A,
        b=b,
        m_max=args.m_max,
        tolerance=args.tolerance,
        max_cycles=args.max_cycles,
        cte=args.cte,
        convergence_bonus=args.convergence_bonus,
        include_log_residual_in_state=args.include_log_residual,
    )

    # AK-SLRL replay buffer = min(n/2, 20000) (§7.2)
    if args.buffer_size is None:
        buffer_size = int(min(max(A.shape[0] // 2, 1), 20_000))
    else:
        buffer_size = int(args.buffer_size)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=1,
        gradient_steps=1,
        ent_coef=args.ent_coef,
        target_update_interval=args.target_update_interval,
        policy_kwargs={"net_arch": list(args.net_arch)},
        verbose=0,
        seed=seed,
        device=args.device,
    )

    logger = _StopOnDone()
    logger.set_b_norm(float(np.linalg.norm(b)))
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.max_cycles, callback=logger, progress_bar=False)
    elapsed = time.perf_counter() - t0

    residuals = np.array(logger.residuals, dtype=np.float64)
    relative_residuals = np.array(logger.relative_residuals, dtype=np.float64)
    ms = np.array(logger.ms, dtype=np.int64)

    converged_mask = residuals < args.tolerance
    if np.any(converged_mask):
        idx = int(np.argmax(converged_mask)) + 1
        used_ms = ms[:idx]
        return {
            "converged": True,
            "cycles_to_tol": idx,
            "total_arnoldi": int(used_ms.sum()),
            "mean_m": float(used_ms.mean()),
            "elapsed_seconds": float(elapsed),
            "buffer_size": buffer_size,
            "final_residual_norm": float(residuals[idx - 1]),
            "final_relative_residual_norm": (
                float(relative_residuals[idx - 1]) if relative_residuals.size else float("nan")
            ),
        }

    return {
        "converged": False,
        "cycles_to_tol": args.max_cycles,
        "total_arnoldi": int(ms.sum()) if len(ms) else 0,
        "mean_m": float(ms.mean()) if len(ms) else float("nan"),
        "elapsed_seconds": float(elapsed),
        "buffer_size": buffer_size,
        "final_residual_norm": float(residuals[-1]) if len(residuals) else float("inf"),
        "final_relative_residual_norm": (
            float(relative_residuals[-1]) if relative_residuals.size else float("inf")
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices-dir", type=str, default="matrices")
    parser.add_argument("--matrix-names", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--m-max", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Threshold on absolute residual norm ||r|| (AKSLRLEnv convention).")
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--force-consistent-rhs", action="store_true",
                        help="Use b = A @ ones regardless of any packaged RHS.")
    # AKSLRLEnv reward / observation
    parser.add_argument("--cte", type=float, default=1.0,
                        help="Constant c in R = cte/||r|| + (||r_{k-1}|| - ||r_k||).")
    parser.add_argument("--convergence-bonus", type=float, default=0.0)
    parser.add_argument("--include-log-residual", action="store_true", default=True,
                        help="Append log(||r||) to the observation (default: on).")
    parser.add_argument("--no-log-residual", dest="include_log_residual",
                        action="store_false")
    # SAC hyperparameters. Paper-specified (Keramati & Hamdullahpur 2025):
    # gamma=0.97, buffer_size=min(n//2, 20000), ent_coef='auto'. Everything
    # else falls back to stable-baselines3 SAC defaults (lr 3e-4, batch 256,
    # tau 0.005, target update every step, MLP [256, 256]).
    parser.add_argument("--gamma", type=float, default=0.97,
                        help="Paper: 0.97.")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="SB3 SAC default; paper does not specify.")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Paper: min(n // 2, 20_000). Default None auto-computes this.")
    parser.add_argument("--learning-starts", type=int, default=100,
                        help="SB3 SAC default; paper does not specify.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="SB3 SAC default; paper does not specify.")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="SB3 SAC default; paper does not specify.")
    parser.add_argument("--ent-coef", type=str, default="auto",
                        help="Paper: 'auto' for automatic entropy tuning.")
    parser.add_argument("--target-update-interval", type=int, default=1,
                        help="SB3 SAC default; paper does not specify.")
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256],
                        help="Hidden-layer widths for actor/critic MLPs. SB3 SAC default.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="results/train_sac_collection.json")
    args = parser.parse_args()

    # accept either 'auto' or a float for ent_coef
    try:
        args.ent_coef = float(args.ent_coef)
    except ValueError:
        pass

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
        f"matrices_dir={matrices_root}, m_max={args.m_max}, "
        f"tolerance={args.tolerance} (absolute), max_cycles={args.max_cycles}, "
        f"force_consistent_rhs={args.force_consistent_rhs}"
    )

    for idx, config in enumerate(configs, start=1):
        print(f"\n{'='*72}")
        print(f"[{idx}/{len(configs)}] {config['name']}")
        print(f"{'='*72}")
        A, b = load_problem(config, matrices_root,
                            force_consistent_rhs=args.force_consistent_rhs)
        print(f"n={A.shape[0]}, nnz={A.nnz}, ||b||={float(np.linalg.norm(b)):.3e}")

        runs = []
        for offset in range(args.num_seeds):
            seed = args.base_seed + offset
            run = run_sac(A, b, args, seed)
            runs.append(run)
            print(
                f"  seed={seed:02d}: arnoldi={run['total_arnoldi']:8d}  "
                f"time={run['elapsed_seconds']:8.2f}s  mean_m={run['mean_m']:5.2f}  "
                f"final_relres={run['final_relative_residual_norm']:.3e}  "
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
            "m_max": args.m_max,
            "tolerance": args.tolerance,
            "tolerance_kind": "absolute_residual_norm",
            "max_cycles": args.max_cycles,
            "num_seeds": args.num_seeds,
            "base_seed": args.base_seed,
            "device": args.device,
            "force_consistent_rhs": args.force_consistent_rhs,
            "env": {
                "class": "AKSLRLEnv",
                "cte": args.cte,
                "convergence_bonus": args.convergence_bonus,
                "include_log_residual_in_state": args.include_log_residual,
            },
            "sac_hyperparams": {
                "learning_rate": args.learning_rate,
                "buffer_size": args.buffer_size,
                "learning_starts": args.learning_starts,
                "batch_size": args.batch_size,
                "gamma": args.gamma,
                "tau": args.tau,
                "ent_coef": args.ent_coef,
                "target_update_interval": args.target_update_interval,
                "net_arch": list(args.net_arch),
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

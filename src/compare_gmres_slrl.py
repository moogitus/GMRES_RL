"""
Per-matrix runner for the §4.1 six-matrix AK-SLRL comparison (Table 2 +
Figure 2). Runs DQN, SAC (AK-SLRL baseline), and fixed-restart GMRES(20)
on the six matrices for five seeds each and emits per-matrix convergence
traces (relative residual vs cumulative Arnoldi steps and wall-clock time).

RHS is b = A·1 (§4.2); writes traces and aggregate JSON to
results/slrl_six_matrix/.

Built on stable-baselines3 (https://github.com/DLR-RM/stable-baselines3).
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import csr_matrix
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback

from akslrl_env import AKSLRLEnv
from env import GMRESEnv
from train_dqn import load_problem as load_dqn_problem


MATRIX_ORDER = [
    "1138_bus",
    "finance256",
    "ct20stif",
    "olesnik0",
    "ex19",
    "crankseg_1",
]


def _dqn_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        m_max=args.m_max,
        history_length=args.dqn_history_length,
        tolerance=args.tolerance,
        max_cycles=args.max_cycles,
        gamma=args.dqn_gamma,
        lambda_work=args.dqn_lambda_work,
        convergence_bonus=args.dqn_convergence_bonus,
        learning_rate=args.dqn_learning_rate,
        buffer_size=args.dqn_buffer_size,
        learning_starts=args.dqn_learning_starts,
        batch_size=args.dqn_batch_size,
        target_update_interval=args.dqn_target_update_interval,
        exploration_fraction=args.dqn_exploration_fraction,
        exploration_final_eps=args.dqn_exploration_final_eps,
        device=args.device,
    )


def _sac_args(args, b_norm: float) -> SimpleNamespace:
    return SimpleNamespace(
        m_max=args.m_max,
        tolerance=args.tolerance * max(float(b_norm), 1e-12),
        max_cycles=args.max_cycles,
        cte=args.sac_cte,
        convergence_bonus=args.sac_convergence_bonus,
        include_log_residual=args.sac_include_log_residual,
        gamma=args.sac_gamma,
        learning_rate=args.sac_learning_rate,
        buffer_size=args.sac_buffer_size,
        learning_starts=args.sac_learning_starts,
        batch_size=args.sac_batch_size,
        tau=args.sac_tau,
        ent_coef=args.sac_ent_coef,
        target_update_interval=args.sac_target_update_interval,
        net_arch=list(args.sac_net_arch),
        device=args.device,
    )


# sb3 callback that records a full convergence trace (residual, m, time)
# and stops learning once the solve terminates
class _TraceOnDone(BaseCallback):
    def __init__(self, b_norm: float, relative_from_info: bool):
        super().__init__()
        self.b_norm = max(float(b_norm), 1e-12)
        self.relative_from_info = bool(relative_from_info)
        self._done = False
        self._t0 = None
        self.residual_norms: list[float] = []
        self.relative_residuals: list[float] = []
        self.ms: list[int] = []
        self.step_times: list[float] = []

    def start_timer(self) -> None:
        self._t0 = time.perf_counter()

    def _on_step(self) -> bool:
        if self._t0 is None:
            self.start_timer()
        now = time.perf_counter() - self._t0
        for info, done in zip(
            self.locals.get("infos", []),
            self.locals.get("dones", [False]),
        ):
            residual_norm = float(info.get("residual_norm", np.nan))
            self.residual_norms.append(residual_norm)
            if self.relative_from_info and "relative_residual_norm" in info:
                rel = float(info["relative_residual_norm"])
            else:
                rel = residual_norm / self.b_norm
            self.relative_residuals.append(rel)
            self.ms.append(int(info.get("current_m", 0)))
            self.step_times.append(float(now))
            if done:
                self._done = True
        return not self._done


def _trace_payload(
    initial_relative_residual: float,
    initial_residual_norm: float,
    logger: _TraceOnDone,
) -> dict:
    ms = np.asarray(logger.ms, dtype=np.int64)
    arnoldi = np.concatenate([[0], np.cumsum(ms, dtype=np.int64)])
    wallclock = np.concatenate([[0.0], np.asarray(logger.step_times, dtype=np.float64)])
    residual_norm = np.concatenate([[initial_residual_norm], np.asarray(logger.residual_norms, dtype=np.float64)])
    relative_residual = np.concatenate(
        [[initial_relative_residual], np.asarray(logger.relative_residuals, dtype=np.float64)]
    )
    return {
        "arnoldi_steps": arnoldi.astype(np.int64).tolist(),
        "wallclock_seconds": wallclock.astype(np.float64).tolist(),
        "residual_norm": residual_norm.astype(np.float64).tolist(),
        "relative_residual_norm": relative_residual.astype(np.float64).tolist(),
    }


def run_dqn_trace(A, b, args, seed: int) -> dict:
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
    _, reset_info = env.reset(seed=seed)

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

    logger = _TraceOnDone(b_norm=np.linalg.norm(b), relative_from_info=True)
    logger.start_timer()
    model.learn(total_timesteps=args.max_cycles, callback=logger, progress_bar=False)

    trace = _trace_payload(
        initial_relative_residual=float(reset_info["relative_residual_norm"]),
        initial_residual_norm=float(reset_info["residual_norm"]),
        logger=logger,
    )
    final_rel = trace["relative_residual_norm"][-1]
    return {
        "converged": bool(final_rel < args.tolerance),
        "cycles_to_tol": max(0, len(trace["arnoldi_steps"]) - 1),
        "total_arnoldi": int(trace["arnoldi_steps"][-1]),
        "elapsed_seconds": float(trace["wallclock_seconds"][-1]),
        "final_residual_norm": float(trace["residual_norm"][-1]),
        "final_relative_residual_norm": float(final_rel),
        "trace": trace,
    }


def run_sac_trace(A, b, args, seed: int) -> dict:
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
    _, reset_info = env.reset(seed=seed)

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

    logger = _TraceOnDone(b_norm=np.linalg.norm(b), relative_from_info=False)
    logger.start_timer()
    model.learn(total_timesteps=args.max_cycles, callback=logger, progress_bar=False)

    initial_residual_norm = float(reset_info["residual_norm"])
    initial_relative = initial_residual_norm / max(float(np.linalg.norm(b)), 1e-12)
    trace = _trace_payload(
        initial_relative_residual=initial_relative,
        initial_residual_norm=initial_residual_norm,
        logger=logger,
    )
    final_rel = trace["relative_residual_norm"][-1]
    return {
        "converged": bool(final_rel < args.relative_tolerance),
        "cycles_to_tol": max(0, len(trace["arnoldi_steps"]) - 1),
        "total_arnoldi": int(trace["arnoldi_steps"][-1]),
        "elapsed_seconds": float(trace["wallclock_seconds"][-1]),
        "final_residual_norm": float(trace["residual_norm"][-1]),
        "final_relative_residual_norm": float(final_rel),
        "buffer_size": buffer_size,
        "trace": trace,
    }


def run_fixed_gmres20_trace(A, b, args) -> dict:
    env = GMRESEnv(
        A=csr_matrix(A),
        b=b,
        m_max=args.m_max,
        tolerance=args.tolerance,
        max_cycles=args.max_cycles,
        history_length=1,
        lambda_work=0.0,
        gamma_shape=1.0,
        convergence_bonus=0.0,
    )
    _, reset_info = env.reset()
    arnoldi = [0]
    wallclock = [0.0]
    residual_norm = [float(reset_info["residual_norm"])]
    relative_residual = [float(reset_info["relative_residual_norm"])]
    t0 = time.perf_counter()

    while True:
        _, _, terminated, truncated, info = env.step(19)
        arnoldi.append(int(info["total_arnoldi"]))
        wallclock.append(float(time.perf_counter() - t0))
        residual_norm.append(float(info["residual_norm"]))
        relative_residual.append(float(info["relative_residual_norm"]))
        if terminated or truncated:
            break

    return {
        "converged": bool(relative_residual[-1] < args.tolerance),
        "cycles_to_tol": int(info["cycle_count"]),
        "total_arnoldi": int(arnoldi[-1]),
        "elapsed_seconds": float(wallclock[-1]),
        "final_residual_norm": float(residual_norm[-1]),
        "final_relative_residual_norm": float(relative_residual[-1]),
        "trace": {
            "arnoldi_steps": arnoldi,
            "wallclock_seconds": wallclock,
            "residual_norm": residual_norm,
            "relative_residual_norm": relative_residual,
        },
    }


def _resample_traces(runs: list[dict], x_key: str, num_points: int = 250) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_max = max(float(run["trace"][x_key][-1]) for run in runs)
    if x_max <= 0:
        x_grid = np.linspace(0.0, 1.0, num_points)
    else:
        x_grid = np.linspace(0.0, x_max, num_points)

    curves = []
    for run in runs:
        xs = np.asarray(run["trace"][x_key], dtype=np.float64)
        ys = np.asarray(run["trace"]["relative_residual_norm"], dtype=np.float64)
        ys = np.maximum(ys, 1e-16)
        log_ys = np.log10(ys)
        interp = np.interp(x_grid, xs, log_ys, left=log_ys[0], right=log_ys[-1])
        curves.append(10.0 ** interp)

    curve_array = np.asarray(curves, dtype=np.float64)
    return x_grid, curve_array.mean(axis=0), curve_array.std(axis=0)


def _plot_matrix_curve(matrix_row: dict, metric: str, out_path: Path) -> None:
    metric_label = {
        "arnoldi_steps": "Arnoldi steps",
        "wallclock_seconds": "Wall-clock time (s)",
    }[metric]
    colors = {
        "dqn": "#C0504D",
        "sac": "#4F81BD",
        "gmres20": "#4BACC6",
    }
    labels = {
        "dqn": "DQN",
        "sac": "SAC",
        "gmres20": "GMRES(20)",
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 6.0), constrained_layout=True)

    for method_key in ("dqn", "sac", "gmres20"):
        runs = matrix_row["methods"][method_key]["runs"]
        x_grid, mean_curve, std_curve = _resample_traces(runs, metric)
        ax.plot(x_grid, mean_curve, linewidth=2.8, color=colors[method_key], label=labels[method_key])
        if len(runs) > 1:
            lower = np.maximum(mean_curve - std_curve, 1e-16)
            upper = np.maximum(mean_curve + std_curve, 1e-16)
            ax.fill_between(x_grid, lower, upper, color=colors[method_key], alpha=0.18)

    ax.set_xlabel(metric_label, fontsize=13, fontweight="bold")
    ax.set_ylabel("Relative residual norm", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    ax.grid(True, which="major", axis="both", color="#B0B0B0", alpha=0.5)
    ax.set_title(
        f"{matrix_row['name']}: convergence vs {metric_label.lower()}\n"
        f"means over {matrix_row['num_seeds']} seed(s) for DQN/SAC",
        fontsize=13,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrices-dir",
        type=str,
        default=str(repo_root / "matrices" / "slrl_benchmark_matrices"),
    )
    parser.add_argument(
        "--matrix-names",
        nargs="+",
        default=MATRIX_ORDER,
    )
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Relative residual tolerance target.")
    parser.add_argument("--max-cycles", type=int, default=10000)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--m-max", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--dqn-history-length", type=int, default=5)
    parser.add_argument("--dqn-gamma", type=float, default=0.925)
    parser.add_argument("--dqn-lambda-work", type=float, default=0.01)
    parser.add_argument("--dqn-convergence-bonus", type=float, default=9.0)
    parser.add_argument("--dqn-learning-rate", type=float, default=3e-3)
    parser.add_argument("--dqn-buffer-size", type=int, default=10_000)
    parser.add_argument("--dqn-learning-starts", type=int, default=25)
    parser.add_argument("--dqn-batch-size", type=int, default=32)
    parser.add_argument("--dqn-target-update-interval", type=int, default=100)
    parser.add_argument("--dqn-exploration-fraction", type=float, default=0.10)
    parser.add_argument("--dqn-exploration-final-eps", type=float, default=0.01)

    parser.add_argument("--sac-cte", type=float, default=1.0)
    parser.add_argument("--sac-convergence-bonus", type=float, default=0.0)
    parser.add_argument("--sac-gamma", type=float, default=0.97)
    parser.add_argument("--sac-learning-rate", type=float, default=3e-4)
    parser.add_argument("--sac-buffer-size", type=int, default=None)
    parser.add_argument("--sac-learning-starts", type=int, default=100)
    parser.add_argument("--sac-batch-size", type=int, default=256)
    parser.add_argument("--sac-tau", type=float, default=0.005)
    parser.add_argument("--sac-ent-coef", type=str, default="auto")
    parser.add_argument("--sac-target-update-interval", type=int, default=1)
    parser.add_argument("--sac-net-arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--sac-include-log-residual", action="store_true", default=True)
    parser.add_argument("--no-sac-log-residual", dest="sac_include_log_residual", action="store_false")

    parser.add_argument(
        "--out-json",
        type=str,
        default=str(repo_root / "results" / "slrl_six_matrix" / "results.json"),
    )
    parser.add_argument(
        "--out-dir-arnoldi",
        type=str,
        default=str(repo_root / "results" / "slrl_six_matrix" / "arnoldi"),
    )
    parser.add_argument(
        "--out-dir-wallclock",
        type=str,
        default=str(repo_root / "results" / "slrl_six_matrix" / "wallclock"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        args.sac_ent_coef = float(args.sac_ent_coef)
    except ValueError:
        pass

    matrices_root = Path(args.matrices_dir).expanduser().resolve()
    selected_names = [name for name in MATRIX_ORDER if name in set(args.matrix_names)]
    if not selected_names:
        raise ValueError("No matrices selected.")

    dqn_args = _dqn_args(args)
    arnoldi_dir = Path(args.out_dir_arnoldi)
    wallclock_dir = Path(args.out_dir_wallclock)

    rows = []
    for idx, name in enumerate(selected_names, start=1):
        print(f"\n{'=' * 72}")
        print(f"[{idx}/{len(selected_names)}] {name}")
        print(f"{'=' * 72}")
        A, b = load_dqn_problem({"name": name}, matrices_root)
        b_norm = float(np.linalg.norm(b))
        print(f"n={A.shape[0]}, nnz={A.nnz}, ||b||={b_norm:.3e}")

        gmres_run = run_fixed_gmres20_trace(A, b, args)
        dqn_runs = []
        sac_runs = []
        for offset in range(args.num_seeds):
            seed = args.base_seed + offset
            dqn_run = run_dqn_trace(A, b, dqn_args, seed)
            sac_args = _sac_args(args, b_norm)
            sac_args.relative_tolerance = args.tolerance
            sac_run = run_sac_trace(A, b, sac_args, seed)
            dqn_runs.append(dqn_run)
            sac_runs.append(sac_run)
            print(
                f"  seed={seed:02d}  "
                f"DQN arnoldi={dqn_run['total_arnoldi']:7d} time={dqn_run['elapsed_seconds']:8.2f}s  "
                f"SAC arnoldi={sac_run['total_arnoldi']:7d} time={sac_run['elapsed_seconds']:8.2f}s"
            )

        matrix_row = {
            "name": name,
            "shape": [int(A.shape[0]), int(A.shape[1])],
            "nnz": int(A.nnz),
            "b_norm": b_norm,
            "num_seeds": int(args.num_seeds),
            "methods": {
                "gmres20": {"runs": [gmres_run]},
                "dqn": {"runs": dqn_runs},
                "sac": {"runs": sac_runs},
            },
        }
        rows.append(matrix_row)
        _plot_matrix_curve(matrix_row, "arnoldi_steps", arnoldi_dir / f"{name}.png")
        _plot_matrix_curve(matrix_row, "wallclock_seconds", wallclock_dir / f"{name}.png")

    payload = {
        "meta": {
            "matrix_order": selected_names,
            "matrices_dir": str(matrices_root),
            "tolerance_relative": args.tolerance,
            "max_cycles": args.max_cycles,
            "m_max": args.m_max,
            "num_seeds": args.num_seeds,
            "base_seed": args.base_seed,
            "device": args.device,
            "dqn_hyperparams": vars(dqn_args),
            "sac_hyperparams": {
                "cte": args.sac_cte,
                "convergence_bonus": args.sac_convergence_bonus,
                "gamma": args.sac_gamma,
                "learning_rate": args.sac_learning_rate,
                "buffer_size": args.sac_buffer_size,
                "learning_starts": args.sac_learning_starts,
                "batch_size": args.sac_batch_size,
                "tau": args.sac_tau,
                "ent_coef": args.sac_ent_coef,
                "target_update_interval": args.sac_target_update_interval,
                "net_arch": list(args.sac_net_arch),
                "include_log_residual": args.sac_include_log_residual,
            },
            "notes": [
                "All methods use the consistent RHS b = A @ 1.",
                "Plots are per-matrix convergence traces with relative residual norm on the y-axis.",
                "DQN and GMRES(20) terminate on relative residual tolerance.",
                "SAC uses the implemented absolute-residual environment with tolerance scaled by ||b|| to match the relative target.",
            ],
        },
        "results": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved trace log to {out_json}")
    print(f"Saved Arnoldi plots to {arnoldi_dir}")
    print(f"Saved wall-clock plots to {wallclock_dir}")


if __name__ == "__main__":
    main()

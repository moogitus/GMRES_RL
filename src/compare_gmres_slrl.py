"""
compare_gmres_slrl.py

Run DQN, SAC, and fixed GMRES(20) on a six-matrix hard benchmark, then
generate paper-style comparison plots for:
  - total Arnoldi steps
  - wall-clock time

The benchmark always uses the consistent RHS b = A @ 1.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from env import GMRESEnv
from train_dqn import load_problem as load_dqn_problem
from train_dqn import run_dqn
from train_sac import run_sac


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


def run_fixed_gmres20(A, b, args) -> dict:
    env = GMRESEnv(
        A=A,
        b=b,
        m_max=args.m_max,
        tolerance=args.tolerance,
        max_cycles=args.max_cycles,
        history_length=1,
        lambda_work=0.0,
        gamma_shape=1.0,
        convergence_bonus=0.0,
    )
    _, info = env.reset()
    t0 = time.perf_counter()
    while True:
        _, _, terminated, truncated, info = env.step(19)
        if terminated or truncated:
            break
    elapsed = time.perf_counter() - t0
    return {
        "converged": bool(terminated),
        "cycles_to_tol": int(info["cycle_count"]),
        "total_arnoldi": int(info["total_arnoldi"]),
        "mean_m": 20.0,
        "elapsed_seconds": float(elapsed),
        "final_residual_norm": float(info["residual_norm"]),
        "final_relative_residual_norm": float(info["relative_residual_norm"]),
    }


def _summarize_runs(runs: list[dict]) -> dict:
    def _avg(key: str) -> float:
        return float(np.mean([run[key] for run in runs]))

    def _std(key: str) -> float:
        return float(np.std([run[key] for run in runs]))

    return {
        "convergence_rate": _avg("converged"),
        "arnoldi_mean": _avg("total_arnoldi"),
        "arnoldi_std": _std("total_arnoldi"),
        "time_mean": _avg("elapsed_seconds"),
        "time_std": _std("elapsed_seconds"),
        "cycles_mean": _avg("cycles_to_tol"),
        "final_residual_norm_mean": _avg("final_residual_norm"),
        "final_relative_residual_norm_mean": _avg("final_relative_residual_norm"),
    }


def _plot_metric(payload: dict, out_path: Path, metric_key: str, ylabel: str, title_suffix: str) -> None:
    labels = [row["name"] for row in payload["results"]]
    x = np.arange(len(labels))

    series = {
        "DQN": np.array([row["methods"]["dqn"]["summary"][metric_key] for row in payload["results"]], dtype=np.float64),
        "SAC": np.array([row["methods"]["sac"]["summary"][metric_key] for row in payload["results"]], dtype=np.float64),
        "GMRES(20)": np.array([row["methods"]["gmres20"]["summary"][metric_key] for row in payload["results"]], dtype=np.float64),
    }
    colors = {
        "DQN": "#C0504D",
        "SAC": "#4F81BD",
        "GMRES(20)": "#4BACC6",
    }
    markers = {
        "DQN": "^",
        "SAC": "o",
        "GMRES(20)": "s",
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    for label, values in series.items():
        ax.plot(x, values, linewidth=3, marker=markers[label], markersize=7, label=label, color=colors[label])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=11)
    ax.set_xlabel("matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    ax.grid(True, axis="y", color="#999999", alpha=0.6, linewidth=1)
    ax.grid(False, axis="x")
    ax.set_title(
        "Six-matrix hard benchmark: DQN vs SAC vs GMRES(20)\n"
        f"{title_suffix}",
        fontsize=14,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices-dir", type=str, default=str(repo_root / "matrices"))
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Relative residual tolerance used for DQN and GMRES(20). "
                             "SAC receives the matrix-specific absolute equivalent.")
    parser.add_argument("--max-cycles", type=int, default=10000)
    parser.add_argument("--num-seeds", type=int, default=1)
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

    parser.add_argument("--out-json", type=str, default=str(repo_root / "src" / "logs" / "six_matrix_rl_comparison.json"))
    parser.add_argument("--out-arnoldi", type=str, default=str(repo_root / "src" / "logs" / "six_matrix_rl_comparison_arnoldi.png"))
    parser.add_argument("--out-wallclock", type=str, default=str(repo_root / "src" / "logs" / "six_matrix_rl_comparison_wallclock.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        args.sac_ent_coef = float(args.sac_ent_coef)
    except ValueError:
        pass

    matrices_root = Path(args.matrices_dir).expanduser().resolve()
    dqn_args = _dqn_args(args)

    rows = []
    for idx, name in enumerate(MATRIX_ORDER, start=1):
        print(f"\n{'=' * 72}")
        print(f"[{idx}/{len(MATRIX_ORDER)}] {name}")
        print(f"{'=' * 72}")
        A, b = load_dqn_problem({"name": name}, matrices_root)
        b_norm = float(np.linalg.norm(b))
        print(f"n={A.shape[0]}, nnz={A.nnz}, ||b||={b_norm:.3e}")

        gmres_runs = []
        dqn_runs = []
        sac_runs = []
        for offset in range(args.num_seeds):
            seed = args.base_seed + offset
            gmres_run = run_fixed_gmres20(A, b, args)
            dqn_run = run_dqn(A, b, dqn_args, seed)
            sac_run = run_sac(A, b, _sac_args(args, b_norm), seed)
            gmres_runs.append(gmres_run)
            dqn_runs.append(dqn_run)
            sac_runs.append(sac_run)
            print(
                f"  seed={seed:02d}  "
                f"GMRES20 arnoldi={gmres_run['total_arnoldi']:7d} time={gmres_run['elapsed_seconds']:8.2f}s  "
                f"DQN arnoldi={dqn_run['total_arnoldi']:7d} time={dqn_run['elapsed_seconds']:8.2f}s  "
                f"SAC arnoldi={sac_run['total_arnoldi']:7d} time={sac_run['elapsed_seconds']:8.2f}s"
            )

        rows.append({
            "name": name,
            "shape": [int(A.shape[0]), int(A.shape[1])],
            "nnz": int(A.nnz),
            "b_norm": b_norm,
            "methods": {
                "gmres20": {"runs": gmres_runs, "summary": _summarize_runs(gmres_runs)},
                "dqn": {"runs": dqn_runs, "summary": _summarize_runs(dqn_runs)},
                "sac": {"runs": sac_runs, "summary": _summarize_runs(sac_runs)},
            },
        })

    payload = {
        "meta": {
            "matrix_order": MATRIX_ORDER,
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
                "DQN and GMRES(20) terminate on relative residual tolerance.",
                "SAC uses the implemented absolute-residual environment with tolerance scaled by ||b|| to match the relative target.",
            ],
        },
        "results": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    _plot_metric(
        payload,
        Path(args.out_arnoldi),
        metric_key="arnoldi_mean",
        ylabel="total Arnoldi steps (log scale)",
        title_suffix=f"relative tol={args.tolerance:.0e}, max_cycles={args.max_cycles}",
    )
    _plot_metric(
        payload,
        Path(args.out_wallclock),
        metric_key="time_mean",
        ylabel="wall-clock time in seconds (log scale)",
        title_suffix=f"relative tol={args.tolerance:.0e}, max_cycles={args.max_cycles}",
    )

    print(f"\nSaved results to {out_json}")
    print(f"Saved Arnoldi plot to {args.out_arnoldi}")
    print(f"Saved wall-clock plot to {args.out_wallclock}")


if __name__ == "__main__":
    main()

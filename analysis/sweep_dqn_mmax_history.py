"""
Sensitivity sweep over the DQN action range m_max and observation depth d
(history length). Internal sanity check; not a paper figure or table.
Compares the DQN restart controller across m_max in {20, 40, 60} and
history_length in {1, 5, 10} on a small subset of the benchmark suite.

Reuses train_dqn.run_dqn (built on stable-baselines3).
"""
import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_dqn import discover_matrix_names, load_problem, run_dqn

# summary statistics over seeds for a single matrix/config
def summarise_runs(runs: list[dict]) -> dict:
    return {
        "convergence_rate": float(np.mean([run["converged"] for run in runs])),
        "arnoldi_mean": float(np.mean([run["total_arnoldi"] for run in runs])),
        "arnoldi_std": float(np.std([run["total_arnoldi"] for run in runs])),
        "time_mean": float(np.mean([run["elapsed_seconds"] for run in runs])),
        "time_std": float(np.std([run["elapsed_seconds"] for run in runs])),
        "cycles_mean": float(np.mean([run["cycles_to_tol"] for run in runs])),
        "final_relative_residual_norm_mean": float(
            np.mean([run["final_relative_residual_norm"] for run in runs])
        ),
    }

def build_dqn_args(base_args, m_max: int, history_length: int):
    args = deepcopy(base_args)
    args.m_max = int(m_max)
    args.history_length = int(history_length)
    return args

def rank_combo(summary: dict) -> tuple:
    return (
        -summary["convergence_rate_mean"],
        summary["arnoldi_mean"],
        summary["time_mean"],
        summary["final_relative_residual_norm_mean"],
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices-dir", type=str, default="matrices/full_benchmark")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gamma", type=float, default=0.9239181596082815)
    parser.add_argument("--lambda-work", type=float, default=0.00977496121089123)
    parser.add_argument("--convergence-bonus", type=float, default=8.78377065661212)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    parser.add_argument("--exploration-final-eps", type=float, default=0.01)
    parser.add_argument("--m-max-values", nargs="+", type=int, default=[20, 40, 60])
    parser.add_argument("--history-length-values", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--out", type=str, default="results/dqn_mmax_history/results.json")
    args = parser.parse_args()

    matrices_root = Path(args.matrices_dir)
    matrix_names = discover_matrix_names(matrices_root)[: args.limit]
    if not matrix_names:
        raise ValueError("No matrices selected for sweep.")

    base_dqn_args = SimpleNamespace(
        m_max=20,
        history_length=5,
        tolerance=args.tolerance,
        max_cycles=args.max_cycles,
        gamma=args.gamma,
        lambda_work=args.lambda_work,
        convergence_bonus=args.convergence_bonus,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        device=args.device,
    )

    results = {
        "meta": {
            "matrices_dir": str(matrices_root.resolve()),
            "matrix_names": matrix_names,
            "max_cycles": args.max_cycles,
            "tolerance": args.tolerance,
            "num_seeds": args.num_seeds,
            "base_seed": args.base_seed,
            "device": args.device,
            "reward_params": {
                "gamma_shape": args.gamma,
                "lambda_work": args.lambda_work,
                "convergence_bonus": args.convergence_bonus,
            },
            "dqn_hyperparams": {
                "learning_rate": args.learning_rate,
                "buffer_size": args.buffer_size,
                "learning_starts": args.learning_starts,
                "batch_size": args.batch_size,
                "target_update_interval": args.target_update_interval,
                "exploration_fraction": args.exploration_fraction,
                "exploration_final_eps": args.exploration_final_eps,
            },
            "m_max_values": args.m_max_values,
            "history_length_values": args.history_length_values,
        },
        "dqn_grid": {},
        "ranking": [],
    }

    loaded = {}
    for name in matrix_names:
        print(f"\nLoading {name}")
        loaded[name] = load_problem({"name": name}, matrices_root)

    for m_max in args.m_max_values:
        for history_length in args.history_length_values:
            combo_key = f"mmax_{m_max}_k_{history_length}"
            dqn_args = build_dqn_args(base_dqn_args, m_max, history_length)
            combo_entry = {"per_matrix": {}, "summary": {}}
            print(f"\nRunning DQN sweep for {combo_key}")
            all_runs = []
            for idx, name in enumerate(matrix_names, start=1):
                A, b = loaded[name]
                runs = []
                for offset in range(args.num_seeds):
                    seed = args.base_seed + offset
                    run = run_dqn(A, b, dqn_args, seed)
                    runs.append(run)
                    all_runs.append(run)
                    print(
                        f"  [{idx:02d}/{len(matrix_names):02d}] {name:15s} seed={seed:02d} "
                        f"arnoldi={run['total_arnoldi']:7d} time={run['elapsed_seconds']:8.2f}s "
                        f"mean_m={run['mean_m']:5.2f} final_relres={run['final_relative_residual_norm']:.3e} "
                        f"conv={run['converged']}"
                    )
                combo_entry["per_matrix"][name] = {
                    "runs": runs,
                    "summary": summarise_runs(runs),
                }

            combo_entry["summary"] = {
                **summarise_runs(all_runs),
                "convergence_rate_mean": float(
                    np.mean(
                        [
                            item["summary"]["convergence_rate"]
                            for item in combo_entry["per_matrix"].values()
                        ]
                    )
                ),
            }
            results["dqn_grid"][combo_key] = combo_entry

    ranking = []
    for combo_key, combo_entry in results["dqn_grid"].items():
        ranking.append(
            {
                "combo": combo_key,
                **combo_entry["summary"],
            }
        )
    ranking.sort(key=rank_combo)
    results["ranking"] = ranking

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved sweep results to {out_path}")

    if ranking:
        best = ranking[0]
        print(
            "\nBest combo by convergence-first ranking: "
            f"{best['combo']}  conv={best['convergence_rate_mean']:.3f}  "
            f"arnoldi_mean={best['arnoldi_mean']:.1f}  "
            f"time_mean={best['time_mean']:.2f}s"
        )

if __name__ == "__main__":
    main()

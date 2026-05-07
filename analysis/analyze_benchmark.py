"""
Analyze the Peairs-style 159-matrix benchmark with rliable.

Expected input: JSON emitted by analysis/run_peairs_style_159_suite.py
"""

from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from pandas.util import _decorators as _pd_decorators
except ImportError:  # pragma: no cover
    _pd_decorators = None
else:
    _deprecate_kwarg = getattr(_pd_decorators, "deprecate_kwarg", None)
    if _deprecate_kwarg is not None:
        params = list(inspect.signature(_deprecate_kwarg).parameters)
        if params and params[0] == "klass":
            def _compat_deprecate_kwarg(old_arg_name, new_arg_name, mapping=None, stacklevel=2):
                return _deprecate_kwarg(
                    FutureWarning,
                    old_arg_name,
                    new_arg_name,
                    mapping=mapping,
                    stacklevel=stacklevel,
                )

            _pd_decorators.deprecate_kwarg = _compat_deprecate_kwarg

try:
    from rliable import library as rly
    from rliable import metrics
    from rliable import plot_utils
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "rliable is not installed. Install it with:\n"
        "  .venv/bin/python -m pip install rliable"
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="src/logs/peairs_style_159_suite.json",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="gmres20",
        choices=["gmres20", "gmres60"],
    )
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument(
        "--out-json",
        type=str,
        default="src/logs/peairs_style_159_rliable_summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="src/logs/peairs_style_159_rliable",
    )
    return parser.parse_args()


def _method_run_count(payload: dict, method: str) -> int:
    return max(
        len(matrix_row["methods"][method]["runs"])
        for matrix_row in payload["results"].values()
    )


def _tile_runs(values: list[float], target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == target_len:
        return arr
    if arr.size == 1:
        return np.repeat(arr, target_len)
    raise ValueError(f"Cannot tile {arr.size} runs to target length {target_len}.")


def build_metric_matrices(payload: dict, baseline: str):
    results = payload["results"]
    matrix_names = list(results.keys())
    methods = payload["meta"]["methods"]
    num_runs = max(_method_run_count(payload, method) for method in methods)

    arnoldi_scores = {method: np.zeros((num_runs, len(matrix_names))) for method in methods}
    wallclock_scores = {method: np.zeros((num_runs, len(matrix_names))) for method in methods}
    success_scores = {method: np.zeros((num_runs, len(matrix_names))) for method in methods}

    for col, matrix_name in enumerate(matrix_names):
        row = results[matrix_name]
        baseline_runs = row["methods"][baseline]["runs"]
        baseline_arnoldi = _tile_runs(
            [run["total_arnoldi"] for run in baseline_runs], num_runs
        )
        baseline_time = _tile_runs(
            [run["elapsed_seconds"] for run in baseline_runs], num_runs
        )

        for method in methods:
            runs = row["methods"][method]["runs"]
            method_arnoldi = _tile_runs(
                [max(float(run["total_arnoldi"]), 1.0) for run in runs], num_runs
            )
            method_time = _tile_runs(
                [max(float(run["elapsed_seconds"]), 1e-12) for run in runs], num_runs
            )
            method_success = _tile_runs(
                [float(run["converged"]) for run in runs], num_runs
            )
            arnoldi_scores[method][:, col] = baseline_arnoldi / method_arnoldi
            wallclock_scores[method][:, col] = baseline_time / method_time
            success_scores[method][:, col] = method_success

    return matrix_names, methods, arnoldi_scores, wallclock_scores, success_scores


def interval_summary(score_dict: dict[str, np.ndarray], reps: int):
    aggregate_fn = lambda scores: np.array(
        [
            metrics.aggregate_iqm(scores),
            metrics.aggregate_mean(scores),
        ]
    )
    estimates, cis = rly.get_interval_estimates(score_dict, aggregate_fn, reps=reps)
    summary = {}
    for method in score_dict:
        summary[method] = {
            "iqm": float(estimates[method][0]),
            "iqm_ci_low": float(cis[method][0][0]),
            "iqm_ci_high": float(cis[method][1][0]),
            "mean": float(estimates[method][1]),
            "mean_ci_low": float(cis[method][0][1]),
            "mean_ci_high": float(cis[method][1][1]),
        }
    return summary


def probability_of_improvement_summary(
    score_dict: dict[str, np.ndarray], reference: str, reps: int
):
    pair_dict = {
        method: (score_dict[method], score_dict[reference])
        for method in score_dict
        if method != reference
    }
    if not pair_dict:
        return {}
    estimates, cis = rly.get_interval_estimates(
        pair_dict,
        metrics.probability_of_improvement,
        reps=reps,
    )
    summary = {}
    for method in pair_dict:
        estimate = float(np.asarray(estimates[method]).squeeze())
        ci = np.asarray(cis[method], dtype=np.float64).squeeze()
        if ci.ndim == 0:
            ci_low = ci_high = float(ci)
        else:
            ci_low = float(ci[0])
            ci_high = float(ci[-1])
        summary[method] = {
            "probability": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return summary


def save_performance_profile(
    score_dict: dict[str, np.ndarray],
    xlabel: str,
    out_path: Path,
    reps: int,
):
    thresholds = np.linspace(0.0, 2.0, 101)
    distributions, cis = rly.create_performance_profile(
        score_dict, thresholds, reps=reps
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    plot_utils.plot_performance_profiles(
        distributions,
        thresholds,
        performance_profile_cis=cis,
        xlabel=xlabel,
        ax=ax,
    )
    ax.set_ylabel("Fraction of matrix-seed pairs")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_interval_plot(summary: dict, title: str, out_path: Path):
    methods = list(summary.keys())
    iqm = [summary[method]["iqm"] for method in methods]
    lower = [summary[method]["iqm"] - summary[method]["iqm_ci_low"] for method in methods]
    upper = [summary[method]["iqm_ci_high"] - summary[method]["iqm"] for method in methods]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    y = np.arange(len(methods))
    ax.errorbar(iqm, y, xerr=[lower, upper], fmt="o", capsize=4)
    ax.set_yticks(y, methods)
    ax.set_xlabel("IQM")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    payload = json.loads(Path(args.input).read_text())

    matrix_names, methods, arnoldi_scores, wallclock_scores, success_scores = (
        build_metric_matrices(payload, baseline=args.baseline)
    )

    arnoldi_summary = interval_summary(arnoldi_scores, reps=args.reps)
    wallclock_summary = interval_summary(wallclock_scores, reps=args.reps)
    success_summary = interval_summary(success_scores, reps=args.reps)

    dqn_arnoldi_poi = probability_of_improvement_summary(
        arnoldi_scores, reference="gmres_rl", reps=args.reps
    )
    dqn_wallclock_poi = probability_of_improvement_summary(
        wallclock_scores, reference="gmres_rl", reps=args.reps
    )

    out_dir = Path(args.out_dir)
    save_performance_profile(
        arnoldi_scores,
        xlabel=f"Arnoldi speedup vs {args.baseline}",
        out_path=out_dir / "arnoldi_performance_profile.png",
        reps=args.reps,
    )
    save_performance_profile(
        wallclock_scores,
        xlabel=f"Wall-clock speedup vs {args.baseline}",
        out_path=out_dir / "wallclock_performance_profile.png",
        reps=args.reps,
    )
    save_interval_plot(
        arnoldi_summary,
        title="IQM Arnoldi speedup",
        out_path=out_dir / "arnoldi_iqm_intervals.png",
    )
    save_interval_plot(
        wallclock_summary,
        title="IQM wall-clock speedup",
        out_path=out_dir / "wallclock_iqm_intervals.png",
    )
    save_interval_plot(
        success_summary,
        title="IQM success rate",
        out_path=out_dir / "success_iqm_intervals.png",
    )

    summary_payload = {
        "meta": {
            "input": str(Path(args.input).resolve()),
            "baseline": args.baseline,
            "reps": args.reps,
            "matrix_count": len(matrix_names),
            "methods": methods,
        },
        "arnoldi_speedup": arnoldi_summary,
        "wallclock_speedup": wallclock_summary,
        "success_rate": success_summary,
        "probability_of_improvement_vs_dqn": {
            "arnoldi": dqn_arnoldi_poi,
            "wallclock": dqn_wallclock_poi,
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary_payload, indent=2))
    print(f"Saved rliable summary to {out_json}")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()

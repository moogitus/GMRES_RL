"""
analysis/validate_variance_figures.py

Computes a documented set of variance and tail-event statistics from the
Peairs-style 159-matrix GMRES benchmark and writes them to three CSVs.
Each summary statistic is paired with the slice of the data it is computed
on and the formula that produced it, so any single number can be
re-derived from the per-cell or per-matrix tables.

The script compares the two stochastic adaptive restart-selection methods
in the benchmark: ``rand_gmres`` (uniform random m) and ``gmres_rl`` (the
DQN restart controller). Methods with deterministic schedules
(``gmres20``, ``gmres60``, ``angle_gmres``) are excluded since they have no
seed-to-seed variance to compare.

Input
-----
A JSON file produced by ``analysis/run_peairs_style_159_suite.py`` with the
schema:
  meta:
    max_total_arnoldi : int
    methods           : list[str]
    ...
  results:
    <matrix-name>:
      methods:
        <method-name>:
          runs : list of per-seed dicts with total_arnoldi,
                 elapsed_seconds, converged, final_relative_residual_norm

Outputs (in ``--out-dir``)
--------------------------
1. ``per_cell_runs.csv``
   One row per (matrix, method, seed). Source of truth; every other CSV
   is computed from this one.

2. ``per_matrix_statistics.csv``
   One row per matrix. Per-method mean / std / coefficient of variation
   across the 5 seeds, plus convergence and cap counts and per-matrix
   tail ratios.

3. ``summary_statistics.csv``
   One row per scalar statistic, with: stable identifier, description,
   value, units, slice, sample size, formula, and a one-line numerical
   interpretation.

Run
---
    python analysis/validate_variance_figures.py
    python analysis/validate_variance_figures.py --input src/logs/159-partial.json

Memory: streams the input via ``ijson`` so peak RAM is well under 100 MB
even for the 787 MB partial dump.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import ijson  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "ijson is required for streaming the 787 MB JSON. Install with:\n"
        "    pip install ijson"
    ) from exc

import ijson  # type: ignore  # re-import after the install-guard

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "src" / "logs" / "159-partial.json"
DEFAULT_OUT = REPO_ROOT / "analysis" / "results" / "variance_validation"

# Stochastic restart-selection methods. Deterministic schedules
# (gmres20, gmres60, angle_gmres) have no seed-to-seed variance and are
# omitted from this analysis.
METHODS = ("rand_gmres", "gmres_rl")

# The Arnoldi cap from meta.max_total_arnoldi. A run that reports
# total_arnoldi >= CAP is treated as "hit the cap" regardless of the
# `converged` flag (the recorder allows a few-Arnoldi-step overshoot).
ARNOLDI_CAP = 100_000


# --------------------------------------------------------------------------- #
# Streaming load
# --------------------------------------------------------------------------- #

def load_per_cell_records(json_path: Path) -> list[dict]:
    """Stream the partial-results JSON and yield one dict per (matrix, method,
    seed) cell. We deliberately ignore the per-cycle traces (which are the
    bulk of the file size) -- they are not needed for any figure here.
    """
    records: list[dict] = []
    with json_path.open("rb") as f:
        for matrix_name, entry in ijson.kvitems(f, "results"):
            for method in METHODS:
                method_block = entry.get("methods", {}).get(method)
                if method_block is None:
                    continue
                for seed_idx, run in enumerate(method_block.get("runs", [])):
                    records.append({
                        "matrix": matrix_name,
                        "method": method,
                        "seed": seed_idx,
                        "total_arnoldi": int(run["total_arnoldi"]),
                        "elapsed_seconds": float(run["elapsed_seconds"]),
                        "converged": bool(run["converged"]),
                        "final_relative_residual": float(run["final_relative_residual_norm"]),
                    })
    return records


# --------------------------------------------------------------------------- #
# Per-matrix aggregation
# --------------------------------------------------------------------------- #

def build_per_matrix_table(per_cell: list[dict]) -> list[dict]:
    """Aggregate per-cell records into one row per matrix with both methods'
    mean / std / CV / convergence count, plus matched indicators used by the
    summary statistics.

    Returns a list of dicts, sorted alphabetically by matrix name.
    """
    by_matrix: dict[str, dict[str, list[dict]]] = {}
    for r in per_cell:
        by_matrix.setdefault(r["matrix"], {m: [] for m in METHODS})
        by_matrix[r["matrix"]][r["method"]].append(r)

    rows: list[dict] = []
    for matrix in sorted(by_matrix):
        cells = by_matrix[matrix]
        row: dict = {"matrix": matrix}
        per_method_arr = {}
        per_method_conv = {}
        for m in METHODS:
            arn = np.array([c["total_arnoldi"] for c in cells[m]], dtype=np.float64)
            conv = np.array([c["converged"] for c in cells[m]], dtype=bool)
            per_method_arr[m] = arn
            per_method_conv[m] = conv
            mean = float(arn.mean()) if arn.size else np.nan
            std = float(arn.std(ddof=0)) if arn.size else np.nan
            cv = std / mean if mean and not np.isnan(mean) and mean != 0 else np.nan
            row.update({
                f"{m}_n_seeds":       int(arn.size),
                f"{m}_arnoldi_mean":  mean,
                f"{m}_arnoldi_std":   std,
                f"{m}_arnoldi_cv":    cv,
                f"{m}_arnoldi_min":   float(arn.min()) if arn.size else np.nan,
                f"{m}_arnoldi_max":   float(arn.max()) if arn.size else np.nan,
                f"{m}_n_converged":   int(conv.sum()),
                f"{m}_n_capped":      int((arn >= ARNOLDI_CAP).sum()),
                # Per-matrix tail statistics, only meaningful when mean > 0.
                f"{m}_worst_over_mean":
                    (float(arn.max()) / mean) if (arn.size and mean and mean != 0) else np.nan,
                f"{m}_minmax_spread_over_mean":
                    ((float(arn.max()) - float(arn.min())) / mean)
                    if (arn.size and mean and mean != 0) else np.nan,
            })

        rand_conv_all = bool(per_method_conv["rand_gmres"].all()) and per_method_arr["rand_gmres"].size == 5
        rl_conv_all = bool(per_method_conv["gmres_rl"].all()) and per_method_arr["gmres_rl"].size == 5
        row["both_fully_converged"] = int(rand_conv_all and rl_conv_all)

        # Per-matrix winners. Defined only when both methods fully converged
        # (otherwise std vs cap-hit comparisons are not apples-to-apples).
        if row["both_fully_converged"]:
            row["rl_std_lower_than_rand"] = int(
                row["gmres_rl_arnoldi_std"] < row["rand_gmres_arnoldi_std"]
            )
            row["rl_cv_lower_than_rand"] = int(
                row["gmres_rl_arnoldi_cv"] < row["rand_gmres_arnoldi_cv"]
            )
        else:
            row["rl_std_lower_than_rand"] = ""
            row["rl_cv_lower_than_rand"] = ""

        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Helpers used by summary computations
# --------------------------------------------------------------------------- #

def _arr(per_matrix: list[dict], col: str, where: Iterable[bool] | None = None) -> np.ndarray:
    """Pull column ``col`` from per_matrix rows, optionally masked."""
    vals = np.array([r[col] for r in per_matrix], dtype=np.float64)
    if where is not None:
        vals = vals[np.fromiter(where, dtype=bool, count=len(per_matrix))]
    return vals


def _safe_geomean(ratios: np.ndarray) -> float:
    """Geometric mean of a 1-D array of strictly-positive ratios."""
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size == 0:
        return float("nan")
    return float(np.exp(np.log(ratios).mean()))


def _percentile(arr: np.ndarray, q: float) -> float:
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else float("nan")


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #

def compute_summary_statistics(per_cell: list[dict],
                               per_matrix: list[dict]) -> list[dict]:
    """Compute the scalar variance / tail statistics emitted to CSV.

    Each entry in the returned list documents one statistic with:
      - id             stable identifier
      - description    what the number means
      - method         which method (or pair) the figure is about
      - value          the scalar result
      - units          unit of the value (count, fraction, ratio, arnoldi, ...)
      - slice          the subset of cells / matrices the value was computed on
      - sample_size    how many records contributed
      - formula        a short equation pointing into the per_cell or
                       per_matrix table, sufficient to re-derive the value
      - interpretation a one-line numerical restatement of the value
    """
    out: list[dict] = []

    # Index per_matrix by name for fast lookup.
    n_matrices_total = len(per_matrix)
    converged_mask = np.array([r["both_fully_converged"] for r in per_matrix], dtype=bool)
    n_both_converged = int(converged_mask.sum())

    # Also restrict the "safe" slice further to matrices where both methods
    # have nonzero mean Arnoldi -- this is the slice used for the worst/mean
    # tail figures, since the ratio is undefined when mean == 0.
    safe_mask = converged_mask & np.array([
        r["rand_gmres_arnoldi_mean"] > 0 and r["gmres_rl_arnoldi_mean"] > 0
        for r in per_matrix
    ], dtype=bool)
    n_safe = int(safe_mask.sum())

    # ---------------------------------------------------------------- 1. Convergence rates ---
    for m in METHODS:
        n_full = sum(1 for r in per_matrix
                     if r[f"{m}_n_converged"] == r[f"{m}_n_seeds"]
                     and r[f"{m}_n_seeds"] > 0)
        out.append({
            "id": f"convergence_rate_full_seed_{m}",
            "description": f"Matrices where all 5 seeds of {m} reached the relative-residual tolerance",
            "method": m,
            "value": n_full,
            "units": "count",
            "slice": f"all {n_matrices_total} matrices",
            "sample_size": n_matrices_total,
            "formula": f"sum(per_matrix.{m}_n_converged == per_matrix.{m}_n_seeds)",
            "interpretation":
                f"{m} fully converged on {n_full}/{n_matrices_total} matrices "
                f"({100*n_full/n_matrices_total:.1f}%).",
        })

    # ----------------------------------------- 2. Per-cell catastrophic events (cap hits) ---
    cells_by_method: dict[str, list[dict]] = {m: [] for m in METHODS}
    for r in per_cell:
        cells_by_method[r["method"]].append(r)
    for m in METHODS:
        cells = cells_by_method[m]
        n_cells = len(cells)
        n_capped = sum(1 for c in cells if c["total_arnoldi"] >= ARNOLDI_CAP)
        out.append({
            "id": f"per_cell_cap_hits_{m}",
            "description": f"Number of (matrix, seed) cells in which {m} hit the {ARNOLDI_CAP}-Arnoldi cap",
            "method": m,
            "value": n_capped,
            "units": "count",
            "slice": f"all {n_cells} (matrix, seed) cells across {n_matrices_total} matrices",
            "sample_size": n_cells,
            "formula": f"sum(per_cell_runs.method == '{m}' AND per_cell_runs.total_arnoldi >= {ARNOLDI_CAP})",
            "interpretation":
                f"{m} failed to reach tolerance in {n_capped}/{n_cells} "
                f"({100*n_capped/n_cells:.2f}%) seed-cells.",
        })

    # Reduction figure
    n_capped_rand = sum(1 for c in cells_by_method["rand_gmres"] if c["total_arnoldi"] >= ARNOLDI_CAP)
    n_capped_rl = sum(1 for c in cells_by_method["gmres_rl"] if c["total_arnoldi"] >= ARNOLDI_CAP)
    out.append({
        "id": "per_cell_cap_hits_reduction_rl_vs_rand",
        "description": "Fractional reduction in cap-hits going from rand_gmres to gmres_rl",
        "method": "comparison",
        "value": 1.0 - n_capped_rl / n_capped_rand if n_capped_rand else float("nan"),
        "units": "fraction",
        "slice": "all (matrix, seed) cells",
        "sample_size": len(cells_by_method["rand_gmres"]),
        "formula": "1 - (per_cell_cap_hits_gmres_rl / per_cell_cap_hits_rand_gmres)",
        "interpretation":
            f"DQN reduces catastrophic seed-cell failures by "
            f"{100*(1 - n_capped_rl / n_capped_rand):.1f}% relative to randGMRES "
            f"({n_capped_rand} -> {n_capped_rl}).",
    })

    # ----------------------------------------- 3. Full-matrix failures (all 5 seeds capped) ---
    for m in METHODS:
        n_full_fail = sum(1 for r in per_matrix
                          if r[f"{m}_n_seeds"] > 0 and r[f"{m}_n_capped"] == r[f"{m}_n_seeds"])
        out.append({
            "id": f"full_matrix_failures_{m}",
            "description": f"Matrices where every seed of {m} hit the cap",
            "method": m,
            "value": n_full_fail,
            "units": "count",
            "slice": f"all {n_matrices_total} matrices",
            "sample_size": n_matrices_total,
            "formula": f"sum(per_matrix.{m}_n_capped == per_matrix.{m}_n_seeds)",
            "interpretation":
                f"{m} gave up entirely on {n_full_fail}/{n_matrices_total} matrices.",
        })

    # ---------------------- 4. Partial failures (some seeds capped, others converged) ---
    for m in METHODS:
        n_partial = sum(
            1 for r in per_matrix
            if r[f"{m}_n_seeds"] > 0
            and 0 < r[f"{m}_n_capped"] < r[f"{m}_n_seeds"]
        )
        out.append({
            "id": f"partial_matrix_failures_{m}",
            "description": f"Matrices where some but not all seeds of {m} hit the cap",
            "method": m,
            "value": n_partial,
            "units": "count",
            "slice": f"all {n_matrices_total} matrices",
            "sample_size": n_matrices_total,
            "formula": f"sum(0 < per_matrix.{m}_n_capped < per_matrix.{m}_n_seeds)",
            "interpretation":
                f"{m} produced inconsistent (some pass, some fail) outcomes on "
                f"{n_partial}/{n_matrices_total} matrices.",
        })

    # ----------------------------------------- 5. Per-cell percentiles of total Arnoldi ---
    for m in METHODS:
        arn = np.array([c["total_arnoldi"] for c in cells_by_method[m]], dtype=np.float64)
        for q in (50, 75, 90, 95, 99):
            out.append({
                "id": f"per_cell_p{q}_arnoldi_{m}",
                "description": f"{q}th percentile of total Arnoldi across all {m} (matrix, seed) cells",
                "method": m,
                "value": float(np.percentile(arn, q)),
                "units": "arnoldi_iterations",
                "slice": f"all {len(arn)} (matrix, seed) cells",
                "sample_size": int(arn.size),
                "formula": f"numpy.percentile(per_cell_runs[per_cell_runs.method == '{m}'].total_arnoldi, {q})",
                "interpretation":
                    f"At p{q}, {m}'s seed-cell needed {int(np.percentile(arn, q))} Arnoldi iterations.",
            })

    # ----------------------------------------- 6. Per-matrix seed-std (Arnoldi) ---
    rand_std = _arr(per_matrix, "rand_gmres_arnoldi_std", converged_mask)
    rl_std = _arr(per_matrix, "gmres_rl_arnoldi_std", converged_mask)

    out.extend([
        {
            "id": "median_seed_std_rand",
            "description": "Median (across matrices) of seed-to-seed std in total Arnoldi for rand_gmres",
            "method": "rand_gmres",
            "value": float(np.median(rand_std)),
            "units": "arnoldi_iterations",
            "slice": f"{n_both_converged} matrices where both methods fully converged on all 5 seeds",
            "sample_size": n_both_converged,
            "formula": "numpy.median(per_matrix[both_fully_converged].rand_gmres_arnoldi_std)",
            "interpretation":
                f"On a typical matrix, randGMRES's per-seed Arnoldi count varies by std "
                f"{int(np.median(rand_std))}.",
        },
        {
            "id": "median_seed_std_rl",
            "description": "Median (across matrices) of seed-to-seed std in total Arnoldi for gmres_rl",
            "method": "gmres_rl",
            "value": float(np.median(rl_std)),
            "units": "arnoldi_iterations",
            "slice": f"{n_both_converged} matrices where both methods fully converged on all 5 seeds",
            "sample_size": n_both_converged,
            "formula": "numpy.median(per_matrix[both_fully_converged].gmres_rl_arnoldi_std)",
            "interpretation":
                f"On a typical matrix, DQN's per-seed Arnoldi count varies by std "
                f"{int(np.median(rl_std))}.",
        },
        {
            "id": "median_seed_std_reduction_rl_vs_rand",
            "description": "Fractional reduction in median seed-std going from rand_gmres to gmres_rl",
            "method": "comparison",
            "value": 1.0 - float(np.median(rl_std)) / float(np.median(rand_std)),
            "units": "fraction",
            "slice": f"{n_both_converged} matrices where both methods fully converged",
            "sample_size": n_both_converged,
            "formula": "1 - median_seed_std_rl / median_seed_std_rand",
            "interpretation":
                f"DQN's typical seed-to-seed std is "
                f"{100*(1 - float(np.median(rl_std)) / float(np.median(rand_std))):.1f}% lower than randGMRES.",
        },
        {
            "id": "mean_seed_std_rand",
            "description": "Arithmetic mean of seed-std across matrices for rand_gmres",
            "method": "rand_gmres",
            "value": float(rand_std.mean()),
            "units": "arnoldi_iterations",
            "slice": f"{n_both_converged} matrices where both methods fully converged",
            "sample_size": n_both_converged,
            "formula": "numpy.mean(per_matrix[both_fully_converged].rand_gmres_arnoldi_std)",
            "interpretation":
                f"Mean per-matrix seed-std (random): {rand_std.mean():.0f}.",
        },
        {
            "id": "mean_seed_std_rl",
            "description": "Arithmetic mean of seed-std across matrices for gmres_rl",
            "method": "gmres_rl",
            "value": float(rl_std.mean()),
            "units": "arnoldi_iterations",
            "slice": f"{n_both_converged} matrices where both methods fully converged",
            "sample_size": n_both_converged,
            "formula": "numpy.mean(per_matrix[both_fully_converged].gmres_rl_arnoldi_std)",
            "interpretation":
                f"Mean per-matrix seed-std (DQN): {rl_std.mean():.0f}.",
        },
    ])

    # Geo-mean of per-matrix std ratio. Both arrays masked to strictly-positive
    # entries first; otherwise a zero std on either side would produce a
    # degenerate 0 or infinity that contaminates the geometric mean.
    _std_mask = (rand_std > 0) & (rl_std > 0)
    _gm_std = (
        _safe_geomean(rl_std[_std_mask] / rand_std[_std_mask])
        if _std_mask.any() else float("nan")
    )
    out.append({
        "id": "geomean_std_ratio_rl_over_rand",
        "description": "Geometric mean of per-matrix std ratio gmres_rl_std / rand_gmres_std",
        "method": "comparison",
        "value": _gm_std,
        "units": "ratio",
        "slice": f"{int(_std_mask.sum())} matrices where both methods fully converged AND both stds > 0",
        "sample_size": int(_std_mask.sum()),
        "formula": "exp(mean(log(per_matrix.gmres_rl_arnoldi_std / per_matrix.rand_gmres_arnoldi_std))), "
                   "filtered to rows where both stds > 0 (a zero std would otherwise "
                   "produce a degenerate ratio of 0 or infinity).",
        "interpretation":
            f"Geo-mean per-matrix std ratio (rl/rand) = {_gm_std:.3f}; "
            f"DQN's per-matrix std is on average ~"
            f"{100*(1 - _gm_std):.1f}% lower.",
    })

    # Per-matrix wins
    n_rl_lower_std = sum(
        1 for r in per_matrix
        if r["both_fully_converged"] == 1 and r["rl_std_lower_than_rand"] == 1
    )
    n_rl_lower_cv = sum(
        1 for r in per_matrix
        if r["both_fully_converged"] == 1 and r["rl_cv_lower_than_rand"] == 1
    )
    out.extend([
        {
            "id": "n_matrices_rl_std_lower_than_rand",
            "description": "Matrices where DQN's seed-std is strictly less than rand's",
            "method": "comparison",
            "value": n_rl_lower_std,
            "units": "count",
            "slice": f"{n_both_converged} matrices where both methods fully converged",
            "sample_size": n_both_converged,
            "formula": "sum(per_matrix[both_fully_converged].rl_std_lower_than_rand)",
            "interpretation":
                f"DQN had lower seed-std than random on "
                f"{n_rl_lower_std}/{n_both_converged} ({100*n_rl_lower_std/n_both_converged:.0f}%) matrices.",
        },
        {
            "id": "n_matrices_rl_cv_lower_than_rand",
            "description": "Matrices where DQN's coefficient of variation is strictly less than rand's",
            "method": "comparison",
            "value": n_rl_lower_cv,
            "units": "count",
            "slice": f"{n_both_converged} matrices where both methods fully converged",
            "sample_size": n_both_converged,
            "formula": "sum(per_matrix[both_fully_converged].rl_cv_lower_than_rand)",
            "interpretation":
                f"DQN had lower coefficient of variation than random on "
                f"{n_rl_lower_cv}/{n_both_converged} ({100*n_rl_lower_cv/n_both_converged:.0f}%) matrices.",
        },
    ])

    # Coefficient of variation medians.
    # CV = std / mean is undefined when mean = 0, so per_matrix carries NaN
    # in those rows. Filter NaNs before the median; the sample size reported
    # in the CSV reflects the actual number of finite-CV rows used.
    rand_cv_all = _arr(per_matrix, "rand_gmres_arnoldi_cv", converged_mask)
    rl_cv_all = _arr(per_matrix, "gmres_rl_arnoldi_cv", converged_mask)
    rand_cv = rand_cv_all[np.isfinite(rand_cv_all)]
    rl_cv = rl_cv_all[np.isfinite(rl_cv_all)]
    out.extend([
        {
            "id": "median_cv_rand",
            "description": "Median (across matrices) of std/mean for rand_gmres",
            "method": "rand_gmres",
            "value": float(np.median(rand_cv)) if rand_cv.size else float("nan"),
            "units": "ratio",
            "slice": f"{int(rand_cv.size)} matrices where both methods fully converged "
                     f"AND rand_gmres mean Arnoldi > 0",
            "sample_size": int(rand_cv.size),
            "formula": "numpy.median(per_matrix[both_fully_converged "
                       "AND rand_gmres_arnoldi_mean > 0].rand_gmres_arnoldi_cv)",
            "interpretation":
                f"Typical randGMRES coefficient of variation: "
                f"{100 * float(np.median(rand_cv)):.2f}%."
                if rand_cv.size else "no rows with finite CV",
        },
        {
            "id": "median_cv_rl",
            "description": "Median (across matrices) of std/mean for gmres_rl",
            "method": "gmres_rl",
            "value": float(np.median(rl_cv)) if rl_cv.size else float("nan"),
            "units": "ratio",
            "slice": f"{int(rl_cv.size)} matrices where both methods fully converged "
                     f"AND gmres_rl mean Arnoldi > 0",
            "sample_size": int(rl_cv.size),
            "formula": "numpy.median(per_matrix[both_fully_converged "
                       "AND gmres_rl_arnoldi_mean > 0].gmres_rl_arnoldi_cv)",
            "interpretation":
                f"Typical DQN coefficient of variation: "
                f"{100 * float(np.median(rl_cv)):.2f}%."
                if rl_cv.size else "no rows with finite CV",
        },
    ])

    # ----------------------------------------- 7. Per-matrix tail of worst/mean ratio ---
    for m in METHODS:
        col = f"{m}_worst_over_mean"
        vals = _arr(per_matrix, col, safe_mask)
        for q in (50, 90, 95, 99):
            out.append({
                "id": f"worst_over_mean_p{q}_{m}",
                "description": f"{q}th percentile of (max_seed / mean_seed) Arnoldi for {m}",
                "method": m,
                "value": _percentile(vals, q),
                "units": "ratio",
                "slice": f"{n_safe} matrices where both methods fully converged AND both means > 0",
                "sample_size": n_safe,
                "formula": f"numpy.percentile(per_matrix[safe].{col}, {q})",
                "interpretation":
                    f"At p{q}, {m}'s worst seed within a matrix is "
                    f"{_percentile(vals, q):.3f}x the per-matrix mean.",
            })

    # ----------------------------------------- 8. Per-matrix (max - min) / mean spread ---
    for m in METHODS:
        col = f"{m}_minmax_spread_over_mean"
        vals = _arr(per_matrix, col, safe_mask)
        for q in (50, 90, 99):
            out.append({
                "id": f"minmax_spread_over_mean_p{q}_{m}",
                "description": f"{q}th percentile of (max_seed - min_seed)/mean Arnoldi for {m}",
                "method": m,
                "value": _percentile(vals, q),
                "units": "ratio",
                "slice": f"{n_safe} matrices where both methods fully converged AND both means > 0",
                "sample_size": n_safe,
                "formula": f"numpy.percentile(per_matrix[safe].{col}, {q})",
                "interpretation":
                    f"At p{q}, the seed-band of {m} on a matrix spans "
                    f"{_percentile(vals, q):.3f} * its per-matrix mean.",
            })

    # ----------------------------------------- 9. Geo-mean Arnoldi ratio (rl/rand) ---
    rand_mean = _arr(per_matrix, "rand_gmres_arnoldi_mean", converged_mask)
    rl_mean = _arr(per_matrix, "gmres_rl_arnoldi_mean", converged_mask)
    valid = (rand_mean > 0) & (rl_mean > 0)
    arn_ratio = _safe_geomean(rl_mean[valid] / rand_mean[valid])
    out.append({
        "id": "geomean_arnoldi_ratio_rl_over_rand",
        "description": "Geometric mean of per-matrix Arnoldi mean ratio gmres_rl / rand_gmres "
                       "(<1 means DQN uses fewer Arnoldi iterations on average)",
        "method": "comparison",
        "value": arn_ratio,
        "units": "ratio",
        "slice": f"{int(valid.sum())} matrices where both methods fully converged AND both means > 0",
        "sample_size": int(valid.sum()),
        "formula": "exp( mean( log(per_matrix.gmres_rl_arnoldi_mean / per_matrix.rand_gmres_arnoldi_mean) ) ), "
                   "filtered to rows where both means > 0",
        "interpretation":
            f"DQN uses ~{(1 - arn_ratio) * 100:+.1f}% the Arnoldi count of randGMRES on average "
            f"(geo-mean ratio = {arn_ratio:.3f}; values <1 favor DQN).",
    })

    return out


# --------------------------------------------------------------------------- #
# CSV writers
# --------------------------------------------------------------------------- #

def _write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_per_cell_csv(per_cell: list[dict], out_dir: Path) -> Path:
    fieldnames = [
        "matrix", "method", "seed",
        "total_arnoldi", "elapsed_seconds",
        "converged", "final_relative_residual",
    ]
    path = out_dir / "per_cell_runs.csv"
    _write_csv(per_cell, path, fieldnames)
    return path


def write_per_matrix_csv(per_matrix: list[dict], out_dir: Path) -> Path:
    base = ["matrix", "both_fully_converged",
            "rl_std_lower_than_rand", "rl_cv_lower_than_rand"]
    method_cols = []
    for m in METHODS:
        method_cols.extend([
            f"{m}_n_seeds", f"{m}_n_converged", f"{m}_n_capped",
            f"{m}_arnoldi_mean", f"{m}_arnoldi_std", f"{m}_arnoldi_cv",
            f"{m}_arnoldi_min", f"{m}_arnoldi_max",
            f"{m}_worst_over_mean", f"{m}_minmax_spread_over_mean",
        ])
    fieldnames = base + method_cols
    path = out_dir / "per_matrix_statistics.csv"
    _write_csv(per_matrix, path, fieldnames)
    return path


def write_summary_csv(summary: list[dict], out_dir: Path) -> Path:
    fieldnames = ["id", "description", "method", "value", "units",
                  "slice", "sample_size", "formula", "interpretation"]
    path = out_dir / "summary_statistics.csv"
    _write_csv(summary, path, fieldnames)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="Path to the partial / full Peairs-style benchmark JSON.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT),
                        help="Directory to write the three CSV outputs into.")
    args = parser.parse_args()

    json_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    print(f"Streaming {json_path} ...")
    per_cell = load_per_cell_records(json_path)
    print(f"  collected {len(per_cell):,} (matrix, method, seed) cells")

    print("Aggregating to per-matrix statistics ...")
    per_matrix = build_per_matrix_table(per_cell)
    print(f"  {len(per_matrix)} matrices")

    print("Computing summary statistics ...")
    summary = compute_summary_statistics(per_cell, per_matrix)
    print(f"  {len(summary)} summary rows")

    cell_csv = write_per_cell_csv(per_cell, out_dir)
    matrix_csv = write_per_matrix_csv(per_matrix, out_dir)
    summary_csv = write_summary_csv(summary, out_dir)
    print(f"\nwrote:")
    print(f"  {cell_csv}")
    print(f"  {matrix_csv}")
    print(f"  {summary_csv}")


if __name__ == "__main__":
    main()

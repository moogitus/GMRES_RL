"""
Log-log slope of mean total Arnoldi iterations vs. κ_2(A) for each
method on the κ-sweep (§4.3, slopes quoted in the discussion of Figure 1).
Reads kappa_sweep.json, fits a power law to each method's per-matrix
mean, and writes a companion CSV with bootstrap 95% CIs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT = REPO_ROOT / "results" / "kappa_sweep" / "kappa_sweep.json"
OUTPUT = REPO_ROOT / "results" / "kappa_sweep" / "kappa_sweep_slopes.csv"

METHODS = [
    ("gmres20_seeds",   "gmres20",    "GMRES(20)"),
    ("rand_gmres_seeds", "rand_gmres", "randGMRES"),
    ("dqn_seeds",       "gmres_rl",   "DQN"),
]


def main(n_bootstrap: int = 2000, seed: int = 0) -> None:
    data = json.loads(INPUT.read_text())
    rows = sorted(data["results"], key=lambda r: r["condition_number"])
    kappas = np.array([r["condition_number"] for r in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)

    csv_rows = []
    for alg_key, short_name, label in METHODS:
        means = np.array([
            np.mean([run["total_arnoldi"] for rhs in r["per_rhs"] for run in rhs[alg_key]])
            for r in rows
        ], dtype=np.float64)
        valid = (kappas > 0) & (means > 0)
        x = np.log10(kappas[valid])
        y = np.log10(means[valid])
        slope, intercept = np.polyfit(x, y, 1)

        # percentile-bootstrap CI by resampling matrices with replacement
        boot_slopes = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            idx = rng.integers(0, x.size, size=x.size)
            boot_slopes[i] = np.polyfit(x[idx], y[idx], 1)[0]
        ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])

        csv_rows.append({
            "id": f"slope_arnoldi_vs_kappa_{short_name}",
            "method": short_name,
            "label": label,
            "slope": f"{slope:.4f}",
            "intercept": f"{intercept:.4f}",
            "ci95_low": f"{ci_low:.4f}",
            "ci95_high": f"{ci_high:.4f}",
            "n_matrices": int(valid.sum()),
            "formula": "numpy.polyfit(log10(condition_number), log10(mean_total_arnoldi), 1)",
            "interpretation": f"{label} mean Arnoldi grows as kappa^{slope:.2f} (95% CI [{ci_low:.2f}, {ci_high:.2f}]).",
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"wrote {OUTPUT}")
    for row in csv_rows:
        print(f"  {row['label']:10s}  slope={row['slope']}  "
              f"95% CI=[{row['ci95_low']}, {row['ci95_high']}]")


if __name__ == "__main__":
    main()

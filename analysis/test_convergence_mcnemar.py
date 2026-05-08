"""
analysis/test_convergence_mcnemar.py

Tests whether the per-cell convergence rates of two restart-selection
methods differ under matched-pair structure. Each (matrix, seed) cell is
the same linear system seen by both methods, so the appropriate test for
the binary outcome ``converged at relative residual <= 1e-6 within the
Arnoldi cap`` is McNemar's test on the paired 2x2 contingency table.

H0: the two methods are equally likely to flip a discordant pair, i.e.,
        P(method_A converges and method_B does not)
    =   P(method_B converges and method_A does not).

Under H0, the discordant pairs follow a Binomial(n_disc, 0.5)
distribution. The script reports both the exact (binomial) p-value and
the continuity-corrected asymptotic chi-squared p-value, the marginal
rates, the rate difference, and a 95% Newcombe-Wilson hybrid CI on the
difference.

Inputs
------
  analysis/results/variance_validation/per_cell_runs.csv
  Produced by analysis/validate_variance_figures.py.

Outputs
-------
  analysis/results/variance_validation/mcnemar_convergence.csv
  One row per scalar quantity (counts, marginals, test statistics,
  CIs, effect sizes), each with a stable id and a description.

Run
---
    python analysis/test_convergence_mcnemar.py
    python analysis/test_convergence_mcnemar.py \\
        --method-a gmres_rl --method-b rand_gmres
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest, chi2, norm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "analysis" / "results" / "variance_validation" / "per_cell_runs.csv"
DEFAULT_OUT = REPO_ROOT / "analysis" / "results" / "variance_validation" / "mcnemar_convergence.csv"

# Default: hypothesise gmres_rl converges more often than rand_gmres.
# Swap with --method-a / --method-b for any other matched comparison.
DEFAULT_METHOD_A = "gmres_rl"
DEFAULT_METHOD_B = "rand_gmres"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_paired_outcomes(csv_path: Path,
                          method_a: str,
                          method_b: str) -> list[tuple[str, int, bool, bool]]:
    """Return a list of (matrix, seed, A_converged, B_converged) tuples,
    one per matched (matrix, seed) cell where both methods have a run.
    Cells missing one method are dropped from the test.
    """
    by_cell: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row["method"] not in (method_a, method_b):
                continue
            key = (row["matrix"], int(row["seed"]))
            # The CSV stores Python booleans as 'True'/'False' strings.
            by_cell[key][row["method"]] = row["converged"].strip().lower() == "true"

    pairs: list[tuple[str, int, bool, bool]] = []
    dropped = 0
    for (matrix, seed), d in sorted(by_cell.items()):
        if method_a in d and method_b in d:
            pairs.append((matrix, seed, d[method_a], d[method_b]))
        else:
            dropped += 1
    if dropped:
        print(f"  dropped {dropped} unmatched cells (only one method present)")
    return pairs


# --------------------------------------------------------------------------- #
# Contingency table
# --------------------------------------------------------------------------- #

def contingency_table(pairs: list[tuple[str, int, bool, bool]]
                       ) -> tuple[int, int, int, int]:
    """Return (n_AA, n_AB_only, n_BA_only, n_neither):

        n_AA       both methods converged
        n_AB_only  method A converged, method B did not  (favors A)
        n_BA_only  method B converged, method A did not  (favors B)
        n_neither  both methods failed
    """
    n_AA = sum(1 for _, _, a, b in pairs if a and b)
    n_AB_only = sum(1 for _, _, a, b in pairs if a and not b)
    n_BA_only = sum(1 for _, _, a, b in pairs if (not a) and b)
    n_neither = sum(1 for _, _, a, b in pairs if (not a) and (not b))
    return n_AA, n_AB_only, n_BA_only, n_neither


# --------------------------------------------------------------------------- #
# McNemar's test
# --------------------------------------------------------------------------- #

def mcnemar_exact(n_disc_a: int, n_disc_b: int) -> float:
    """Two-sided exact McNemar p-value via the binomial test on the
    discordant cells. Recommended whenever n_disc_a + n_disc_b is small
    (rule of thumb < 25); always conservative.

    Under H0, n_disc_a ~ Binomial(n_disc_a + n_disc_b, 0.5).
    """
    n = n_disc_a + n_disc_b
    if n == 0:
        return float("nan")
    k = min(n_disc_a, n_disc_b)
    return float(binomtest(k, n, p=0.5, alternative="two-sided").pvalue)


def mcnemar_asymptotic(n_disc_a: int, n_disc_b: int,
                       continuity: bool = True) -> tuple[float, float]:
    """McNemar's chi-squared statistic and asymptotic two-sided p-value.

    With Edwards' continuity correction (default), the statistic is

        chi^2 = (|n_disc_a - n_disc_b| - 1)^2 / (n_disc_a + n_disc_b)

    distributed under H0 as chi-squared with 1 d.f. The continuity
    correction matters only for moderate sample sizes; for the exact
    p-value, use ``mcnemar_exact``.
    """
    n = n_disc_a + n_disc_b
    if n == 0:
        return float("nan"), float("nan")
    diff = abs(n_disc_a - n_disc_b)
    if continuity:
        diff = max(diff - 1, 0)
    chi_sq = (diff ** 2) / n
    p_value = float(chi2.sf(chi_sq, df=1))
    return chi_sq, p_value


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion k/n at level 1 - alpha."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = float(norm.ppf(1 - alpha / 2))
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    halfwidth = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (center - halfwidth, center + halfwidth)


def newcombe_paired_ci(n_AA: int, n_AB_only: int, n_BA_only: int, n_neither: int,
                       alpha: float = 0.05) -> tuple[float, float]:
    """Newcombe (1998) hybrid Wilson interval for the difference in paired
    proportions, p_A - p_B. Recommended over the Wald paired CI when
    proportions are near 0 or 1, which is the case here (rates ~0.9).

    Reference: Newcombe, R.G. (1998). "Improved confidence intervals
    for the difference between binomial proportions based on paired data."
    Statistics in Medicine, 17(22), 2635-2650.
    """
    n = n_AA + n_AB_only + n_BA_only + n_neither
    if n == 0:
        return (float("nan"), float("nan"))

    n_A_yes = n_AA + n_AB_only      # method A converged
    n_B_yes = n_AA + n_BA_only      # method B converged
    p_A = n_A_yes / n
    p_B = n_B_yes / n

    lA, uA = wilson_ci(n_A_yes, n, alpha=alpha)
    lB, uB = wilson_ci(n_B_yes, n, alpha=alpha)

    # Phi (Pearson) correlation of the paired binary outcomes; falls back
    # to 0 when any marginal vanishes (the formula's denominator becomes 0).
    row1 = n_A_yes
    row0 = n - n_A_yes
    col1 = n_B_yes
    col0 = n - n_B_yes
    denom_phi = math.sqrt(max(row1 * row0 * col1 * col0, 1e-12))
    if min(row1, row0, col1, col0) == 0:
        phi = 0.0
    else:
        phi = (n_AA * n_neither - n_AB_only * n_BA_only) / denom_phi

    # Newcombe's "method 10": clamp the two-by-two phi-adjusted variance
    # contributions to be non-negative under the radical.
    L = math.sqrt(max((p_A - lA) ** 2
                      - 2 * phi * (p_A - lA) * (uB - p_B)
                      + (uB - p_B) ** 2, 0.0))
    U = math.sqrt(max((uA - p_A) ** 2
                      - 2 * phi * (uA - p_A) * (p_B - lB)
                      + (p_B - lB) ** 2, 0.0))

    delta = p_A - p_B
    return (delta - L, delta + U)


# --------------------------------------------------------------------------- #
# CSV writer
# --------------------------------------------------------------------------- #

def write_results_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "description", "value", "interpretation"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="per_cell_runs.csv produced by validate_variance_figures.py")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                        help="Where to write the test results CSV.")
    parser.add_argument("--method-a", type=str, default=DEFAULT_METHOD_A,
                        help="Method whose convergence rate we hypothesise is higher.")
    parser.add_argument("--method-b", type=str, default=DEFAULT_METHOD_B,
                        help="Comparison method.")
    args = parser.parse_args()

    csv_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    print(f"Reading {csv_path}")
    pairs = load_paired_outcomes(csv_path, args.method_a, args.method_b)
    print(f"  matched cells: {len(pairs)}")

    n_AA, n_AB, n_BA, n_NN = contingency_table(pairs)
    n_total = n_AA + n_AB + n_BA + n_NN
    n_disc = n_AB + n_BA

    p_A = (n_AA + n_AB) / n_total
    p_B = (n_AA + n_BA) / n_total
    delta = p_A - p_B

    chi_sq, p_asymp = mcnemar_asymptotic(n_AB, n_BA, continuity=True)
    p_exact = mcnemar_exact(n_AB, n_BA)
    ci_low, ci_high = newcombe_paired_ci(n_AA, n_AB, n_BA, n_NN)

    discordant_ratio = (n_AB / n_BA) if n_BA > 0 else float("inf")

    # Console summary -------------------------------------------------------
    print()
    print("=== McNemar's test on convergence rate ===")
    print(f"  H0: P({args.method_a}-only conv) == P({args.method_b}-only conv)")
    print(f"  H1: those probabilities differ (two-sided)")
    print()
    print(f"  Contingency table over {n_total} matched (matrix, seed) cells:")
    print(f"                            {args.method_b}: conv     {args.method_b}: fail")
    print(f"    {args.method_a}: conv         {n_AA:>8d}              {n_AB:>8d}")
    print(f"    {args.method_a}: fail         {n_BA:>8d}              {n_NN:>8d}")
    print()
    print(f"  Marginal rates: {args.method_a} = {p_A:.4f},  {args.method_b} = {p_B:.4f}")
    print(f"  Difference (A - B): {delta:+.4f}")
    print(f"  Newcombe 95% CI on the difference: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print()
    print(f"  Discordant cells: {n_disc} total ({n_AB} favor {args.method_a}, {n_BA} favor {args.method_b})")
    print(f"  Discordant ratio (A-only / B-only): {discordant_ratio:.3f}")
    print()
    print(f"  McNemar exact (binomial)  p = {p_exact:.6e}")
    print(f"  McNemar asymptotic chi^2  = {chi_sq:.4f},  p = {p_asymp:.6e}")
    print(f"     (continuity-corrected, df = 1)")

    # Results CSV -----------------------------------------------------------
    interpretation = (
        f"{args.method_a} converged on {100*p_A:.2f}% of cells vs "
        f"{args.method_b}'s {100*p_B:.2f}%. Of {n_disc} discordant cells, "
        f"{n_AB} favor {args.method_a} and {n_BA} favor {args.method_b}. "
        f"Exact two-sided McNemar p = {p_exact:.3e}."
    )
    rows = [
        {"id": "method_a", "description": "First method label",
         "value": args.method_a, "interpretation": ""},
        {"id": "method_b", "description": "Second method label",
         "value": args.method_b, "interpretation": ""},
        {"id": "n_paired_cells",
         "description": "Number of (matrix, seed) cells with runs from both methods",
         "value": n_total,
         "interpretation": f"Test sample size: {n_total} matched cells."},
        {"id": "n_AA_both_converged",
         "description": "Cells where both methods converged",
         "value": n_AA,
         "interpretation": f"{n_AA} cells: both methods reached tolerance."},
        {"id": "n_AB_only_method_a_converged",
         "description": f"Cells where {args.method_a} converged but {args.method_b} did not",
         "value": n_AB,
         "interpretation": f"{n_AB} discordant cells favor {args.method_a}."},
        {"id": "n_BA_only_method_b_converged",
         "description": f"Cells where {args.method_b} converged but {args.method_a} did not",
         "value": n_BA,
         "interpretation": f"{n_BA} discordant cells favor {args.method_b}."},
        {"id": "n_neither_converged",
         "description": "Cells where neither method converged",
         "value": n_NN,
         "interpretation": f"{n_NN} cells: both methods hit the cap."},
        {"id": f"convergence_rate_{args.method_a}",
         "description": f"Marginal convergence rate of {args.method_a}",
         "value": p_A,
         "interpretation": f"{args.method_a} converges on {100*p_A:.2f}% of cells."},
        {"id": f"convergence_rate_{args.method_b}",
         "description": f"Marginal convergence rate of {args.method_b}",
         "value": p_B,
         "interpretation": f"{args.method_b} converges on {100*p_B:.2f}% of cells."},
        {"id": "convergence_rate_difference_a_minus_b",
         "description": "Marginal rate difference, method A minus method B",
         "value": delta,
         "interpretation":
             f"{args.method_a} converges on {100*delta:+.2f} percentage points more cells "
             f"than {args.method_b}."},
        {"id": "ci95_diff_low",
         "description": "Newcombe (1998) hybrid Wilson 95% CI lower bound on the rate difference",
         "value": ci_low,
         "interpretation": f"95% lower CI on the difference: {ci_low:+.4f}."},
        {"id": "ci95_diff_high",
         "description": "Newcombe (1998) hybrid Wilson 95% CI upper bound on the rate difference",
         "value": ci_high,
         "interpretation": f"95% upper CI on the difference: {ci_high:+.4f}."},
        {"id": "n_discordant",
         "description": "Total discordant cells (n_AB + n_BA); the only cells contributing to the McNemar statistic",
         "value": n_disc,
         "interpretation": f"{n_disc} discordant cells; "
                           f"under H0 these split as Binomial({n_disc}, 0.5)."},
        {"id": "discordant_ratio_a_over_b",
         "description": "Ratio n_AB / n_BA: multiplicative advantage of method A on discordant pairs",
         "value": discordant_ratio,
         "interpretation":
             f"{args.method_a} wins discordant cells {discordant_ratio:.2f}:1 over {args.method_b}."
             if math.isfinite(discordant_ratio)
             else f"{args.method_a} wins all discordant cells (ratio = infinity)."},
        {"id": "mcnemar_chi_squared_continuity_corrected",
         "description": "Edwards' continuity-corrected McNemar chi-squared statistic on discordant cells",
         "value": chi_sq,
         "interpretation": f"chi^2 = {chi_sq:.4f}."},
        {"id": "mcnemar_p_value_asymptotic",
         "description": "Two-sided p-value from the chi-squared (1 df) distribution",
         "value": p_asymp,
         "interpretation": f"asymptotic p = {p_asymp:.4e}."},
        {"id": "mcnemar_p_value_exact",
         "description": "Exact two-sided binomial p-value on the n_AB vs n_BA split (preferred for any sample size)",
         "value": p_exact,
         "interpretation": f"exact p = {p_exact:.4e}."},
        {"id": "summary",
         "description": "One-line natural-language summary of the test",
         "value": "",
         "interpretation": interpretation},
    ]
    write_results_csv(rows, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

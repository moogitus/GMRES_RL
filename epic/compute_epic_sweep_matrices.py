"""
EPIC analysis on the heterogeneous SuiteSparse HPO sweep matrices
(§3.3, §7.3, Table 1, HPO column). Companion to compute_epic_rewards.py
(synthetic convdiff coverage); reuses its reward definitions, EPIC
canonicalization, Pearson distance, and bootstrap.

Differences from the convdiff variant:
  - transition distribution is the 16 SuiteSparse matrices in
    matrices/hpo_matrices/ instead of synthetic 1D convection-diffusion;
  - uses GMRESEnv (discrete action; relative residual norms ρ = ||r||/||b||)
    so transitions are scale-invariant across the heterogeneous suite;
  - default --convergence-bonus is 9 (the value used in DQN training);
  - writes epic_sweep_*.csv to avoid overwriting the convdiff outputs.

Transition representation: each row is (ρ_{t-1}, m, ρ_t, converged).
Initial residual is ρ_0 = 1 by construction, convergence threshold is
relative (default 1e-6).

Coverage modes (same semantics as compute_epic_rewards.py):
  'random'  m ~ Uniform{1, ..., m_max} each cycle
  'fixed20' m = m_max every cycle
  'mixed'   num_rollouts of 'random' followed by num_rollouts of 'fixed20'
            per matrix; broad action coverage, fixed-m anchor present.

Sanity check: D_EPIC(R_PBRS, R_work) should be ≈ 0 when convergence_bonus
is 0, since EPIC is invariant to PBRS and R_PBRS - R_work = γ Φ(s') - Φ(s).

Examples (from repo root):
    python epic/compute_epic_sweep_matrices.py
    python epic/compute_epic_sweep_matrices.py \\
        --matrices-dir matrices/hpo_matrices \\
        --convergence-bonus 9 --coverage mixed --bootstrap 200 \\
        --out-matrix epic/epic_sweep_distance_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import sys
import tarfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import loadmat, mmread
from scipy.sparse import csr_matrix, issparse

# path setup: make both epic/ and src/ importable regardless of cwd
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_GMRES_SRC = _PROJECT_ROOT / "src"

for _p in [str(_SCRIPT_DIR), str(_GMRES_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# shared EPIC machinery (reward defs, canonicalization, bootstrap, IO).
# collect_transitions is provided locally; everything else is reused.
from compute_epic_rewards import (  # noqa: E402
    RewardCfg,
    REWARD_REGISTRY,
    eval_reward_vec,
    canonicalize,
    epic_distance,
    bootstrap_epic,
    pearson_distance,
    save_distance_matrix_csv,
    save_bootstrap_csv,
    save_transitions_csv,
    print_distance_matrix,
    print_interpretation,
    Transition,
)

from env import GMRESEnv  # noqa: E402


# default sweep matrix set; mirrors matrices/hpo_matrices/
SWEEP_MATRIX_NAMES: List[str] = [
    "1138_bus",
    "G2_circuit",
    "Pres_Poisson",
    "aft01",
    "bundle1",
    "cfd1",
    "crankseg_1",
    "ct20stif",
    "ex13",
    "finan512",
    "mhd1280b",
    "nasa2910",
    "offshore",
    "parabolic_fem",
    "raefsky4",
    "thermal2",
]


# matrix loading: SuiteSparse .tar.gz, .mtx[.gz], or .mat
def _largest_matrix_member(tar: tarfile.TarFile) -> tarfile.TarInfo:
    members = [
        m for m in tar.getmembers()
        if m.name.endswith(".mtx") and not m.name.endswith("_b.mtx")
    ]
    if not members:
        raise ValueError("No matrix .mtx file found in archive.")
    return max(members, key=lambda m: m.size)


def _rhs_member(tar: tarfile.TarFile) -> Optional[tarfile.TarInfo]:
    members = [m for m in tar.getmembers() if m.name.endswith("_b.mtx")]
    return max(members, key=lambda m: m.size) if members else None


def _flatten_rhs(data) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _load_mtx_file(path: Path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as handle:
        return mmread(io.BytesIO(handle.read()))


def _load_mat_file(path: Path) -> Tuple:
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


def _validate_problem(name: str, A, b) -> Tuple:
    A = csr_matrix(A.astype(np.float64))
    b = _flatten_rhs(b)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{name}: matrix is not square ({A.shape})")
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"{name}: RHS dimension mismatch")
    return A, b


def _consistent_rhs(A) -> np.ndarray:
    x_true = np.ones(A.shape[1], dtype=np.float64)
    return np.asarray(A @ x_true, dtype=np.float64).reshape(-1)


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _recursive_candidates(root: Path, patterns: List[str]) -> List[Path]:
    candidates: List[Path] = []
    seen: set = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(path)
    return candidates


def _find_local_archive(name: str, matrices_dir: Path) -> Optional[Path]:
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


def _find_local_mat(name: str, matrices_dir: Path) -> Optional[Path]:
    direct = _first_existing([
        matrices_dir / f"{name}.mat",
        matrices_dir / name / f"{name}.mat",
    ])
    if direct is not None:
        return direct
    matches = _recursive_candidates(matrices_dir, [f"{name}.mat"])
    return matches[0] if matches else None


def load_problem(
    config: dict,
    matrices_dir: Path,
    force_consistent_rhs: bool = False,
) -> Tuple:
    """
    Load a named matrix problem from local files.
    Returns (A, b) with A as CSR float64 and b as 1-D float64 ndarray.
    """
    matrices_dir = matrices_dir.expanduser().resolve()
    if not matrices_dir.exists():
        raise FileNotFoundError(f"Matrix directory not found: {matrices_dir}")

    name = config["name"]

    mat_path = _find_local_mat(name, matrices_dir)
    if mat_path is not None:
        A, b = _load_mat_file(mat_path)
        if force_consistent_rhs or b is None:
            b = _consistent_rhs(A)
        return _validate_problem(name, A, b)

    tar_path = _find_local_archive(name, matrices_dir)
    if tar_path is not None:
        with tarfile.open(tar_path) as tar:
            matrix_member = _largest_matrix_member(tar)
            rhs_member_info = _rhs_member(tar)
            with tar.extractfile(matrix_member) as handle:
                A = mmread(io.BytesIO(handle.read()))
            if force_consistent_rhs or rhs_member_info is None:
                b = _consistent_rhs(A)
            else:
                with tar.extractfile(rhs_member_info) as handle:
                    b = mmread(io.BytesIO(handle.read()))
        return _validate_problem(name, A, b)

    raise FileNotFoundError(
        f"Could not find matrix {name!r} in {matrices_dir}. "
        "Expected a .mat or .tar.gz / .tgz archive."
    )


# transition collection over the sweep matrices
def _sample_discrete_action(coverage: str, m_max: int, rng: np.random.Generator) -> int:
    """
    Return a discrete action index in {0, ..., m_max-1}.
    action index i  ->  restart length m = i + 1.
    'fixed20' always returns m_max - 1 (i.e. m = m_max).
    'random'  samples uniformly.
    """
    if coverage == "fixed20":
        return m_max - 1
    return int(rng.integers(0, m_max))


def collect_transitions_from_sweep(
    matrix_names: List[str],
    matrices_dir: Path,
    m_max: int,
    tolerance: float,
    max_cycles: int,
    num_rollouts: int,
    coverage: str,
    rng: np.random.Generator,
    max_n: Optional[int] = None,
    verbose: bool = True,
) -> List[Transition]:
    """
    Load each sweep matrix and run GMRESEnv rollouts to collect transitions.

    Transitions store *relative* residual norms ρ = ||r|| / ||b||, so they
    are scale-invariant across the heterogeneous matrix set. The tolerance
    argument is therefore also a relative threshold.

    For 'mixed' coverage: runs num_rollouts random-action rollouts AND
    num_rollouts fixed-m_max rollouts per matrix, giving balanced action
    diversity while anchoring the distribution to the standard GMRES(m_max)
    baseline.

    Parameters
    ----------
    matrix_names  : names of matrices to load (matched against matrices_dir)
    matrices_dir  : directory containing .tar.gz or .mat files
    m_max         : maximum restart length; action i -> m = i+1
    tolerance     : relative convergence threshold (terminated when ρ < tol)
    max_cycles    : max GMRES cycles per rollout (hard cap)
    num_rollouts  : rollouts per matrix per coverage pass
    coverage      : 'random' | 'fixed20' | 'mixed'
    rng           : numpy random generator
    max_n         : skip matrices with n > max_n  (None = no limit)
    verbose       : print per-matrix progress
    """
    transitions: List[Transition] = []
    skipped: List[str] = []

    # 'mixed' = one random pass + one fixed20 pass
    passes = ["random", "fixed20"] if coverage == "mixed" else [coverage]

    for name in matrix_names:
        try:
            A, b = load_problem({"name": name}, matrices_dir)
        except (FileNotFoundError, ValueError) as exc:
            if verbose:
                print(f"  [skip] {name}: {exc}")
            skipped.append(name)
            continue

        n = int(A.shape[0])
        if max_n is not None and n > max_n:
            if verbose:
                print(f"  [skip] {name}: n={n} > --max-n={max_n}")
            skipped.append(name)
            continue

        matrix_count = 0

        for pass_coverage in passes:
            for _ in range(num_rollouts):
                env = GMRESEnv(
                    A, b,
                    m_max=m_max,
                    tolerance=tolerance,
                    max_cycles=max_cycles,
                )
                obs, info = env.reset()
                done = False
                cycle = 0

                while not done and cycle < max_cycles:
                    action = _sample_discrete_action(pass_coverage, m_max, rng)
                    obs, _, terminated, truncated, step_info = env.step(action)

                    prev_rel = step_info.get("prev_relative_residual_norm")
                    curr_rel = step_info.get("relative_residual_norm")
                    m_val = step_info.get("current_m")

                    if prev_rel is not None and curr_rel is not None and m_val is not None:
                        transitions.append(Transition(
                            prev_norm=float(prev_rel),
                            m=int(m_val),
                            curr_norm=float(curr_rel),
                            converged=bool(terminated),
                        ))
                        matrix_count += 1

                    done = terminated or truncated
                    cycle += 1

        if verbose:
            print(f"  {name}: n={n}, transitions={matrix_count}")

    if skipped:
        print(f"\n  Skipped {len(skipped)} matrices: {', '.join(skipped)}")

    return transitions


# output helpers (supplement those imported from compute_epic_rewards)
def print_sweep_transition_stats(transitions: List[Transition], args: argparse.Namespace) -> None:
    prev_norms = np.array([t.prev_norm for t in transitions])
    curr_norms = np.array([t.curr_norm for t in transitions])
    ms = np.array([t.m for t in transitions])
    m_counts = {int(v): int(np.sum(ms == v)) for v in sorted(np.unique(ms))}

    print("\n=== Transition dataset (sweep matrices) ===")
    print(f"  num_transitions        : {len(transitions)}")
    print(f"  rho_prev  min/med/max  : "
          f"{prev_norms.min():.3e} / {np.median(prev_norms):.3e} / {prev_norms.max():.3e}")
    print(f"  rho_curr  min/med/max  : "
          f"{curr_norms.min():.3e} / {np.median(curr_norms):.3e} / {curr_norms.max():.3e}")
    print(f"  m distribution         : {m_counts}")
    print(f"  coverage mode          : {args.coverage}")
    print(f"  action dist mode       : {args.action_dist}")
    print(f"  convergence_bonus      : {args.convergence_bonus}")
    print(f"  gamma                  : {args.gamma}")
    print(f"  lambda_work            : {args.lambda_work}")
    print(f"  tolerance tau          : {args.tolerance}")
    print(f"  converged frac         : {np.mean([t.converged for t in transitions]):.3f}")
    print(f"  norm type              : relative (ρ = ||r|| / ||b||)")


def build_parser() -> argparse.ArgumentParser:
    default_matrices_dir = str(_PROJECT_ROOT / "matrices" / "hpo_matrices")

    p = argparse.ArgumentParser(
        description=(
            "Compute EPIC distances between GMRES reward functions on the "
            "heterogeneous sweep matrices (matrices/hpo_matrices/)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Matrix selection
    p.add_argument(
        "--matrices-dir", type=str, default=default_matrices_dir,
        help="Directory containing the sweep matrix archives (.tar.gz or .mat).",
    )
    p.add_argument(
        "--matrix-names", type=str, nargs="+", default=None,
        help="Subset of matrix names to use. Default: all 16 sweep matrices.",
    )
    p.add_argument(
        "--max-n", type=int, default=None,
        help="Skip matrices with dimension n > max_n. Default: use all.",
    )

    # Environment settings
    p.add_argument("--m-max", type=int, default=20)
    p.add_argument("--tolerance", type=float, default=1e-6,
                   help="Relative residual convergence threshold.")
    p.add_argument("--max-cycles", type=int, default=50,
                   help="Max GMRES cycles per rollout (for transition collection only).")
    p.add_argument("--num-rollouts", type=int, default=2,
                   help="Rollouts per matrix per coverage pass.")

    # Coverage
    p.add_argument(
        "--coverage", choices=["random", "fixed20", "mixed"], default="mixed",
        help=(
            "Action coverage strategy. 'mixed' = random pass + fixed-m_max pass "
            "per matrix, giving broad action diversity anchored to the GMRES(20) baseline."
        ),
    )

    # Reward configuration
    p.add_argument(
        "--convergence-bonus", type=float, default=9.0,
        help="Sparse terminal bonus B (B=9 matches DQN training; use 0 for PBRS sanity check).",
    )
    p.add_argument("--lambda-work", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--cte", type=float, default=1.0)

    # EPIC computation
    p.add_argument(
        "--action-dist", choices=["empirical", "uniform"], default="empirical",
        help="Distribution used to sample actions for the EPIC canonicalization integral.",
    )
    p.add_argument("--num-canon-samples", type=int, default=2000,
                   help="K: number of independent samples for EPIC canonicalization.")
    p.add_argument("--bootstrap", type=int, default=200,
                   help="Number of bootstrap resamples for 95%% CI. 0 = skip.")
    p.add_argument("--seed", type=int, default=42)

    # Output paths
    p.add_argument(
        "--save-transitions", type=str, default=None,
        help="If set, save raw transitions to this CSV path.",
    )
    p.add_argument(
        "--out-matrix", type=str,
        default=str(_SCRIPT_DIR / "results" / "epic_sweep_distance_matrix.csv"),
    )
    p.add_argument(
        "--out-bootstrap", type=str,
        default=str(_SCRIPT_DIR / "results" / "epic_sweep_bootstrap_summary.csv"),
    )

    return p


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    matrices_dir = Path(args.matrices_dir)
    matrix_names = args.matrix_names if args.matrix_names else SWEEP_MATRIX_NAMES

   # collect transitions from the sweep matrices
    print(
        f"Collecting transitions from {len(matrix_names)} sweep matrices\n"
        f"  matrices_dir  = {matrices_dir}\n"
        f"  coverage      = {args.coverage}\n"
        f"  num_rollouts  = {args.num_rollouts} per matrix per pass\n"
        f"  max_cycles    = {args.max_cycles}\n"
        f"  max_n         = {args.max_n!r}"
    )

    transitions = collect_transitions_from_sweep(
        matrix_names=matrix_names,
        matrices_dir=matrices_dir,
        m_max=args.m_max,
        tolerance=args.tolerance,
        max_cycles=args.max_cycles,
        num_rollouts=args.num_rollouts,
        coverage=args.coverage,
        rng=rng,
        max_n=args.max_n,
        verbose=True,
    )

    if len(transitions) == 0:
        print(
            "ERROR: No transitions collected. "
            "Check --matrices-dir and that at least one matrix was loaded.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = RewardCfg(
        cte=args.cte,
        lambda_work=args.lambda_work,
        gamma_shape=args.gamma,
        convergence_bonus=args.convergence_bonus,
        tolerance=args.tolerance,
    )

    print_sweep_transition_stats(transitions, args)

    if args.save_transitions:
        save_transitions_csv(transitions, args.save_transitions)

    # build eval arrays
    eval_prev = np.array([t.prev_norm for t in transitions], dtype=np.float64)
    eval_ms   = np.array([t.m         for t in transitions], dtype=np.int64)
    eval_curr = np.array([t.curr_norm for t in transitions], dtype=np.float64)

    # state distribution D_S: pool of all observed residual norms.
    all_norms = np.concatenate([eval_prev, eval_curr])

    # action distribution D_A
    if args.action_dist == "empirical":
        action_pool = eval_ms.astype(np.int64)
    else:  # uniform over 1..m_max
        action_pool = rng.integers(1, args.m_max + 1, size=len(eval_ms))

    # sub-sample for canonicalization; S and S' are drawn independently 
    K = min(args.num_canon_samples, len(all_norms), len(action_pool))

    S_samples      = all_norms[rng.choice(len(all_norms),   size=K, replace=True)]
    Sp_samples     = all_norms[rng.choice(len(all_norms),   size=K, replace=True)]
    action_samples = action_pool[rng.choice(len(action_pool), size=K, replace=True)]

    print(f"\nCanonicalizing {len(REWARD_REGISTRY)} reward functions "
          f"with K={K} independent samples per transition ...")

    # compute canonicalized reward vectors for each reward function
    names = list(REWARD_REGISTRY.keys())
    canonical: dict[str, np.ndarray] = {}

    for name, fn in REWARD_REGISTRY.items():
        print(f"  Canonicalizing {name} ...", end=" ", flush=True)
        canonical[name] = canonicalize(
            reward_fn=fn,
            eval_prev=eval_prev,
            eval_ms=eval_ms,
            eval_curr=eval_curr,
            S_samples=S_samples,
            Sp_samples=Sp_samples,
            action_samples=action_samples,
            gamma=args.gamma,
            cfg=cfg,
        )
        print("done")

   # compute EPIC distance matrix
    print("\nComputing EPIC distance matrix ...")
    n_r = len(names)
    dist_matrix = np.full((n_r, n_r), 0.0)
    boot_results: dict = {}

    for i in range(n_r):
        for j in range(i, n_r):
            if i == j:
                dist_matrix[i, j] = 0.0
                continue
            ca = canonical[names[i]]
            cb = canonical[names[j]]
            d = epic_distance(ca, cb)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

            if args.bootstrap > 0:
                pt, bm, bs, ci_lo, ci_hi = bootstrap_epic(ca, cb, args.bootstrap, rng)
                pair_key = f"{names[i]} vs {names[j]}"
                boot_results[pair_key] = (pt, bm, bs, ci_lo, ci_hi)

    print_distance_matrix(names, dist_matrix)
    print_interpretation(names, dist_matrix)

    # cross-reference convdiff results for the key pairs
    print("\n=== Comparison with convdiff baseline ===")
    idx = {n: i for i, n in enumerate(names)}

    def _d(a: str, b: str) -> str:
        i, j = idx.get(a), idx.get(b)
        if i is None or j is None:
            return "n/a"
        return f"{dist_matrix[i, j]:.6f}"

    print(f"  D(pbrs, base_work)       = {_d('pbrs','base_work')}")
    print(f"    [PBRS sanity check: should be ≈ 0.000000 when convergence_bonus=0]")
    print(f"    [convdiff reference    :   0.000000]")
    print()
    print(f"  D(original, pbrs)        = {_d('original','pbrs')}")
    print(f"    [main finding: should be large (convdiff reference ≈ 0.722893, B=0)]")
    print()
    print(f"  D(original, pbrs) [B=9]  = {_d('original','pbrs')}")
    print(f"    [with convergence bonus B={args.convergence_bonus:.0f}]")
    print(f"    [convdiff reference B=9:   0.399874 (bootstrap CI ≈ [0.353, 0.452])]")

    # save outputs
    print("\nSaving outputs ...")
    save_distance_matrix_csv(names, dist_matrix, args.out_matrix)

    if args.bootstrap > 0 and boot_results:
        save_bootstrap_csv(boot_results, args.out_bootstrap)

    print("\nDone.")
    print(f"  Distance matrix -> {args.out_matrix}")
    if args.bootstrap > 0 and boot_results:
        print(f"  Bootstrap CI    -> {args.out_bootstrap}")


if __name__ == "__main__":
    main()

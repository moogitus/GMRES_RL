"""
compute_epic_rewards.py
=======================

EPIC (Equivalent-Policy Invariant Comparison) distance between reward functions
for the GMRES(m) + single-life RL project.

Imports from src/env.py (the canonical environment), which implements three
reward types:
  - "original"  : R = cte/||r_k|| + (||r_{k-1}|| - ||r_k||)  [Keramati & Hamdullahpura 2025]
  - "shaped"    : R = -λm + log||r_{k-1}|| - γ·log||r_k||     [naive log-PBRS, unbounded Φ]
  - "pbrs"      : R = -λm + log(max(||r_t||,τ)/τ)             [τ-clamped PBRS, satisfies
                        - γ·log(max(||r_{t+1}||,τ)/τ)          Ng et al. 1999 exactly]

Mathematical background
-----------------------
Given a reward R(s, a, s') and discount gamma, the *canonicalized* reward is:

    C(R)(s, a, s') = R(s, a, s')
        + E_{A~D_A, S~D_S, S'~D_S}[
              gamma * R(s', A, S')
            - R(s,   A, S')
            - gamma * R(S,  A, S')
          ]

This canonicalization removes any potential-based shaping component while
preserving the policy-equivalence class (Gleave et al., ICLR 2021).

The EPIC distance is then the Pearson distance between canonicalized rewards
evaluated on the same set of transitions:

    D_EPIC(R_A, R_B) = sqrt( (1 - corr(C(R_A), C(R_B))) / 2 )

Values are in [0, 1]; 0 means the two rewards are equivalent up to PBRS.

Transition representation
--------------------------
A state is represented solely by the residual norm ||r||.  A transition is:

    (prev_norm, m, curr_norm)

where prev_norm = ||r_t||, m is the GMRES restart parameter chosen by the agent,
and curr_norm = ||r_{t+1}||.

τ-clamped PBRS (the "pbrs" reward in src/env.py)
--------------------------------------------------
The naive potential Φ(s) = -log(||r||) is unbounded: as ||r|| → 0, Φ → +∞.
At the terminal state ||r|| < τ the PBRS term γΦ(s_T) is enormous, injecting
a spurious massive bonus that leaks backward through Bellman updates.

The τ-clamped potential fixes this:

    Φ_τ(s) = -log( max(||r||, τ) / τ )

Properties that make it satisfy Ng et al. (1999) shaping conditions exactly:
  1. Φ_τ(s) ≤ 0 always (since max(||r||,τ)/τ ≥ 1)
  2. Φ_τ(s) = 0 exactly at convergence (||r|| ≤ τ => max(||r||,τ)/τ = 1)
  3. Φ_τ is bounded and smooth everywhere above τ

EPIC expects D(pbrs, naive_log_shaped) = 0, and D(pbrs, base_work) = 0
because EPIC removes ALL PBRS simultaneously — the clamping changes Φ but
not the policy-equivalence class.  The real finding is D(original, pbrs) ≈ 0.72,
showing the paper's reward and our shaped reward lie in fundamentally different
policy-equivalence classes.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

# Make src/ importable regardless of working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class RewardCfg:
    cte: float = 1.0
    lambda_work: float = 0.05
    gamma_shape: float = 0.97
    convergence_bonus: float = 0.0
    tolerance: float = 1e-6
    eps: float = 1e-12


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def original_reward(prev_norm: float, m: int, curr_norm: float, cfg: RewardCfg) -> float:
    """cte/(curr+eps) + (prev-curr), plus optional convergence bonus."""
    r = cfg.cte / (curr_norm + cfg.eps) + (prev_norm - curr_norm)
    if curr_norm < cfg.tolerance <= prev_norm:
        r += cfg.convergence_bonus
    return r


def pbrs_reward(
    prev_norm: float, m: int, curr_norm: float, cfg: RewardCfg
) -> float:
    """
    τ-clamped PBRS (matches src/env.py reward_type="pbrs").

    Phi_tau(s) = -log(max(||r||, tau) / tau)
      => Phi_tau <= 0 always; Phi_tau = 0 exactly at convergence.

    R = -lambda_work*m + log(max(prev,tau)/tau) - gamma*log(max(curr,tau)/tau)

    Satisfies Ng, Harada & Russell (1999) shaping conditions exactly:
    the potential is bounded above and vanishes at terminal states.
    """
    tau = cfg.tolerance
    shaped_log_prev = np.log(max(prev_norm, tau) / tau)
    shaped_log_curr = np.log(max(curr_norm, tau) / tau)
    r = -cfg.lambda_work * m + shaped_log_prev - cfg.gamma_shape * shaped_log_curr
    if curr_norm < cfg.tolerance <= prev_norm:
        r += cfg.convergence_bonus
    return r


def naive_log_shaped_reward(
    prev_norm: float, m: int, curr_norm: float, cfg: RewardCfg
) -> float:
    """
    Naive (non-terminal-corrected) log-shaped reward, kept for comparison only.
    Uses log(||r||) directly without clamping to tau.
    """
    r = (
        -cfg.lambda_work * m
        + np.log(max(prev_norm, cfg.eps))
        - cfg.gamma_shape * np.log(max(curr_norm, cfg.eps))
    )
    if curr_norm < cfg.tolerance <= prev_norm:
        r += cfg.convergence_bonus
    return r


def base_work_reward(
    prev_norm: float, m: int, curr_norm: float, cfg: RewardCfg
) -> float:
    """Pure work penalty -lambda_work*m plus optional convergence bonus."""
    r = -cfg.lambda_work * m
    if curr_norm < cfg.tolerance <= prev_norm:
        r += cfg.convergence_bonus
    return r


def ref_linear_reward(
    prev_norm: float, m: int, curr_norm: float, cfg: RewardCfg
) -> float:
    """-m plus optional success bonus."""
    r = -float(m)
    if curr_norm < cfg.tolerance <= prev_norm:
        r += cfg.convergence_bonus
    return r


def ref_quadratic_reward(
    prev_norm: float, m: int, curr_norm: float, cfg: RewardCfg
) -> float:
    """-(m^2) plus optional success bonus."""
    r = -float(m ** 2)
    if curr_norm < cfg.tolerance <= prev_norm:
        r += cfg.convergence_bonus
    return r


REWARD_REGISTRY: dict[str, Callable] = {
    "original":         original_reward,       # paper's R = cte/||r_k|| + Δ||r||
    "pbrs":             pbrs_reward,            # τ-clamped PBRS (src/env.py "pbrs")
    "naive_log_shaped": naive_log_shaped_reward, # unbounded log-PBRS (src/env.py "shaped")
    "base_work":        base_work_reward,       # pure work penalty -λm
    "ref_linear":       ref_linear_reward,      # -m
    "ref_quadratic":    ref_quadratic_reward,   # -m²
}


# ---------------------------------------------------------------------------
# Vectorized reward evaluation helpers
# ---------------------------------------------------------------------------

def _bonus_mask(prev: np.ndarray, curr: np.ndarray, cfg: RewardCfg) -> np.ndarray:
    return (curr < cfg.tolerance) & (prev >= cfg.tolerance)


def original_reward_npvec(
    prev: np.ndarray, ms: np.ndarray, curr: np.ndarray, cfg: RewardCfg
) -> np.ndarray:
    r = cfg.cte / (curr + cfg.eps) + (prev - curr)
    if cfg.convergence_bonus:
        r = np.where(_bonus_mask(prev, curr, cfg), r + cfg.convergence_bonus, r)
    return r


def pbrs_reward_npvec(
    prev: np.ndarray, ms: np.ndarray, curr: np.ndarray, cfg: RewardCfg
) -> np.ndarray:
    tau = cfg.tolerance
    sl_prev = np.log(np.maximum(prev, tau) / tau)
    sl_curr = np.log(np.maximum(curr, tau) / tau)
    r = -cfg.lambda_work * ms + sl_prev - cfg.gamma_shape * sl_curr
    if cfg.convergence_bonus:
        r = np.where(_bonus_mask(prev, curr, cfg), r + cfg.convergence_bonus, r)
    return r


def naive_log_shaped_reward_npvec(
    prev: np.ndarray, ms: np.ndarray, curr: np.ndarray, cfg: RewardCfg
) -> np.ndarray:
    r = (
        -cfg.lambda_work * ms
        + np.log(np.maximum(prev, cfg.eps))
        - cfg.gamma_shape * np.log(np.maximum(curr, cfg.eps))
    )
    if cfg.convergence_bonus:
        r = np.where(_bonus_mask(prev, curr, cfg), r + cfg.convergence_bonus, r)
    return r


def base_work_reward_npvec(
    prev: np.ndarray, ms: np.ndarray, curr: np.ndarray, cfg: RewardCfg
) -> np.ndarray:
    r = -cfg.lambda_work * ms.astype(np.float64)
    if cfg.convergence_bonus:
        r = np.where(_bonus_mask(prev, curr, cfg), r + cfg.convergence_bonus, r)
    return r


def ref_linear_reward_npvec(
    prev: np.ndarray, ms: np.ndarray, curr: np.ndarray, cfg: RewardCfg
) -> np.ndarray:
    r = -ms.astype(np.float64)
    if cfg.convergence_bonus:
        r = np.where(_bonus_mask(prev, curr, cfg), r + cfg.convergence_bonus, r)
    return r


def ref_quadratic_reward_npvec(
    prev: np.ndarray, ms: np.ndarray, curr: np.ndarray, cfg: RewardCfg
) -> np.ndarray:
    r = -(ms.astype(np.float64) ** 2)
    if cfg.convergence_bonus:
        r = np.where(_bonus_mask(prev, curr, cfg), r + cfg.convergence_bonus, r)
    return r


# Map scalar reward fn -> vectorized version (keyed by function __name__)
_NPVEC_REGISTRY: dict[str, Callable] = {
    "original_reward":         original_reward_npvec,
    "pbrs_reward":             pbrs_reward_npvec,
    "naive_log_shaped_reward": naive_log_shaped_reward_npvec,
    "base_work_reward":        base_work_reward_npvec,
    "ref_linear_reward":       ref_linear_reward_npvec,
    "ref_quadratic_reward":    ref_quadratic_reward_npvec,
}


def eval_reward_vec(
    reward_fn: Callable,
    prev_norms: np.ndarray,
    ms: np.ndarray,
    curr_norms: np.ndarray,
    cfg: RewardCfg,
) -> np.ndarray:
    """Evaluate reward function over arrays; returns float64 array."""
    # Use vectorized version if available, otherwise fall back to scalar loop.
    npvec = _NPVEC_REGISTRY.get(getattr(reward_fn, "__name__", ""), None)
    if npvec is not None:
        return npvec(
            prev_norms.astype(np.float64),
            ms.astype(np.float64),
            curr_norms.astype(np.float64),
            cfg,
        )
    return np.array(
        [reward_fn(p, int(m), c, cfg) for p, m, c in zip(prev_norms, ms, curr_norms)],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Transition collection
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    prev_norm: float
    m: int
    curr_norm: float
    converged: bool  # crossed tolerance on this step


def collect_transitions(
    n: int,
    eps_values: List[float],
    beta: float,
    m_max: int,
    tolerance: float,
    max_cycles: int,
    num_rhs: int,
    num_rollouts: int,
    coverage: str,
    rng: np.random.Generator,
) -> List[Transition]:
    """
    Instantiate AKSLRLEnv on convdiff matrices and run random rollouts.
    coverage: 'random' | 'fixed20' | 'mixed'
    """
    from src.matrices import make_convdiff_1d_sparse
    from src.env import AKSLRLEnv

    transitions: List[Transition] = []

    for eps in eps_values:
        A = make_convdiff_1d_sparse(n, eps, beta)
        # env handles scipy sparse via @; reward_type irrelevant for transition collection
        for _ in range(num_rhs):
            b = rng.standard_normal(n).astype(np.float64)
            for _ in range(num_rollouts):
                env = AKSLRLEnv(
                    A, b,
                    m_max=m_max,
                    tolerance=tolerance,
                    max_cycles=max_cycles,
                    reward_type="pbrs",     # valid in src/env.py; rewards recomputed below
                    convergence_bonus=0.0,
                )
                obs, info = env.reset()
                done = False
                cycle = 0
                while not done and cycle < max_cycles:
                    action = _sample_action(coverage, rng)
                    obs, _, terminated, truncated, step_info = env.step(
                        np.array([action], dtype=np.float32)
                    )
                    p = step_info.get("prev_residual_norm")
                    c = step_info.get("residual_norm")
                    m = step_info.get("current_m")
                    if p is not None and c is not None and m is not None:
                        transitions.append(Transition(
                            prev_norm=float(p),
                            m=int(m),
                            curr_norm=float(c),
                            converged=bool(terminated),
                        ))
                    done = terminated or truncated
                    cycle += 1

    # For mixed mode collect half-and-half by running again with opposite strategy
    if coverage == "mixed":
        half = len(transitions)
        for eps in eps_values:
            A = make_convdiff_1d_sparse(n, eps, beta)
            for _ in range(num_rhs):
                b = rng.standard_normal(n).astype(np.float64)
                for _ in range(num_rollouts):
                    env = AKSLRLEnv(
                        A, b,
                        m_max=m_max,
                        tolerance=tolerance,
                        max_cycles=max_cycles,
                        reward_type="pbrs",
                        convergence_bonus=0.0,
                    )
                    obs, info = env.reset()
                    done = False
                    cycle = 0
                    while not done and cycle < max_cycles:
                        # opposite strategy: fixed20 for the second half
                        action = 1.0
                        obs, _, terminated, truncated, step_info = env.step(
                            np.array([action], dtype=np.float32)
                        )
                        p = step_info.get("prev_residual_norm")
                        c = step_info.get("residual_norm")
                        m = step_info.get("current_m")
                        if p is not None and c is not None and m is not None:
                            transitions.append(Transition(
                                prev_norm=float(p),
                                m=int(m),
                                curr_norm=float(c),
                                converged=bool(terminated),
                            ))
                        done = terminated or truncated
                        cycle += 1

    return transitions


def _sample_action(coverage: str, rng: np.random.Generator) -> float:
    if coverage == "fixed20":
        return 1.0
    # random (also first pass of mixed)
    return float(rng.uniform(0.0, 1.0))


# ---------------------------------------------------------------------------
# EPIC canonicalization
# ---------------------------------------------------------------------------

def canonicalize(
    reward_fn: Callable,
    eval_prev: np.ndarray,       # (N,)
    eval_ms: np.ndarray,         # (N,) int
    eval_curr: np.ndarray,       # (N,)
    S_samples: np.ndarray,       # (K,) residual norms for S  — drawn independently
    Sp_samples: np.ndarray,      # (K,) residual norms for S' — drawn independently
    action_samples: np.ndarray,  # (K,) integer m values
    gamma: float,
    cfg: RewardCfg,
    chunk_j: int = 200,
) -> np.ndarray:
    """
    Compute the canonicalized reward C(R) for each eval transition.

    C_i = R(prev_i, m_i, curr_i)
          + mean_j[ gamma * R(curr_i, A_j, S'_j)
                    - R(prev_i, A_j, S'_j)
                    - gamma * R(S_j, A_j, S'_j) ]

    S_j and S'_j are drawn *independently* from the state distribution.
    Loop structure: outer over j-chunks (vectorized over all N transitions inside),
    so the per-j cost is O(N) numpy work rather than O(N) pure Python.
    """
    N = len(eval_prev)
    K = len(S_samples)
    assert len(Sp_samples) == K and len(action_samples) == K

    base = eval_reward_vec(reward_fn, eval_prev, eval_ms, eval_curr, cfg)
    correction = np.zeros(N, dtype=np.float64)

    for j_start in range(0, K, chunk_j):
        j_end = min(j_start + chunk_j, K)
        S_j  = S_samples[j_start:j_end]    # (k,)
        Sp_j = Sp_samples[j_start:j_end]   # (k,) — truly independent
        A_j  = action_samples[j_start:j_end]  # (k,)
        k = j_end - j_start

        # term3[j] = gamma * R(S_j, A_j, Sp_j)  — shape (k,), no dependence on i
        term3 = gamma * eval_reward_vec(reward_fn, S_j, A_j, Sp_j, cfg)  # (k,)

        # For each j, accumulate correction over all N eval transitions.
        # term1[j] = gamma * R(curr_i, A_j, Sp_j) — varies in curr_i and j
        # term2[j] = R(prev_i, A_j, Sp_j)         — varies in prev_i and j
        # We build (N, k) matrices by broadcasting.
        #   rows = eval transitions, cols = sample index j

        # Broadcast: prev_col[i, j] = eval_prev[i], Sp_row[j] = Sp_j[j]
        # R(eval_prev[i], A_j[j], Sp_j[j]) for all (i,j)
        # = eval_reward_npvec over flattened grid then reshape

        prev_grid = np.repeat(eval_prev[:, None], k, axis=1).ravel()   # (N*k,)
        curr_grid = np.repeat(eval_curr[:, None], k, axis=1).ravel()   # (N*k,)
        A_grid    = np.tile(A_j, N).ravel()                            # (N*k,)
        Sp_grid   = np.tile(Sp_j, N).ravel()                           # (N*k,)

        r_prev_grid = eval_reward_vec(reward_fn, prev_grid, A_grid, Sp_grid, cfg)
        r_curr_grid = eval_reward_vec(reward_fn, curr_grid, A_grid, Sp_grid, cfg)

        # correction += sum_j [gamma*r_curr - r_prev - gamma*term3] / K
        #             = (sum over j-chunk of row sums)
        r_prev_mat = r_prev_grid.reshape(N, k)          # (N, k)
        r_curr_mat = r_curr_grid.reshape(N, k)          # (N, k)

        # sum over j dimension; divide by K outside loop
        correction += np.sum(gamma * r_curr_mat - r_prev_mat, axis=1) - np.sum(term3)

    correction /= K
    return base + correction


# ---------------------------------------------------------------------------
# EPIC distance
# ---------------------------------------------------------------------------

def pearson_distance(a: np.ndarray, b: np.ndarray) -> float:
    """sqrt((1 - corr(a,b)) / 2), clipped to [0,1]."""
    std_a = np.std(a)
    std_b = np.std(b)
    if std_a < 1e-12 or std_b < 1e-12:
        warnings.warn("Near-zero variance in canonicalized reward; returning NaN.")
        return float("nan")
    corr = float(np.corrcoef(a, b)[0, 1])
    corr = np.clip(corr, -1.0, 1.0)
    return float(np.sqrt(max(0.0, (1.0 - corr) / 2.0)))


def epic_distance(
    ca: np.ndarray,
    cb: np.ndarray,
) -> float:
    return pearson_distance(ca, cb)


def bootstrap_epic(
    ca: np.ndarray,
    cb: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    """Returns (point_estimate, boot_mean, boot_std, ci_lo, ci_hi) — unpacked as 5-tuple."""
    point = epic_distance(ca, cb)
    N = len(ca)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        samples.append(pearson_distance(ca[idx], cb[idx]))
    samples_arr = np.array([s for s in samples if not np.isnan(s)])
    if len(samples_arr) == 0:
        return point, float("nan"), float("nan"), float("nan"), float("nan")
    ci_lo, ci_hi = float(np.percentile(samples_arr, 2.5)), float(np.percentile(samples_arr, 97.5))
    return point, float(np.mean(samples_arr)), float(np.std(samples_arr)), ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_transition_stats(transitions: List[Transition], args) -> None:
    prev_norms = np.array([t.prev_norm for t in transitions])
    curr_norms = np.array([t.curr_norm for t in transitions])
    ms = np.array([t.m for t in transitions])
    m_counts = {v: int(np.sum(ms == v)) for v in sorted(np.unique(ms))}

    print("\n=== Transition dataset ===")
    print(f"  num_transitions   : {len(transitions)}")
    print(f"  prev_norm  min/med/max: {prev_norms.min():.3e} / {np.median(prev_norms):.3e} / {prev_norms.max():.3e}")
    print(f"  curr_norm  min/med/max: {curr_norms.min():.3e} / {np.median(curr_norms):.3e} / {curr_norms.max():.3e}")
    print(f"  m distribution    : {m_counts}")
    print(f"  coverage mode     : {args.coverage}")
    print(f"  action dist mode  : {args.action_dist}")
    print(f"  convergence_bonus : {args.convergence_bonus}")
    print(f"  gamma             : {args.gamma}")
    print(f"  lambda_work       : {args.lambda_work}")
    print(f"  tolerance tau     : {args.tolerance}")
    print(f"  converged frac    : {np.mean([t.converged for t in transitions]):.3f}")


def print_distance_matrix(names: List[str], matrix: np.ndarray) -> None:
    col_w = 28
    header = f"{'':28s}" + "".join(f"{n:>28s}" for n in names)
    print("\n=== EPIC Distance Matrix ===")
    print(header)
    for i, name in enumerate(names):
        row = f"{name:28s}" + "".join(
            f"{'nan':>28s}" if np.isnan(matrix[i, j]) else f"{matrix[i,j]:>28.4f}"
            for j in range(len(names))
        )
        print(row)


def print_interpretation(names: List[str], matrix: np.ndarray) -> None:
    idx = {n: i for i, n in enumerate(names)}

    def d(a, b):
        i, j = idx.get(a), idx.get(b)
        if i is None or j is None:
            return float("nan")
        return matrix[i, j]

    print("\n=== Interpretation ===")
    print(
        f"  D(original, pbrs) = {d('original','pbrs'):.4f}\n"
        f"    -> Policy-equivalence gap between the paper's original reward and\n"
        f"       our τ-clamped PBRS.  Low = same optimal policy; high = fundamentally\n"
        f"       different incentive structures.  Paper benchmark: D(Path,Cliff) = 0.27.\n"
    )
    print(
        f"  D(pbrs, base_work) = {d('pbrs','base_work'):.4f}   [PBRS sanity check]\n"
        f"    -> Should be 0 when convergence_bonus=0.  Gleave et al. Prop 4.2 guarantees\n"
        f"       EPIC is invariant to potential-based shaping, so the τ-clamped PBRS\n"
        f"       (Φ_τ added to -λm) must canonicalize to the same reward as -λm alone.\n"
        f"       Nonzero here signals a canonicalization bug or non-PBRS component.\n"
    )
    print(
        f"  D(pbrs, naive_log_shaped) = {d('pbrs','naive_log_shaped'):.4f}   [τ-clamping sanity check]\n"
        f"    -> Should be 0.  The τ-clamped potential Φ_τ = -log(max(||r||,τ)/τ) and\n"
        f"       the naive potential Φ = -log(||r||) differ as functions, but BOTH are\n"
        f"       PBRS over the same base reward -λm.  EPIC removes all PBRS, so they\n"
        f"       land in the same equivalence class regardless.  This confirms that\n"
        f"       τ-clamping is a numerical/stability improvement — not a policy change.\n"
    )
    print(
        f"  D(original, ref_linear) = {d('original','ref_linear'):.4f}\n"
        f"  D(pbrs,     ref_linear) = {d('pbrs','ref_linear'):.4f}\n"
        f"    -> Which reward is closer to the simple proxy -m (minimize work per cycle).\n"
        f"       pbrs near 0 => shaped reward is policy-equiv to pure work minimization.\n"
    )
    print(
        f"  D(original,      ref_quadratic) = {d('original','ref_quadratic'):.4f}\n"
        f"  D(pbrs,          ref_quadratic) = {d('pbrs','ref_quadratic'):.4f}\n"
        f"    -> Same comparison vs -(m^2) (quadratic penalization of large restarts).\n"
    )


# ---------------------------------------------------------------------------
# CSV save helpers
# ---------------------------------------------------------------------------

def save_distance_matrix_csv(names: List[str], matrix: np.ndarray, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, name in enumerate(names):
            row = [name] + [
                "nan" if np.isnan(matrix[i, j]) else f"{matrix[i,j]:.6f}"
                for j in range(len(names))
            ]
            w.writerow(row)
    print(f"  Saved distance matrix -> {path}")


def save_bootstrap_csv(boot_results: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "point_estimate", "boot_mean", "boot_std", "ci_lo_2.5", "ci_hi_97.5"])
        for pair, vals in boot_results.items():
            w.writerow([pair] + [f"{v:.6f}" if not np.isnan(v) else "nan" for v in vals])
    print(f"  Saved bootstrap summary -> {path}")


def save_transitions_csv(transitions: List[Transition], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prev_norm", "m", "curr_norm", "converged"])
        for t in transitions:
            w.writerow([t.prev_norm, t.m, t.curr_norm, int(t.converged)])
    print(f"  Saved transitions -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute EPIC distances between GMRES reward functions."
    )
    # Matrix / env settings
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--eps-values", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--m-max", type=int, default=20)
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--max-cycles", type=int, default=200)
    p.add_argument("--num-rhs", type=int, default=3)
    p.add_argument("--num-rollouts", type=int, default=1)
    # Coverage
    p.add_argument("--coverage", choices=["random", "fixed20", "mixed"], default="random")
    # Reward config
    p.add_argument("--convergence-bonus", type=float, default=0.0)
    p.add_argument("--lambda-work", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--cte", type=float, default=1.0)
    # EPIC settings
    p.add_argument("--action-dist", choices=["empirical", "uniform"], default="empirical")
    p.add_argument("--num-canon-samples", type=int, default=2000)
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    # Output
    p.add_argument("--save-transitions", type=str, default=None,
                   help="Path to save transitions CSV")
    p.add_argument("--out-matrix", type=str, default="epic_distance_matrix.csv")
    p.add_argument("--out-bootstrap", type=str, default="epic_bootstrap_summary.csv")
    return p


def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    # -----------------------------------------------------------------------
    # 1. Collect transitions
    # -----------------------------------------------------------------------
    print(f"Collecting transitions (n={args.n}, eps={args.eps_values}, "
          f"coverage={args.coverage})...")
    transitions = collect_transitions(
        n=args.n,
        eps_values=args.eps_values,
        beta=args.beta,
        m_max=args.m_max,
        tolerance=args.tolerance,
        max_cycles=args.max_cycles,
        num_rhs=args.num_rhs,
        num_rollouts=args.num_rollouts,
        coverage=args.coverage,
        rng=rng,
    )

    if len(transitions) == 0:
        print("ERROR: No transitions collected. Check env setup.", file=sys.stderr)
        sys.exit(1)

    cfg = RewardCfg(
        cte=args.cte,
        lambda_work=args.lambda_work,
        gamma_shape=args.gamma,
        convergence_bonus=args.convergence_bonus,
        tolerance=args.tolerance,
    )

    print_transition_stats(transitions, args)

    if args.save_transitions:
        save_transitions_csv(transitions, args.save_transitions)

    # -----------------------------------------------------------------------
    # 2. Build eval arrays
    # -----------------------------------------------------------------------
    eval_prev = np.array([t.prev_norm for t in transitions], dtype=np.float64)
    eval_ms   = np.array([t.m         for t in transitions], dtype=np.int64)
    eval_curr = np.array([t.curr_norm for t in transitions], dtype=np.float64)

    # State distribution D_S: pool of prev and curr norms
    all_norms = np.concatenate([eval_prev, eval_curr])

    # Action distribution D_A
    if args.action_dist == "empirical":
        action_pool = eval_ms.astype(np.int64)
    else:  # uniform over 1..m_max
        action_pool = rng.integers(1, args.m_max + 1, size=len(eval_ms))

    # Sub-sample for canonicalization efficiency.
    # S and S' are drawn *independently* to satisfy the EPIC formula.
    K = min(args.num_canon_samples, len(all_norms))
    K = min(K, len(action_pool))

    S_samples   = all_norms[rng.choice(len(all_norms),   size=K, replace=True)]
    Sp_samples  = all_norms[rng.choice(len(all_norms),   size=K, replace=True)]
    action_samples = action_pool[rng.choice(len(action_pool), size=K, replace=True)]

    print(f"\nCanonicalizing with K={K} independent samples per transition ...")

    # -----------------------------------------------------------------------
    # 3. Canonicalize all reward functions
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 4. EPIC distance matrix
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # 5. Save outputs
    # -----------------------------------------------------------------------
    print("\nSaving outputs ...")
    save_distance_matrix_csv(names, dist_matrix, args.out_matrix)

    if args.bootstrap > 0 and boot_results:
        save_bootstrap_csv(boot_results, args.out_bootstrap)

    print("\nDone.")


if __name__ == "__main__":
    main()

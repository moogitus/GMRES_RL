"""
Continuous-action GMRES(m) environment matching the AK-SLRL baseline of
Keramati & Hamdullahpur (2025), arXiv:2502.00227. Used for the SAC
baseline in §4.1 and the SAC reward ablation in §7.4.

Action a_t ∈ [0, 1] is discretized to m = ⌊a_t (m_max - 1)⌋ + 1 (§3.1).
The observation is the full residual vector (rescaled to [0,1]), with
log ||r|| optionally appended; this is the O(n) controller-side memory
that motivates the discrete-action DQN env in env.py (§4.1).

Built on top of gymnasium (https://github.com/Farama-Foundation/Gymnasium).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# continuous-action GMRES(m) gym env (AK-SLRL baseline, §3.2). default
# reward is the inverse-residual reward of Keramati & Hamdullahpur (2025):
#   R_t = cte / ||r_k|| + (||r_{k-1}|| - ||r_k||).
class AKSLRLEnv(gym.Env):

    metadata = {"render_modes": ["console"]}

    def __init__(
        self,
        A,
        b,
        m_max=20,
        tolerance=1e-5,
        max_cycles=200,
        cte=1.0,
        convergence_bonus=0.0,
        include_log_residual_in_state=True,
    ):
        # A, b: linear system Ax = b
        # m_max: max restart length (Krylov subspace dimension)
        # tolerance: convergence threshold on ||r||  (NB: absolute, not relative)
        # max_cycles: truncation cap on restart cycles per episode
        # cte: constant c_te in the original reward (paper convention: 1.0)
        # convergence_bonus: sparse terminal bonus on first crossing tolerance
        # include_log_residual_in_state: append log(||r||) to the obs vector
        super().__init__()

        self.A = A
        self.b = b
        self.n = A.shape[0]
        self.m_max = m_max
        self.tolerance = tolerance
        self.max_cycles = max_cycles
        self.cte = cte
        self.convergence_bonus = convergence_bonus
        self.include_log_residual_in_state = include_log_residual_in_state

        obs_dim = self.n + (1 if include_log_residual_in_state else 0)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.ones(obs_dim, dtype=np.float32)
        if include_log_residual_in_state:
            low[-1] = -50.0
            high[-1] = 50.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # runtime state, populated on reset()
        self.x = None
        self.current_residual_vector = None
        self.current_residual_norm = None
        self.cycle_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = np.zeros(self.n)
        self.current_residual_vector = self.b - self.A @ self.x
        self.current_residual_norm = float(np.linalg.norm(self.current_residual_vector))
        self.cycle_count = 0

        obs = self._build_observation(self.current_residual_vector,
                                      self.current_residual_norm)
        info = {"residual_norm": self.current_residual_norm}
        return obs, info

    def step(self, action):
        # discretize a_t in [0,1] to m in {1,...,m_max} per §3.1
        a_t = float(np.clip(action[0], 0.0, 1.0))
        m = int(np.floor(a_t * (self.m_max - 1)) + 1)
        m = max(1, min(m, self.m_max))

        prev_norm = self.current_residual_norm

        # already at tolerance: terminate cleanly with zero reward
        if prev_norm < self.tolerance:
            obs = self._build_observation(self.current_residual_vector, prev_norm)
            return obs, 0.0, True, False, {
                "current_m": m,
                "residual_norm": prev_norm,
                "cycle_count": self.cycle_count,
            }

        self.x, self.current_residual_vector = self._execute_gmres_cycle(
            self.x, self.current_residual_vector, m
        )
        self.current_residual_norm = float(np.linalg.norm(self.current_residual_vector))
        self.cycle_count += 1

        reward = self._compute_reward(prev_norm, self.current_residual_norm)

        terminated = bool(self.current_residual_norm < self.tolerance)
        truncated = bool(self.cycle_count >= self.max_cycles) and not terminated

        obs = self._build_observation(self.current_residual_vector,
                                      self.current_residual_norm)
        info = {
            "current_m": m,
            "residual_norm": self.current_residual_norm,
            "prev_residual_norm": prev_norm,
            "cycle_count": self.cycle_count,
        }
        return obs, float(reward), terminated, truncated, info

    # AK-SLRL inverse-residual reward (Keramati & Hamdullahpur 2025)
    def _compute_reward(self, prev_norm, curr_norm):
        eps = 1e-12
        reward = (self.cte / (curr_norm + eps)) + (prev_norm - curr_norm)
        if curr_norm < self.tolerance <= prev_norm:
            reward += self.convergence_bonus
        return reward

    def _build_observation(self, residual_vector, residual_norm):
        normalized = self._normalize_state(residual_vector)
        if self.include_log_residual_in_state:
            log_r = np.log(max(residual_norm, 1e-12))
            log_r = np.clip(log_r, -50.0, 50.0)
            normalized = np.concatenate([normalized, np.array([log_r], dtype=np.float32)])
        return normalized

    # min-max rescale the residual vector into [0,1]^n
    @staticmethod
    def _normalize_state(vector):
        v_min = float(np.min(vector))
        v_max = float(np.max(vector))
        if v_max - v_min == 0:
            return np.zeros_like(vector, dtype=np.float32)
        return ((vector - v_min) / (v_max - v_min)).astype(np.float32)

    def _execute_gmres_cycle(self, x_current, r_current, m):
        V, H, beta, actual_m, breakdown = self._arnoldi_iteration(
            self.A, r_current, m, tol=1e-14
        )

        if actual_m == 0:
            return x_current, r_current

        e1 = np.zeros(actual_m + 1)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(H, e1, rcond=None)

        x_next = x_current + V[:, :actual_m] @ y
        r_next = self.b - self.A @ x_next
        return x_next, r_next

    @staticmethod
    def _arnoldi_iteration(A, r, m, tol=1e-14):
        n = r.shape[0]
        beta = float(np.linalg.norm(r))

        if beta < tol:
            V = np.zeros((n, 1))
            H = np.zeros((1, 0))
            return V, H, beta, 0, True

        V = np.zeros((n, m + 1), dtype=float)
        H = np.zeros((m + 1, m), dtype=float)
        V[:, 0] = r / beta

        actual_m = 0
        breakdown = False

        for j in range(m):
            vj = V[:, j]
            w = A(vj) if callable(A) else A @ vj

            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = float(np.linalg.norm(w))
            actual_m = j + 1

            if H[j + 1, j] < tol:
                breakdown = True
                break

            V[:, j + 1] = w / H[j + 1, j]

        V = V[:, : actual_m + 1]
        H = H[: actual_m + 1, :actual_m]
        return V, H, beta, actual_m, breakdown


# quick smoke test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 100
    A = rng.standard_normal((n, n)) + n * np.eye(n)
    b = rng.standard_normal(n)

    env = AKSLRLEnv(A, b, m_max=20, tolerance=1e-6, max_cycles=100)
    obs, info = env.reset()
    total_r = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        total_r += r
        if terminated or truncated:
            break
    print(f"cycles={info['cycle_count']:3d}  "
          f"final ||r||={info['residual_norm']:.3e}  "
          f"total reward={total_r:.3f}  "
          f"terminated={terminated}  truncated={truncated}")

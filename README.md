# Learning to Restart: Reinforcement Learning for Adaptive GMRES(m)
This is the companion repository with code and experiments from the paper "Reinforcement Learning for Adaptive GMRES(m)" by Ryan Divan, Rishabh Mohapatra, and Tyler Pellek.

## Overview

We use an online reinforcement learning algorithm to pick m in GMRES(m) with a per-cycle adaptive policy trained online using a Deep Q-Network (DQN). The agent observes the current residual vector norm and selects a restart length m ∈ {1, …, m_max} for each GMRES cycle. Training happens during the solve itself (single-life RL) — no pretraining required for the core result.

---

## Dependencies

```
numpy scipy torch stable-baselines3 gymnasium matplotlib
```

Install: `pip install numpy scipy torch stable-baselines3 gymnasium matplotlib`

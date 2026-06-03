# Learning to Restart: Reinforcement Learning for Adaptive GMRES(m)

Companion repository for *Learning to Restart: Reinforcement Learning for Adaptive
GMRES(m)* by Ryan Divan, Rishabh Mohapatra, and Tyler Pellek (Princeton, COS435 / ECE433).

We learn an adaptive rule for the restart parameter $m$ in GMRES($m$) using
single-life reinforcement learning: a fresh DQN agent is trained online during
each linear-system solve, observing a compact history of relative-residual norms
and recent restart choices and choosing $m \in \{1, \ldots, m_{\max}\}$ at each
GMRES cycle. The agent's overhead is $\mathcal{O}(1)$ in $n$, whereas the
AK-SLRL SAC baseline of [Keramati & Hamdullahpur (2025)](https://arxiv.org/abs/2502.00227)
incurs $\mathcal{O}(n)$ per replay-buffer transition.

## Quick install

This repo uses [Git LFS](https://git-lfs.com/) for the ~2.6 GB of data files
(SuiteSparse `.tar.gz` matrix archives and the 825 MB `results/peairs_155/full_run.json`).
Install `git-lfs` *before* cloning, otherwise you will only get small pointer
stubs in place of the matrices and the raw benchmark JSON.

```bash
# install git-lfs once on your machine
brew install git-lfs    # macOS; Linux: see https://git-lfs.com/
git lfs install

git clone https://github.com/moogitus/GMRES_RL.git
cd GMRES_RL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If you already cloned without LFS, run `git lfs pull` from inside the repo to
download the actual files.

The matrices in `matrices/` are SuiteSparse `.tar.gz` archives (or `.mtx` files
for the synthetic convection-diffusion suite). Convection-diffusion matrices can
be regenerated locally with `python src/make_convdiff_matrices.py`.

### Components of Git LFS

- `matrices/full_benchmark/*.tar.gz` (155-matrix Peairs suite)
- `matrices/hpo_matrices/*.tar.gz` (16-matrix HPO suite)
- `matrices/slrl_benchmark_matrices/*.tar.gz` (6-matrix AK-SLRL suite)
- `results/peairs_155/full_run.json` (raw per-cell data, ~825 MB)

## Repository structure

```
GMRES_RL/
├── src/                                Sources: env, training, matrix generation
│   ├── env.py                          GMRES env (DQN, discrete action, PBRS reward)
│   ├── akslrl_env.py                   GMRES env (SAC baseline, continuous action,
│   │                                   AK-SLRL inverse-residual reward)
│   ├── train_dqn.py                    DQN single-life training driver
│   ├── train_sac.py                    SAC single-life training driver
│   ├── compare_gmres_slrl.py           Per-matrix convergence-trace runner
│   ├── make_convdiff_matrices.py       Generates the 20 synthetic 1D conv-diff matrices
│   └── utils.py                        Shared matrix-IO helpers + SB3 callback
│
├── analysis/                           Sweep, comparison, and statistical analysis
│   ├── benchmark.py                    Peairs-style benchmark over the 155-matrix suite
│   ├── analyze_benchmark.py            rliable analysis of benchmark.py output
│   ├── bayesian_hyperparameter_search.py
│   │                                   Optuna TPE search over (lambda, gamma, B) for the
│   │                                   PBRS reward
│   ├── kappa_sweep.py                  20-matrix conv-diff sweep over condition number
│   ├── compute_kappa_sweep_slopes.py   Log-log slope of Arnoldi vs kappa with bootstrap CI
│   ├── sac_reward_sweep.py             Reward-function ablation under SAC
│   ├── sweep_dqn_mmax_history.py       Sensitivity sweep over m_max / history length
│   ├── test_convergence_mcnemar.py     McNemar's test on per-cell convergence
│   └── compute_variance_statistics.py  Variance / tail statistics from per-cell data
│
├── epic/                               EPIC reward-distance analysis (Gleave et al. 2021)
│   ├── compute_epic_rewards.py         Synthetic conv-diff coverage
│   ├── compute_epic_sweep_matrices.py  HPO-coverage variant
│   └── results/                        EPIC distance matrices and bootstrap CSVs
│
├── matrices/                           All test matrices
│   ├── slrl_benchmark_matrices/        6 AK-SLRL benchmark matrices
│   ├── hpo_matrices/                   16 SuiteSparse matrices for reward HPO
│   ├── kappa_sweep_20_matrices/        20 1D conv-diff matrices, kappa sweep
│   └── full_benchmark/                 155 SuiteSparse matrices for the Peairs-style
│                                       comparison (some via Git LFS)
│
└── results/                            All experiment outputs
    ├── slrl_six_matrix/                AK-SLRL six-matrix comparison
    │   ├── results.json
    │   ├── arnoldi/                    paper-ready convergence-vs-Arnoldi PNGs
    │   └── wallclock/                  paper-ready convergence-vs-wallclock PNGs
    ├── peairs_155/                     155-matrix benchmark
    │   ├── full_run.json               raw per-cell data (large, may need Git LFS)
    │   ├── full_run.meta.json
    │   ├── rliable_summary.json
    │   └── rliable/                    IQM / performance-profile / success-rate PNGs
    ├── kappa_sweep/                    Kappa-sweep figure, json, slope CSV
    ├── hpo_dqn_pbrs/                   Optuna study artifacts (json + sqlite)
    ├── reward_sweep_sac/               SAC reward ablation
    ├── variance_validation/            Per-cell, per-matrix, summary statistics + McNemar
    ├── std_scatter/                    rl-vs-rand seed-std scatter
    ├── wallclock_sweep/                Wall-clock distribution under fixed-restart sweep
    └── dqn_mmax_history/               m_max / history-length sensitivity sweep output
```

## Reproducing the paper's experiments

Each script accepts a `--help` flag listing all CLI options. The defaults
below match the configurations used in the paper.

### §4.1 — Six-matrix AK-SLRL benchmark (Table 2)

```bash
python src/compare_gmres_slrl.py
```

Runs DQN, SAC, and fixed-restart GMRES(20) on the six AK-SLRL matrices for
five seeds each. Writes per-matrix traces and aggregate JSON to
`results/slrl_six_matrix/`, plus per-matrix convergence figures under
`arnoldi/` and `wallclock/`.

### §4.2 — 155-matrix benchmark (Table 3, Table 4, Figure 7)

```bash
# 1. run the full benchmark (long; uses up to all CPU cores)
python analysis/benchmark.py

# 2. produce the rliable IQM / probability-of-improvement summary
python analysis/analyze_benchmark.py

# 3. variance and tail statistics
python analysis/compute_variance_statistics.py

# 4. McNemar test on the convergence-rate gap
python analysis/test_convergence_mcnemar.py
```

Outputs land in `results/peairs_155/` and `results/variance_validation/`.

### §4.3 — Kappa sweep on 1D convection-diffusion (Figure 1)

```bash
python analysis/kappa_sweep.py
python analysis/compute_kappa_sweep_slopes.py
```

Outputs `results/kappa_sweep/{kappa_sweep.json, kappa_sweep_arnoldi_vs_kappa.png,
kappa_sweep_slopes.csv}`.

### Reward hyperparameter selection (Appendix 7.2)

```bash
python analysis/bayesian_hyperparameter_search.py --n-trials 60
```

Optuna TPE over $(\lambda_{\text{work}}, \gamma, B)$ on the 16 HPO matrices.
The resumable Optuna SQLite store is at
`results/hpo_dqn_pbrs/gmres_dqn_pbrs_hpo_16mat.db`.

### Reward EPIC analysis (Appendix 7.3)

```bash
python epic/compute_epic_rewards.py            # synthetic conv-diff coverage
python epic/compute_epic_sweep_matrices.py     # HPO-matrix coverage
```

Outputs go to `epic/results/`.

### SAC reward ablation (Appendix 7.4)

```bash
python analysis/sac_reward_sweep.py
```

Independent Optuna sweeps over $R_{\mathrm{AK}}$ and $R_{\mathrm{ours}}$ under
SAC, on the convdiff `Easy`–`Extreme` ladder. Writes
`results/reward_sweep_sac/results.json`.

## Dependencies

All third-party requirements are in [`requirements.txt`](requirements.txt):

| Purpose | Packages |
|---|---|
| Core numerics | `numpy`, `scipy` |
| RL framework | `gymnasium`, `stable-baselines3`, `torch` |
| Plotting + tabular analysis | `matplotlib`, `pandas` |
| Streaming JSON parser | `ijson` (for the ~800 MB `full_run.json`) |
| Hyperparameter search | `optuna`, `tqdm` |
| RL evaluation framework | `rliable` (Agarwal et al., NeurIPS 2021) |

Tested on Python 3.11+ with PyTorch 2.x. CPU-only; no GPU is needed.

## Citation

If you use this code, please cite our project:

```bibtex
@misc{divan2025learning,
  author       = {Divan, Ryan and Mohapatra, Rishabh and Pellek, Tyler},
  title        = {Learning to Restart: Reinforcement Learning for Adaptive GMRES(m)},
  howpublished = {\url{https://github.com/moogitus/GMRES_RL}},
  year         = {2025},
  note         = {COS435 / ECE433 final project, Princeton University}
}
```

and the prior work this builds on:

- Saad, Y. and Schultz, M. H. (1986). *GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems.* SIAM J. Sci. Stat. Comput. 7(3).
- Keramati, H. and Hamdullahpur, F. (2025). *AK-SLRL: Adaptive Krylov subspace exploration using single-life reinforcement learning.* arXiv:2502.00227.
- Ng, A. Y., Harada, D., and Russell, S. J. (1999). *Policy invariance under reward transformations.* ICML.
- Gleave, A. et al. (2021). *Quantifying differences in reward functions.* ICLR.
- Agarwal, R. et al. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS.

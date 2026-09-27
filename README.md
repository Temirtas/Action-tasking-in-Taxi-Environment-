# Action Masking in Taxi Environment (Q-Learning)

A from-scratch tabular Q-learning implementation for Gymnasium's `Taxi-v4` environment, comparing agent performance **with** and **without** action masking.

## What this does

The script trains multiple independent Q-learning agents on the Taxi environment across several random seeds, once using **action masking** (restricting exploration/exploitation to only valid actions at each state) and once without it (considering all actions, valid or not). It then plots the training reward curves for both conditions to visually compare learning speed and stability.

## How it works

- **`train_q_learning(...)`** trains a single tabular Q-learning agent:
  - Initializes a Q-table of shape `(n_states, n_actions)`.
  - Uses an epsilon-greedy policy for action selection (`epsilon = 0.1` by default).
  - When `use_action_mask=True`, both action selection and the Q-learning bootstrap target are restricted to the environment's currently valid actions (via `info["action_mask"]`).
  - Runs for a configurable number of episodes, updating Q-values with the standard Q-learning update rule.
  - Returns the per-episode rewards along with their mean and standard deviation.

- **Experiment loop**: runs `n_runs = 12` independent trials (each with a different seed derived from `BASE_RANDOM_SEED`), training one masked and one unmasked agent per seed, for `episodes = 5000` episodes each.

- **Visualization**: plots every individual run's reward curve at low opacity, overlaid with the mean curve across all runs for both masked (blue) and unmasked (red) conditions. The resulting figure is saved to `_static/img/tutorials/taxi_v3_action_masking_comparison.png`.

## Requirements

- Python 3.12
- numpy
- matplotlib
- gymnasium

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib gymnasium
```

## Usage

```bash
python3 taxi_qlearning.py
```

This will print progress for each of the 12 runs, then display and save a plot comparing training performance with and without action masking.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `episodes` | 5000 | Number of training episodes per run |
| `n_runs` | 12 | Number of independent runs (different seeds) |
| `learning_rate` | 0.1 | Q-learning step size (α) |
| `discount_factor` | 0.95 | Reward discount factor (γ) |
| `epsilon` | 0.1 | Exploration rate for epsilon-greedy policy |
| `seed` | 58922320 (+ run index) | Random seed for reproducibility |

## Output

A comparison plot showing total reward per episode for both conditions, saved as a PNG and displayed on screen, illustrating whether action masking improves training speed and/or stability on the Taxi task.

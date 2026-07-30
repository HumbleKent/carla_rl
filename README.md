# CARLA Cone Avoidance RL (PPO Algorithm)

> **Reinforcement learning agents for autonomous driving in the [CARLA simulator](https://carla.org/) (v0.9.15)**  
> Built on [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) and [Gymnasium](https://gymnasium.farama.org/).

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python: 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![CARLA: 0.9.15](https://img.shields.io/badge/CARLA-0.9.15-orange.svg)

---

## Overview

CARLA Cone Avoidance RL (PPO Algorithm) is a framework for training and evaluating reinforcement learning driving agents to avoid from traffic cone inside the CARLA simulator. It provides:

- A **Gymnasium-compatible environment** (`carla-rl-gym-v0`) wrapping CARLA's Python API
- A **PPO agent** with an **EfficientNet-B0 visual backbone** fused with a scalar feature MLP
- **Parallel training** via `SubprocVecEnv` across multiple CARLA server instances
- **Traffic cone avoidance** as the primary training scenario, with support for custom scenario layouts
- Utility scripts for waypoint planning, cone placement, reward monitoring, and scenario visualisation

---

## Project Structure

```
CARLA-RL-Agents/
│
├── configuration.py            # Global hyperparameters and path constants
├── efficientnet_architecture.py # EfficientNet-B0 PPO feature extractor
│
├── train_cone_avoidance.py     # Main training entry point (PPO + parallel envs)
├── evaluate_best_model.py      # Evaluate a saved model over N episodes
├── run.ps1                     # PowerShell automation: server launch + scenario menu
│
├── env/                        # Gymnasium environment
│   ├── environment.py          # CarlaEnv — core gym.Env implementation
│   ├── reward.py               # Modular reward functions
│   ├── pre_processing.py       # Observation pre-processing
│   ├── observation_action_space.py
│   ├── cone_spawn.json         # Cone placement for training scenarios
│   ├── vehicle_spawn.json      # Vehicle spawn points and scenarios
│   └── vehicle_config/         # Sensor and physics JSON configs
│
├── src/                        # CARLA world/actor management modules
│   ├── world.py                # World wrapper (cone spawning, settings)
│   ├── vehicle.py              # Vehicle actor manager
│   ├── sensors.py              # Sensor suite management
│   ├── server.py               # CARLA server lifecycle helpers
│   ├── route_planner.py        # Waypoint-based route planning
│   ├── display.py              # Pygame sensor data display
│   └── ...
│
├── utils/                      # Development & scenario helper scripts
│   ├── debug_session.py        # Interactive debug session with visual overlays
│   ├── draw_waypoints.py
│   ├── generate_scenarios.py
│   ├── spawn_cone_json.py
│   ├── reward_monitor.py
│   └── visualize_all_scenarios.py
│
├── checkpoints/                # Model snapshot directory (.gitkeep)
├── logs_cone/                  # Training logs directory (.gitkeep)
├── tensorboard_cone/           # TensorBoard directory (.gitkeep)
│
├── environment.yml             # Conda environment (recommended)
├── requirements.txt            # Pip dependencies with version pins
└── QUICKSTART.md               # Common commands reference
```

---

## Prerequisites

- **CARLA 0.9.15** — [Download here](https://github.com/carla-simulator/carla/releases/tag/0.9.15)
- **Python 3.10**
- **NVIDIA GPU** with CUDA support (strongly recommended)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/CARLA-RL-Agents.git
   cd CARLA-RL-Agents
   ```

2. **Create & activate the Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate carla_blackwell
   ```

3. **Install the CARLA Python API wheel:**
   *(Replace `<CARLA_ROOT>` with your local CARLA installation path)*
   ```bash
   pip install <CARLA_ROOT>/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-win_amd64.whl
   ```

---

> See [QUICKSTART.md](QUICKSTART.md) for execution and command examples.

---

## Architecture

The agent uses a **multi-modal PPO policy**:

```
Observation Dict
├── rgb_data  (360×640×3)  ──► EfficientNet-B0 backbone (frozen)
│                               → GlobalAvgPool → [1280-dim]
│                                                         ┐
└── rest      (23-dim)     ──► MLP (23→128→256)          ├─► Concat [1536-dim] ──► PPO Head
                                → [256-dim]               ┘
```

- **Visual stream**: Pre-trained EfficientNet-B0 feature extractor (frozen weights, ImageNet-normalised)
- **Scalar stream**: 23-dimensional vector covering velocity, heading error, cone proximity, and last action
- **Output**: Continuous action space — `[steer, throttle/brake]` ∈ `[-1, 1]²`

---

## Training Outputs

All runtime artifacts are excluded from version control (see `.gitignore`):

| Directory | Contents |
|---|---|
| `checkpoints/` | Periodic model snapshots (`.zip`) per run |
| `logs_cone/` | Episode logs and `best_model.zip` per run |
| `tensorboard_cone/` | TensorBoard event files |

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This project is heavily based on and extends the following open-source works by [Ângelo Morgado](https://github.com/angelomorgado):

- **[CARLA-RL-Agents](https://github.com/angelomorgado/CARLA-RL-Agents)** — The primary reference repository. The agent architecture, training loop, parallel environment setup, and evaluation scripts in this project are adapted and extended from this work.
- **[CARLA-GymDrive](https://github.com/angelomorgado/CARLA-GymDrive)** — The gymnasium environment wrapper for CARLA. The `env/` and `src/` modules are built on top of this framework.

Additional thanks to:
- [CARLA Simulator](https://carla.org/) — for the autonomous driving simulation platform
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) by DLR-RM — for the reliable RL algorithm implementations
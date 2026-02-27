# CARLA RL Environment

## Introduction

This is a unified CARLA RL environment based on `gymnasium.Env`. It is designed for training agents to navigate and avoid obstacles (like traffic cones) using RGB input and a comprehensive feature vector.

## Instructions

The environment is registered as `carla-rl-gym-v0`.

Example usage:

```python
import gymnasium as gym
import env.environment

env = gym.make('carla-rl-gym-v0', port=2000)
obs, info = env.reset()

for _ in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Observation Space

The observation space is a `Dict`:
- **rgb_data**: (360, 640, 3) RGB image.
- **rest**: (23,) float32 vector containing:
    1. Distance to target
    2. Distance to next waypoint
    3. Yaw error (heading difference)
    4. Angular velocity (Z-axis)
    5. Forward velocity (local)
    6. Lateral velocity (local)
    7-8. Last action taken (Steer, Throttle/Brake)
    9-23. Top 5 closest cones (rel_x, rel_y, distance for each)

## Action Space

The environment uses a continuous action space:
- **Continuous**: `spaces.Box(low=-1.0, high=1.0, shape=(2,))`
    - Index 0: Steering (-1.0 to 1.0)
    - Index 1: Throttle/Brake (-1.0 to 1.0)

## Features

- **Obstacle Avoidance**: Built-in support for traffic cone detection and penalty logic.
- **Advanced Kinematics**: Local velocity and yaw error calculation for stable driving.
- **Parallel Support**: Designed to work with `SubprocVecEnv` and staggered worker initialization.
- **Dynamic Scenario Loading**: Loads scenarios from JSON configuration.

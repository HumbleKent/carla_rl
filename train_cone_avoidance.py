import sys
import os
import glob
import traceback
import argparse
import time
import json
import random

import carla
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from gymnasium.envs.registration import register

import configuration as config
from env.environment_cone import ConeCarlaEnv
from agent.cone_architecture import CustomExtractor_PPO_Cone
from src.world import World

# Register the environment if not already
try:
    register(
        id="carla-cone-rl-gym-v0",
        entry_point="env.environment_cone:ConeCarlaEnv",
        max_episode_steps=config.ENV_MAX_STEPS,
    )
except Exception:
    pass

class CustomEvalCallback(EvalCallback):
    def __init__(self, env, eval_freq, log_path, n_eval_episodes=5, deterministic=True, render=False):
        super().__init__(env, best_model_save_path=log_path, log_path=log_path, eval_freq=eval_freq, 
                         n_eval_episodes=n_eval_episodes, deterministic=deterministic, render=render)
        self.eval_results = []
        self.episode_numbers = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.eval_freq == 0:
            self.eval_results.append(self.last_mean_reward)
            self.episode_numbers.append(self.n_calls // self.eval_freq)
        return result

def make_env(port, rank=0, spawn_cones=False):
    def _init():
        env = gym.make('carla-cone-rl-gym-v0', 
                       port=port, 
                       time_limit=30, 
                       initialize_server=False, 
                       synchronous_mode=True, 
                       show_sensor_data=False, 
                       spawn_cones=spawn_cones,
                       verbose=False,
                       action_jitter=0.05)  # Add small noise to actions for exploration
        # Seed the environment to ensure each parallel instance is unique
        env.reset(seed=int(time.time()) + rank)
        env = Monitor(env)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", nargs='+', type=int, default=[2000], help="List of CARLA server ports")
    parser.add_argument("--run-name", type=str, default="v1", help="Unique name for this training run")
    parser.add_argument("--load-path", type=str, default=None, help="Path to a previous model to continue training")
    parser.add_argument("--eval-freq", type=int, default=1000, help="Number of steps between evaluations")
    parser.add_argument("--total-steps", type=int, default=1000000, help="Total timesteps to train for")
    args = parser.parse_args()

    print(f"Starting training on ports {args.ports} with run name '{args.run_name}'")

    # CUDA Check
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("WARNING: CUDA not found in this environment. Training will be slow on CPU.")
        device = "cpu"

    # 1. PRE-CHECK: Ensure CARLA servers are reachable and SPAWN CONES
    for port in args.ports:
        print(f"Connecting to CARLA on 127.0.0.1:{port} for setup...")
        time.sleep(1)
        try:
            client = carla.Client('127.0.0.1', port)
            client.set_timeout(15.0)
            # Create a temporary World to spawn cones
            world_obj = World(client=client, synchronous_mode=True)
            world_obj.spawn_cones_from_json()
            # Tick once to ensure they are registered
            client.get_world().tick()
            print(f"Successfully spawned cones and verified port {port}")
        except Exception as e:
            print(f"CRITICAL: Could not setup CARLA on port {port}. Error: {e}")
            sys.exit(1)

    # 2. CREATE VECTOR ENV
    print("Creating training environments...")
    if len(args.ports) > 1:
        env = SubprocVecEnv([make_env(port=p, rank=i, spawn_cones=False) for i, p in enumerate(args.ports)])
    else:
        env = DummyVecEnv([make_env(port=args.ports[0], spawn_cones=False)])
    
    env = VecTransposeImage(env)

    # 3. DIRECTORIES
    log_dir = f"./logs_cone/{args.run_name}/"
    checkpoint_dir = f"./checkpoints/ppo_cone_{args.run_name}/"
    tensorboard_dir = f"./tensorboard_cone/{args.run_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # 4. MODEL
    policy_kwargs = dict(features_extractor_class=CustomExtractor_PPO_Cone)
    if args.load_path and os.path.exists(args.load_path):
        print(f"Loading previous model from: {args.load_path}")
        model = PPO.load(
            args.load_path, 
            env=env, 
            device=device, 
            custom_objects={"tensorboard_log": tensorboard_dir}
        )
    else:
        print("Creating fresh model...")
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            n_steps=1024,
            batch_size=64,
            device=device
        )

    # 5. CALLBACKS
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=checkpoint_dir, name_prefix=f"ppo_{args.run_name}")
    eval_callback = CustomEvalCallback(env, eval_freq=args.eval_freq, log_path=log_dir)
    callback = CallbackList([checkpoint_callback, eval_callback])

    # 6. LEARN
    try:
        model.learn(total_timesteps=args.total_steps, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("Closing environment and cleaning up...")
        try:
            env.close()
        except:
            pass
        
        # Final cleanup of all CARLA actors on all ports
        for port in args.ports:
            try:
                print(f"Performing deep cleanup on port {port}...")
                client = carla.Client('127.0.0.1', port)
                client.set_timeout(5.0)
                world = client.get_world()
                
                # Turn off sync mode to ensure actors are destroyed immediately
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
                
                # Destroy all vehicles, walkers, sensors and cones
                actors = world.get_actors()
                for actor in actors.filter('vehicle.*'):
                    if actor.is_alive: actor.destroy()
                for actor in actors.filter('sensor.*'):
                    if actor.is_alive: actor.destroy()
                for actor in actors.filter('walker.*'):
                    if actor.is_alive: actor.destroy()
                for actor in actors.filter('static.prop.constructioncone'):
                    if actor.is_alive: actor.destroy()
                
                time.sleep(1) # Give time for destruction
                print(f"Cleanup finished on port {port}")
            except Exception as e:
                print(f"Failed to cleanup port {port}: {e}")

if __name__ == "__main__":
    main()

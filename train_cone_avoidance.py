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
from env.environment import CarlaEnv
from agent.cone_architecture import CustomExtractor_PPO_Cone
from src.world import World

# Register the environment if not already
# Register the environment if not already
try:
    register(
        id="carla-rl-gym-v0",
        entry_point="env.environment:CarlaEnv",
        max_episode_steps=config.ENV_MAX_STEPS,
    )
except Exception:
    pass

class EpisodeEvalCallback(EvalCallback):
    def __init__(self, env, eval_ep_freq, save_ep_freq, log_path, save_path, n_eval_episodes=5, deterministic=True, render=False):
        # We set eval_freq to a very large number so the parent class doesn't trigger it via steps
        super().__init__(env, best_model_save_path=log_path, log_path=log_path, eval_freq=int(1e12), 
                         n_eval_episodes=n_eval_episodes, deterministic=deterministic, render=render)
        self.eval_ep_freq = eval_ep_freq
        self.save_ep_freq = save_ep_freq
        self.save_path = save_path
        self.episodes_finished = 0
        self.eval_results = []
        self.episode_log = []

    def _on_step(self) -> bool:
        # local 'dones' is an array of booleans for each parallel environment
        for done in self.locals['dones']:
            if done:
                self.episodes_finished += 1
                
                # Trigger evaluation every N episodes
                if self.episodes_finished % self.eval_ep_freq == 0:
                    print(f"\n[Eval] {self.episodes_finished} episodes completed. Starting evaluation...")
                    time.sleep(1.0) # Small jitter to avoid simultaneous resets on shared port
                    
                    # Force parent evaluation logic by temporarily setting freq to 1
                    # (since any number % 1 == 0, it always triggers)
                    self.eval_freq = 1 
                    super()._on_step()
                    self.eval_freq = int(1e12) 

                    self.eval_results.append(self.last_mean_reward)
                    self.episode_log.append(self.episodes_finished)

                # TRIGGER CHECKPOINT every N episodes
                if self.save_ep_freq > 0 and self.episodes_finished % self.save_ep_freq == 0:
                    path = os.path.join(self.save_path, f"ppo_ep{self.episodes_finished}.zip")
                    print(f"[Checkpoint] Saving model to {path}")
                    self.model.save(path)
        
        return True

def make_env(port, rank=0, spawn_cones=False):
    def _init():
        log_file = f"logs_cone/worker_{port}_log.txt"
        os.makedirs("logs_cone", exist_ok=True)
        with open(log_file, "a") as f:
            f.write(f"\n--- Worker {port} starting up ---\n")
            f.flush()
        try:
            with open(log_file, "a") as f: f.write(f"Calling gym.make for port {port}...\n"); f.flush()
            env = gym.make('carla-rl-gym-v0', 
                           port=port, 
                           time_limit=30, 
                           initialize_server=False, 
                           synchronous_mode=True, 
                           show_sensor_data=False, 
                           spawn_cones=True,
                           verbose=False,
                           action_jitter=0.0)
            
            with open(log_file, "a") as f: f.write(f"gym.make succeeded. Calling env.reset()...\n"); f.flush()
            env.reset(seed=int(time.time()) + rank)
            
            with open(log_file, "a") as f: f.write(f"env.reset() succeeded. Wrapping Monitor...\n"); f.flush()
            env = Monitor(env)
            
            with open(log_file, "a") as f: f.write(f"Worker {port} initialization fully complete.\n"); f.flush()
            return env
        except BaseException as e:
            with open(log_file, "a") as f: 
                f.write(f"FATAL CRASH IN WORKER {port}:\n{str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
                f.flush()
            raise
    return _init

def make_eval_env(port):
    """Create a dedicated single-process eval env on the given port.
    
    IMPORTANT: Never pass the training env to EvalCallback — doing so causes
    evaluate_policy to step training workers mid-episode, which triggers
    KeyError: 'collision' (sensors not ready) and kills the subprocess pipe.
    """
    def _init():
        env = gym.make('carla-rl-gym-v0',
                       port=port,
                       time_limit=30,
                       initialize_server=False,
                       synchronous_mode=True,
                       show_sensor_data=False,
                       spawn_cones=True,
                       verbose=False,
                       action_jitter=0.0)  # No jitter during eval
        env = Monitor(env)
        return env
    return _init

def main():
    def make_env_with_delay(p, i):
        def _init():
            # A staggered delay to prevent simultaneous connections hitting CARLA
            time.sleep(i * 2.0)
            env_fn = make_env(port=p, rank=i, spawn_cones=False)
            return env_fn()
        return _init

    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", nargs='+', type=int, default=[2000], help="List of CARLA server ports")
    parser.add_argument("--run-name", type=str, default="v1", help="Unique name for this training run")
    parser.add_argument("--load-path", type=str, default=None, help="Path to a previous model to continue training")
    parser.add_argument("--eval-ep-freq", type=int, default=25, help="Number of episodes between evaluations")
    parser.add_argument("--save-ep-freq", type=int, default=50, help="Number of episodes between model checkpoints")
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
            
            # CRITICAL: Turn OFF synchronous mode before this client disconnects!
            # Otherwise, the CARLA server will hang indefinitely waiting for a tick.
            settings = client.get_world().get_settings()
            settings.synchronous_mode = False
            client.get_world().apply_settings(settings)
            
            print(f"Successfully spawned cones and verified port {port}")
        except Exception as e:
            print(f"CRITICAL: Could not setup CARLA on port {port}. Error: {e}")
            sys.exit(1)

    # 2. CREATE VECTOR ENV (Parallel mode)
    print("Creating training environments...")

    if len(args.ports) > 1:
        env = SubprocVecEnv([make_env_with_delay(p, i) for i, p in enumerate(args.ports)])
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
    print("Initializing model...")
    policy_kwargs = dict(features_extractor_class=CustomExtractor_PPO_Cone)
    if args.load_path and os.path.exists(args.load_path):
        model = PPO.load(args.load_path, env=env, device=device, custom_objects={"tensorboard_log": tensorboard_dir})
    else:
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

    # 5. CALLBACKS (Both Eval and Saving are now done in one episode-based callback)
    eval_port = args.ports[0]  # Use first port for evaluation
    print(f"Creating dedicated eval env on port {eval_port}...")
    eval_env = DummyVecEnv([make_eval_env(eval_port)])
    eval_env = VecTransposeImage(eval_env)
    
    callback = EpisodeEvalCallback(
        eval_env, 
        eval_ep_freq=args.eval_ep_freq, 
        save_ep_freq=args.save_ep_freq,
        log_path=log_dir,
        save_path=checkpoint_dir
    )

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

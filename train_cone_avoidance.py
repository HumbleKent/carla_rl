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
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register

import configuration as config
from env.environment import CarlaEnv
from efficientnet_architecture import CustomExtractor_PPO_EfficientNet
from src.world import World
from src.server import CarlaServer

# Register the environment if not already
try:
    register(
        id="carla-rl-gym",
        entry_point="env.environment:CarlaEnv",
        max_episode_steps=config.ENV_MAX_STEPS,
    )
except Exception:
    pass

class StatusCheckpointCallback(BaseCallback):
    def __init__(self, save_step_freq, save_path, initial_episode=0):
        super().__init__(verbose=1)
        self.save_step_freq = save_step_freq
        self.last_save_step = 0
        self.save_path = save_path
        self.episodes_finished = initial_episode
        self.local_episodes = 0

    def _on_step(self) -> bool:
        # Check termination and logic...
        for idx, done in enumerate(self.locals['dones']):
            if done:
                self.episodes_finished += 1
                self.local_episodes += 1
                info = self.locals['infos'][idx]
                scenario = info.get('scenario_name', 'Unknown')
                port = info.get('port', 'Unknown')
                
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    print(f"  [EPISODE] {self.local_episodes} (Global: {self.episodes_finished}) [Port {port}] finished! Status: {scenario} | R: {ep_reward:.1f}")

                # Trigger checkpoint every N steps (checked at episode end)
                if self.save_step_freq > 0 and self.num_timesteps >= self.last_save_step + self.save_step_freq:
                    self.last_save_step = (self.num_timesteps // self.save_step_freq) * self.save_step_freq
                    path = os.path.join(self.save_path, f"ppo_step{self.num_timesteps}_ep{self.episodes_finished}.zip")
                    print(f"  [Checkpoint] Saving model at step {self.num_timesteps} (Ep: {self.episodes_finished}) to {path}")
                    self.model.save(path)
        
        return True

class SurgicalEvalCallback(BaseCallback):
    def __init__(self, eval_port, eval_freq, log_path, run_name, train_scenarios, worker_verbose=False, deterministic=True, verbose=1):
        super().__init__(verbose=verbose)
        self.eval_port = eval_port
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.run_name = run_name
        self.train_scenarios = train_scenarios
        self.worker_verbose = worker_verbose
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"\n--- [EVALUATION] Starting surgical evaluation on Port {self.eval_port} ---")
            server_proc = None
            eval_env = None
            try:
                # 1. Start Server
                print(f"  [Eval] Starting CARLA server on Port {self.eval_port}...")
                server_proc = CarlaServer.initialize_server(port=self.eval_port, low_quality=config.SIM_LOW_QUALITY, sleep_time=8)
                
                # 2. Create Env
                print(f"  [Eval] Creating environment...")
                eval_env_init = make_env(port=self.eval_port, rank=99, spawn_cones=True, scenarios=self.train_scenarios, show_sensor_data=False, verbose=self.worker_verbose)()
                eval_env = Monitor(eval_env_init)
                eval_env = DummyVecEnv([lambda: eval_env])
                eval_env = VecTransposeImage(eval_env)
                
                # 3. Run Evaluation
                
                print(f"  [Eval] Running episodes (deterministic={self.deterministic})...")
                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    eval_env,
                    n_eval_episodes=5,
                    render=False,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                )
                
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                mean_len = np.mean(episode_lengths)
                
                print(f"  [Eval] Results: Reward {mean_reward:.2f} +/- {std_reward:.2f} | Length {mean_len:.1f}")
                
                # 4. Save best model if improved
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    save_path = os.path.join(self.log_path, "best_model.zip")
                    self.model.save(save_path)
                    print(f"  [Eval] New best model saved to {save_path}")
                
            except Exception as e:
                print(f"  [Eval] CRITICAL ERROR during evaluation: {e}")
            finally:
                # 5. Surgical Shutdown
                print(f"  [Eval] Shutting down evaluation resources...")
                if eval_env:
                    try: eval_env.close()
                    except: pass
                if server_proc:
                    try: CarlaServer.close_server(server_proc, silent=True)
                    except: pass
                print("--- [EVALUATION] Finished ---\n")
                
        return True

def make_env(port, rank=0, spawn_cones=False, scenarios=None, show_sensor_data=False, verbose=False):
    def _init():
        log_file = f"logs_cone/worker_{port}_log.txt"
        os.makedirs("logs_cone", exist_ok=True)
        with open(log_file, "a") as f:
            f.write(f"\n--- Worker {port} starting up ---\n")
            f.flush()
        try:
            with open(log_file, "a") as f: f.write(f"Calling gym.make for port {port}...\n"); f.flush()
            env = gym.make('carla-rl-gym', 
                           port=port, 
                           time_limit=30, 
                           initialize_server=False, 
                           synchronous_mode=True, 
                           show_sensor_data=show_sensor_data,  # Passed from main()
                           spawn_cones=spawn_cones,
                           verbose=verbose,
                           scenarios=scenarios,
                           camera=config.CAMERA_VIEW,
                           debug_features=[])
            
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

# Evaluation environment creation removed

def main():
    def make_env_with_delay(p, i):
        def _init():
            # A staggered delay to prevent simultaneous connections hitting CARLA
            time.sleep(i * 2.0)
            env_fn = make_env(port=p, rank=i, spawn_cones=True, scenarios=train_scenarios, show_sensor_data=args.show_sensor_data, verbose=worker_verbose)
            return env_fn()
        return _init

    parser = argparse.ArgumentParser()
    parser.add_argument("--ports","-p", nargs='+', type=int, default=[2000], help="List of CARLA server ports")
    parser.add_argument("--run-name","-n", type=str, default="v1", help="Unique name for this training run")
    parser.add_argument("--load-path","-l", type=str, default=None, help="Path to a previous model to continue training")
    parser.add_argument("--save-step-freq","-ss", type=int, default=20000, help="Number of steps between model checkpoints [Episode equivalent: ~50]")
    parser.add_argument("--total-steps","-t", type=int, default=5000000, help="Total timesteps to train for")
    parser.add_argument("--scenario","-sc", type=str, default=None, help="Specific scenario to train on (default: all)")
    parser.add_argument("--camera","-c", type=str, default=config.CAMERA_VIEW, choices=["front", "bev"], help="Camera view to use (front, bev)")
    parser.add_argument("--eval-port","-ep", type=int, default=4000, help="DEDICATED Port for the evaluation environment (avoids training conflicts)")
    parser.add_argument("--show-sensor-data","-s", action="store_true", help="Enable Pygame window for sensor visualization (single port recommended)")
    args = parser.parse_args()

    config.CAMERA_VIEW = args.camera

    train_scenarios = [args.scenario] if args.scenario else []


    worker_verbose = False
    print(f"Starting training on ports {args.ports} with run name '{args.run_name}'")

    # CUDA Check
    import torch
    # ... rest of CUDA check code ...
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("WARNING: CUDA not found in this environment. Training will be slow on CPU.")
        device = "cpu"

    # 1. PRE-CHECK: Ensure training CARLA servers are reachable and SPAWN CONES
    setup_ports = list(args.ports)
    spawned_servers = []

    for port in setup_ports:
        purpose = "Training"
        print(f"Connecting to CARLA on 127.0.0.1:{port} for {purpose} setup...")
        
        connected = False
        for attempt in range(2):
            try:
                client = carla.Client('127.0.0.1', port)
                client.set_timeout(10.0)
                # Check connection
                client.get_world()
                connected = True
                break
            except Exception:
                print(f"CRITICAL: Training Port {port} is not running. Please start it manually.")
                sys.exit(1)
        
        try:
            # Clear out any old debug markers from previous runs (like plan_waypoints.py)
            print(f"Reloading world on port {port} to clear old debug markers...")
            client.reload_world(reset_settings=False)
            
            # Create a temporary World to spawn cones
            from src.world import World as SetupWorld
            world_obj = SetupWorld(client=client, synchronous_mode=True)
            world_obj.spawn_cones_from_json()
            # Tick once to ensure they are registered
            client.get_world().tick()
            
            # CRITICAL: Turn OFF synchronous mode before this client disconnects!
            settings = client.get_world().get_settings()
            settings.synchronous_mode = False
            client.get_world().apply_settings(settings)
            
            print(f"Successfully verified and prepared port {port}")
        except Exception as e:
            print(f"CRITICAL: Error during setup of port {port}: {e}")
            sys.exit(1)

    # 2. CREATE VECTOR ENV (Parallel mode)
    print("Creating training environments...")

    if len(args.ports) > 1:
        def make_env_wrapper(p, i):
            def _init():
                try:
                    time.sleep(i * 2.5) # Staggered startup
                    return Monitor(gym.make('carla-rl-gym', 
                                   port=p, 
                                   time_limit=30, 
                                    initialize_server=False, 
                                    synchronous_mode=True, 
                                    show_sensor_data=args.show_sensor_data,
                                    spawn_cones=True,
                                    verbose=worker_verbose,
                                    scenarios=train_scenarios,
                                    camera=config.CAMERA_VIEW,
                                    debug_features=["term"]))
                except Exception as e:
                    print(f"FAILED to create environment on port {p}: {e}")
                    raise
            return _init
        
        env = SubprocVecEnv([make_env_wrapper(p, i) for i, p in enumerate(args.ports)])
    else:
        # For single port, no delay needed
        env = DummyVecEnv([lambda: Monitor(gym.make('carla-rl-gym', 
                                                   port=args.ports[0], 
                                                   time_limit=30, 
                                                   initialize_server=False, 
                                                   synchronous_mode=True, 
                                                   show_sensor_data=args.show_sensor_data,
                                                   spawn_cones=True,
                                                   verbose=worker_verbose,
                                                   scenarios=train_scenarios,
                                                   camera=config.CAMERA_VIEW,
                                                   debug_features=["term"]))])
    
    env = VecTransposeImage(env)

    # 3. DIRECTORIES
    log_dir = f"./logs_cone/{args.run_name}/"
    checkpoint_dir = f"./checkpoints/ppo_cone_{args.run_name}/"
    tensorboard_dir = f"./tensorboard_cone/{args.run_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # 4. MODEL
    print(f"Initializing model with EfficientNet backbone...")
    extractor_class = CustomExtractor_PPO_EfficientNet

    policy_kwargs = dict(features_extractor_class=extractor_class)
    
    initial_step = 0
    initial_episode = 0
    is_new_run = True # Default to fresh run
    
    if args.load_path:
        # Check if file exists, if not try checkpoints/ path
        load_path = args.load_path
        if not os.path.exists(load_path):
            candidate = os.path.join("checkpoints", f"ppo_cone_{args.run_name}", os.path.basename(load_path))
            if os.path.exists(candidate):
                load_path = candidate
            else:
                # Try one more: maybe the user provided just the zip name
                for root, dirs, files in os.walk("checkpoints"):
                    if os.path.basename(load_path) in files:
                        load_path = os.path.join(root, os.path.basename(load_path))
                        break
        
        if os.path.exists(load_path):
            print(f"Loading previous model from: {load_path}")
            model = PPO.load(load_path, env=env, device=device, custom_objects={"tensorboard_log": tensorboard_dir, "learning_rate": 1e-4})
            model.batch_size = 128
            is_new_run = False # Found a model, continuation
            import re
            filename = os.path.basename(load_path)
            
            # Better regex for step (\d+) and ep (\d+)
            step_match = re.search(r"step(\d+)", filename)
            ep_match = re.search(r"_ep(\d+)", filename) # Use _ep to avoid 'step' match
            if not ep_match: # Fallback
                ep_match = re.search(r"ep(\d+)", filename.replace("step", "xxxx"))
                
            if step_match:
                initial_step = int(step_match.group(1))
                # Sync model.num_timesteps in case it wasn't in the zip or we want to override
                model.num_timesteps = initial_step
                
            if ep_match:
                initial_episode = int(ep_match.group(1))
                print(f"Resuming training: Step {initial_step}, Episode {initial_episode}")
        else:
            print(f"ERROR: Could not find model at '{args.load_path}'. Starting fresh...")
            model = PPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=tensorboard_dir,
                learning_rate=1e-4,
                n_steps=2048,
                batch_size=128,
                ent_coef=0.1,
                device=device
            )
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=128,
            ent_coef=0.1,
            device=device
        )

    # 5. CALLBACKS
    status_callback = StatusCheckpointCallback(
        save_step_freq=args.save_step_freq,
        save_path=checkpoint_dir,
        initial_episode=initial_episode
    )

    # Path for evaluation logs (official evaluations.npz and best_model.zip)
    eval_log_dir = os.path.join("logs_cone", args.run_name)
    os.makedirs(eval_log_dir, exist_ok=True)

    # 4b. SURGICAL EVALUATION CALLBACK
    # This replacement ensures Port 4000 is only active during the evaluation phase.
    eval_callback = SurgicalEvalCallback(
        eval_port=args.eval_port,
        eval_freq=max(1000, 10000 // len(args.ports)),
        log_path=eval_log_dir,
        run_name=args.run_name,
        train_scenarios=train_scenarios,
        worker_verbose=worker_verbose,
        deterministic=True
    )

    callback = CallbackList([status_callback, eval_callback])
    try:
        # Calculate remaining steps if we want total-steps to be the absolute target
        remaining_steps = args.total_steps - model.num_timesteps
        if remaining_steps <= 0:
            print(f"Target steps {args.total_steps} already reached (current: {model.num_timesteps}). Nothing to do.")
        else:
            print(f"Training for {remaining_steps} more steps (Total Target: {args.total_steps})")
            # Using tb_log_name ensures SB3 creates numbered subfolders like PPO_0, PPO_1 inside your run_name folder
            model.learn(total_timesteps=remaining_steps, callback=callback, reset_num_timesteps=is_new_run, tb_log_name="PPO")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nCRITICAL FAILURE during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing environment and cleaning up...")
        
        # 1. Kill specifically spawned servers FIRST to prevent cleanup hangs
        for proc in spawned_servers:
            try:
                CarlaServer.close_server(proc)
            except:
                pass

        # 2. Close training environments
        try:
            env.close()
        except:
            pass
        
        # 3. Final cleanup of all CARLA actors on training ports
        for port in args.ports:
            try:
                print(f"Performing deep cleanup on port {port}...")
                client = carla.Client('127.0.0.1', port)
                client.set_timeout(5.0)
                world = client.get_world()
                
                # Turn off sync mode before clearing and reloading
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
                
                # Reloading the world is the only way in CARLA to clear persistent debug markers
                client.reload_world(reset_settings=False)
                print(f"Cleanup finished on port {port}")
            except Exception as e:
                print(f"Failed to cleanup port {port}: {e}")

if __name__ == "__main__":
    main()

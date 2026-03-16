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
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.envs.registration import register

import configuration as config
from env.environment import CarlaEnv
from agent.efficient_architecture import CustomExtractor_PPO_EfficientNet
from src.world import World

# Register the environment if not already
try:
    register(
        id="carla-rl-gym-v0",
        entry_point="env.environment:CarlaEnv",
        max_episode_steps=config.ENV_MAX_STEPS,
    )
except Exception:
    pass

class EpisodeEvalCallback(BaseCallback):
    def __init__(self, eval_port, eval_ep_freq, save_ep_freq, log_path, save_path, n_eval_episodes=5, deterministic=True, initial_episode=0, scenarios=None):
        super().__init__(verbose=1)
        self.eval_port = eval_port
        self.eval_ep_freq = eval_ep_freq
        self.save_ep_freq = save_ep_freq
        self.log_path = log_path
        self.save_path = save_path
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.initial_episode = initial_episode
        self.episodes_finished = initial_episode
        self.local_episodes = 0
        self.best_mean_reward = -np.inf
        self.eval_results = []
        self.episode_log = []
        self.eval_server_process = None
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_length = []
        self.scenarios = scenarios

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
                    print(f"  [EPISODE] {self.local_episodes} (Global: {self.episodes_finished}) [Port {port}] finished! Status: {scenario} | R: {info['episode']['r']:.1f}")

                # LAZY EVALUATION TRIGGER
                if self.episodes_finished % self.eval_ep_freq == 0:
                    print(f"\n>>>> Starting EVALUATION on Port: {self.eval_port}...")
                    
                    from src.server import CarlaServer
                    # 1. Ephemeral Server Launch (if separate)
                    # We check if something is already there first to be safe, but launch if not
                    self.eval_server_process = None
                    try:
                        # Try to connect to see if one is already running
                        c = carla.Client('127.0.0.1', self.eval_port)
                        c.set_timeout(2.0)
                        c.get_world()
                        print(f">>>> Using existing CARLA server on port {self.eval_port}")
                    except:
                        print(f">>>> Launching ephemeral EVAL server on port {self.eval_port}...")
                        self.eval_server_process = CarlaServer.initialize_server(
                            port=self.eval_port, 
                            low_quality=True, 
                            offscreen_rendering=True, 
                            sleep_time=15
                        )

                    # Create and Close Eval Env (Lazy)
                    try:
                        import carla
                        # 1. Setup port 4000 (Spawn cones)
                        temp_client = carla.Client('127.0.0.1', self.eval_port)
                        temp_client.set_timeout(15.0)
                        settings = temp_client.get_world().get_settings()
                        was_sync = settings.synchronous_mode
                        
                        settings.synchronous_mode = False # ensure off for setup
                        temp_client.get_world().apply_settings(settings)
                        temp_client.reload_world(reset_settings=False)
                        
                        from src.world import World as SetupWorld
                        sw = SetupWorld(client=temp_client, synchronous_mode=True)
                        sw.spawn_cones_from_json()
                        temp_client.get_world().tick()
                        
                        # 2. Create the Env for stable_baselines
                        eval_env = DummyVecEnv([lambda: Monitor(gym.make('carla-rl-gym-v0', 
                                                                         port=self.eval_port,
                                                                         time_limit=30,
                                                                         initialize_server=False,
                                                                         synchronous_mode=True,
                                                                         spawn_cones=True,
                                                                         is_eval=True,
                                                                         verbose=False,
                                                                         scenarios=self.scenarios,
                                                                         debug_features=['target']))])
                        eval_env = VecTransposeImage(eval_env)
                        
                        # 3. RUN EVALUATION
                        episode_rewards, episode_lengths = evaluate_policy(
                            self.model, 
                            eval_env, 
                            n_eval_episodes=self.n_eval_episodes, 
                            deterministic=self.deterministic,
                            return_episode_rewards=True
                        )
                        mean_reward = np.mean(episode_rewards)
                        std_reward = np.std(episode_rewards)
                        print(f">>>> EVAL COMPLETE. Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

                        # 4. Save best model
                        if mean_reward > self.best_mean_reward:
                             self.best_mean_reward = mean_reward
                             self.model.save(os.path.join(self.log_path, "best_model.zip"))
                        
                        # Add to Tensorboard
                        self.logger.record("eval/mean_reward", float(mean_reward))
                        self.logger.record("eval/std_reward", float(std_reward))
                        self.logger.dump(step=self.num_timesteps)

                        # Store results for .npz
                        self.evaluations_timesteps.append(self.num_timesteps)
                        self.evaluations_results.append(episode_rewards)
                        self.evaluations_length.append(episode_lengths)

                        np.savez(
                            os.path.join(self.log_path, "evaluations.npz"),
                            timesteps=self.evaluations_timesteps,
                            results=self.evaluations_results,
                            ep_lengths=self.evaluations_length,
                        )
                        
                        # 5. CLOSE and Clean (Release port 4000)
                        eval_env.close()
                        
                        # 6. Shut down ephemeral server
                        if self.eval_server_process:
                            print(f">>>> Closing ephemeral EVAL server on port {self.eval_port}...")
                            CarlaServer.close_server(self.eval_server_process)
                            self.eval_server_process = None
                            
                        print(f">>>> Evaluation cycle on Port {self.eval_port} finished.\n")
                        
                    except Exception as e:
                        print(f">>>> [ERROR] Evaluation on port {self.eval_port} failed: {e}")
                        # Ensure cleanup if failed
                        if self.eval_server_process:
                             try: CarlaServer.close_server(self.eval_server_process)
                             except: pass
                             self.eval_server_process = None
                        import traceback
                        traceback.print_exc()

                # Trigger checkpoint every N episodes
                if self.save_ep_freq > 0 and self.episodes_finished % self.save_ep_freq == 0:
                    path = os.path.join(self.save_path, f"ppo_ep{self.episodes_finished}.zip")
                    print(f"  [Checkpoint] Saving model to {path}")
                    self.model.save(path)
        
        return True

def make_env(port, rank=0, spawn_cones=False, scenarios=None):
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
                           show_sensor_data=False,  # Disabled: subprocesses can't display Pygame windows
                           spawn_cones=True,
                           verbose=worker_verbose,
                           scenarios=scenarios,
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

def make_eval_env(port, scenarios=None):
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
                       show_sensor_data=False,  # Disabled: eval env also runs in subprocess during training
                       spawn_cones=True,
                       scenarios=scenarios,
                       is_eval=True)  # No jitter during eval
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
    parser.add_argument("--ports","-p", nargs='+', type=int, default=[2000], help="List of CARLA server ports")
    parser.add_argument("--run-name","-n", type=str, default="v1", help="Unique name for this training run")
    parser.add_argument("--load-path","-l", type=str, default=None, help="Path to a previous model to continue training")
    parser.add_argument("--eval-ep-freq","-e", type=int, default=25, help="Number of episodes between evaluations")
    parser.add_argument("--save-ep-freq","-s", type=int, default=50, help="Number of episodes between model checkpoints")
    parser.add_argument("--total-steps","-t", type=int, default=1000000, help="Total timesteps to train for")
    parser.add_argument("--eval-port","-ep", type=int, default=4000, help="Dedicated port for evaluation (optional)")
    parser.add_argument("--scenario","-sc", type=str, default=None, help="Specific scenario to train on (default: all)")
    args = parser.parse_args()

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

    # 1. PRE-CHECK: Ensure CARLA servers are reachable and SPAWN CONES
    setup_ports = list(args.ports)

    for port in setup_ports:
        print(f"Connecting to CARLA on 127.0.0.1:{port} for setup...")
        time.sleep(1)
        try:
            client = carla.Client('127.0.0.1', port)
            client.set_timeout(15.0)
            
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
            
            print(f"Successfully spawned cones and verified port {port}")
        except Exception as e:
            print(f"CRITICAL: Could not setup CARLA on port {port}. Error: {e}")
            sys.exit(1)

    # 2. CREATE VECTOR ENV (Parallel mode)
    print("Creating training environments...")

    if len(args.ports) > 1:
        def make_env_wrapper(p, i):
            def _init():
                try:
                    time.sleep(i * 2.5) # Staggered startup
                    return Monitor(gym.make('carla-rl-gym-v0', 
                                   port=p, 
                                   time_limit=30, 
                                    initialize_server=False, 
                                    synchronous_mode=True, 
                                    show_sensor_data=False,
                                    spawn_cones=True,
                                    verbose=worker_verbose,
                                    scenarios=train_scenarios,
                                    debug_features=[]))
                except Exception as e:
                    print(f"FAILED to create environment on port {p}: {e}")
                    raise
            return _init
        
        env = SubprocVecEnv([make_env_wrapper(p, i) for i, p in enumerate(args.ports)])
    else:
        # For single port, no delay needed
        env = DummyVecEnv([lambda: Monitor(gym.make('carla-rl-gym-v0', 
                                                   port=args.ports[0], 
                                                   time_limit=30, 
                                                   initialize_server=False, 
                                                   synchronous_mode=True, 
                                                   show_sensor_data=False,
                                                   spawn_cones=True,
                                                   verbose=worker_verbose,
                                                   scenarios=train_scenarios,
                                                   debug_features=[]))])
    
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
    policy_kwargs = dict(features_extractor_class=CustomExtractor_PPO_EfficientNet)
    if args.load_path and os.path.exists(args.load_path):
        model = PPO.load(args.load_path, env=env, device=device, custom_objects={"tensorboard_log": tensorboard_dir})
        model.batch_size = 128 # Explicitly update batch size when loading
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            n_steps=1024,
            batch_size=128,
            device=device
        )

    import re
    initial_episode = 0
    if args.load_path:
        # Try to extract episode number from filename (e.g., ppo_ep1650.zip)
        match = re.search(r"ep(\d+)", os.path.basename(args.load_path))
        if match:
            initial_episode = int(match.group(1))
            print(f"Resuming training from episode {initial_episode}")

    # 5. CALLBACKS
    eval_port = args.eval_port if args.eval_port else args.ports[-1]
    callback = EpisodeEvalCallback(
        eval_port=eval_port, 
        eval_ep_freq=args.eval_ep_freq, 
        save_ep_freq=args.save_ep_freq,
        log_path=log_dir,
        save_path=checkpoint_dir,
        initial_episode=initial_episode,
        scenarios=train_scenarios
    )

    # 6. LEARN
    try:
        model.learn(total_timesteps=args.total_steps, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nCRITICAL FAILURE during training: {e}")
        import traceback
        traceback.print_exc()
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

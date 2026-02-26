import os
import argparse
import time
import carla
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from gymnasium.envs.registration import register

import configuration as config
from env.environment_cone import ConeCarlaEnv
from agent.cone_architecture import CustomExtractor_PPO_Cone
from src.world import World

# 1. Register the environment
try:
    register(
        id="carla-cone-rl-gym-v0",
        entry_point="env.environment_cone:ConeCarlaEnv",
        max_episode_steps=config.ENV_MAX_STEPS,
    )
except Exception:
    pass

def main():
    parser = argparse.ArgumentParser(description="Evaluate the best trained CARLA RL model")
    parser.add_argument("--port", "-p", type=int, default=config.SIM_PORT, help="CARLA server port")
    parser.add_argument("--name", "-n", type=str, default="blackwell_fast_v1", help="Name of the training run")
    parser.add_argument("--episodes", "-ep", type=int, default=5, help="Number of evaluation episodes")
    args = parser.parse_args()

    # 2. Define path to the best model
    model_path = os.path.abspath(f"logs_cone/{args.name}/best_model.zip")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please check your run-name or ensure a best_model.zip exists in the logs_cone folder.")
        return

    print(f"Loading best model from: {model_path}")

    # 3. Initialize the environment
    # We use DummyVecEnv and VecTransposeImage to match the training observation processing
    def make_env():
        env = gym.make('carla-cone-rl-gym-v0', 
                       port=args.port, 
                       time_limit=120, 
                       initialize_server=False, 
                       synchronous_mode=True, 
                       show_sensor_data=True, 
                       spawn_cones=True, 
                       verbose=True)
        return env

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

    # 4. Load the PPO model
    # We specify the custom architecture so SB3 knows how to load the weights
    policy_kwargs = dict(features_extractor_class=CustomExtractor_PPO_Cone)
    model = PPO.load(model_path, env=env, custom_objects={"policy_kwargs": policy_kwargs})

    print(f"Starting evaluation for {args.episodes} episodes...")

    # 5. Evaluation Loop
    try:
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            print(f"\n--- Episode {episode} ---")
            
            while not done:
                # Predict action using the loaded model
                # deterministic=True is usually preferred for evaluation
                action, _states = model.predict(obs, deterministic=True)
                
                # Take action in environment
                obs, rewards, dones, infos = env.step(action)
                
                total_reward += rewards[0]
                steps += 1
                done = dones[0]

                if steps % 10 == 0:
                    print(f"Step: {steps} | Current Reward: {total_reward:.2f}", end="\r")

            print(f"\nEpisode {episode} finished!")
            print(f"Total Steps: {steps}")
            print(f"Total Reward: {total_reward:.2f}")
            time.sleep(2) # Brief pause between episodes
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        # 6. Cleanup
        print("Cleaning up environment and destroying actors...")
        try:
            env.close()
        except Exception:
            pass
        
        # Deep cleanup: Applying world-level destruction logic directly for maximum robustness
        try:
            client = carla.Client('127.0.0.1', args.port)
            client.set_timeout(5.0)
            world = client.get_world()
            
            # Switch to asynchronous mode to ensure immediate destruction
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            
            # Find and destroy all lingering actors in the world
            actors = world.get_actors()
            to_destroy = []
            to_destroy.extend(list(actors.filter('vehicle.*')))
            to_destroy.extend(list(actors.filter('sensor.*')))
            to_destroy.extend(list(actors.filter('static.prop.constructioncone')))
            to_destroy.extend(list(actors.filter('walker.*')))
            to_destroy.extend(list(actors.filter('controller.ai.walker')))
            
            print(f"Deep cleanup: Found {len(to_destroy)} actors to destroy.")
            for actor in to_destroy:
                if actor.is_alive:
                    try:
                        actor.destroy()
                    except Exception:
                        pass # Ignore actors that already disappeared
            
            print("Deep cleanup: CARLA world is now clean.")
        except Exception as e:
            print(f"Notice: Secondary deep cleanup encountered an issue: {e}")
            
        print("Evaluation complete.")

if __name__ == "__main__":
    main()

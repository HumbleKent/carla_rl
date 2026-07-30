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
from env.environment import CarlaEnv
from efficient_architecture import CustomExtractor_PPO_EfficientNet

# 1. Register the environment
try:
    register(
        id="carla-rl-gym-v0",
        entry_point="env.environment:CarlaEnv",
        max_episode_steps=config.ENV_MAX_STEPS,
    )
except Exception:
    pass

def main():
    parser = argparse.ArgumentParser(description="Dedicated Debug Session for CARLA RL Agents")
    parser.add_argument("--port", "-p", type=int, default=config.SIM_PORT, help="CARLA server port")
    parser.add_argument("--name", "-n", type=str, default="huge_stand_penalty", help="Name of the training run")
    parser.add_argument("--episodes", "-ep", type=int, default=3, help="Number of debug episodes")
    parser.add_argument("--efficient", "-eff", action="store_true", help="Force use of EfficientNet architecture")
    parser.add_argument("--debug", "-d", nargs="+", default=["waypoints", "target", "term"], 
                        help="Debug features: waypoints, target, sensors, nn, reward, term")
    args = parser.parse_args()

    # 2. Find the model
    model_path = os.path.abspath(f"logs_cone/{args.name}/best_model.zip")
    if not os.path.exists(model_path):
        # Fallback to latest checkpoint if best_model doesn't exist
        checkpoint_dir = f"checkpoints/ppo_cone_{args.name}/"
        if os.path.exists(checkpoint_dir):
            import glob
            list_of_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
            if list_of_files:
                model_path = max(list_of_files, key=os.path.getctime)
                print(f"best_model.zip not found, using latest checkpoint: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found for run '{args.name}'")
        return

    # 3. Setup Environment with Debug Features
    def make_env():
        env = gym.make('carla-rl-gym-v0', 
                       port=args.port, 
                       time_limit=120, 
                       initialize_server=False, 
                       synchronous_mode=True, 
                       show_sensor_data=True, 
                       spawn_cones=True, 
                       verbose=True,
                       debug_features=args.debug) # <--- Debug features passed here
        return env

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

    # 4. Load Model
    print(f"Loading model: {model_path}")
    # Architecture detection
    policy_kwargs = dict(features_extractor_class=CustomExtractor_PPO_EfficientNet)
    model = PPO.load(model_path, env=env, custom_objects={"policy_kwargs": policy_kwargs})

    # 5. Debug Loop
    try:
        for ep in range(args.episodes):
            print(f"\n--- Starting Debug Episode {ep+1}/{args.episodes} ---")
            obs = env.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                step += 1
                
                # The DebugManager inside CarlaEnv handles the visuals (waypoints, NN window, etc.)
                
                # Slow down the debug simulation slightly to make it easier to see
                time.sleep(0.05)
                
                if done:
                    print(f"Episode {ep+1} Finished | Total Reward: {total_reward:.2f} | Steps: {step}")
                    time.sleep(1.0)
    finally:
        env.close()
        print("\nDebug session complete.")

if __name__ == "__main__":
    main()

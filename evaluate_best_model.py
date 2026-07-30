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
from src.world import World

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
    parser = argparse.ArgumentParser(description="Evaluate the best trained CARLA RL model")
    parser.add_argument("--port", "-p", type=int, default=config.SIM_PORT, help="CARLA server port")
    parser.add_argument("--name", "-n", type=str, default="blackwell_fast_v1", help="Name of the training run")
    parser.add_argument("--episodes", "-ep", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--scenario","-sc", type=str, default=None, help="Specific scenario to evaluate on (default: all)")
    parser.add_argument("--backbone","-b", type=str, default="efficientnet", choices=["efficientnet"], help="CNN backbone for visual extraction (efficientnet)")
    parser.add_argument("--efficient", "-eff", action="store_true", help="Force use of EfficientNet backbone (deprecated, use --backbone)")
    parser.add_argument("--camera","-c", type=str, default=config.CAMERA_VIEW, choices=["front", "bev"], help="Camera view to use (front, bev)")
    parser.add_argument("--weather","-w", type=str, default="Clear Noon", help="Force a specific weather condition (e.g. 'Hard Rain Noon')")
    parser.add_argument("--lights", action="store_true", help="Force vehicle headlights ON")
    args = parser.parse_args()

    config.CAMERA_VIEW = args.camera

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
        scenarios = [args.scenario] if args.scenario else []
        env = gym.make('carla-rl-gym-v0', 
                       port=args.port, 
                       time_limit=120, 
                       initialize_server=False, 
                       synchronous_mode=True, 
                       show_sensor_data=False, 
                       spawn_cones=True, 
                       scenarios=scenarios,
                       camera=config.CAMERA_VIEW,
                       weather_override=args.weather,
                       force_lights=args.lights,
                       verbose=False)
        return env

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)

    # 4. Load the PPO model
    # Detect architecture based on argument or name
    if args.backbone == "efficientnet" or args.efficient or "efficient" in args.name.lower() or "5m" in args.name.lower():
        print("Architecture detected/forced: EfficientNet-B0")
        extractor_class = CustomExtractor_PPO_EfficientNet
    else:
        print("WARNING: Defaulting to EfficientNet as other architectures were removed.")
        extractor_class = CustomExtractor_PPO_EfficientNet
        
    policy_kwargs = dict(features_extractor_class=extractor_class)
    model = PPO.load(model_path, env=env, custom_objects={"policy_kwargs": policy_kwargs})

    print(f"Starting evaluation for {args.episodes} episodes...")

    # 5. Evaluation Loop
    eval_stats = []
    try:
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            termination_reason = "Unknown"
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                
                # SB3 automatic reset: when dones[0] is True, the info dict contains the terminal info
                if dones[0]:
                    termination_reason = infos[0].get('termination_reason', 'Success/Timeout')
                    completion_pct = infos[0].get('completion_pct', 0)
                    avg_speed = infos[0].get('avg_speed', 0)
                    steer_jerk = infos[0].get('steer_jerk', 0)
                    throttle_jerk = infos[0].get('throttle_jerk', 0)
                
                total_reward += rewards[0]
                steps += 1
                done = dones[0]

                if steps % 5 == 0:
                    print(f"  [Eval] Episode {episode} | Score: {total_reward:.1f}", end="\r")

            # Clear live line and print episode summary
            print(f"                                                                ", end="\r")
            print(f"  [FINISHED] Episode: {episode} | Steps: {steps} | Total Reward: {total_reward:.1f} | Reason: {termination_reason}")
            
            eval_stats.append({
                'episode': episode,
                'steps': steps,
                'reward': total_reward,
                'reason': termination_reason,
                'completion': completion_pct,
                'speed': avg_speed,
                'steer_jerk': steer_jerk,
                'throttle_jerk': throttle_jerk
            })
            time.sleep(0.5)

        # 6. Print Summary Table
        print("\n" + "="*105)
        print(f"{'EVALUATION SUMMARY':^104}|")
        print("="*105)
        header = f"{'Ep':^4} | {'Steps':^5} | {'Reward':^8} | {'Comp %':^7} | {'Speed':^7} | {'S-Jerk':^7} | {'T-Jerk':^7} | {'Termination Reason':^23}|"
        print(header)
        print("-" * 104 + "|")
        
        for stat in eval_stats:
            print(f"{stat['episode']:^4} | {stat['steps']:^5} | {stat['reward']:^8.1f} | {stat['completion']:^7.1f}% | {stat['speed']:^7.1f} | {stat['steer_jerk']:^7.2f} | {stat['throttle_jerk']:^7.2f} | {stat['reason']:^23}|")
        
        print("-" * 104 + "|")
        mean_steps = np.mean([s['steps'] for s in eval_stats])
        mean_reward = np.mean([s['reward'] for s in eval_stats])
        mean_comp = np.mean([s['completion'] for s in eval_stats])
        mean_speed = np.mean([s['speed'] for s in eval_stats])
        mean_s_jerk = np.mean([s['steer_jerk'] for s in eval_stats])
        mean_t_jerk = np.mean([s['throttle_jerk'] for s in eval_stats])
        
        success_count = sum(1 for s in eval_stats if "Reached Target Destination" in s['reason'] or "SUCCESS" in s['reason'])
        total_episodes = len(eval_stats)
        success_rate = (success_count / total_episodes) * 100 if total_episodes > 0 else 0
        
        footer = f"{'AVG':^4} | {mean_steps:^5.1f} | {mean_reward:^8.1f} | {mean_comp:^7.1f}% | {mean_speed:^7.1f} | {mean_s_jerk:^7.2f} | {mean_t_jerk:^7.2f} | Success: {success_count}/{total_episodes} ({success_rate:.0f}%) |"
        print(footer)
        print("="*105 + "\n")

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
            
            # Use batch destruction to avoid race conditions with other environments
            batch = [carla.command.DestroyActor(x) for x in to_destroy]
            client.apply_batch(batch)
            
            print("Deep cleanup: CARLA world is now clean.")
        except Exception as e:
            print(f"Notice: Secondary deep cleanup encountered an issue: {e}")
            
        print("Evaluation complete.")

if __name__ == "__main__":
    main()

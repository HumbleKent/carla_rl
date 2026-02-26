"""
Test script to verify traffic cone spawning in Town05.

This script spawns the environment once to trigger cone loading,
then keeps it running for 30 seconds to allow visual inspection.

Usage:
    1. Start CARLA server: CarlaUE4.exe
    2. Activate environment: conda activate carla_rl
    3. Run: python test_cone_spawning.py
"""

from env.environment import CarlaEnv
import time

def main():
    print("=" * 60)
    print("Traffic Cone Spawning Test - Town05 Work Zones")
    print("=" * 60)
    print()
    print("Creating environment...")
    
    # Create environment with verbose output
    env = CarlaEnv(
        time_limit=60,
        initialize_server=False,  # Assumes CARLA is already running
        random_weather=False,
        synchronous_mode=True,
        show_sensor_data=False,
        has_traffic=False,
        verbose=True
    )
    
    print()
    print("Forcing map to Town05 (cones are positioned for Town05)...")
    # Access the world object and set map to Town05
    env._CarlaEnv__world.set_active_map('Town05')
    time.sleep(2)  # Give time for map to load
    
    print()
    print("Resetting environment with Town05 scenario...")
    # Explicitly request a Town05 scenario in reset options to prevent random Town01 loading
    obs, info = env.reset(options={'scenario_name': 'Town05-ClearNoon-WorkZone-0'})
    
    print()
    print(f"Scenario loaded: {info['scenario_name']}")
    print(f"Total active cones: {env._CarlaEnv__world.get_cone_count()}")
    
    print()
    print("=" * 60)
    print("Cones are now visible in CARLA!")
    print("Inspect the following work zones in Town05:")
    print("  1. Lane Closure (Progressive) - around (-50, 140)")
    print("  2. Pothole Repair Zone - single cone at (25, -15)")
    print("  3. Incomplete Barrier (Zigzag) - around (60-90, 42-45)")
    print("  4. Sudden Lane Diversion - around (-80 to -60, -25 to -35)")
    print("  5. Narrow Passage Gate - at (10, 70-74)")
    print("  6. Merge Zone (Converging) - around (-100 to -80, 85-90)")
    print()
    print("Environment will stay active for 30 seconds...")
    print("=" * 60)
    
    # Keep environment alive for inspection
    time.sleep(30)
    
    print()
    print("Closing environment and destroying cones...")
    env.close()
    
    print()
    print("Test complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()

"""
Simple cone spawning test - No ego vehicle, just cones!

This directly spawns traffic cones in Town05 without using the environment system.

Usage:
    1. Start CARLA server: CarlaUE4.exe
    2. Activate environment: conda activate carla_rl
    3. Run: python test_cone_spawning_simple.py
"""

import carla
import json
import time

def main():
    print("=" * 60)
    print("Simple Traffic Cone Spawning Test - Town05")
    print("=" * 60)
    print()
    
    # Connect to CARLA
    print("Connecting to CARLA server...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Load Town05
    print("Loading Town05...")
    client.load_world('Town05')
    time.sleep(3)  # Wait for map to load
    
    world = client.get_world()
    print("Town05 loaded!")
    print()
    
    # Load cone configuration
    print("Loading cone configuration from town05_workzone_cones.json...")
    with open('env/town05_workzone_cones.json', 'r') as f:
        cone_config = json.load(f)
    
    # Get cone blueprint
    blueprint_library = world.get_blueprint_library()
    cone_bp = blueprint_library.find('static.prop.trafficcone01')
    
    # Spawn all cones
    spawned_cones = []
    total_cones = 0
    
    print("Spawning traffic cones...")
    print()
    for scenario_name, scenario_data in cone_config.items():
        if 'cones' not in scenario_data:
            continue
            
        print(f"  {scenario_name}:")
        print(f"    {scenario_data['description']}")
        
        for cone_pos in scenario_data['cones']:
            try:
                location = carla.Location(x=cone_pos['x'], y=cone_pos['y'], z=cone_pos['z'])
                rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
                transform = carla.Transform(location, rotation)
                
                cone = world.spawn_actor(cone_bp, transform)
                spawned_cones.append(cone)
                total_cones += 1
            except RuntimeError as e:
                print(f"    Failed to spawn cone at ({cone_pos['x']}, {cone_pos['y']}): {e}")
        
        print(f"    Spawned: {len(scenario_data['cones'])} cones")
        print()
    
    print("=" * 60)
    print(f"Total cones spawned: {total_cones}")
    print()
    print("Cones are now visible in CARLA!")
    print("Check the spectator view at these locations:")
    print("  1. Lane Closure: (-50, 140) to (-5, 144)")
    print("  2. Pothole Repair: (25, -15)")
    print("  3. Zigzag Barrier: (60-90, 42-45)")
    print("  4. Lane Diversion: (-80 to -60, -25 to -35)")
    print("  5. Narrow Gate: (10, 70-74)")
    print("  6. Merge Zone: (-100 to -80, 85-90)")
    print()
    print("Environment will stay active for 60 seconds...")
    print("Press Ctrl+C to exit early")
    print("=" * 60)
    
    # Keep cones visible
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    # Cleanup
    print()
    print("Destroying cones...")
    for cone in spawned_cones:
        try:
            cone.destroy()
        except:
            pass
    
    print(f"Destroyed {len(spawned_cones)} cones")
    print()
    print("Test complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()

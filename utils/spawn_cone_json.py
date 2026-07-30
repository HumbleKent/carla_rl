"""
spawn_cone_json.py
------------------
Standalone verification tool that loads cone positions from `env/cone_spawn.json`
and physically spawns them as `static.prop.constructioncone` actors in a running
CARLA instance, then holds them in place until the user presses Ctrl+C.

Useful for:
  - Visually confirming the cone layout is correct before starting a training run
  - Testing that the JSON positions align with the intended road geometry in Town05
  - Verifying cone spawning under a specific weather condition

The spectator camera is automatically positioned above the first cone after spawning
so you can immediately see the result in the CARLA window.

Usage:
    python utils/spawn_cone_json.py [--port 2000] [--weather ClearNoon]

Run from the project root directory with a CARLA server already running.
Press Ctrl+C to destroy all spawned cones and exit cleanly.
"""

import os
import sys
import glob

# Add Carla egg to path
try:
    sys.path.append(glob.glob('C:/Users/User/Documents/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    sys.path.append('C:/Users/User/Documents/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.15-py3.7-win-amd64.egg')

import carla
import time
import json
import argparse

def test_cone_spawn(port=2000, weather_name=None):
    """Test that cones are loaded from JSON and spawned correctly"""
    
    try:
        # 1. Connect to CARLA
        print(f"Connecting to CARLA on port {port}...")
        client = carla.Client('127.0.0.1', port)
        client.set_timeout(15.0)
        
        # 2. Get world and setup weather
        print("Loading Town05...")
        world = client.load_world('Town05')

        if weather_name:
            print(f"Setting weather to {weather_name}...")
            # Try to find the weather preset in carla.WeatherParameters
            preset = getattr(carla.WeatherParameters, weather_name, None)
            if preset:
                world.set_weather(preset)
            else:
                # Handle names with spaces by removing them (e.g., 'Clear Noon' -> 'ClearNoon')
                clean_name = weather_name.replace(" ", "")
                preset = getattr(carla.WeatherParameters, clean_name, None)
                if preset:
                    world.set_weather(preset)
                else:
                    print(f"Warning: Weather preset '{weather_name}' not found. Using default.")

        cone_layout_path = 'env/cone_spawn.json'
        print(f"Loading cone layout from {cone_layout_path}...")
        
        if not os.path.exists(cone_layout_path):
            print(f"ERROR: {cone_layout_path} not found!")
            return
        
        with open(cone_layout_path, 'r') as f:
            cone_transforms = json.load(f)
        
        print(f"Loaded {len(cone_transforms)} cone positions from JSON")
        
        # 4. Get blueprint library
        blueprint_library = world.get_blueprint_library()
        cone_bp = blueprint_library.filter('static.prop.constructioncone')[0]
        
        # 5. Spawn cones from JSON
        spawned_cones = []
        failed = 0
        
        print("Spawning cones...")
        for i, cone_data in enumerate(cone_transforms):
            location = carla.Location(
                x=cone_data['x'],
                y=cone_data['y'],
                z=0.0
            )
            rotation = carla.Rotation(
                pitch=0.0,
                yaw=0.0,
                roll=0.0
            )
            transform = carla.Transform(location, rotation)
            
            cone = world.try_spawn_actor(cone_bp, transform)
            if cone:
                spawned_cones.append(cone)
                if i < 5:  # Print first 5 positions
                    print(f"  Cone {i+1}: spawned at ({cone_data['x']:.2f}, {cone_data['y']:.2f}, {0.0})")
            else:
                failed += 1
        
        print(f"\n✓ Successfully spawned {len(spawned_cones)} cones")
        if failed > 0:
            print(f"✗ Failed to spawn {failed} cones")
        
        # 6. Position spectator to view the cones
        spectator = world.get_spectator()
        if spawned_cones:
            # Position above the first cone
            first_cone_loc = spawned_cones[0].get_location()
            spectator_location = carla.Location(
                x=first_cone_loc.x,
                y=first_cone_loc.y,
                z=first_cone_loc.z + 50
            )
            spectator.set_transform(carla.Transform(
                spectator_location,
                carla.Rotation(pitch=-90)
            ))
            print(f"\n✓ Spectator positioned above first cone at ({first_cone_loc.x:.2f}, {first_cone_loc.y:.2f})")

        # 7. Keep cones visible for inspection
        print("\nCones are now visible in CARLA. Press Ctrl+C to cleanup and exit...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nCleaning up...")
        
        # 8. Cleanup
        print("Destroying spawned cones...")
        for cone in spawned_cones:
            if cone.is_alive:
                cone.destroy()
        
        print("✓ Test completed successfully!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spawn Cones from JSON in CARLA")
    parser.add_argument("--port", "-p", type=int, default=2000, help="CARLA port")
    parser.add_argument("--weather", "-w", type=str, default=None, help="Weather preset (e.g. ClearNoon, HardRainNoon)")
    args = parser.parse_args()
    
    test_cone_spawn(port=args.port, weather_name=args.weather)

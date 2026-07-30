"""
visualize_all_scenarios.py
--------------------------
Interactive CARLA spectator tool that spawns the full cone layout from
`env/cone_spawn.json` and lets the user browse through each named scenario
by pressing the arrow keys (or entering a number if pynput is unavailable).

For each scenario the spectator camera is automatically repositioned to an
overhead / angled bird's-eye view centred on that scenario's cone cluster,
making it easy to visually audit every situation in the layout without
manually flying the camera around.

Scenario groupings are parsed from `_scenario` metadata tags embedded in the
cone layout JSON. Each group's centre, cone count, and a predefined camera
angle are used to compose the view.

Dependencies:
  - pynput  (optional, for real-time arrow-key navigation)
    Install with: pip install pynput
    Without it the script falls back to typed number input.

Usage:
    python utils/visualize_all_scenarios.py

Run from the project root directory with a CARLA server already running on port 2000.
Press Ctrl+C or ESC to exit and destroy all spawned cones.
"""

import os
import sys
import json
import time
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
import numpy as np

try:
    from pynput import keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

def parse_scenarios_from_json(cone_transforms):
    """Group cones into scenarios based on the '_scenario' tag in the data."""
    scenario_groups = []
    current_group = None
    
    for cone in cone_transforms:
        # Check if this cone marks the start of a new scenario
        if "_scenario" in cone:
            if current_group:
                scenario_groups.append(current_group)
            
            current_group = {
                "name": cone["_scenario"],
                "cones": []
            }
        
        # If we have an active group, add the cone coordinates
        if current_group:
            current_group["cones"].append(cone)
            
    # Add the last group
    if current_group:
        scenario_groups.append(current_group)
    
    # Extract names, centers (average x,y), and counts
    names = []
    centers = []
    counts = []
    
    for group in scenario_groups:
        names.append(group["name"])
        cones = group["cones"]
        count = len(cones)
        counts.append(count)
        
        if count > 0:
            avg_x = sum(c['x'] for c in cones) / count
            avg_y = sum(c['y'] for c in cones) / count
            centers.append((avg_x, avg_y))
        else:
            centers.append((0.0, 0.0))
            
    return names, centers, counts

class ScenarioViewer:
    def __init__(self):
        self.current_idx = 0
        self.should_update = True
        self.running = True
        
    def on_press(self, key):
        """Handle keyboard press events"""
        try:
            if key == keyboard.Key.right or key == keyboard.Key.down:
                self.current_idx += 1
                self.should_update = True
            elif key == keyboard.Key.left or key == keyboard.Key.up:
                self.current_idx -= 1
                self.should_update = True
        except AttributeError:
            pass

def spectator_manual_control():
    spawned_actors = []
    
    try:
        # Connect to CARLA
        print("=" * 60)
        print("CARLA Scenario Spectator - Manual Control")
        print("=" * 60)
        print("\nConnecting to CARLA...")
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)
        
        # Load world
        print("Loading Town05...")
        world = client.load_world('Town05')
        blueprint_library = world.get_blueprint_library()
        spectator = world.get_spectator()
        
        # Load cone layout
        cone_layout_path = 'env/cone_spawn.json'
        if not os.path.exists(cone_layout_path):
            print(f"ERROR: {cone_layout_path} not found!")
            return
        
        with open(cone_layout_path, 'r') as f:
            cone_transforms = json.load(f)
        
        print(f"✓ Loaded {len(cone_transforms)} cone positions\n")
        
        # Parse scenarios from tags
        print("Parsing scenarios from metadata...")
        scenario_names, scenario_centers, scenario_cones = parse_scenarios_from_json(cone_transforms)
        print(f"✓ Found {len(scenario_centers)} scenarios\n")
        
        print("Scenarios detected:")
        for i, (name, center, num_cones) in enumerate(zip(scenario_names, scenario_centers, scenario_cones)):
            print(f"  {i+1}. {name}")
            print(f"     Location: ({center[0]:.1f}, {center[1]:.1f}) | {num_cones} cones")
        print()
        
        # Spawn cones
        print("Spawning cones...")
        cone_bp = blueprint_library.filter('static.prop.constructioncone')[0]
        
        for cone_data in cone_transforms:
            transform = carla.Transform(
                carla.Location(x=cone_data['x'], y=cone_data['y'], z=0.0),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )
            
            actor = world.spawn_actor(cone_bp, transform)
            spawned_actors.append(actor)
        
        print(f"✓ Successfully spawned {len(spawned_actors)} cones\n")
        
        # Define camera positions for each scenario
        camera_configs = [
            {"height": 30, "pitch": -70, "distance": 0},   # Lane Closure - top down
            {"height": 30, "pitch": -60, "distance": 0},   # Pothole - angled
            {"height": 30, "pitch": -50, "distance": 0},   # Chicane - lower angle
            {"height": 30, "pitch": -70, "distance": 0},   # Barrier - top down
            {"height": 30, "pitch": -60, "distance": 0},   # Diversion - angled
            {"height": 30, "pitch": -65, "distance": 0},   # Lane Change - medium
        ]
        
        print("=" * 60)
        print("MANUAL CONTROL MODE")
        print("=" * 60)
        print("\nKeyboard Controls:")
        print("  ← → (Left/Right Arrow) - Switch between scenarios")
        print("  ↑ ↓ (Up/Down Arrow)    - Switch between scenarios")
        print("  ESC                    - Exit and cleanup")
        print("\nView scenarios in the CARLA window\n")
        
        if not HAS_PYNPUT:
            print("WARNING: pynput not installed. Install with: pip install pynput")
            print("Falling back to number key control (press 1-6 + Enter)\n")
        
        # Create viewer instance
        viewer = ScenarioViewer()
        
        if HAS_PYNPUT:
            # Start keyboard listener in background thread
            listener = keyboard.Listener(on_press=viewer.on_press)
            listener.start()
        else:
            print("Use fallback mode: Type scenario number (1-6) and press Enter, or 'q' to quit")
        
        # Initial view
        viewer.current_idx = 0
        viewer.should_update = True
        
        print("-" * 60)
        
        try:
            while viewer.running:
                if viewer.should_update:
                    # Wrap around
                    viewer.current_idx = viewer.current_idx % len(scenario_centers)
                    
                    center = scenario_centers[viewer.current_idx]
                    config = camera_configs[viewer.current_idx] if viewer.current_idx < len(camera_configs) else camera_configs[0]
                    name = scenario_names[viewer.current_idx] if viewer.current_idx < len(scenario_names) else f"Scenario {viewer.current_idx+1}"
                    
                    # Position spectator
                    spec_transform = carla.Transform(
                        carla.Location(
                            x=center[0] + config["distance"], 
                            y=center[1], 
                            z=config["height"]
                        ),
                        carla.Rotation(pitch=config["pitch"], yaw=0, roll=0)
                    )
                    spectator.set_transform(spec_transform)
                    
                    # Display current view
                    print(f"\n[{viewer.current_idx + 1}/{len(scenario_centers)}] Viewing: {name}")
                    print(f"Location: ({center[0]:.1f}, {center[1]:.1f}) | {scenario_cones[viewer.current_idx]} cones")
                    print("Use arrow keys to switch...")
                    
                    viewer.should_update = False
                
                if not HAS_PYNPUT:
                    # Fallback: manual input
                    user_input = input("\nEnter scenario number (1-6) or 'q' to quit: ").strip()
                    if user_input.lower() == 'q':
                        break
                    try:
                        num = int(user_input)
                        if 1 <= num <= len(scenario_centers):
                            viewer.current_idx = num - 1
                            viewer.should_update = True
                    except ValueError:
                        pass
                else:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("STOPPED BY USER (Ctrl+C)")
            print("=" * 60)
        
        if HAS_PYNPUT and listener.is_alive():
            listener.stop()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        print("Destroying spawned cones...")
        for actor in spawned_actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except:
                pass
        print("✓ Cleanup completed!")

if __name__ == "__main__":
    spectator_manual_control()

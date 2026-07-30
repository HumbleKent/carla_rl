"""
draw_waypoints.py
-----------------
Developer visualisation tool that connects to a running CARLA server and continuously
draws the planned route (waypoints, start marker, and target marker) for every scenario
defined in `env/vehicle_spawn.json` directly onto the CARLA world using debug primitives.

Useful for:
  - Verifying that the AdvancedRoutePlanner produces correct, collision-free paths
  - Visually inspecting start/target positions and cone avoidance corridors
  - Checking waypoint spacing and cone safe-distance thresholds

Usage:
    python utils/draw_waypoints.py [--port 2000] [--scenario "Lane Change to Left"]
                                   [--no-shadows] [--cone-radius 1.5]

Run from the project root directory with a CARLA server already running.
"""

import carla
import json
import time
import argparse
import sys
import os

# Add the project root to sys.path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.route_planner import AdvancedRoutePlanner
from src.debug_manager import DebugManager
import re

def get_cone_safe_distance():
    """Extracts the safe_distance default value from reward_pothole.py via regex."""
    reward_file = 'env/reward_pothole.py'
    if not os.path.exists(reward_file):
        return 1.5
    try:
        with open(reward_file, 'r') as f:
            content = f.read()
            # Looks for safe_distance=X.X in the __proximity_cone_penalty definition
            match = re.search(r'def __proximity_cone_penalty\(.*safe_distance=([\d.]+)\)', content)
            if match:
                return float(match.group(1))
    except Exception:
        pass
    return 1.5

def main():
    parser = argparse.ArgumentParser(description="Standalone Waypoint Visualizer")
    parser.add_argument("--town", type=str, default="Town05", help="CARLA town name")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario name (from vehicle_spawn.json)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--no-shadows", action="store_false", dest="shadows", help="Keep shadows enabled (default: False/Shadow-Free)")
    parser.add_argument("--target-threshold", type=float, default=0.6, help="Success radius for target destination (meters)")
    parser.add_argument("--wp-threshold", type=float, default=0.4, help="Success radius for waypoints (meters)")
    parser.add_argument("--cone-radius", type=float, default=get_cone_safe_distance(), help="Safe distance for cones (meters) [Auto-detected from reward_pothole.py]")
    parser.set_defaults(shadows=False) # Default to shadow-free
    args = parser.parse_args()

    # 1. Connect to CARLA
    print(f"Connecting to CARLA on port {args.port}...")
    try:
        client = carla.Client('127.0.0.1', args.port)
        client.set_timeout(10.0)
        
        # Check current world to avoid unnecessary load
        current_world = client.get_world()
        if not current_world.get_map().name.endswith(args.town):
            print(f"Loading {args.town}...")
            world = client.load_world(args.town)
        else:
            world = current_world
            print(f"Using current world: {args.town}")
            
    except Exception as e:
        print(f"ERROR: Could not connect to CARLA: {e}")
        return
    
    # 1.5 Apply Shadow-Free Lighting if requested
    if not args.shadows:
        try:
            weather = world.get_weather()
            weather.sun_altitude_angle = 90.0
            weather.cloudiness = 100.0
            weather.sun_intensity = 0.0
            weather.sky_light_intensity = 50.0
            world.set_weather(weather)
            print("✓ Lighting adjusted to Shadow-Free mode for clear visualization.")
        except Exception as e:
            print(f"Warning: Could not adjust lighting: {e}")
    
    # 2. Load Scenarios
    spawn_json = 'env/vehicle_spawn.json'
    if not os.path.exists(spawn_json):
        print(f"ERROR: {spawn_json} not found!")
        return
        
    with open(spawn_json, 'r') as f:
        scenarios = json.load(f)
    
    # Pick scenario
    if args.scenario:
        if args.scenario in scenarios:
            s_name = args.scenario
        else:
            print(f"Warning: Scenario '{args.scenario}' not found. Using first available.")
            s_name = list(scenarios.keys())[0]
    else:
        s_name = list(scenarios.keys())[0]
        
    s_data = scenarios[s_name]
    print(f"Viewing waypoints for: {s_name}")

    # 3. Initialize Planner and Debugger
    planner = AdvancedRoutePlanner(world, sampling_res=2.0)
    
    # Load cones if they exist
    cone_json = 'env/cone_spawn.json'
    if os.path.exists(cone_json):
        with open(cone_json, 'r') as f:
            cones = json.load(f)
            planner.set_cones(cones)
            print(f"Loaded {len(cones)} cones for avoidance logic.")
    
    dm = DebugManager(debug_list=["waypoints", "target"])

    # 5. Continuous Drawing Loop
    print(f"\nDrawing waypoints for ALL {len(scenarios)} scenarios... (Press Ctrl+C to stop)")
    print("You can now move your CARLA window freely to see the cyan breadcrumbs.")
    
    try:
        while True:
            for s_name, s_data in scenarios.items():
                # Define Start and End for this scenario
                start_pos = s_data['initial_position']
                target_pos = s_data['target_position']
                
                start_loc = carla.Location(x=start_pos['x'], y=start_pos['y'], z=start_pos['z'])
                target_loc = carla.Location(x=target_pos['x'], y=target_pos['y'], z=target_pos['z'])

                # Generate the path
                waypoints = planner.plan_route(start_loc, target_loc, avoid_cones=True)
                
                # 1. Draw Start (Blue) and Goal (Green) posts
                dm.draw_start(world, [start_loc.x, start_loc.y, start_loc.z], life_time=1.1)
                dm.draw_target(world, [target_loc.x, target_loc.y, target_loc.z], life_time=1.1, radius=args.target_threshold)

                # 2. Draw path waypoints (Cyan dots) - Circles removed for cleaner view
                dm.draw_waypoints(world, waypoints, life_time=1.1, threshold=0.0)

                # (Cone debug drawing removed: relying on physical actors in CARLA)
            
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()

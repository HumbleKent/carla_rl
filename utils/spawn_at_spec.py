"""
spawn_at_spec.py
----------------
Interactive development tool for manually placing vehicles and traffic cones in CARLA
at the current spectator camera position or at user-supplied coordinates.

Spawned actor IDs are persisted inside the script file itself (in VEHICLE_IDS / CONE_IDS
lists) so that a subsequent `--destroy` or `--destroy-all` call can clean them up even
after the terminal is closed.

Modes (pass one flag at a time):
  --spawn      (-s)  Spawn a Tesla Model 3 at the current spectator transform
  --destroy    (-d)  Destroy the most recently spawned vehicle (LIFO)
  --destroy-all(-a)  Destroy all tracked vehicles and cones
  --monitor    (-m)  Print the spectator's live position/rotation to the terminal
  --cone       (-c)  Spawn a single traffic cone at the spectator's XY position (z=0)
  --interactive(-i)  Enter a prompt loop to spawn cones by typing X/Y coordinates

Usage:
    python utils/spawn_at_spec.py --spawn
    python utils/spawn_at_spec.py --monitor
    python utils/spawn_at_spec.py --interactive
    python utils/spawn_at_spec.py --destroy-all

Run from the project root directory with a CARLA server already running on port 2000.
"""

import os
import sys
import argparse
import time
import glob

# THIS LIST IS AUTO-UPDATED BY THE SCRIPT
VEHICLE_IDS = []
CONE_IDS = []

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    try:
        sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass

import carla

# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.world import World
from src.vehicle import Vehicle

def update_storage(v_ids, c_ids):
    """
    Reads this script's source code, updates the VEHICLE_IDS and CONE_IDS lists,
    and writes them back to the file.
    """
    file_path = os.path.abspath(__file__)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    with open(file_path, 'w') as f:
        for line in lines:
            if line.startswith("VEHICLE_IDS ="):
                f.write(f"VEHICLE_IDS = {v_ids}\n")
            elif line.startswith("CONE_IDS ="):
                f.write(f"CONE_IDS = {c_ids}\n")
            else:
                f.write(line)

def destroy_all_actors(carla_world):
    """Destroys all vehicles and cones tracked in the script's storage lists."""
    if VEHICLE_IDS or CONE_IDS:
        print(f"Destroying {len(VEHICLE_IDS)} vehicles and {len(CONE_IDS)} cones...")
        while VEHICLE_IDS:
            vehicle_id = VEHICLE_IDS.pop()
            actor_vehicle = carla_world.get_actor(vehicle_id)
            if actor_vehicle:
                actor_vehicle.destroy()
                print(f"Destroyed vehicle with ID: {vehicle_id}")

        while CONE_IDS:
            cone_id = CONE_IDS.pop()
            actor_cone = carla_world.get_actor(cone_id)
            if actor_cone:
                actor_cone.destroy()
                print(f"Destroyed cone with ID: {cone_id}")
        
        update_storage(VEHICLE_IDS, CONE_IDS)
        print("✓ Cleanup completed!")
    else:
        print("No spawned objects to destroy.")

def interactive_cone_spawn(carla_world):
    """Function to interactively ask for X and Y coordinates to spawn cones"""
    blueprint_library = carla_world.get_blueprint_library()
    cone_bp = blueprint_library.find('static.prop.constructioncone')
    
    print("\n" + "=" * 60)
    print("INTERACTIVE CONE SPAWNER")
    print("Enter X and Y coordinates to spawn cones. Use 'q' to exit.")
    print("=" * 60)
    
    try:
        while True:
            x_input = input("Enter X coordinate (or 'q'): ").strip().lower()
            if x_input == 'q':
                break
            
            y_input = input("Enter Y coordinate (or 'q'): ").strip().lower()
            if y_input == 'q':
                break
            
            try:
                x = float(x_input)
                y = float(y_input)
                
                spawn_transform = carla.Transform(
                    carla.Location(x=x, y=y, z=0.0),
                    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
                )
                
                cone = carla_world.try_spawn_actor(cone_bp, spawn_transform)
                if cone:
                    print(f"✓ Spawned cone {cone.id} at x:{x:.2f}, y:{y:.2f} (z=0)")
                    CONE_IDS.append(cone.id)
                    update_storage(VEHICLE_IDS, CONE_IDS)
                else:
                    print(f"✗ Failed to spawn cone (Collision detected)")
            except ValueError:
                print("! Invalid input. Enter numbers only.")
    finally:
        destroy_all_actors(carla_world)
        print("\nExiting interactive mode.")

def monitor_spectator(carla_world):
    """Connect to CARLA and print the spectator's transform in real-time"""
    try:
        spectator = carla_world.get_spectator()
        
        print("\n" + "=" * 40)
        print("SPECTATOR TRANSFORM MONITOR")
        print("Move the spectator in the CARLA window")
        print("Press Ctrl+C to stop")
        print("=" * 40 + "\n")
        
        while True:
            transform = spectator.get_transform()
            loc = transform.location
            rot = transform.rotation
            print(f"Location(x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}) | Rotation(pitch={rot.pitch:.2f}, yaw={rot.yaw:.2f}, roll={rot.roll:.2f})", end='\r')
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\nERROR: {e}")


def main():
    argparser = argparse.ArgumentParser(
        description='Spawn a vehicle at the spectator location or monitor spectator transform')
    argparser.add_argument(
        '-s','--spawn',
        action='store_true',
        help='Spawn a vehicle at the current spectator position')
    argparser.add_argument(
        '-d','--destroy',
        action='store_true',
        help='Destroy the previously spawned vehicle')
    argparser.add_argument(
        '-a', '--destroy-all',
        action='store_true',
        help='Destroy all previously spawned vehicles')
    argparser.add_argument(
        '-m', '--monitor',
        action='store_true',
        help='Monitor and print spectator transform in real-time')
    argparser.add_argument(
        '-c', '--cone',
        action='store_true',
        help='Spawn a traffic cone at the current spectator position (z=0, rotation=0)')
    argparser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Spawn cones interactively by entering X and Y coordinates')

    args = argparser.parse_args()

    if not args.spawn and not args.destroy and not args.destroy_all and not args.monitor and not args.cone and not args.interactive:
        print("Please specify --spawn, --destroy, --destroy-all, --monitor, --cone, or --interactive")
        return

    # Initialize World (connects to client)
    try:
        world_wrapper = World()
        carla_world = world_wrapper.get_world()
    except Exception as e:
        print(f"Error connecting to CARLA: {e}")
        return

    if args.monitor:
        monitor_spectator(carla_world)
        return

    if args.interactive:
        interactive_cone_spawn(carla_world)
        return

    if args.destroy:
        if VEHICLE_IDS:
            vehicle_id = VEHICLE_IDS.pop() # LIFO
            actor = carla_world.get_actor(vehicle_id)
            if actor:
                actor.destroy()
                print(f"Destroyed vehicle with ID: {vehicle_id}")
            else:
                print(f"Vehicle with ID {vehicle_id} not found (maybe already destroyed?)")
            
            update_storage(VEHICLE_IDS, CONE_IDS)
        else:
            print("No previously spawned vehicle ID found in internal storage.")

    if args.destroy_all:
        destroy_all_actors(carla_world)

    if args.spawn:
        spectator = carla_world.get_spectator()
        transform = spectator.get_transform()
        
        print(f"Spectator Transform: {transform}")
        
        # Use existing Vehicle class
        vehicle = Vehicle(world_wrapper.get_world())
        
        # spawn_vehicle takes location (x,y,z) and rotation (pitch, yaw, roll)
        loc = (transform.location.x, transform.location.y, transform.location.z)
        rot = (transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll)
        
        vehicle.spawn_vehicle(location=loc, rotation=rot)
        
        spawned_vehicle = vehicle.get_vehicle()
        if spawned_vehicle:
            print(f"Spawned vehicle at: {spawned_vehicle.get_transform()}")
            VEHICLE_IDS.append(spawned_vehicle.id)
            update_storage(VEHICLE_IDS, CONE_IDS)
        else:
            print("Failed to spawn vehicle.")

    if args.cone:
        spectator = carla_world.get_spectator()
        transform = spectator.get_transform()
        
        # Override z and rotation as requested
        spawn_location = carla.Location(x=transform.location.x, y=transform.location.y, z=0.0)
        spawn_rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        spawn_transform = carla.Transform(spawn_location, spawn_rotation)
        
        blueprint_library = carla_world.get_blueprint_library()
        cone_bp = blueprint_library.find('static.prop.constructioncone')
        
        cone = carla_world.try_spawn_actor(cone_bp, spawn_transform)
        
        if cone:
            print(f"Spawned cone at: {spectator.get_transform()}")
            CONE_IDS.append(cone.id)
            update_storage(VEHICLE_IDS, CONE_IDS)
        else:
            print("Failed to spawn cone.")

if __name__ == '__main__':
    main()


import glob
import os
import sys
import argparse
import time

# THIS LIST IS AUTO-UPDATED BY THE SCRIPT
VEHICLE_IDS = []

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

def update_storage(new_ids):
    """
    Reads this script's source code, updates the VEHICLE_IDS list,
    and writes it back to the file.
    """
    file_path = os.path.abspath(__file__)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    with open(file_path, 'w') as f:
        for line in lines:
            if line.startswith("VEHICLE_IDS ="):
                f.write(f"VEHICLE_IDS = {new_ids}\n")
            else:
                f.write(line)

def main():
    argparser = argparse.ArgumentParser(
        description='Spawn a vehicle at the spectator location')
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

    args = argparser.parse_args()

    if not args.spawn and not args.destroy and not args.destroy_all:
        print("Please specify either --spawn, --destroy, or --destroy-all")
        return

    # Initialize World (connects to client)
    try:
        world_wrapper = World()
        carla_world = world_wrapper.get_world()
    except Exception as e:
        print(f"Error connecting to CARLA: {e}")
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
            
            update_storage(VEHICLE_IDS)
        else:
            print("No previously spawned vehicle ID found in internal storage.")

    if args.destroy_all:
        if VEHICLE_IDS:
            print(f"Destroying {len(VEHICLE_IDS)} vehicles...")
            while VEHICLE_IDS:
                vehicle_id = VEHICLE_IDS.pop()
                actor = carla_world.get_actor(vehicle_id)
                if actor:
                    actor.destroy()
                    print(f"Destroyed vehicle with ID: {vehicle_id}")
                else:
                    print(f"Vehicle with ID {vehicle_id} not found.")
            
            update_storage(VEHICLE_IDS)
        else:
            print("No previously spawned vehicle IDs found in internal storage.")

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
            update_storage(VEHICLE_IDS)
        else:
            print("Failed to spawn vehicle.")

if __name__ == '__main__':
    main()

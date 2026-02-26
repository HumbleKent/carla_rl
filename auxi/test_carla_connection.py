import sys
import os
import glob

# --- CARLA EGG SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
carla_dist_path = os.path.abspath(os.path.join(script_dir, '../../carla/dist'))
egg_file = glob.glob(os.path.join(carla_dist_path, 'carla-*.egg'))
if egg_file:
    sys.path.append(egg_file[0])

import carla
import sys

def test_connection(port):
    print(f"Testing connection to {port}...")
    try:
        client = carla.Client('127.0.0.1', port)
        client.set_timeout(5.0)
        world = client.get_world()
        print(f"Success! Map name: {world.get_map().name}")
    except Exception as e:
        print(f"Failed to connect to {port}: {e}")

if __name__ == "__main__":
    test_connection(2000)
    test_connection(2002)

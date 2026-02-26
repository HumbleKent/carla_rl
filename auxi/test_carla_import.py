import sys
import os
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
carla_dist_path = os.path.abspath(os.path.join(script_dir, '../../carla/dist'))
egg_files = glob.glob(os.path.join(carla_dist_path, 'carla-*.egg'))
if egg_files:
    print(f"Adding egg: {egg_files[0]}")
    sys.path.append(egg_files[0])

try:
    import carla
    print("CARLA imported successfully!")
    client = carla.Client('127.0.0.1', 2000)
    print(f"CARLA Version: {client.get_client_version()}")
except Exception as e:
    print(f"Failed to import CARLA: {e}")
    import traceback
    traceback.print_exc()

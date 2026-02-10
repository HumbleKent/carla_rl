import glob
import os
import sys

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

def test_cone_spawn():
    """Test that cones are loaded from JSON and spawned correctly"""
    
    try:
        # 1. Connect to CARLA
        print("Connecting to CARLA...")
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)
        
        # 2. Get world
        print("Loading Town05...")
        world = client.load_world('Town05')
        cone_layout_path = 'env/cone_layout.json'
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
                z=cone_data['z']
            )
            rotation = carla.Rotation(
                pitch=cone_data['pitch'],
                yaw=cone_data['yaw'],
                roll=cone_data['roll']
            )
            transform = carla.Transform(location, rotation)
            
            cone = world.try_spawn_actor(cone_bp, transform)
            if cone:
                spawned_cones.append(cone)
                if i < 5:  # Print first 5 positions
                    print(f"  Cone {i+1}: spawned at ({cone_data['x']:.2f}, {cone_data['y']:.2f}, {cone_data['z']:.2f})")
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
    test_cone_spawn()

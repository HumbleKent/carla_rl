import carla
import numpy as np
import math
import configuration as config

class ConeControl:
    def __init__(self, world):
        self.__world = world
        self.__map = None
        self.__active_cones = []
        
    def update_map(self, map):
        self.__map = map
        
    def spawn_cones_at_waypoints(self, waypoint_positions):
        for position in waypoint_positions:
            self.__spawn_single_cone(position)
    
    def spawn_single_cone(self, position):
        cone_bp = self.__world.get_blueprint_library().find('static.prop.constructioncone')
        
        carla_location = carla.Location(x=position[0], y=position[1], z=position[2])
        carla_rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        transform = carla.Transform(carla_location, carla_rotation)
        
        try:
            cone = self.__world.spawn_actor(cone_bp, transform)
            self.__active_cones.append(cone)
            if config.VERBOSE:
                print(f"Spawned traffic cone at {carla_location}")
        except RuntimeError as e:
            if config.VERBOSE:
                print(f"Failed to spawn cone at {carla_location}: {e}")
    
    def spawn_cone_in_road(self, ego_location, offset_distance=3.0, distance_ahead=20.0):
        ego_yaw = ego_location.rotation.yaw
        
        offset_rad = math.radians(ego_yaw + 90)
        
        cone_x = ego_location.x + distance_ahead * math.cos(math.radians(ego_yaw)) + offset_distance * math.cos(offset_rad)
        cone_y = ego_location.y + distance_ahead * math.sin(math.radians(ego_yaw)) + offset_distance * math.sin(offset_rad)
        cone_z = ego_location.z
        
        self.spawn_single_cone([cone_x, cone_y, cone_z])
    
    def spawn_cone_barrier(self, ego_location, num_cones=3, spacing=5.0, distance_ahead=20.0, offset=0.0):
        ego_yaw = ego_location.rotation.yaw
        
        for i in range(num_cones):
            cone_distance = distance_ahead + (i * spacing)
            
            offset_rad = math.radians(ego_yaw + 90)
            lateral_offset = offset + (i % 2) * 0.5 if num_cones > 3 else offset
            
            cone_x = ego_location.x + cone_distance * math.cos(math.radians(ego_yaw)) + lateral_offset * math.cos(offset_rad)
            cone_y = ego_location.y + cone_distance * math.sin(math.radians(ego_yaw)) + lateral_offset * math.sin(offset_rad)
            cone_z = ego_location.z
            
            self.spawn_single_cone([cone_x, cone_y, cone_z])
    
    def spawn_cone_zigzag(self, ego_location, num_cones=3, spacing=8.0, distance_ahead=20.0, lateral_spacing=3.0):
        ego_yaw = ego_location.rotation.yaw
        
        for i in range(num_cones):
            cone_distance = distance_ahead + (i * spacing)
            
            lateral_offset = lateral_spacing if i % 2 == 0 else -lateral_spacing
            
            offset_rad = math.radians(ego_yaw + 90)
            
            cone_x = ego_location.x + cone_distance * math.cos(math.radians(ego_yaw)) + lateral_offset * math.cos(offset_rad)
            cone_y = ego_location.y + cone_distance * math.sin(math.radians(ego_yaw)) + lateral_offset * math.sin(offset_rad)
            cone_z = ego_location.z
            
            self.spawn_single_cone([cone_x, cone_y, cone_z])
    
    def spawn_cone_narrow_gap(self, ego_location, distance_ahead=20.0, gap_width=2.5):
        ego_yaw = ego_location.rotation.yaw
        offset_rad = math.radians(ego_yaw + 90)
        
        left_offset = gap_width / 2.0
        right_offset = -gap_width / 2.0
        
        left_cone_x = ego_location.x + distance_ahead * math.cos(math.radians(ego_yaw)) + left_offset * math.cos(offset_rad)
        left_cone_y = ego_location.y + distance_ahead * math.sin(math.radians(ego_yaw)) + left_offset * math.sin(offset_rad)
        left_cone_z = ego_location.z
        
        right_cone_x = ego_location.x + distance_ahead * math.cos(math.radians(ego_yaw)) + right_offset * math.cos(offset_rad)
        right_cone_y = ego_location.y + distance_ahead * math.sin(math.radians(ego_yaw)) + right_offset * math.sin(offset_rad)
        right_cone_z = ego_location.z
        
        self.spawn_single_cone([left_cone_x, left_cone_y, left_cone_z])
        self.spawn_single_cone([right_cone_x, right_cone_y, right_cone_z])
    
    def destroy_cones(self):
        for cone in self.__active_cones:
            try:
                cone.destroy()
            except RuntimeError as e:
                if config.VERBOSE:
                    print(f"Error destroying cone: {e}")
        
        self.__active_cones = []
        if config.VERBOSE:
            print("Destroyed all traffic cones")
    
    def get_active_cones(self):
        return self.__active_cones
    
    def get_cone_count(self):
        return len(self.__active_cones)

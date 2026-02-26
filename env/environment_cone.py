import numpy as np
import math
import carla
import gymnasium as gym
from env.environment import CarlaEnv
from env.pre_processing import PreProcessing
import env.observation_action_space_cone as cone_obs_space

class ConePreProcessing(PreProcessing):
    def preprocess_data(self, observation_data, cone_data, last_action=None):
        # observation_data keys: rgb_data, position, target_position, next_waypoint_position, velocity, angular_velocity, rotation
        
        target_distance = self.distance(observation_data['position'], observation_data['target_position'])
        next_waypoint_distance = self.distance(observation_data['position'], observation_data['next_waypoint_position'])
        
        # 1. Yaw Error (Heading Difference)
        # Vector to next waypoint
        wp_dir = observation_data['next_waypoint_position'] - observation_data['position']
        target_yaw = math.atan2(wp_dir[1], wp_dir[0])
        # Vehicle yaw (comes in degrees, convert to radians)
        vehicle_yaw = math.radians(observation_data['rotation'][1]) # rotation[1] is yaw
        
        yaw_error = target_yaw - vehicle_yaw
        # Normalize to [-pi, pi]
        while yaw_error > math.pi: yaw_error -= 2 * math.pi
        while yaw_error < -math.pi: yaw_error += 2 * math.pi
        
        # 2. Local Velocities (Forward and Lateral)
        # We project the world velocity onto the vehicle's forward/right vectors
        curr_yaw = vehicle_yaw
        vx_world = observation_data['velocity'][0]
        vy_world = observation_data['velocity'][1]
        
        # rotation matrix: [[cos, -sin], [sin, cos]]
        # to get local: world_v dot unit_vectors
        forward_v = vx_world * math.cos(curr_yaw) + vy_world * math.sin(curr_yaw)
        lateral_v = -vx_world * math.sin(curr_yaw) + vy_world * math.cos(curr_yaw)
        
        # 3. Angular Velocity Z
        ang_v_z = observation_data['angular_velocity'][2] # Z is vertical axis in CARLA
        
        # Action history (2 features)
        if last_action is None:
            last_action = np.array([0.0, 0.0], dtype=np.float32)
        
        # Flatten cone data
        # Cone data expected to be list of dicts: {'rel_x', 'rel_y', 'dist'}
        cone_features = []
        for cone in cone_data:
            cone_features.extend([cone['rel_x'], cone['rel_y'], cone['dist']])
            
        # Ensure we have fixed size. If fewer cones, pad with huge distance or 0
        num_expected_cones = 5
        current_num = len(cone_data)
        if current_num < num_expected_cones:
            for _ in range(num_expected_cones - current_num):
                # Padding: 0 rel_x, 0 rel_y, 1000 dist (far away)
                cone_features.extend([0.0, 0.0, 1000.0])
        elif current_num > num_expected_cones:
            # Should be handled before by sorting, but just in case
            cone_features = cone_features[:num_expected_cones*3]
            
        rest_list = [
            target_distance, 
            next_waypoint_distance, 
            yaw_error, 
            ang_v_z, 
            forward_v, 
            lateral_v
        ] + list(last_action) + cone_features
        
        rest_vector = np.array(rest_list, dtype=np.float32)
        
        # Original keys were 'position', 'target_position' etc. but mapped to 'rest' in PreProcessing.
        # We replace the 'preprocess_data' signature from base class, which takes only 'observation_data'.
        # However, calling it with extra arg might break if called generically.
        # Better to put cone_data inside observation_data before calling.
        
        neo_observation_data = {
            'rgb_data': observation_data['rgb_data'],
            'rest': rest_vector
        }
        return neo_observation_data

class ConeCarlaEnv(CarlaEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override observation space
        self.observation_space = cone_obs_space.observation_space
        self.cone_pre_processing = ConePreProcessing()
        self.num_cones_to_track = 5
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        return super().reset(seed=seed, options=options)

    def _update_observation(self):        
        # Access private members using name mangling
        vehicle = self._CarlaEnv__vehicle
        world = self._CarlaEnv__world
        active_scenario_dict = self._CarlaEnv__active_scenario_dict
        waypoints = self._CarlaEnv__waypoints
        situations_map = self._CarlaEnv__situations_map
        
        observation_space = vehicle.get_observation_data()
        rgb_image = observation_space['rgb_data']
        vehicle_loc = vehicle.get_location()
        current_position = np.array([vehicle_loc.x, vehicle_loc.y, vehicle_loc.z])
        target_position = np.array([active_scenario_dict['target_position']['x'], active_scenario_dict['target_position']['y'], active_scenario_dict['target_position']['z']])
        
        # Handle waypoints being empty potentially
        if waypoints and len(waypoints) > 0:
            next_waypoint_position = np.array([waypoints[0][0], waypoints[0][1], waypoints[0][2]])
        else:
            next_waypoint_position = target_position # Fallback

        # Kinematics
        velocity = vehicle.get_vehicle().get_velocity()
        ang_vel = vehicle.get_vehicle().get_angular_velocity()
        transform = vehicle.get_vehicle().get_transform()
        
        observation = {
            'rgb_data': np.uint8(rgb_image),
            'position': np.float32(current_position),
            'target_position': np.float32(target_position),
            'next_waypoint_position': np.float32(next_waypoint_position),
            'velocity': np.array([velocity.x, velocity.y, velocity.z], dtype=np.float32),
            'angular_velocity': np.array([ang_vel.x, ang_vel.y, ang_vel.z], dtype=np.float32),
            'rotation': np.array([transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll], dtype=np.float32),
            'situation': situation
        }
        
        # --- Cone Logic ---
        active_cones = world.get_active_cones()
        cone_data = []
        
        if active_cones:
            # Calculate distance to all cones
            cones_with_dist = []
            for cone in active_cones:
                if not cone.is_alive:
                    continue
                cone_loc = cone.get_location()
                dist = vehicle_loc.distance(cone_loc)
                cones_with_dist.append((cone, dist))
              
            # Sort by distance
            cones_with_dist.sort(key=lambda x: x[1])
            
            # Take top N
            closest_cones = cones_with_dist[:self.num_cones_to_track]
            
            for cone, dist in closest_cones:
                cone_loc = cone.get_location()
                # Global relative
                rel_x = cone_loc.x - vehicle_loc.x
                rel_y = cone_loc.y - vehicle_loc.y
                
                cone_data.append({
                    'rel_x': rel_x,
                    'rel_y': rel_y,
                    'dist': dist
                })
        
        # Use our custom pre-processing
        self._CarlaEnv__observation = self.cone_pre_processing.preprocess_data(observation, cone_data, last_action=self.last_action)
        
        # Aux variables for reward function
        self._CarlaEnv__reward_target_pos = target_position
        self._CarlaEnv__reward_current_pos = current_position
        self._CarlaEnv__reward_next_waypoint_pos = next_waypoint_position
        
        # Calculate speed for reward function (Km/h)
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self._CarlaEnv__reward_speed = speed_kmh
        self.current_cone_data = cone_data # Store for step()

    def step(self, action):
        # Use parent logic for control and observation update
        obs, _, terminated, truncated, info = super().step(action)
        
        # recalculate reward using cone data
        reward = self._CarlaEnv__reward_func.calculate_reward(
            self._CarlaEnv__vehicle, 
            self._CarlaEnv__reward_current_pos, 
            self._CarlaEnv__reward_target_pos, 
            self._CarlaEnv__reward_next_waypoint_pos, 
            self._CarlaEnv__reward_speed,
            cone_data=self.current_cone_data
        )
        
        # Store action for next step's observation
        self.last_action = np.array(action, dtype=np.float32)
        
        return obs, reward, terminated, truncated, info

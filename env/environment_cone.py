import numpy as np
import math
import carla
import gymnasium as gym
from env.environment import CarlaEnv
from env.pre_processing import PreProcessing
import env.observation_action_space_cone as cone_obs_space

class ConePreProcessing(PreProcessing):
    def preprocess_data(self, observation_data, cone_data):
        # reuse standard logic for basic parts
        # observation_data keys: rgb_data, position, target_position, next_waypoint_position, speed, situation
        
        target_distance = self.distance(observation_data['position'], observation_data['target_position'])
        next_waypoint_distance = self.distance(observation_data['position'], observation_data['next_waypoint_position'])
        speed = observation_data['speed'][0]
        
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
            
        rest_list = [target_distance, next_waypoint_distance, speed] + cone_features
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

        speed = np.array([vehicle.get_speed()])
        situation = situations_map[active_scenario_dict['situation']]

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
        
        observation = {
            'rgb_data': np.uint8(rgb_image),
            'position': np.float32(current_position),
            'target_position': np.float32(target_position),
            'next_waypoint_position': np.float32(next_waypoint_position),
            'speed': np.float32(speed),
            'situation': situation
        }
        
        # Use our custom pre-processing
        self._CarlaEnv__observation = self.cone_pre_processing.preprocess_data(observation, cone_data)
        
        # Aux variables for reward function
        self._CarlaEnv__reward_target_pos = target_position
        self._CarlaEnv__reward_current_pos = current_position
        self._CarlaEnv__reward_next_waypoint_pos = next_waypoint_position
        self._CarlaEnv__reward_speed = speed[0]
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
        
        return obs, reward, terminated, truncated, info

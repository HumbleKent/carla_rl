'''
Pre-processing Module:
    - This module is used to preprocess the observation data before feeding it to the policy network
'''
import numpy as np
import math
from env.env_aux.farthest_sampler import FarthestSampler
from env.env_aux.point_net import PointNetfeat
import cv2
import torch

class PreProcessing:
    def __init__(self) -> None:
        pass
    
    def preprocess_data(self, observation_data, cone_data=None, last_action=None):
        '''
        Preprocesses raw CARLA data into the 23-dimension vector for the RL agent.
        '''
        target_distance = self.distance(observation_data['position'], observation_data['target_position'])
        next_waypoint_distance = self.distance(observation_data['position'], observation_data['next_waypoint_position'])
        
        # 1. Yaw Error (Heading Difference)
        wp_dir = observation_data['next_waypoint_position'] - observation_data['position']
        target_yaw = math.atan2(wp_dir[1], wp_dir[0])
        vehicle_yaw = math.radians(observation_data['rotation'][1]) # rotation[1] is yaw
        
        yaw_error = target_yaw - vehicle_yaw
        while yaw_error > math.pi: yaw_error -= 2 * math.pi
        while yaw_error < -math.pi: yaw_error += 2 * math.pi
        
        # 2. Local Velocities (Forward and Lateral)
        vx_world = observation_data['velocity'][0]
        vy_world = observation_data['velocity'][1]
        forward_v = vx_world * math.cos(vehicle_yaw) + vy_world * math.sin(vehicle_yaw)
        lateral_v = -vx_world * math.sin(vehicle_yaw) + vy_world * math.cos(vehicle_yaw)
        
        # 3. Angular Velocity Z
        ang_v_z = observation_data['angular_velocity'][2]
        
        # 4. Action history (2 features)
        if last_action is None:
            last_action = np.array([0.0, 0.0], dtype=np.float32)
        
        # 5. Flatten cone data (15 features)
        cone_features = []
        if cone_data:
            for cone in cone_data:
                cone_features.extend([cone['rel_x'], cone['rel_y'], cone['dist']])
            
        # Ensure we have fixed size (5 cones * 3 features = 15)
        num_expected_cones = 5
        current_num = len(cone_data) if cone_data else 0
        if current_num < num_expected_cones:
            for _ in range(num_expected_cones - current_num):
                cone_features.extend([0.0, 0.0, 1000.0]) # Padding
        elif current_num > num_expected_cones:
            cone_features = cone_features[:num_expected_cones*3]
            
        # Assemble 'rest' vector (Total: 2 + 4 + 2 + 15 = 23)
        rest_list = [
            target_distance, 
            next_waypoint_distance, 
            yaw_error, 
            ang_v_z, 
            forward_v, 
            lateral_v
        ] + list(last_action) + cone_features
        
        neo_observation_data = {
            'rgb_data': observation_data['rgb_data'],
            'rest': np.array(rest_list, dtype=np.float32)
        }
        
        return neo_observation_data

    # Distance function between two lists of 3 points
    def distance(self, a, b):
        return np.linalg.norm(a - b)
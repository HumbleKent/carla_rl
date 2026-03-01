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
        
        # Normalize values to be roughly between [-1.0, 1.0] for stable neural network training
        norm_target_dist = min(target_distance / 200.0, 1.0) # Assume max meaningful distance is ~200m
        norm_wp_dist = min(next_waypoint_distance / 20.0, 1.0)
        
        # Yaw is [-pi, pi], divide by pi to get [-1, 1]
        norm_yaw_error = yaw_error / math.pi
        
        # Assume max realistic ang_v_z is ~3.0
        norm_ang_v_z = np.clip(ang_v_z / 3.0, -1.0, 1.0)
        
        # Assume max forward velocity is ~30 m/s (108 km/h), lateral rarely exceeds ~10 m/s
        norm_forward_v = np.clip(forward_v / 30.0, -1.0, 1.0)
        norm_lateral_v = np.clip(lateral_v / 10.0, -1.0, 1.0)
        
        # 5. Flatten and normalize cone data (15 features)
        # We normalize distance relative to 20 meters. 
        # If no cone exists, we output [0, 0, 1.0] instead of raw distances
        cone_features = []
        max_cone_dist = 20.0
        if cone_data:
            for cone in cone_data:
                norm_rel_x = np.clip(cone['rel_x'] / max_cone_dist, -1.0, 1.0)
                norm_rel_y = np.clip(cone['rel_y'] / max_cone_dist, -1.0, 1.0)
                norm_dist  = np.clip(cone['dist'] / max_cone_dist, 0.0, 1.0)
                cone_features.extend([norm_rel_x, norm_rel_y, norm_dist])
            
        # Ensure we have fixed size (5 cones * 3 features = 15)
        num_expected_cones = 5
        current_num = len(cone_data) if cone_data else 0
        if current_num < num_expected_cones:
            for _ in range(num_expected_cones - current_num):
                # Padding: If no cone is in slot, distance is considered "far" (1.0 in normalized space)
                cone_features.extend([0.0, 0.0, 1.0]) 
        elif current_num > num_expected_cones:
            cone_features = cone_features[:num_expected_cones*3]
            
        # Assemble 'rest' vector (Total: 2 + 4 + 2 + 15 = 23)
        rest_list = [
            norm_target_dist, 
            norm_wp_dist, 
            norm_yaw_error, 
            norm_ang_v_z, 
            norm_forward_v, 
            norm_lateral_v
        ] + list(last_action) + cone_features
        
        neo_observation_data = {
            'rgb_data': observation_data['rgb_data'],
            'rest': np.array(rest_list, dtype=np.float32)
        }
        
        return neo_observation_data

    # Distance function between two lists of 3 points
    def distance(self, a, b):
        return np.linalg.norm(a - b)
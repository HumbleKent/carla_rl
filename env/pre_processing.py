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
        Preprocesses raw CARLA data into the 8-dimension vector for the RL agent.
        '''
        target_distance = self.distance(observation_data['position'], observation_data['target_position'])
        
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
        
        # Normalize values
        norm_target_dist = min(target_distance / 200.0, 1.0)
        norm_yaw_error = yaw_error / math.pi
        norm_ang_v_z = np.clip(ang_v_z / 3.0, -1.0, 1.0)
        norm_forward_v = np.clip(forward_v / 30.0, -1.0, 1.0)
        norm_lateral_v = np.clip(lateral_v / 10.0, -1.0, 1.0)
            
        # Extract actions
        last_steering = last_action[0]
        last_throttle = max(0.0, float(last_action[1]))
        last_brake    = max(0.0, float(-last_action[1]))
            
        # Assemble 'rest' vector (Total: 8 features)
        rest_list = [
            norm_forward_v, 
            norm_lateral_v,
            norm_ang_v_z,
            norm_yaw_error,
            norm_target_dist,
            last_steering, 
            last_throttle, 
            last_brake
        ]
        
        # 6. Process RGB image (Resize to 224x224 to save memory and match EfficientNet)
        rgb_image = observation_data['rgb_data']
        if rgb_image.shape[0] != 224 or rgb_image.shape[1] != 224:
            rgb_image = cv2.resize(rgb_image, (224, 224), interpolation=cv2.INTER_AREA)
            
        neo_observation_data = {
            'rgb_data': rgb_image,
            'rest': np.array(rest_list, dtype=np.float32)
        }
        
        return neo_observation_data, yaw_error

    # Distance function between two lists of 3 points
    def distance(self, a, b):
        return np.linalg.norm(a - b)
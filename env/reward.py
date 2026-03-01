from src.vehicle import Vehicle
from src.world import World
import configuration as config
import carla
import numpy as np

# ======================================== Global Variables =================================================================
class Reward:
    def __init__(self) -> None:
        self.terminated           = False
        self.inside_stop_area     = False
        self.has_stopped          = False
        self.current_steering     = 0.0
        self.current_throttle     = 0.0
        self.waypoints            = []      
        self.total_ep_reward      = 0  
        self.prev_target_distance = None # Used to calculate distance progress
        
        self.countint = 0

    # ======================================== Main Reward Function ==========================================================
    def calculate_reward(self, vehicle: Vehicle, current_pos, target_pos, next_waypoint_pos, speed, cone_data=None) -> float:   
        target_distance = self.distance(current_pos, target_pos)
        next_waypoint_distance = self.distance(current_pos, next_waypoint_pos)
        
        if self.terminated:
            self.countint += 1
            print("The episode already ended!!!, count: ", self.countint)
            
        # 1. Living Penalty (The 'Cost of Time')
        # We keep this so the agent still wants to finish the level, but it won't be paralyzing anymore.
        living_penalty = -20.0 / config.ENV_MAX_STEPS
        
        # 2. Stand-Still Penalty 
        stand_still_penalty = self.__stand_still_penalty(speed)

        # 3. Distance Delta (The 'Progress Reward')
        distance_reward = 0.0
        if self.prev_target_distance is not None:
            delta = self.prev_target_distance - target_distance
            
            ### CHANGED: Massively boosted the progress reward. 
            ### Now, moving forward easily overshadows the living and stand-still penalties.
            ### Was: delta * 0.5
            distance_reward = delta * 10.0 
            
        self.prev_target_distance = target_distance

        # 4. Proximity Cone Penalty (Near-misses)
        cone_proximity_penalty = self.__proximity_cone_penalty(cone_data)

        reward = self.__collision_reward(vehicle) + \
            self.__steering_jerk(vehicle) + \
            self.__throttle_brake_jerk(vehicle) + \
            self.__speed_reward(speed) + \
            self.__target_destination(target_distance) + \
            self.__waypoint_reached(next_waypoint_distance) + \
            living_penalty + \
            stand_still_penalty + \
            distance_reward + \
            cone_proximity_penalty
        
        self.total_ep_reward += reward
        
        return reward
        
    # ============================================= Reward Functions ==========================================================
    def __collision_reward(self, vehicle):
        '''
        Penalizes the vehicle differently depending on the type of line crossed:

        - Hard collision (wall/car): heavy penalty, no instant termination
          (agent can learn to recover rather than always dying at step 1).

        - Broken line crossed: small recurring penalty per step.
          Allowed — overtaking, lane changing for cone avoidance, etc.
          The agent is discouraged but not stopped.

        - Solid line crossed: IMMEDIATE episode termination + large penalty.
          Never allowed — this is a hard road rule violation.
        '''
        penalty = 0.0

        if vehicle.collision_occurred():
            penalty -= 15.0

        if vehicle.solid_line_crossed():
            # Hard violation — end episode immediately
            self.terminated = True
            penalty -= 20.0

        elif vehicle.lane_invasion_occurred():
            # Broken line — soft recurring penalty, episode continues
            penalty -= 5.0

        return penalty

        
    def __steering_jerk(self, vehicle, threshold=0.2):
        lbd = 10/config.ENV_MAX_STEPS
        steering_diff = abs(vehicle.get_steering() - self.current_steering)
        self.current_steering = vehicle.get_steering()
        return -lbd if steering_diff > threshold else 0.0

    def __throttle_brake_jerk(self, vehicle, threshold=0.1):
        lbd = 10/config.ENV_MAX_STEPS
        throttle_diff = abs(vehicle.get_throttle_brake() - self.current_throttle)
        self.current_throttle = vehicle.get_throttle_brake()
        return -lbd if throttle_diff > threshold else 0.0

    def __speed_reward(self, speed, speed_limit=50):
        # Give a substantial positive reward for moving, scaling with speed
        if speed < 2:
            return 0.0
        elif speed >= 2 and speed <= speed_limit:
            # Reward smooth driving (peaks at +1.0 per frame at speed_limit)
            return (speed / speed_limit)
        else:
            return -1.0  # Excessive speed over limit

    def __stand_still_penalty(self, speed):
        """Strongly penalize the agent for not moving when it should be."""
        if speed < 1.0:
            # -1.0 per step is massively worse than the total penalty for a lane error.
            # This completely breaks the "safest option is to never move" local optimum!
            return -1.0
        return 0.0
 
    def __proximity_cone_penalty(self, cone_data, safe_distance=2.5):
        """Penalize being too close to a cone (near-miss penalty)."""
        if not cone_data:
            return 0.0
        penalty = 0.0
        for cone in cone_data:
            dist = cone['dist']
            if dist < safe_distance:
                penalty -= (1.0 - (dist / safe_distance)) * (10.0 / config.ENV_MAX_STEPS)
        return penalty

    def __target_destination(self, target_distance, threshold=5.0):
        if target_distance <= threshold:
            self.terminated = True
            return 500.0 # Massive bonus to ensure the model prioritizes finishing
        elif target_distance > threshold and target_distance <= 50.0:
            return (-7.0*target_distance + 395.0) / (9.0 * config.ENV_MAX_STEPS)
        elif target_distance > 50.0 and target_distance <= 100.0:
            return (100.0 - target_distance) / (10.0 * config.ENV_MAX_STEPS)
        else:
            return 0.0
        
    def __waypoint_reached(self, next_waypoint_distance, threshold=1.0):
        '''
        Rewards the agent for hitting a waypoint.
        '''
        if next_waypoint_distance < threshold:
            self.waypoints.pop(0)
            
            ### CHANGED: Increased from 2.0 to 5.0 to give a stronger "breadcrumb" signal 
            ### that following the path is highly desirable.
            return 5.0
        else:
            return 0.0
        
    def __light_pole_trangression(self, map, vehicle, world):
        lbd = 20.0
        current_waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True)
        traffic_lights = world.get_world().get_traffic_lights_from_waypoint(current_waypoint, distance=10.0)

        for traffic_light in traffic_lights:
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                stop_waypoints = traffic_light.get_stop_waypoints()
                for stop_waypoint in stop_waypoints:
                    if current_waypoint.transform.location.distance(stop_waypoint.transform.location) < 2.0 and vehicle.get_speed() > 0.3:
                        ### Note: We leave termination here because running a red light is a critical failure.
                        self.terminated = True
                        return -lbd
        return 0.0

    def __stop_sign_transgression(self, vehicle, map):
        lbd = 20.0
        distance = 20.0  
        current_location = vehicle.get_location()
        current_waypoint = map.get_waypoint(current_location, project_to_road=True)
        
        stop_signs_on_same_road = []
        for landmark in current_waypoint.get_landmarks_of_type(distance, carla.LandmarkType.StopSign):
            landmark_waypoint = map.get_waypoint(landmark.transform.location, project_to_road=True)
            if landmark_waypoint.road_id == current_waypoint.road_id:
                stop_signs_on_same_road.append(landmark)

        if len(stop_signs_on_same_road) == 0:
            if self.inside_stop_area and self.has_stopped:
                self.has_stopped = False
                self.inside_stop_area = False
                return 0
            elif self.inside_stop_area and not self.has_stopped:
                self.has_stopped = False
                self.inside_stop_area = False
                ### Note: We leave termination here because running a stop sign is a critical failure.
                self.terminated = True
                return -lbd
            else:            
                return 0.0
        else:
            self.inside_stop_area = True

        for stop_sign in stop_signs_on_same_road:
            if vehicle.get_speed() < 1.0:
                self.has_stopped = True
        
    # ==================================== Helper Functions ================================================================
    def distance(self, a, b):
        return np.linalg.norm(a - b)

    def get_waypoints(self):
        return self.waypoints
    
    def reset(self, waypoints):
        self.terminated           = False
        self.inside_stop_area     = False
        self.has_stopped          = False
        self.current_steering     = 0.0
        self.current_throttle     = 0.0
        self.waypoints            = waypoints
        self.total_ep_reward      = 0
        self.prev_target_distance = None
    
    def get_terminated(self):
        return self.terminated
    
    def get_total_ep_reward(self):
        return self.total_ep_reward
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
        self.termination_reason   = "None"
        
        self.countint = 0

    # ======================================== Main Reward Function ==========================================================
    def calculate_reward(self, vehicle: Vehicle, current_pos, target_pos, next_waypoint_pos, speed, yaw_error=0.0, cone_data=None, step_count=0, verbose=False, debug_manager=None) -> float:   
        target_distance = self.distance(current_pos, target_pos)
        next_waypoint_distance = self.distance(current_pos, next_waypoint_pos)
        
        if self.terminated:
            self.countint += 1
            if debug_manager:
                debug_manager.log("term", f"Episode already ended! count: {self.countint}")
            
        # 1. Stand-Still Penalty (Harsh)
        stand_still_penalty = self.__stand_still_penalty(speed)
        
        # 2. Moving Forward Bonus
        moving_bonus = 0.0
        if speed > 2.0:
            moving_bonus = 0.1 

        # 3. Progress Reward
        distance_reward = 0.0
        if self.prev_target_distance is not None:
            delta = self.prev_target_distance - target_distance
            distance_reward = delta * 10.0 
            
        self.prev_target_distance = target_distance

        # 4. Proximity Cone Penalty
        cone_proximity_penalty = self.__proximity_cone_penalty(cone_data, safe_distance=3.0)

        # 5. Core components
        coll_reward = self.__collision_reward(vehicle, verbose, debug_manager)
        
        # --- PREVENT SUICIDE SHORTCUT ---
        if self.terminated and step_count < config.ENV_MAX_STEPS:
            lost_time_penalty = -2.0 * (config.ENV_MAX_STEPS - step_count) / config.ENV_MAX_STEPS
            coll_reward += lost_time_penalty

        steering_reward = self.__steering_jerk(vehicle)
        throttle_reward = self.__throttle_brake_jerk(vehicle)
        target_bonus = self.__target_destination(target_distance, threshold=2.0, verbose=verbose, debug_manager=debug_manager)
        waypoint_reward = self.__waypoint_reached(next_waypoint_distance)
        speed_reward = self.__speed_reward(speed)
        
        # Penalizes looking away from the safe path
        heading_penalty = -abs(yaw_error) * 0.2

        # 6. Safety rules
        light_reward = self.__light_pole_trangression(vehicle, verbose=verbose, debug_manager=debug_manager)
        stop_reward = self.__stop_sign_transgression(vehicle, verbose=verbose, debug_manager=debug_manager)

        reward = coll_reward + \
                 steering_reward + \
                 throttle_reward + \
                 speed_reward + \
                 target_bonus + \
                 waypoint_reward + \
                 moving_bonus + \
                 stand_still_penalty + \
                 distance_reward + \
                 cone_proximity_penalty + \
                 heading_penalty + \
                 light_reward + \
                 stop_reward
        
        if debug_manager and debug_manager.is_active("reward"):
            debug_manager.log("reward", f"Step {step_count} | R: {reward:.3f} (S:{speed_reward:.2f} P:{distance_reward:.2f} C:{coll_reward:.2f} Con:{cone_proximity_penalty:.2f})")

        self.total_ep_reward += reward
        
        return reward
        
    # ============================================= Reward Functions ==========================================================
    def __collision_reward(self, vehicle, verbose=False, debug_manager=None):
        '''
        Redesigned collision logic:
        - Cone hit: CRITICAL FAILURE. -30 penalty and immediate termination.
        - Solid line: CRITICAL FAILURE. -50 penalty and immediate termination.
        - Hard collision (wall/car): -20 penalty. No termination (allows learning to recover).
        - Broken line: -2 penalty (soft discouragement).
        '''
        penalty = 0.0

        if vehicle.hit_cone():
            # New termination condition: Hitting a cone ends the episode
            self.terminated = True
            self.termination_reason = "Hit a traffic cone"
            penalty -= 30.0
            if debug_manager:
                debug_manager.log("term", "Hit a traffic cone!")
            elif verbose: 
                print("\n[REWARD] TERMINATED: Hit a traffic cone!")
            return penalty

        if vehicle.solid_line_crossed():
            self.terminated = True
            self.termination_reason = "Crossed a solid line"
            penalty -= 50.0
            if debug_manager:
                debug_manager.log("term", "Crossed a solid line!")
            elif verbose: 
                print("\n[REWARD] TERMINATED: Crossed a solid line!")
            return penalty

        if vehicle.collision_occurred():
            penalty -= 20.0

        if vehicle.lane_invasion_occurred():
            penalty -= 2.0

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
        # Always reward moving. 
        if speed <= 0.1:
            return 0.0
        elif speed <= speed_limit:
            # Linear reward that peaks at +1.0 at speed limit
            return (speed / speed_limit)
        else:
            # Gentle penalty for overspeeding rather than a hard drop
            return 1.0 - (speed - speed_limit) * 0.1

    def __stand_still_penalty(self, speed):
        """Strongly penalize the agent for not moving when it should be."""
        if speed < 1.0:
            # Increased to -10.0 to make it extremely painful to stay still.
            # This forces the agent to explore even if it's scared of cones.
            return -1.0
        return 0.0
 
    def __proximity_cone_penalty(self, cone_data, safe_distance=3.0):
        """Penalize being too close to a cone (near-miss penalty)."""
        if not cone_data:
            return 0.0
        penalty = 0.0
        for cone in cone_data:
            dist = cone['dist']
            if dist < safe_distance:
                # Stronger exponential penalty as the car gets closer to a cone
                # At dist=0, it's -1.0 per frame. At dist=safe_distance, it's 0.
                penalty -= (1.0 - (dist / safe_distance)) ** 2
        return penalty

    def __target_destination(self, target_distance, threshold=2.0, verbose=False, debug_manager=None):
        if target_distance <= threshold:
            self.terminated = True
            self.termination_reason = "Reached Target Destination"
            if debug_manager:
                debug_manager.log("term", "Reached Target Destination! (Success)")
            elif verbose: 
                print("\n[REWARD] TERMINATED: Reached Target Destination! (Success)")
            return 500.0 # Massive bonus to ensure the model prioritizes finishing
        elif target_distance > threshold and target_distance <= 50.0:
            return (-7.0*target_distance + 395.0) / (9.0 * config.ENV_MAX_STEPS)
        elif target_distance > 50.0 and target_distance <= 100.0:
            return (100.0 - target_distance) / (10.0 * config.ENV_MAX_STEPS)
        else:
            return 0.0
        
    def __waypoint_reached(self, next_waypoint_distance, threshold=1.0):
        '''
        Waypoints are popped to update the heading guidance (breadcrumbs), 
        but we no longer give an explicit reward for hitting them.
        '''
        if next_waypoint_distance < threshold:
            if self.waypoints:
                self.waypoints.pop(0)
            return 0.0
        else:
            return 0.0
        
    def __light_pole_trangression(self, vehicle, verbose=False, debug_manager=None):
        lbd = 20.0
        world_obj = vehicle.get_world_obj() 
        map = world_obj.get_map()
        current_waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True)
        traffic_lights = world_obj.get_traffic_lights_from_waypoint(current_waypoint, distance=10.0)

        for traffic_light in traffic_lights:
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                stop_waypoints = traffic_light.get_stop_waypoints()
                for stop_waypoint in stop_waypoints:
                    if current_waypoint.road_id == stop_waypoint.road_id and current_waypoint.lane_id == stop_waypoint.lane_id:
                        vec_x = vehicle.get_location().x - stop_waypoint.transform.location.x
                        vec_y = vehicle.get_location().y - stop_waypoint.transform.location.y
                        forward_vec = stop_waypoint.transform.get_forward_vector()
                        dot = vec_x * forward_vec.x + vec_y * forward_vec.y
                        dist = current_waypoint.transform.location.distance(stop_waypoint.transform.location)
                        
                        if dot > 0.0 and dist < 4.0:
                            self.terminated = True
                            self.termination_reason = "Ran a Red Light"
                            if debug_manager:
                                debug_manager.log("term", "Ran a Red Light!")
                            elif verbose: 
                                print("\n[REWARD] TERMINATED: Ran a Red Light!")
                            return -lbd
        return 0.0

    def __stop_sign_transgression(self, vehicle, verbose=False, debug_manager=None):
        lbd = 20.0
        distance = 20.0  
        world_obj = vehicle.get_world_obj()
        map = world_obj.get_map()
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
                self.terminated = True
                self.termination_reason = "Ran a Stop Sign"
                if debug_manager:
                    debug_manager.log("term", "Ran a Stop Sign!")
                elif verbose: 
                    print("\n[REWARD] TERMINATED: Ran a Stop Sign!")
                return -lbd
            else:            
                return 0.0
        else:
            self.inside_stop_area = True

        for stop_sign in stop_signs_on_same_road:
            if vehicle.get_speed() < 1.0:
                self.has_stopped = True
        return 0.0
        
    # ==================================== Helper Functions ================================================================
    def distance(self, a, b):
        return np.linalg.norm(a - b)

    def get_waypoints(self):
        return self.waypoints
    
    def reset(self, waypoints):
        self.terminated           = False
        self.termination_reason   = "None"
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
    def get_termination_reason(self):
        return self.termination_reason

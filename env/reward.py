from src.vehicle import Vehicle
from src.world import World
import configuration as config
import carla
import numpy as np

# ======================================== Global Variables =================================================================
class Reward:
    def __init__(self) -> None:
        self.terminated             = False
        self.current_steering       = 0.0
        self.current_throttle       = 0.0
        self.waypoints              = []
        self.total_ep_reward        = 0
        self.prev_target_distance   = None # Used to calculate distance progress
        self.prev_waypoint_distance = None # Used to calculate waypoint progress
        self.prev_lateral_dist      = None # Used to calculate lateral progress toward target lane
        self.start_pos_x            = None # X position at episode start (used to detect lane change phase)
        self.termination_reason     = "None"

    # ======================================== Main Reward Function ==========================================================
    def calculate_reward(self, vehicle: Vehicle, current_pos, target_pos, next_waypoint_pos, speed, yaw_error=0.0, cte=0.0, cone_data=None, step_count=0, verbose=False, debug_manager=None) -> float:
        target_distance = self.distance(current_pos, target_pos)
        next_waypoint_distance = self.distance(current_pos, next_waypoint_pos)

        # Store start X on first step so we know the origin of the lane change
        if self.start_pos_x is None:
            self.start_pos_x = current_pos[0]

        # ---- Detect lane change phase ----
        # The vehicle is mid-maneuver when its X sits between the start lane and target lane.
        # We allow a small buffer (0.3m) on each side so boundary noise doesn't flip the flag.
        target_x    = target_pos[0]
        current_x   = current_pos[0]
        x_lo        = min(self.start_pos_x, target_x) - 0.3
        x_hi        = max(self.start_pos_x, target_x) + 0.3
        in_lane_change = (x_lo < current_x < x_hi) and (abs(self.start_pos_x - target_x) > 1.5)

        # 1. Stand-Still Penalty (Harsh)
        stand_still_penalty = self.__stand_still_penalty(speed)

        # 2. Progress Reward (Immediate Waypoint focus)
        distance_reward = 0.0
        if self.prev_waypoint_distance is not None:
            delta = self.prev_waypoint_distance - next_waypoint_distance
            # If delta is extremely negative, it's likely a waypoint pop; ignore that frame
            if delta > -2.0:
                distance_reward = delta * 15.0

        self.prev_waypoint_distance = next_waypoint_distance
        self.prev_target_distance = target_distance

        # 3. Lateral Progress Reward (new — lane change specific)
        # Reward every meter the vehicle moves toward the target lane X.
        lateral_dist = abs(current_x - target_x)
        lateral_reward = 0.0
        if self.prev_lateral_dist is not None:
            lateral_delta = self.prev_lateral_dist - lateral_dist  # positive = closer to target lane
            lateral_reward = lateral_delta * 10.0
        self.prev_lateral_dist = lateral_dist

        # 4. Proximity Cone Penalty
        cone_proximity_penalty = self.__proximity_cone_penalty(cone_data, safe_distance=1.8)

        # 5. Core components (Collisions and Actions)
        coll_reward = self.__collision_reward(vehicle, verbose, debug_manager)

        # --- PREVENT SUICIDE SHORTCUT ---
        if self.terminated and step_count < config.ENV_MAX_STEPS and self.termination_reason != "Reached Target Destination":
            lost_time_penalty = -100.0 * (config.ENV_MAX_STEPS - step_count) / config.ENV_MAX_STEPS
            coll_reward += lost_time_penalty

        # Wider jerk threshold during lane change — sustained steering is required
        steering_reward = self.__steering_jerk(vehicle, threshold=0.4 if in_lane_change else 0.2)
        throttle_reward = self.__throttle_brake_jerk(vehicle)
        target_bonus    = self.__target_destination(target_distance, threshold=2.0, verbose=verbose, debug_manager=debug_manager)
        waypoint_bonus  = self.__waypoint_reached(next_waypoint_distance)
        speed_reward    = self.__speed_reward(speed)

        # 6. Heading Alignment
        HEADING_DEAD_ZONE = 0.15  # radians (~8 degrees)
        # Relax heading penalty mid-maneuver — yaw briefly points left during lane change
        heading_scale   = 0.2 if in_lane_change else 0.5
        heading_penalty = -max(0.0, abs(yaw_error) - HEADING_DEAD_ZONE) * heading_scale

        # 7. CTE — suppressed during lane change to avoid penalising lateral displacement
        cte_penalty = 0.0 if in_lane_change else -abs(cte) * 0.4

        # 8. Summation
        reward = coll_reward + \
                 steering_reward + \
                 throttle_reward + \
                 speed_reward + \
                 target_bonus + \
                 waypoint_bonus + \
                 stand_still_penalty + \
                 distance_reward + \
                 lateral_reward + \
                 cone_proximity_penalty + \
                 heading_penalty + \
                 cte_penalty

        if debug_manager and debug_manager.is_active("reward"):
            debug_manager.log("reward", (
                f"Step {step_count} | R: {reward:.3f} "
                f"(S:{speed_reward:.2f} P:{distance_reward:.2f} Lat:{lateral_reward:.2f} "
                f"C:{coll_reward:.2f} Con:{cone_proximity_penalty:.2f} LC:{in_lane_change})"
            ))

        self.total_ep_reward += reward

        return reward

    # ============================================= Reward Functions ==========================================================
    def __collision_reward(self, vehicle, verbose=False, debug_manager=None):
        penalty = 0.0

        if vehicle.hit_cone():
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

        return penalty

    def __steering_jerk(self, vehicle, threshold=0.05):
        lbd = 10 / config.ENV_MAX_STEPS
        steering_diff = abs(vehicle.get_steering() - self.current_steering)
        self.current_steering = vehicle.get_steering()
        return -lbd if steering_diff > threshold else 0.0

    def __throttle_brake_jerk(self, vehicle):
        throttle_diff = abs(vehicle.get_throttle_brake() - self.current_throttle)
        self.current_throttle = vehicle.get_throttle_brake()
        return -(throttle_diff ** 2) * 0.5

    def __speed_reward(self, speed, speed_limit=config.ENV_SPEED_LIMIT):
        if speed <= 0.1:
            return 0.0
        elif speed <= speed_limit:
            return (speed / speed_limit)
        else:
            return 1.0 - (speed - speed_limit) * 0.1

    def __stand_still_penalty(self, speed):
        if speed < 1.0:
            return -1.0
        return 0.0

    def __proximity_cone_penalty(self, cone_data, safe_distance=1.8):
        if not cone_data:
            return 0.0
        penalty = 0.0
        for cone in cone_data:
            dist = cone['dist']
            if dist < safe_distance:
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
            return 1000.0
        return 0.0

    def __waypoint_reached(self, next_waypoint_distance, threshold=1.0):
        if next_waypoint_distance < threshold:
            if self.waypoints:
                self.waypoints.pop(0)
                self.prev_waypoint_distance = None
            return 2.0
        else:
            return 0.0

    # ==================================== Helper Functions ================================================================
    def distance(self, a, b):
        return np.linalg.norm(a - b)

    def get_waypoints(self):
        return self.waypoints

    def reset(self, waypoints):
        self.terminated             = False
        self.termination_reason     = "None"
        self.current_steering       = 0.0
        self.current_throttle       = 0.0
        self.waypoints              = waypoints
        self.total_ep_reward        = 0
        self.prev_target_distance   = None
        self.prev_waypoint_distance = None
        self.prev_lateral_dist      = None
        self.start_pos_x            = None  # Will be set on first step of each episode

    def get_terminated(self):
        return self.terminated

    def get_total_ep_reward(self):
        return self.total_ep_reward

    def get_termination_reason(self):
        return self.termination_reason

    def get_episode_summary(self):
        return f"Episode Summary | Reason: {self.termination_reason} | Total Reward: {self.total_ep_reward:.2f}"
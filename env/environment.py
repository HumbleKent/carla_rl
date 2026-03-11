import numpy as np
import json
import time
import random
import math
import carla
import gymnasium as gym
from gymnasium.envs.registration import register
import configuration as config

from src.world import World
from src.server import CarlaServer
from src.vehicle import Vehicle
from src.display import Display
from env.reward import Reward
from env.observation_action_space import (
    observation_space, 
    action_space
)
from env.pre_processing import PreProcessing
from src.debug_manager import DebugManager
from src.route_planner import AdvancedRoutePlanner

register(
    id="carla-rl-gym-v0",
    entry_point="env.environment:CarlaEnv",
    max_episode_steps=config.ENV_MAX_STEPS,
)

# Unified Environment for CARLA Training
class CarlaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": config.SIM_FPS}
    
    def __init__(self, scenarios=[], time_limit=60, initialize_server=False, 
                 random_weather=False, random_traffic=False, synchronous_mode=True, 
                 show_sensor_data=False, has_traffic=True, verbose=True, 
                 spawn_cones=True, port=None, is_eval=False,
                 debug_features=[]):
        super().__init__()
        
        self.__port = port if port is not None else config.SIM_PORT
        self.__random_weather = random_weather
        self.__random_traffic = random_traffic
        self.__synchronous_mode = synchronous_mode
        self.__show_sensor_data = show_sensor_data
        self.__has_traffic = has_traffic
        self.__verbose = verbose
        self.__automatic_server_initialization = initialize_server
        self.__cones_spawned = False
        self.__spawn_cones = spawn_cones 
        self.__is_eval = is_eval
        
        # Debugging
        self.debug_manager = DebugManager(debug_features)

        # 1. Server initialization
        if self.__automatic_server_initialization:
            self.__server_process = CarlaServer.initialize_server(
                low_quality=config.SIM_LOW_QUALITY, 
                offscreen_rendering=config.SIM_OFFSCREEN_RENDERING
            )
        
        # 2. Setup World, Vehicle, and Processing
        self.__world = World(synchronous_mode=self.__synchronous_mode, port=self.__port)
        self.__get_situations(scenarios)
        self.__vehicle = Vehicle(self.__world.get_world())
        self.pre_processing = PreProcessing()
        self.__reward_func = Reward()
        
        # 3. Spaces
        self.observation_space = observation_space
        self.action_space = action_space
        
        # 4. State variables
        self._active_scenario_name = None
        self._active_scenario_dict = None
        self._waypoints = None

        self.__episode_number = 0
        self.__truncated = False
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.current_cone_data = []
        self.num_cones_to_track = 5
        
        self.__cone_config = self.__load_cone_config()
        self.__route_planner = AdvancedRoutePlanner(
            self.__world.get_world(), 
            sampling_res=config.ENV_WAYPOINT_SPACING,
            cone_threshold=config.ENV_CONE_THRESHOLD,
            verbose=False
        )
        if self.__cone_config:
            self.__route_planner.set_cones(self.__cone_config)

    def __load_cone_config(self):
        try:
            with open(config.ENV_CONE_LAYOUT_FILE, 'r') as f:
                return json.load(f)
        except:
            return None

    def reset(self, seed=None, options=None):
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        
        # Select Scenario
        if options and 'scenario_name' in options:
            self._active_scenario_name = options['scenario_name']
        else:
            self._active_scenario_name = self.__chose_situation(seed)
        
        if self.__verbose:
            print(f"[Port {self.__port}] Loading scenario {self._active_scenario_name}...")
            
        self.load_scenario(self._active_scenario_name, seed)
        
        # Initialize target position for debugging and rewards
        self._reward_target_pos = np.array([
            self._active_scenario_dict['target_position']['x'], 
            self._active_scenario_dict['target_position']['y'], 
            self._active_scenario_dict['target_position']['z']
        ], dtype=np.float32)
        
        self.place_spectator_above_vehicle()
        
        # Pathfinding
        s_pos = self._active_scenario_dict['initial_position']
        start_loc = carla.Location(x=s_pos['x'], y=s_pos['y'], z=s_pos['z'])
        waypoints_locations = self.get_path_waypoints(spacing=config.ENV_WAYPOINT_SPACING, start_loc=start_loc)
        if self.__verbose:
            print(f"[Port {self.__port}] Generated {len(waypoints_locations)} waypoints.")
        
        self._waypoints = [np.array([w.x, w.y, w.z]) for w in waypoints_locations]
        
        # Draw full planned route at episode start (long lifetime so whole path is visible)
        self.debug_manager.draw_waypoints(self.__world.get_world(), self._waypoints, life_time=30.0)
        self.debug_manager.draw_target(self.__world.get_world(), self._reward_target_pos, life_time=30.0)
        
        # Wait for sensors
        waited = 0
        while not self.__vehicle.sensors_ready() and waited < 50:
            self.__world.tick()
            time.sleep(0.1)
            waited += 1
            
        self._update_observation()
        self.__reward_func.reset(self._waypoints)

        self.__episode_number += 1
        self.__start_timer()
        self.number_of_steps = 0
        
        # Spawn cones AFTER potential map reloads in load_scenario
        if self.__spawn_cones:
            self.__spawn_all_cones()
            
        info = {'scenario_name': self._active_scenario_name, 'waypoints': self._waypoints}
        return self.__observation, info

    def render(self, mode='human'):
        if mode == 'human':
            self.__world.tick()
            if hasattr(self, 'display'):
                self.display.play_window_tick()
        else:
            raise NotImplementedError("This mode is not implemented yet")

    def step(self, action):
        if self.__synchronous_mode:
            self.__world.tick()
            
        # Self-healing: Ensure cones haven't been wiped mid-episode by a parallel process
        if self.__spawn_cones and self.__episode_number > 0 and self.number_of_steps % 10 == 0:
            if self.__world.get_cone_count() == 0 and self.__cone_config is not None:
                if self.__verbose: print(f"[Port {self.__port}] Cones disappeared mid-episode! Respawning...")
                self.__spawn_all_cones()

        self.number_of_steps += 1
        
        # Control
        action_array = np.array(action)
        self.__vehicle.control_vehicle(action_array)

        # Update
        self._update_observation()
        self.place_spectator_above_vehicle()
        
        # Reward
        reward = self.__reward_func.calculate_reward(
            self.__vehicle, 
            self._reward_current_pos, 
            self._reward_target_pos, 
            self._reward_next_waypoint_pos, 
            speed=self._reward_speed,
            yaw_error=self.__current_yaw_error,
            cone_data=self.current_cone_data,
            step_count=self.number_of_steps,
            verbose=self.__verbose,
            debug_manager=self.debug_manager
        )
        
        terminated = self.__reward_func.get_terminated()
        self._waypoints = self.__reward_func.get_waypoints()
        
        # Dynamic rendering of remaining waypoints + target with short lifetime
        # This refreshes every step so waypoints "consume" as the agent progresses
        if self._waypoints:
            self.debug_manager.draw_waypoints(self.__world.get_world(), self._waypoints, life_time=0.2)
        self.debug_manager.draw_target(self.__world.get_world(), self._reward_target_pos, life_time=0.2)
        
        # Use exact step count instead of wall-clock time for reproducible RL episodes
        self.__truncated = (self.number_of_steps >= config.ENV_MAX_STEPS)
        
        if terminated or self.__truncated:
            reason = "SUCCESS (Goal Reached)" if terminated else "TIMEOUT (Max Steps)"
            if self.__vehicle.collision_occurred(): reason = "CRASH (Collision)"
            elif self.__vehicle.lane_invasion_occurred(): reason = "OFF-ROAD (Lane Invasion)"
            
            # If reward function has a more specific reason, use it
            reward_reason = self.__reward_func.get_termination_reason()
            if reward_reason != "None":
                reason = reward_reason

            if self.__verbose:
                print(f"[Port {self.__port}] Episode {self.__episode_number} End | Reason: {reason} | Score: {self.__reward_func.get_total_ep_reward():.2f}")
            self.clean_scenario()
        else:
            reason = "None"
        
        self.last_action = np.array(action, dtype=np.float32)
        info = {
            'scenario_name': self._active_scenario_name, 
            'waypoints': self._waypoints, 
            'port': self.__port,
            'termination_reason': reason
        }
        
        return self.__observation, reward, terminated, self.__truncated, info

    def _update_observation(self):
        obs_dict = self.__vehicle.get_observation_data()
        if 'rgb_data' not in obs_dict: return
            
        rgb_image = obs_dict['rgb_data']
        vehicle_actor = self.__vehicle.get_vehicle()
        if vehicle_actor is None: return
            
        vehicle_loc = vehicle_actor.get_location()
        current_position = np.array([vehicle_loc.x, vehicle_loc.y, vehicle_loc.z])
        target_position = np.array([
            self._active_scenario_dict['target_position']['x'], 
            self._active_scenario_dict['target_position']['y'], 
            self._active_scenario_dict['target_position']['z']
        ])
        
        if self._waypoints and len(self._waypoints) > 0:
            next_waypoint_position = self._waypoints[0]
        else:
            next_waypoint_position = target_position

        velocity = vehicle_actor.get_velocity()
        ang_vel = vehicle_actor.get_angular_velocity()
        transform = vehicle_actor.get_transform()
        
        # Raw Data for Processing
        raw_obs = {
            'rgb_data': np.uint8(rgb_image),
            'position': current_position,
            'target_position': target_position,
            'next_waypoint_position': next_waypoint_position,
            'velocity': np.array([velocity.x, velocity.y, velocity.z], dtype=np.float32),
            'angular_velocity': np.array([ang_vel.x, ang_vel.y, ang_vel.z], dtype=np.float32),
            'rotation': np.array([transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll], dtype=np.float32)
        }
        
        # Cone Detection
        active_cones = self.__world.get_active_cones()
        self.current_cone_data = []
        nearest_cone_loc = None
        if active_cones:
            cones_with_dist = []
            for cone in active_cones:
                if not cone.is_alive: continue
                dist = vehicle_loc.distance(cone.get_location())
                cones_with_dist.append((cone, dist))
            
            cones_with_dist.sort(key=lambda x: x[1])
            if len(cones_with_dist) > 0:
                nearest_cone_loc = cones_with_dist[0][0].get_location()

            for cone, dist in cones_with_dist[:self.num_cones_to_track]:
                loc = cone.get_location()
                self.current_cone_data.append({
                    'rel_x': loc.x - vehicle_loc.x, 
                    'rel_y': loc.y - vehicle_loc.y, 
                    'dist': dist
                })
        
        # Processed Observation
        # Now returns both the observation and the raw yaw_error.
        self.__observation, self.__current_yaw_error = self.pre_processing.preprocess_data(
            raw_obs, 
            cone_data=self.current_cone_data, 
            last_action=self.last_action
        )
        
        # Debugging
        self.debug_manager.show_nn_input(
            raw_obs['rgb_data'], 
            self.__observation['rest'],
            self.last_action
        )
        
        # self.debug_manager.draw_dynamic_distances(
        #     self.__world.get_world(),
        #     vehicle_loc,
        #     target_position,
        #     nearest_cone_loc
        # )
        
        # Refresh the Pygame sensor window every step
        if self.__show_sensor_data and hasattr(self, 'display'):
            self.display.play_window_tick()
        
        # Reward Aux
        self._reward_target_pos = target_position
        self._reward_current_pos = current_position
        self._reward_next_waypoint_pos = next_waypoint_position
        self._reward_speed = 3.6 * velocity.length()

    def close(self):
        self.__vehicle.destroy_vehicle()
        self.__world.destroy_vehicles()
        self.__world.destroy_pedestrians()
        if self.__show_sensor_data and hasattr(self, 'display'):
            self.display.close_window()
        if self.__automatic_server_initialization:
            CarlaServer.close_server(self.__server_process)

    # --- Scenario & World Helpers ---
    def load_scenario(self, scenario_name, seed=None):
        try:
            scenario_dict = self.situations_dict[scenario_name]
        except:
            scenario_name = self.__choose_random_situation(seed)
            scenario_dict = self.situations_dict[scenario_name]
            
        self._active_scenario_name = scenario_name
        self._active_scenario_dict = scenario_dict
        
        self.__load_world(scenario_dict['map_name'])
        self.__world.update_traffic_map()
        time.sleep(2.0)
        
        # Weather
        self.__load_weather(scenario_dict['weather_condition'])
        
        # Ego vehicle
        self.__spawn_vehicle(scenario_dict)
            
        # Create the Display window only once (guard against re-creation every episode)
        if self.__show_sensor_data and not hasattr(self, 'display'):
            self.display = Display('Sensor Feed', self.__vehicle)

        # Traffic
        if self.__has_traffic:
            self.__spawn_traffic(seed=seed)
        
        self.__toggle_lights()

    def clean_scenario(self):
        self.__vehicle.destroy_vehicle()
        self.__world.destroy_vehicles()
        self.__world.destroy_pedestrians()

    def print_all_scenarios(self):
        for idx, i in enumerate(self.situations_list):
            print(idx, ": ", i)
    
    def __load_world(self, name):
        self.__world.set_active_map(name)
        # Refresh all internal handles (World, Controllers, and Mapper)
        self.__world.sync_world_handles()
        if hasattr(self, '__route_planner'):
            self.__route_planner.update_map(self.__world.get_world())

    def __spawn_vehicle(self, s_dict):
        location = (s_dict['initial_position']['x'], s_dict['initial_position']['y'], s_dict['initial_position']['z'])
        rotation = (s_dict['initial_rotation']['pitch'], s_dict['initial_rotation']['yaw'], s_dict['initial_rotation']['roll'])
        try:
            self.__vehicle.spawn_vehicle(location, rotation)
        except Exception as e:
            if self.__verbose:
                print(f"[Port {self.__port}] Spawn area occupied: {e}. Clearing area and retrying...")
            
            # Instead of map reload (which kills cones), just try to clear spawn point actors
            all_actors = self.__world.get_world().get_actors()
            for actor in all_actors.filter('vehicle.*'):
                if actor.get_location().distance(carla.Location(x=location[0], y=location[1], z=location[2])) < 5.0:
                    try: actor.destroy()
                    except: pass
            
            time.sleep(1.0)
            try:
                self.__vehicle.spawn_vehicle(location, rotation)
            except Exception as e2:
                if self.__verbose: print(f"[Port {self.__port}] Failed second spawn, doing fallback map reload...")
                self.__world.reload_map()
                self.__cones_spawned = False
                time.sleep(2.0)
                self.__vehicle.spawn_vehicle(location, rotation)

    def __toggle_lights(self):
        weather = self.__world.get_active_weather().lower()
        if "night" in weather or "noon" in weather:
            self.__world.toggle_lights(lights_on=True)
            self.__vehicle.toggle_lights(lights_on=True)
        else:
            self.__world.toggle_lights(lights_on=False)
            self.__vehicle.toggle_lights(lights_on=False)
        
    def __load_weather(self, weather_name):
        if self.__random_weather:
            self.__world.set_random_weather()
        else:
            self.__world.set_active_weather_preset(weather_name)

    def __spawn_traffic(self, seed):
        if not self.__random_traffic and self._active_scenario_dict.get('traffic_density') == 'None':
            return
        
        if not self.__random_traffic:
            random.seed(self._active_scenario_name)
            seed = self._active_scenario_name
        
        if seed is not None:
            random.seed(seed)
        
        if not self.__random_traffic:
            density = self._active_scenario_dict.get('traffic_density', 'Low')
            num_vehicles = random.randint(1, 5) if density == 'Low' else random.randint(10, 20)
        else:
            num_vehicles = random.randint(1, 20)
        
        self.__world.spawn_vehicles_around_ego(self.__vehicle.get_vehicle(), radius=100, num_vehicles_around_ego=num_vehicles, seed=seed)

    def __spawn_all_cones(self):
        if self.__cone_config is None: return
        
        # Self-healing: Check if cones actually exist in the world
        world_cone_count = self.__world.get_cone_count()
        if world_cone_count >= len(self.__cone_config):
            self.__cones_spawned = True
            return

        if self.__verbose:
            print(f"[Port {self.__port}] Cones missing or map reloaded. Spawning {len(self.__cone_config)} cones...")
            
        for cone in self.__cone_config:
            self.__world.spawn_cones_from_positions([{'x': cone['x'], 'y': cone['y'], 'z': cone.get('z', 0.0)}])
        
        self.__cones_spawned = True
        self.__world.tick()

    def get_path_waypoints(self, spacing=2.0, start_loc=None):
        """
        Plans a route that avoids cones using the AdvancedRoutePlanner.
        """
        if start_loc is None:
            start_loc = self.__vehicle.get_location()
        
        target_loc = carla.Location(
            x=self._active_scenario_dict['target_position']['x'], 
            y=self._active_scenario_dict['target_position']['y'], 
            z=self._active_scenario_dict['target_position']['z']
        )
        
        # Freshly check if cones are available to avoid them in planning
        # We also pass the cone configuration to the planner in case it changed
        if self.__cone_config:
            self.__route_planner.set_cones(self.__cone_config)
            
        waypoints = self.__route_planner.plan_route(
            start_loc, 
            target_loc, 
            avoid_cones=True,
            verbose=self.__verbose
        )
        
        if self.__verbose:
            print(f"[Port {self.__port}] Planner returned {len(waypoints)} points.")
            
        # Return as locations list
        if len(waypoints) > 1:
            return waypoints[1:] # Skip the first one as it's at the car's current pose
        return waypoints

    def get_vehicle(self):
        return self.__vehicle

    def place_spectator_above_vehicle(self):
        self.__world.place_spectator_above_location(self.__vehicle.get_location())

    def output_all_waypoints(self, spacing=5):
        waypoints = self.__world.get_map().generate_waypoints(distance=spacing)
        for w in waypoints:
            self.__world.get_world().debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)

    def draw_waypoints(self, waypoints, life_time=10.0):
        for w in waypoints:
            self.__world.get_world().debug.draw_string(w, 'O', draw_shadow=False,
                                                    color=carla.Color(r=255, g=0, b=0), life_time=life_time,
                                                    persistent_lines=True)

    def __start_timer(self):
        self.start_time = time.time()

    def __get_situations(self, scenarios):
        with open(config.ENV_SCENARIOS_FILE, 'r') as f:
            self.situations_dict = json.load(f)
        if scenarios:
            self.situations_dict = {k: v for k, v in self.situations_dict.items() if v['situation'] in scenarios}
        self.situations_list = list(self.situations_dict.keys())

    def __choose_random_situation(self, seed=None):
        if seed: np.random.seed(seed)
        return np.random.choice(self.situations_list)

    def __chose_situation(self, seed):
        return self.__choose_random_situation(seed)
